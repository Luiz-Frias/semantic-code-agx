//! Tree-sitter splitter adapter.

use semantic_code_ports::{CodeChunk, Language, LineSpan, SplitOptions, SplitterPort};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::Instrument;
use tree_sitter::{Node, Parser, Tree};

const DEFAULT_CHUNK_SIZE: usize = 200;
const DEFAULT_CHUNK_OVERLAP: usize = 40;
// Best-effort cap to keep embedding inputs bounded without truncation.
const DEFAULT_MAX_CHUNK_CHARS: usize = 2_500;

/// Tree-sitter based splitter with line-based fallback.
#[derive(Debug)]
pub struct TreeSitterSplitter {
    chunk_size: AtomicUsize,
    chunk_overlap: AtomicUsize,
    max_chunk_chars: AtomicUsize,
}

impl Default for TreeSitterSplitter {
    fn default() -> Self {
        Self::new(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    }
}

impl TreeSitterSplitter {
    /// Create a splitter with explicit chunk sizing.
    pub const fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: AtomicUsize::new(chunk_size),
            chunk_overlap: AtomicUsize::new(chunk_overlap),
            max_chunk_chars: AtomicUsize::new(DEFAULT_MAX_CHUNK_CHARS),
        }
    }

    /// Configure the maximum number of characters allowed per chunk.
    pub fn set_max_chunk_chars(&self, max_chunk_chars: usize) {
        self.max_chunk_chars
            .store(max_chunk_chars, Ordering::Relaxed);
    }
}

impl SplitterPort for TreeSitterSplitter {
    fn split(
        &self,
        ctx: &RequestContext,
        code: Box<str>,
        language: Language,
        options: SplitOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CodeChunk>>> {
        let chunk_size = self.chunk_size.load(Ordering::Relaxed);
        let chunk_overlap = self.chunk_overlap.load(Ordering::Relaxed);
        let max_chunk_chars = self.max_chunk_chars.load(Ordering::Relaxed);
        let ctx = ctx.clone();
        let span = tracing::info_span!("adapter.splitter.split", language = ?language);
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("splitter.start")?;
                let config = SplitConfig {
                    chunk_size,
                    chunk_overlap,
                };
                config.validate()?;

                let lines = collect_lines(code.as_ref());
                let line_lengths = line_lengths(&lines);
                let total_lines = line_count(lines.len())?;

                let mut ranges = parse_tree(code.as_ref(), language, options.file_path.as_deref())
                    .map_or_else(
                        || split_range(1, total_lines, config.chunk_size, total_lines),
                        |tree| {
                            let spans = spans_from_tree(&tree, total_lines);
                            if spans.is_empty() {
                                split_range(1, total_lines, config.chunk_size, total_lines)
                            } else {
                                merge_ranges(spans, config.chunk_size, total_lines)
                            }
                        },
                    );

                ranges = apply_overlap(ranges, config.chunk_overlap, total_lines);
                ranges = split_ranges_by_char_limit(ranges, &line_lengths, max_chunk_chars);
                ranges = dedupe_ranges_preserve_order(ranges);

                build_chunks(
                    &ctx,
                    &lines,
                    ranges,
                    language,
                    options.file_path.as_deref(),
                    max_chunk_chars,
                )
            }
            .instrument(span),
        )
    }

    fn set_chunk_size(&self, chunk_size: usize) {
        self.chunk_size.store(chunk_size, Ordering::Relaxed);
    }

    fn set_chunk_overlap(&self, chunk_overlap: usize) {
        self.chunk_overlap.store(chunk_overlap, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitConfig {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl SplitConfig {
    fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "chunk size must be greater than zero",
            ));
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "chunk overlap must be smaller than chunk size",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanRange {
    start: u32,
    end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkContentFragment {
    content: Box<str>,
    start_byte: u32,
    end_byte: u32,
}

fn parse_tree(code: &str, language: Language, file_path: Option<&str>) -> Option<Tree> {
    let ts_language = tree_sitter_language(language, file_path)?;
    let mut parser = Parser::new();
    if parser.set_language(&ts_language).is_err() {
        return None;
    }
    parser.parse(code, None)
}

fn tree_sitter_language(
    language: Language,
    file_path: Option<&str>,
) -> Option<tree_sitter::Language> {
    match language {
        Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
        Language::Go => Some(tree_sitter_go::LANGUAGE.into()),
        Language::Java => Some(tree_sitter_java::LANGUAGE.into()),
        Language::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
        Language::TypeScript => {
            if is_tsx(file_path) {
                Some(tree_sitter_typescript::LANGUAGE_TSX.into())
            } else {
                Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            }
        },
        Language::Python => Some(tree_sitter_python::LANGUAGE.into()),
        Language::C => Some(tree_sitter_c::LANGUAGE.into()),
        Language::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
        _ => None,
    }
}

fn is_tsx(file_path: Option<&str>) -> bool {
    let Some(file_path) = file_path else {
        return false;
    };
    let Some((_, ext)) = file_path.rsplit_once('.') else {
        return false;
    };
    ext.eq_ignore_ascii_case("tsx")
}

fn collect_lines(code: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = code.split_inclusive('\n').collect();
    if lines.is_empty() {
        lines.push("");
    }
    lines
}

fn line_count(lines: usize) -> Result<u32> {
    u32::try_from(lines).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "line count overflow",
            ErrorClass::NonRetriable,
        )
    })
}

fn spans_from_tree(tree: &Tree, total_lines: u32) -> Vec<SpanRange> {
    let root = tree.root_node();
    let mut spans = Vec::new();
    collect_spans(root, total_lines, &mut spans);
    spans
}

fn collect_spans(node: Node<'_>, total_lines: u32, spans: &mut Vec<SpanRange>) {
    if is_function_like_node(node) {
        if let Some(span) = span_from_node(node, total_lines) {
            spans.push(span);
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_spans(child, total_lines, spans);
    }
}

fn is_function_like_node(node: Node<'_>) -> bool {
    match node.kind() {
        "decorated_definition" => is_decorated_function_definition(node),
        kind => matches!(
            kind,
            "function_item"
                | "function_declaration"
                | "function_expression"
                | "function_definition"
                | "arrow_function"
                | "method_declaration"
                | "method_definition"
                | "constructor_declaration"
        ),
    }
}

fn is_decorated_function_definition(node: Node<'_>) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .any(|child| child.kind() == "function_definition")
}

fn span_from_node(node: Node<'_>, total_lines: u32) -> Option<SpanRange> {
    let start = to_u32(node.start_position().row).saturating_add(1);
    let mut end = to_u32(node.end_position().row).saturating_add(1);
    if node.end_position().column == 0 && end > start {
        end = end.saturating_sub(1);
    }

    if total_lines == 0 {
        return None;
    }

    let start = start.clamp(1, total_lines);
    let end = end.clamp(start, total_lines);
    Some(SpanRange { start, end })
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn merge_ranges(mut spans: Vec<SpanRange>, chunk_size: usize, total_lines: u32) -> Vec<SpanRange> {
    if spans.is_empty() {
        return spans;
    }

    spans.sort_by_key(|span| span.start);
    let mut output = Vec::new();
    let mut current: Option<SpanRange> = None;

    for span in spans {
        let span = clamp_range(span, total_lines);
        let span_len = span.end.saturating_sub(span.start).saturating_add(1);
        if span_len as usize > chunk_size {
            if let Some(value) = current.take() {
                output.push(value);
            }
            output.extend(split_range(span.start, span.end, chunk_size, total_lines));
            continue;
        }

        match current {
            None => current = Some(span),
            Some(value) => {
                let proposed_end = value.end.max(span.end);
                let proposed_len = proposed_end.saturating_sub(value.start).saturating_add(1);
                if proposed_len as usize > chunk_size {
                    output.push(value);
                    current = Some(span);
                } else {
                    current = Some(SpanRange {
                        start: value.start,
                        end: proposed_end,
                    });
                }
            },
        }
    }

    if let Some(value) = current {
        output.push(value);
    }

    if output.is_empty() {
        return split_range(1, total_lines, chunk_size, total_lines);
    }

    fill_range_gaps(output, chunk_size, total_lines)
}

fn fill_range_gaps(
    mut ranges: Vec<SpanRange>,
    chunk_size: usize,
    total_lines: u32,
) -> Vec<SpanRange> {
    if ranges.is_empty() || total_lines == 0 {
        return ranges;
    }

    ranges.sort_by_key(|range| range.start);
    let mut output = Vec::new();
    let mut cursor = 1u32;

    for range in ranges {
        let range = clamp_range(range, total_lines);
        if cursor < range.start {
            output.extend(split_range(
                cursor,
                range.start.saturating_sub(1),
                chunk_size,
                total_lines,
            ));
        }
        output.push(range);
        cursor = cursor.max(range.end.saturating_add(1));
    }

    if cursor <= total_lines {
        output.extend(split_range(cursor, total_lines, chunk_size, total_lines));
    }

    output
}

fn line_lengths(lines: &[&str]) -> Vec<usize> {
    lines.iter().map(|line| line.len()).collect()
}

fn split_ranges_by_char_limit(
    ranges: Vec<SpanRange>,
    line_lengths: &[usize],
    max_chars: usize,
) -> Vec<SpanRange> {
    if ranges.is_empty() || max_chars == 0 {
        return ranges;
    }

    let mut output = Vec::new();
    for range in ranges {
        let mut current_start = range.start;
        let mut current_len = 0usize;
        let mut line = range.start;

        while line <= range.end {
            let idx = (line.saturating_sub(1)) as usize;
            let len = line_lengths.get(idx).copied().unwrap_or(0);

            if len > max_chars {
                if current_len > 0 {
                    output.push(SpanRange {
                        start: current_start,
                        end: line.saturating_sub(1),
                    });
                    current_len = 0;
                }
                output.push(SpanRange {
                    start: line,
                    end: line,
                });
                current_start = line.saturating_add(1);
                line = line.saturating_add(1);
                continue;
            }

            if current_len > 0 && current_len + len > max_chars {
                output.push(SpanRange {
                    start: current_start,
                    end: line.saturating_sub(1),
                });
                current_start = line;
                current_len = 0;
            }

            current_len += len;
            line = line.saturating_add(1);
        }

        if current_len > 0 && current_start <= range.end {
            output.push(SpanRange {
                start: current_start,
                end: range.end,
            });
        }
    }

    output
}

fn split_content_by_max_chars(
    content: &str,
    max_chars: usize,
) -> Result<Vec<ChunkContentFragment>> {
    if max_chars == 0 || content.len() <= max_chars {
        return Ok(vec![ChunkContentFragment {
            content: content.to_owned().into_boxed_str(),
            start_byte: 0,
            end_byte: usize_to_u32(content.len())?,
        }]);
    }

    let mut output = Vec::new();
    let mut start = 0usize;
    while start < content.len() {
        let mut end = (start + max_chars).min(content.len());
        while end > start && !content.is_char_boundary(end) {
            end = end.saturating_sub(1);
        }
        if end == start {
            end = start.saturating_add(1);
            while end < content.len() && !content.is_char_boundary(end) {
                end = end.saturating_add(1);
            }
        }

        output.push(ChunkContentFragment {
            content: content[start..end].to_owned().into_boxed_str(),
            start_byte: usize_to_u32(start)?,
            end_byte: usize_to_u32(end)?,
        });
        start = end;
    }

    Ok(output)
}

fn dedupe_ranges_preserve_order(ranges: Vec<SpanRange>) -> Vec<SpanRange> {
    let mut seen = BTreeSet::new();
    let mut deduped = Vec::with_capacity(ranges.len());

    for range in ranges {
        if seen.insert((range.start, range.end)) {
            deduped.push(range);
        }
    }

    deduped
}

fn split_range(start: u32, end: u32, chunk_size: usize, total_lines: u32) -> Vec<SpanRange> {
    if total_lines == 0 {
        return Vec::new();
    }

    let chunk_size = u32::try_from(chunk_size).unwrap_or(u32::MAX).max(1);
    let end = end.clamp(1, total_lines);
    let mut current = start.clamp(1, end);
    let mut output = Vec::new();

    while current <= end {
        let mut chunk_end = current.saturating_add(chunk_size).saturating_sub(1);
        if chunk_end > end {
            chunk_end = end;
        }
        output.push(SpanRange {
            start: current,
            end: chunk_end,
        });
        if chunk_end == end {
            break;
        }
        current = chunk_end.saturating_add(1);
    }

    output
}

fn apply_overlap(spans: Vec<SpanRange>, chunk_overlap: usize, total_lines: u32) -> Vec<SpanRange> {
    if spans.len() <= 1 || chunk_overlap == 0 {
        return spans;
    }

    let overlap = u32::try_from(chunk_overlap).unwrap_or(u32::MAX);
    let mut output = Vec::with_capacity(spans.len());

    for (idx, span) in spans.iter().enumerate() {
        let mut start = span.start;
        if idx > 0
            && let Some(prev) = spans.get(idx - 1)
        {
            let candidate = prev.end.saturating_sub(overlap).saturating_add(1);
            if candidate < start {
                start = candidate;
            }
        }
        start = start.clamp(1, span.end);
        let end = span.end.clamp(start, total_lines);
        output.push(SpanRange { start, end });
    }

    output
}

fn clamp_range(span: SpanRange, total_lines: u32) -> SpanRange {
    let start = span.start.clamp(1, total_lines);
    let end = span.end.clamp(start, total_lines);
    SpanRange { start, end }
}

fn build_chunks(
    ctx: &RequestContext,
    lines: &[&str],
    spans: Vec<SpanRange>,
    language: Language,
    file_path: Option<&str>,
    max_chunk_chars: usize,
) -> Result<Vec<CodeChunk>> {
    let file_path = file_path.map(|value| value.to_owned().into_boxed_str());
    let mut chunks = Vec::with_capacity(spans.len());
    for span in spans {
        ctx.ensure_not_cancelled("splitter.build_chunks")?;
        let content = content_for_span(lines, span)?;
        let line_span = LineSpan::new(span.start, span.end).map_err(ErrorEnvelope::from)?;
        let fragments = split_content_by_max_chars(content.as_ref(), max_chunk_chars)?;
        let mark_fragment_offsets = fragments.len() > 1;
        for fragment in fragments {
            chunks.push(CodeChunk {
                content: fragment.content,
                span: line_span,
                fragment_start_byte: mark_fragment_offsets.then_some(fragment.start_byte),
                fragment_end_byte: mark_fragment_offsets.then_some(fragment.end_byte),
                language: Some(language),
                file_path: file_path.clone(),
            });
        }
    }
    Ok(chunks)
}

fn content_for_span(lines: &[&str], span: SpanRange) -> Result<Box<str>> {
    let start_idx = to_usize(span.start.saturating_sub(1))?;
    let end_idx = to_usize(span.end)?;
    if start_idx >= lines.len() || end_idx == 0 || end_idx > lines.len() {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "chunk span out of bounds",
            ErrorClass::NonRetriable,
        ));
    }

    let mut out = String::new();
    let Some(slice) = lines.get(start_idx..end_idx) else {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "chunk span out of bounds",
            ErrorClass::NonRetriable,
        ));
    };

    for line in slice {
        out.push_str(line);
    }
    Ok(out.into_boxed_str())
}

fn to_usize(value: u32) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "index conversion overflow",
            ErrorClass::NonRetriable,
        )
    })
}

fn usize_to_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "fragment byte offset conversion overflow",
            ErrorClass::NonRetriable,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_shared::Result;

    #[tokio::test]
    async fn chunk_overlap_applies_to_line_fallback() -> Result<()> {
        let splitter = TreeSitterSplitter::new(2, 1);
        let ctx = RequestContext::new_request();
        let code = "a\nb\nc\nd\n".into();
        let chunks = splitter
            .split(&ctx, code, Language::Text, SplitOptions::default())
            .await?;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].span.start_line(), 1);
        assert_eq!(chunks[0].span.end_line(), 2);
        assert_eq!(chunks[1].span.start_line(), 2);
        assert_eq!(chunks[1].span.end_line(), 4);
        Ok(())
    }

    #[test]
    fn split_ranges_by_char_limit_splits_ranges() {
        let ranges = vec![SpanRange { start: 1, end: 3 }];
        let line_lengths = vec![3, 3, 3];
        let out = split_ranges_by_char_limit(ranges, &line_lengths, 6);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].start, 1);
        assert_eq!(out[0].end, 2);
        assert_eq!(out[1].start, 3);
        assert_eq!(out[1].end, 3);
    }

    #[test]
    fn split_content_by_max_chars_tracks_fragment_offsets() -> Result<()> {
        let fragments = split_content_by_max_chars("abcdefghij", 4)?;

        assert_eq!(
            fragments,
            vec![
                ChunkContentFragment {
                    content: "abcd".into(),
                    start_byte: 0,
                    end_byte: 4,
                },
                ChunkContentFragment {
                    content: "efgh".into(),
                    start_byte: 4,
                    end_byte: 8,
                },
                ChunkContentFragment {
                    content: "ij".into(),
                    start_byte: 8,
                    end_byte: 10,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn dedupe_ranges_preserves_order_for_identical_spans() {
        let ranges = vec![
            SpanRange { start: 1, end: 3 },
            SpanRange { start: 4, end: 6 },
            SpanRange { start: 4, end: 6 },
            SpanRange { start: 7, end: 8 },
        ];

        let out = dedupe_ranges_preserve_order(ranges);

        assert_eq!(
            out,
            vec![
                SpanRange { start: 1, end: 3 },
                SpanRange { start: 4, end: 6 },
                SpanRange { start: 7, end: 8 },
            ]
        );
    }

    #[test]
    fn overlap_and_char_limit_pipeline_dedupes_identical_spans() {
        let line_lengths = vec![10; 8];
        let ranges = vec![
            SpanRange { start: 1, end: 6 },
            SpanRange { start: 7, end: 8 },
        ];

        let overlapped = apply_overlap(ranges, 3, 8);
        let split = split_ranges_by_char_limit(overlapped, &line_lengths, 30);
        let out = dedupe_ranges_preserve_order(split);

        assert_eq!(
            out,
            vec![
                SpanRange { start: 1, end: 3 },
                SpanRange { start: 4, end: 6 },
                SpanRange { start: 7, end: 8 },
            ]
        );
    }

    #[test]
    fn dedupe_ranges_keeps_overlapping_but_distinct_spans() {
        let ranges = vec![
            SpanRange { start: 1, end: 3 },
            SpanRange { start: 3, end: 5 },
            SpanRange { start: 5, end: 7 },
        ];

        let out = dedupe_ranges_preserve_order(ranges);

        assert_eq!(
            out,
            vec![
                SpanRange { start: 1, end: 3 },
                SpanRange { start: 3, end: 5 },
                SpanRange { start: 5, end: 7 },
            ]
        );
    }

    #[tokio::test]
    async fn long_line_is_split_below_max_chars() -> Result<()> {
        let splitter = TreeSitterSplitter::new(2, 0);
        splitter.set_max_chunk_chars(50);
        let ctx = RequestContext::new_request();
        let line = "a".repeat(123);
        let chunks = splitter
            .split(
                &ctx,
                line.into_boxed_str(),
                Language::Text,
                SplitOptions::default(),
            )
            .await?;

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content.len(), 50);
        assert_eq!(chunks[1].content.len(), 50);
        assert_eq!(chunks[2].content.len(), 23);
        assert!(chunks.iter().all(|chunk| chunk.content.len() <= 50));
        assert_eq!(chunks[0].fragment_start_byte, Some(0));
        assert_eq!(chunks[0].fragment_end_byte, Some(50));
        assert_eq!(chunks[1].fragment_start_byte, Some(50));
        assert_eq!(chunks[1].fragment_end_byte, Some(100));
        assert_eq!(chunks[2].fragment_start_byte, Some(100));
        assert_eq!(chunks[2].fragment_end_byte, Some(123));
        Ok(())
    }

    #[test]
    fn merge_ranges_fills_gaps_around_function_spans() {
        let spans = vec![SpanRange { start: 5, end: 7 }];
        let out = merge_ranges(spans, 10, 12);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].start, 1);
        assert_eq!(out[0].end, 4);
        assert_eq!(out[1].start, 5);
        assert_eq!(out[1].end, 7);
        assert_eq!(out[2].start, 8);
        assert_eq!(out[2].end, 12);
    }

    #[test]
    fn spans_from_tree_recurses_into_classes_and_methods() {
        let code = r"
struct Calculator;

impl Calculator {
    fn multiply(&self, lhs: i32, rhs: i32) -> i32 {
        let product = lhs * rhs;
        product
    }

    fn divide(&self, lhs: i32, rhs: i32) -> i32 {
        let quotient = lhs / rhs;
        quotient
    }
}

fn top_level() -> i32 {
    42
}
";
        let total_lines = line_count(code.lines().count()).expect("code line count");
        let tree = parse_tree(code, Language::Rust, Some("src/lib.rs")).expect("parse tree");

        let spans = spans_from_tree(&tree, total_lines);
        let multiply_line = code
            .lines()
            .position(|line| line.contains("fn multiply"))
            .expect("multiply function") as u32
            + 1;
        let divide_line = code
            .lines()
            .position(|line| line.contains("fn divide"))
            .expect("divide function") as u32
            + 1;
        let top_level_line = code
            .lines()
            .position(|line| line.contains("fn top_level"))
            .expect("top-level function") as u32
            + 1;
        let struct_line = code
            .lines()
            .position(|line| line.contains("struct Calculator"))
            .expect("struct declaration") as u32
            + 1;

        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].start, multiply_line);
        assert!(spans[0].end < divide_line);
        assert_eq!(spans[1].start, divide_line);
        assert!(spans[1].end < top_level_line);
        assert_eq!(spans[2].start, top_level_line);
        assert_ne!(spans[2].start, struct_line);
    }

    #[test]
    fn spans_from_tree_detects_javascript_arrow_functions() {
        let code = r#"
export const handler = async () => {
    return 42;
};

const helper = function () {
    return "ok";
};
"#;
        let total_lines = line_count(code.lines().count()).expect("code line count");
        let tree =
            parse_tree(code, Language::JavaScript, Some("src/handler.js")).expect("parse tree");

        let spans = spans_from_tree(&tree, total_lines);
        let handler_line = code
            .lines()
            .position(|line| line.contains("handler = async () =>"))
            .expect("handler function") as u32
            + 1;
        let helper_line = code
            .lines()
            .position(|line| line.contains("helper = function"))
            .expect("helper function") as u32
            + 1;

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start, handler_line);
        assert_eq!(spans[1].start, helper_line);
    }

    #[test]
    fn spans_from_tree_keeps_python_decorator_context() {
        let code = r#"
@app.route("/api")
@login_required
def handler():
    return "ok"
"#;
        let total_lines = line_count(code.lines().count()).expect("code line count");
        let tree = parse_tree(code, Language::Python, Some("app.py")).expect("parse tree");

        let spans = spans_from_tree(&tree, total_lines);
        let decorator_line = code
            .lines()
            .position(|line| line.contains("@app.route"))
            .expect("decorator line") as u32
            + 1;
        let function_line = code
            .lines()
            .position(|line| line.contains("def handler"))
            .expect("function line") as u32
            + 1;

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, decorator_line);
        assert!(spans[0].end >= function_line);
    }

    #[test]
    fn spans_from_tree_recurses_into_decorated_python_classes() {
        let code = r#"
@dataclass
class Person:
    def greet(self):
        return "hi"
"#;
        let total_lines = line_count(code.lines().count()).expect("code line count");
        let tree = parse_tree(code, Language::Python, Some("person.py")).expect("parse tree");

        let spans = spans_from_tree(&tree, total_lines);
        let class_line = code
            .lines()
            .position(|line| line.contains("class Person"))
            .expect("class line") as u32
            + 1;
        let method_line = code
            .lines()
            .position(|line| line.contains("def greet"))
            .expect("method line") as u32
            + 1;

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, method_line);
        assert_ne!(spans[0].start, class_line);
    }

    #[tokio::test]
    async fn invalid_chunk_config_returns_error() -> Result<()> {
        let splitter = TreeSitterSplitter::new(0, 0);
        let ctx = RequestContext::new_request();
        let result = splitter
            .split(&ctx, "a\n".into(), Language::Text, SplitOptions::default())
            .await;

        let error = result.expect_err("expected invalid chunk size error");
        assert_eq!(error.code, ErrorCode::invalid_input());
        Ok(())
    }
}
