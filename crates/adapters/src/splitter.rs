//! Tree-sitter splitter adapter.

use semantic_code_ports::{CodeChunk, Language, LineSpan, SplitOptions, SplitterPort};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
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
        Box::pin(async move {
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

            build_chunks(&ctx, &lines, ranges, language, options.file_path.as_deref())
        })
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

#[derive(Debug, Clone, Copy)]
struct SpanRange {
    start: u32,
    end: u32,
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
    let mut cursor = root.walk();
    let mut spans = Vec::new();
    for child in root.named_children(&mut cursor) {
        if let Some(span) = span_from_node(child, total_lines) {
            spans.push(span);
        }
    }
    spans
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
) -> Result<Vec<CodeChunk>> {
    let file_path = file_path.map(|value| value.to_owned().into_boxed_str());
    let mut chunks = Vec::with_capacity(spans.len());
    for span in spans {
        ctx.ensure_not_cancelled("splitter.build_chunks")?;
        let content = content_for_span(lines, span)?;
        let line_span = LineSpan::new(span.start, span.end).map_err(ErrorEnvelope::from)?;
        chunks.push(CodeChunk {
            content,
            span: line_span,
            language: Some(language),
            file_path: file_path.clone(),
        });
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
