//! Ignore matcher adapter.

use semantic_code_ports::{IgnoreMatchInput, IgnorePort};

/// Ignore matcher with deterministic normalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct IgnoreMatcher;

impl IgnoreMatcher {
    /// Build a default ignore matcher.
    pub const fn new() -> Self {
        Self
    }
}

impl IgnorePort for IgnoreMatcher {
    fn is_ignored(&self, input: &IgnoreMatchInput) -> bool {
        let normalized_path = normalize_path(input.relative_path.as_ref());
        let path_segments = split_segments(&normalized_path);
        if path_segments.is_empty() {
            return false;
        }

        let patterns = normalize_patterns(&input.ignore_patterns);
        if patterns.is_empty() {
            return false;
        }

        for pattern in patterns {
            let pattern_segments = split_segments(&pattern);
            if pattern_segments.is_empty() {
                continue;
            }
            if matches_segments(&path_segments, &pattern_segments) {
                return true;
            }
        }
        false
    }
}

fn normalize_patterns(patterns: &[Box<str>]) -> Vec<String> {
    let mut normalized = patterns
        .iter()
        .filter_map(|pattern| {
            let trimmed = pattern.trim();
            if trimmed.is_empty() {
                return None;
            }
            Some(normalize_path(trimmed))
        })
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn normalize_path(input: &str) -> String {
    let trimmed = input.trim();
    let replaced = trimmed.replace('\\', "/");
    let collapsed = collapse_forward_slashes(&replaced);
    let collapsed = collapsed.trim_start_matches("./");
    collapsed.trim_matches('/').to_owned()
}

fn split_segments(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|segment| !segment.is_empty() && *segment != ".")
        .collect()
}

fn matches_segments(path: &[&str], pattern: &[&str]) -> bool {
    if pattern.len() > path.len() {
        return false;
    }
    for window in path.windows(pattern.len()) {
        if window == pattern {
            return true;
        }
    }
    false
}

fn collapse_forward_slashes(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut previous_was_slash = false;

    for ch in input.chars() {
        if ch == '/' {
            if previous_was_slash {
                continue;
            }
            previous_was_slash = true;
        } else {
            previous_was_slash = false;
        }
        output.push(ch);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::IgnoreMatchInput;

    #[test]
    fn ignores_common_directories() {
        let matcher = IgnoreMatcher::new();
        let patterns = vec!["node_modules/".into(), "target/".into()];

        let input = IgnoreMatchInput {
            ignore_patterns: patterns,
            relative_path: "src/node_modules/pkg/index.js".into(),
        };
        assert!(matcher.is_ignored(&input));
    }

    #[test]
    fn ignores_nested_segment_sequences() {
        let matcher = IgnoreMatcher::new();
        let input = IgnoreMatchInput {
            ignore_patterns: vec!["src/generated".into()],
            relative_path: "src/generated/code.rs".into(),
        };
        assert!(matcher.is_ignored(&input));
    }

    #[test]
    fn ignore_order_is_deterministic() {
        let matcher = IgnoreMatcher::new();
        let input_a = IgnoreMatchInput {
            ignore_patterns: vec!["target/".into(), "node_modules/".into()],
            relative_path: "node_modules/pkg/index.js".into(),
        };
        let input_b = IgnoreMatchInput {
            ignore_patterns: vec!["node_modules/".into(), "target/".into()],
            relative_path: "node_modules/pkg/index.js".into(),
        };
        assert_eq!(matcher.is_ignored(&input_a), matcher.is_ignored(&input_b));
    }

    #[test]
    fn normalizes_windows_separators() {
        let matcher = IgnoreMatcher::new();
        let input = IgnoreMatchInput {
            ignore_patterns: vec!["target/".into()],
            relative_path: "target\\out\\file.txt".into(),
        };
        assert!(matcher.is_ignored(&input));
    }
}
