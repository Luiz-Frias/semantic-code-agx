//! Domain search types and ordering rules.

use crate::{Language, LineSpan};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Provider-specific search filters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchFilter {
    /// Provider-specific filter expression (e.g. Milvus expr).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_expr: Option<Box<str>>,
}

/// Search query payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchQuery {
    /// Query text.
    pub query: Box<str>,
}

/// Search options controlling the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub top_k: u32,
    /// Optional score threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Optional filter expression.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<SearchFilter>,
    /// Optional hint to include content payloads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_content: Option<bool>,
}

/// Deterministic result key used for ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultKey {
    /// Stable logical path identifier.
    pub relative_path: Box<str>,
    /// Line span of the result.
    pub span: LineSpan,
}

/// Search result with deterministic ordering contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    /// Ordering key.
    pub key: SearchResultKey,
    /// Optional chunk content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Box<str>>,
    /// Optional language hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// Similarity score.
    pub score: f32,
}

/// Deterministic ordering contract:
/// 1) score (desc)
/// 2) relativePath (asc)
/// 3) startLine (asc)
/// 4) endLine (asc)
#[must_use]
pub fn compare_search_results(a: &SearchResult, b: &SearchResult) -> Ordering {
    let score_order = b.score.total_cmp(&a.score);
    if score_order != Ordering::Equal {
        return score_order;
    }

    let path_order = a.key.relative_path.cmp(&b.key.relative_path);
    if path_order != Ordering::Equal {
        return path_order;
    }

    let start_order = a.key.span.start_line().cmp(&b.key.span.start_line());
    if start_order != Ordering::Equal {
        return start_order;
    }

    a.key.span.end_line().cmp(&b.key.span.end_line())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn search_result_ordering_matches_contract() -> Result<(), Box<dyn Error>> {
        let span = LineSpan::new(1, 2)?;
        let a = SearchResult {
            key: SearchResultKey {
                relative_path: "b.ts".into(),
                span,
            },
            content: None,
            language: None,
            score: 0.9,
        };
        let b = SearchResult {
            key: SearchResultKey {
                relative_path: "a.ts".into(),
                span,
            },
            content: None,
            language: None,
            score: 0.9,
        };
        let c = SearchResult {
            key: SearchResultKey {
                relative_path: "a.ts".into(),
                span: LineSpan::new(5, 10)?,
            },
            content: None,
            language: None,
            score: 0.9,
        };
        let d = SearchResult {
            key: SearchResultKey {
                relative_path: "a.ts".into(),
                span,
            },
            content: None,
            language: None,
            score: 0.95,
        };

        let mut results = vec![a.clone(), b.clone(), c.clone(), d.clone()];
        results.sort_by(compare_search_results);
        assert_eq!(results, vec![d, b, c, a]);
        Ok(())
    }

    #[test]
    fn search_result_serializes_with_camel_case() -> Result<(), Box<dyn Error>> {
        let span = LineSpan::new(1, 2)?;
        let result = SearchResult {
            key: SearchResultKey {
                relative_path: "a.ts".into(),
                span,
            },
            content: None,
            language: Some(Language::TypeScript),
            score: 0.42,
        };

        let value = serde_json::to_value(&result)?;
        let expected = serde_json::json!({
            "key": {
                "relativePath": "a.ts",
                "span": { "startLine": 1, "endLine": 2 }
            },
            "language": "typescript",
            "score": f64::from(result.score)
        });
        assert_eq!(value, expected);
        Ok(())
    }
}
