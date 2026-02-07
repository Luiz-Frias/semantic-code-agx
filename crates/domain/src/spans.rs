//! Span and language helpers for domain metadata.

use crate::primitives::PrimitiveError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Inclusive line span with 1-indexed boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineSpan {
    start_line: u32,
    end_line: u32,
}

impl LineSpan {
    /// Construct a validated line span.
    pub const fn new(start_line: u32, end_line: u32) -> Result<Self, PrimitiveError> {
        if start_line == 0 || end_line == 0 {
            return Err(PrimitiveError::LineSpanNonPositive {
                start_line,
                end_line,
            });
        }

        if start_line > end_line {
            return Err(PrimitiveError::LineSpanStartAfterEnd {
                start_line,
                end_line,
            });
        }

        Ok(Self {
            start_line,
            end_line,
        })
    }

    /// Returns the starting line (1-indexed).
    #[must_use]
    pub const fn start_line(&self) -> u32 {
        self.start_line
    }

    /// Returns the ending line (1-indexed).
    #[must_use]
    pub const fn end_line(&self) -> u32 {
        self.end_line
    }
}

/// Canonical language identifiers derived from file extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// `TypeScript` source files.
    TypeScript,
    /// `JavaScript` source files.
    JavaScript,
    /// `Python` source files.
    Python,
    /// `Java` source files.
    Java,
    /// `Cpp` source files.
    Cpp,
    /// `C` source files.
    C,
    /// `CSharp` source files.
    CSharp,
    /// `Go` source files.
    Go,
    /// `Rust` source files.
    Rust,
    /// `Php` source files.
    Php,
    /// `Ruby` source files.
    Ruby,
    /// `Swift` source files.
    Swift,
    /// `Kotlin` source files.
    Kotlin,
    /// `Scala` source files.
    Scala,
    /// `ObjectiveC` source files.
    #[serde(rename = "objective-c")]
    ObjectiveC,
    /// `Jupyter` notebook files.
    Jupyter,
    /// `Markdown` documents.
    Markdown,
    /// `Text` fallback.
    Text,
}

impl Language {
    /// Returns the canonical string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Python => "python",
            Self::Java => "java",
            Self::Cpp => "cpp",
            Self::C => "c",
            Self::CSharp => "csharp",
            Self::Go => "go",
            Self::Rust => "rust",
            Self::Php => "php",
            Self::Ruby => "ruby",
            Self::Swift => "swift",
            Self::Kotlin => "kotlin",
            Self::Scala => "scala",
            Self::ObjectiveC => "objective-c",
            Self::Jupyter => "jupyter",
            Self::Markdown => "markdown",
            Self::Text => "text",
        }
    }

    /// Derive a language identifier from a file extension.
    #[must_use]
    pub fn from_extension(extension: &str) -> Self {
        let trimmed = extension.trim();
        if trimmed.is_empty() {
            return Self::Text;
        }

        let trimmed = trimmed.trim_start_matches('.');
        if trimmed.is_empty() {
            return Self::Text;
        }

        let lowered = trimmed.to_ascii_lowercase();
        match lowered.as_str() {
            "ts" | "tsx" => Self::TypeScript,
            "js" | "jsx" => Self::JavaScript,
            "py" => Self::Python,
            "java" => Self::Java,
            "cpp" | "hpp" => Self::Cpp,
            "c" | "h" => Self::C,
            "cs" => Self::CSharp,
            "go" => Self::Go,
            "rs" => Self::Rust,
            "php" => Self::Php,
            "rb" => Self::Ruby,
            "swift" => Self::Swift,
            "kt" => Self::Kotlin,
            "scala" => Self::Scala,
            "m" | "mm" => Self::ObjectiveC,
            "ipynb" => Self::Jupyter,
            "md" | "markdown" => Self::Markdown,
            _ => Self::Text,
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn line_span_rejects_non_positive() {
        let error = LineSpan::new(0, 2).err();
        assert!(matches!(
            error,
            Some(PrimitiveError::LineSpanNonPositive { .. })
        ));
    }

    #[test]
    fn line_span_rejects_inverted_bounds() {
        let error = LineSpan::new(3, 2).err();
        assert!(matches!(
            error,
            Some(PrimitiveError::LineSpanStartAfterEnd { .. })
        ));
    }

    #[test]
    fn language_from_extension_maps_values() {
        assert_eq!(Language::from_extension(".ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("jsx"), Language::JavaScript);
        assert_eq!(Language::from_extension(".rs"), Language::Rust);
        assert_eq!(Language::from_extension(""), Language::Text);
    }

    proptest! {
        #[test]
        fn line_span_accepts_valid_ranges((start, end) in valid_line_span()) {
            let span = LineSpan::new(start, end);
            prop_assert!(span.is_ok());
        }
    }

    fn valid_line_span() -> impl Strategy<Value = (u32, u32)> {
        (1u32..2000, 1u32..2000).prop_map(|(start, end)| {
            if start <= end {
                (start, end)
            } else {
                (end, start)
            }
        })
    }
}
