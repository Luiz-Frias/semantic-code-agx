//! # semantic-code-core
//!
//! Core utilities and build information for the semantic-code-agents workspace.
//!
//! This crate provides foundational functionality that has no dependencies on
//! other workspace crates, making it safe to import anywhere.
//!
//! ## Features
//!
//! - [`build_info()`] - Returns build-time metadata about the binary
//! - [`BuildInfo`] - Structured build information

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

/// Build-time information about the binary.
///
/// This struct captures metadata that is determined at compile time and
/// remains constant throughout the lifetime of the running binary.
///
/// # Example
///
/// ```
/// use semantic_code_core::build_info;
///
/// let info = build_info();
/// println!("Running {} v{}", info.name, info.version);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    /// Package name from Cargo.toml
    pub name: &'static str,

    /// Package version from Cargo.toml (semver)
    pub version: &'static str,

    /// Rust compiler version used to build
    pub rustc_version: &'static str,

    /// Target triple (e.g., "x86_64-apple-darwin")
    pub target: &'static str,

    /// Build profile ("debug" or "release")
    pub profile: &'static str,

    /// Git commit hash (short form, if available)
    pub git_hash: Option<&'static str>,

    /// Whether the build had uncommitted changes
    pub git_dirty: bool,
}

impl BuildInfo {
    /// Returns a human-readable version string.
    ///
    /// Format: `name version (git_hash[-dirty])` or `name version` if no git info.
    ///
    /// # Example
    ///
    /// ```
    /// use semantic_code_core::build_info;
    ///
    /// let info = build_info();
    /// // e.g., "semantic-code-core 0.1.0 (abc1234)"
    /// println!("{}", info.version_string());
    /// ```
    #[must_use]
    pub fn version_string(&self) -> String {
        match (self.git_hash, self.git_dirty) {
            (Some(hash), true) => format!("{} {} ({hash}-dirty)", self.name, self.version),
            (Some(hash), false) => format!("{} {} ({hash})", self.name, self.version),
            (None, _) => format!("{} {}", self.name, self.version),
        }
    }

    /// Returns true if this is a debug build.
    #[must_use]
    pub const fn is_debug(&self) -> bool {
        matches!(self.profile.as_bytes(), b"debug")
    }

    /// Returns true if this is a release build.
    #[must_use]
    pub const fn is_release(&self) -> bool {
        matches!(self.profile.as_bytes(), b"release")
    }
}

/// Returns build-time information about the binary.
///
/// This function returns a [`BuildInfo`] struct containing metadata that was
/// captured at compile time. The values are deterministic and will not change
/// during the lifetime of the running process.
///
/// # Example
///
/// ```
/// use semantic_code_core::build_info;
///
/// let info = build_info();
/// assert!(!info.name.is_empty());
/// assert!(!info.version.is_empty());
/// ```
#[must_use]
pub const fn build_info() -> BuildInfo {
    BuildInfo {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        rustc_version: env!("CARGO_PKG_RUST_VERSION"),
        target: target_triple(),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        // Git info would be populated by a build.rs script in the future
        // For now, we leave it as None
        git_hash: option_env!("GIT_HASH"),
        git_dirty: option_env!("GIT_DIRTY").is_some(),
    }
}

/// Returns the target triple at compile time.
///
/// This uses cfg! macros to determine the target platform since
/// env!("TARGET") is only available in build scripts.
const fn target_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        "x86_64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux", target_env = "gnu"))]
    {
        "aarch64-unknown-linux-gnu"
    }

    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
    {
        "x86_64-pc-windows-msvc"
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"),
        all(target_arch = "aarch64", target_os = "linux", target_env = "gnu"),
        all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    )))]
    {
        "unknown"
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// P01.M1.13: `build_info()` returns non-empty fields
    #[test]
    fn build_info_returns_non_empty_fields() {
        let info = build_info();

        assert!(!info.name.is_empty(), "name should not be empty");
        assert!(!info.version.is_empty(), "version should not be empty");
        assert!(
            !info.rustc_version.is_empty(),
            "rustc_version should not be empty"
        );
        assert!(!info.target.is_empty(), "target should not be empty");
        assert!(!info.profile.is_empty(), "profile should not be empty");
    }

    /// P01.M1.14: `build_info()` is deterministic across calls
    #[test]
    fn build_info_is_deterministic() {
        let info1 = build_info();
        let info2 = build_info();

        assert_eq!(info1, info2, "build_info() should return identical values");
    }

    #[test]
    fn version_string_format() {
        let info = build_info();
        let version_str = info.version_string();

        assert!(
            version_str.contains(info.name),
            "version string should contain name"
        );
        assert!(
            version_str.contains(info.version),
            "version string should contain version"
        );
    }

    #[test]
    fn profile_detection() {
        let info = build_info();

        // In test mode, we're always in debug
        assert!(info.is_debug(), "tests run in debug mode");
        assert!(!info.is_release(), "tests should not be release mode");
    }
}
