//! Secret detection and redaction utilities.
//!
//! Provides consistent logic for detecting sensitive keys/variables and
//! redacting their values in error messages and logs.

/// Checks if a key/variable name likely refers to a secret.
///
/// Uses case-insensitive pattern matching to detect common secret-related
/// naming conventions.
///
/// # Examples
///
/// ```
/// use semantic_code_shared::is_secret_key;
///
/// assert!(is_secret_key("API_KEY"));
/// assert!(is_secret_key("password"));
/// assert!(is_secret_key("SCA_EMBEDDING_API_AUTH"));
/// assert!(!is_secret_key("LOG_LEVEL"));
/// ```
pub fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    key.contains("KEY")
        || key.contains("TOKEN")
        || key.contains("SECRET")
        || key.contains("PASSWORD")
        || key.contains("CREDENTIAL")
        || key.contains("AUTH")
}

/// Redacts a value if the key is likely a secret.
///
/// Returns `"[REDACTED]"` for secret keys, or the original value otherwise.
///
/// # Examples
///
/// ```
/// use semantic_code_shared::redact_if_secret;
///
/// assert_eq!(redact_if_secret("API_KEY", "sk-123"), "[REDACTED]");
/// assert_eq!(redact_if_secret("LOG_LEVEL", "debug"), "debug");
/// ```
pub fn redact_if_secret(key: &str, value: &str) -> String {
    if is_secret_key(key) {
        "[REDACTED]".to_string()
    } else {
        value.to_string()
    }
}

/// The redacted placeholder string.
pub const REDACTED: &str = "[REDACTED]";

/// A secret string wrapper that redacts on Display/Debug.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SecretString(Box<str>);

impl SecretString {
    /// Wrap a secret value.
    pub fn new(value: impl Into<Box<str>>) -> Self {
        Self(value.into())
    }

    /// Borrow the underlying secret.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying secret.
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REDACTED)
    }
}

impl std::fmt::Display for SecretString {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REDACTED)
    }
}

impl AsRef<str> for SecretString {
    fn as_ref(&self) -> &str {
        self.expose()
    }
}

impl From<Box<str>> for SecretString {
    fn from(value: Box<str>) -> Self {
        Self(value)
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value.into_boxed_str())
    }
}

/// Generic wrapper that redacts on Display/Debug.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Redacted<T>(T);

impl<T> Redacted<T> {
    /// Wrap a value that must be redacted in logs.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Borrow the underlying value.
    pub const fn expose(&self) -> &T {
        &self.0
    }

    /// Consume and return the underlying value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::fmt::Debug for Redacted<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REDACTED)
    }
}

impl<T> std::fmt::Display for Redacted<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(REDACTED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_secret_patterns() {
        // API keys
        assert!(is_secret_key("API_KEY"));
        assert!(is_secret_key("api_key"));
        assert!(is_secret_key("OPENAI_API_KEY"));

        // Tokens
        assert!(is_secret_key("ACCESS_TOKEN"));
        assert!(is_secret_key("refresh_token"));
        assert!(is_secret_key("JWT_TOKEN"));

        // Secrets
        assert!(is_secret_key("CLIENT_SECRET"));
        assert!(is_secret_key("secret_value"));

        // Passwords
        assert!(is_secret_key("DB_PASSWORD"));
        assert!(is_secret_key("user_password"));

        // Credentials
        assert!(is_secret_key("AWS_CREDENTIAL"));
        assert!(is_secret_key("credentials"));

        // Auth
        assert!(is_secret_key("SCA_EMBEDDING_API_AUTH"));
        assert!(is_secret_key("basic_auth"));
    }

    #[test]
    fn rejects_non_secret_patterns() {
        assert!(!is_secret_key("LOG_LEVEL"));
        assert!(!is_secret_key("PORT"));
        assert!(!is_secret_key("DATABASE_URL"));
        assert!(!is_secret_key("TIMEOUT_MS"));
        assert!(!is_secret_key("MAX_RETRIES"));
    }

    #[test]
    fn redacts_secret_values() {
        assert_eq!(redact_if_secret("API_KEY", "sk-123456"), REDACTED);
        assert_eq!(redact_if_secret("password", "hunter2"), REDACTED);
    }

    #[test]
    fn preserves_non_secret_values() {
        assert_eq!(redact_if_secret("LOG_LEVEL", "debug"), "debug");
        assert_eq!(redact_if_secret("PORT", "8080"), "8080");
    }

    #[test]
    fn secret_string_redacts_display() {
        let secret = SecretString::new("shh");
        assert_eq!(secret.to_string(), REDACTED);
    }
}
