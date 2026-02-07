use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Ok = 0,
    InvalidInput = 2,
    Io = 3,
    Internal = 1,
}

impl ExitCode {
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug)]
pub enum CliError {
    InvalidInput(String),
    Io(std::io::Error),
    Serialization(serde_json::Error),
}

impl CliError {
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        match self {
            Self::InvalidInput(_) => ExitCode::InvalidInput,
            Self::Io(_) => ExitCode::Io,
            Self::Serialization(_) => ExitCode::Internal,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Serialization(error) => write!(formatter, "serialization error: {error}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}
