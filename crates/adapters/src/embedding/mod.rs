//! Embedding adapter implementations.

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "voyage")]
pub mod voyage;

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod fixed;
