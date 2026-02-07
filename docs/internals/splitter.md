# Splitter Adapter

The splitter adapter uses tree-sitter to chunk source files into semantic
segments before embedding.

## Supported languages

Tree-sitter grammars are enabled for:

- Rust
- Go
- Java
- JavaScript
- TypeScript (TS + TSX via file extension)
- Python
- C
- C++

All other languages fall back to line-based chunking.

## Chunk sizing

- Chunk size and overlap are expressed in **lines**.
- Overlap is applied by extending each chunk backwards by N lines.
- Invalid sizing (size `0`, overlap >= size) returns an `invalid_input` error.

## Fallback behavior

If parsing fails or a grammar is unavailable, the splitter uses line-based
chunks with deterministic ordering.
