# Contributing to semantic-code-agx

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- **Rust**: 1.90+ (check `.rust-version`)
- **Tooling**: `mise` for environment setup
- **Git**: For version control

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/Luiz-Frias/semantic-code-agx.git
cd semantic-code-agx

# Install dependencies using mise
mise install

# Set up environment variables
cp .env.example .env.local

# Run tests
cargo test

# Check code quality
just pc-full  # Runs pre-commit hooks + quality gates
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-adapter` for new features
- `fix/resolve-indexing-bug` for bug fixes
- `refactor/improve-error-handling` for refactoring
- `docs/add-architecture-guide` for documentation

### 2. Make Your Changes

Follow these principles:

#### Code Style
- Use `rustfmt` for formatting (enforced by pre-commit)
- Follow the naming conventions in the existing codebase
- Keep lines under 100 characters where practical
- Use meaningful variable and function names

#### Commits
- Use conventional commit format: `type(scope): description`
- Examples:
  - `feat(adapters): add Ollama embedding provider`
  - `fix(core): prevent duplicate index entries`
  - `refactor(ports): simplify trait bounds`
  - `docs(readme): add quick start example`

#### Testing
- Write tests for all new functionality
- Ensure all tests pass: `cargo test`
- Add integration tests for public APIs
- Test edge cases and error paths

#### Documentation
- Update relevant documentation files
- Add doc comments to public APIs
- Include examples in doc comments where helpful

### 3. Run Quality Checks

```bash
# Run full quality gate
just pc-full

# Or run individual checks:
cargo fmt --check        # Code formatting
cargo clippy             # Linting
cargo test               # Unit tests
cargo test --test '*'    # Integration tests
```

### 4. Create a Pull Request

Before submitting:

1. **Rebase your branch** on the latest `main`
2. **Push to your fork**
3. **Create a pull request** with:
   - Clear description of changes
   - Reference to any related issues (#123)
   - List of testing done

### PR Requirements

- All tests must pass
- No clippy warnings
- Code must be formatted with rustfmt
- PR must be reviewed and approved by at least one maintainer
- Documentation must be updated if behavior changed

## Architecture & Design Principles

Before contributing, familiarize yourself with:

- **Hexagonal Architecture**: See `docs/architecture/`
- **Error Handling**: See `docs/reference/error-codes.md`
- **Port Traits**: See `docs/reference/ports.md`
- **Type Safety**: All code uses strong typing; avoid `String` for structured data

Key guidelines:

- **Prefer composition over inheritance**
- **Use Result types** for fallible operations
- **Validate at boundaries** (user input, external APIs)
- **No silent errors**: Log all errors with context
- **Avoid unsafe code** unless absolutely necessary with documentation

## Adding a New Embedding Provider

If you're adding a new embedding adapter:

1. Create a module in `crates/adapters/src/embedding/`
2. Implement the `Embedder` port trait
3. Add configuration in `crates/config/src/schema.rs`
4. Add tests in the module
5. Update documentation in `docs/guides/embedding-providers.md`

See `crates/adapters/src/embedding/openai.rs` as an example.

## Adding a New Vector Database

If you're adding a new vector database adapter:

1. Create a module in `crates/adapters/src/vectordb/`
2. Implement the `VectorDb` port trait
3. Add configuration support
4. Add tests and fixtures
5. Update documentation in `docs/internals/adapters/vectordb.md`

See `crates/adapters/src/vectordb/milvus/` as an example.

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_description() {
        // Arrange
        let input = setup();

        // Act
        let result = perform_action(&input);

        // Assert
        assert_eq!(result, expected_value);
    }
}
```

### Integration Tests

Place integration tests in `tests/` directory at crate root.

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture

# Specific crate
cargo test -p crate-name

# With nextest (faster parallel execution)
cargo nextest run
```

## Documentation

### Updating Existing Docs

- Edit markdown files in `docs/`
- Use clear headings and code blocks
- Include examples where helpful
- Keep technical accuracy

### Writing New Docs

- Use `docs/templates/` as starting point
- Follow the Diataxis structure (tutorials, how-tos, reference, explanation)
- Include examples and diagrams where beneficial
- Link to related documentation

## Code Review Expectations

When your PR is reviewed:

- **Be open to feedback**: Reviewers are helping improve the code
- **Ask questions**: If feedback is unclear, ask for clarification
- **Respond promptly**: Keep the review process moving
- **Make requested changes**: Implement reviewer feedback in new commits

## Release Process

The maintainers follow semantic versioning:

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Changes are released periodically based on accumulated features and fixes.

## Code of Conduct

Please note that this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

- Open a GitHub Discussion for questions
- Check existing Issues and PRs for similar topics
- Read the documentation in `docs/`

Thank you for contributing!
