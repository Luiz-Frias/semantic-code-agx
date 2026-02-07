# Documentation

Welcome to the semantic-code-agx documentation. This guide will help you navigate the project's documentation and find what you need.

## Quick Navigation

### Getting Started
- **[Getting Started](./GETTING_STARTED.md)** - Install and run your first search in 5 minutes
- **[Release & Install](./release.md)** - Release artifacts and install methods
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Common issues and solutions
- **[FAQ](./FAQ.md)** - Frequently asked questions

### Architecture & Design
- **[Architecture Overview](./architecture/README.md)** - System architecture and design principles
- **[Hexagonal Architecture](./architecture/hexagonal.md)** - Ports and adapters pattern
- **[Crate Map](./architecture/crate-map.md)** - Crate organization and dependencies
- **[Data Flow](./architecture/data-flow.md)** - Request lifecycle and sequence diagrams

### User Guides
- **[Indexing Guide](./guides/indexing.md)** - How to index a codebase
- **[Searching Guide](./guides/searching.md)** - How to perform semantic searches
- **[Configuration Guide](./guides/configuration.md)** - Configuration deep-dive
- **[Embedding Providers](./guides/embedding-providers.md)** - Setting up embedding services

### Reference
- **[REST API Reference](./reference/api-v1.md)** - API v1 endpoints and schemas
- **[CLI Reference](./reference/cli.md)** - Command-line interface
- **[CLI Usage](./reference/cli-usage.md)** - End-to-end CLI flows
- **[CLI Commands](./reference/cli-commands.md)** - Command summary + flags
- **[Agent Usage](./reference/agent-usage.md)** - Agent-friendly CLI patterns
- **[Configuration Reference](./reference/config-schema.md)** - Configuration schema
- **[Environment Variables](./reference/env-vars.md)** - Available env vars
- **[Error Codes](./reference/error-codes.md)** - Error codes and meanings
- **[Port Traits](./reference/ports.md)** - Port interface definitions
- **[Observability](./observability.md)** - Structured logs, telemetry, and sampling
- **[Security](./security.md)** - Path policy, state dirs, and redaction rules

### For Developers
- **[Contributing](../CONTRIBUTING.md)** - How to contribute to the project
- **[Architecture Decision Records](./adrs/README.md)** - Why key decisions were made
- **[Internals](./internals/)** - Deep-dive into implementation details:
  - [Embedding Adapters](./internals/adapters/embedding.md)
  - [Vector Database Adapters](./internals/adapters/vectordb.md)
  - [Filesystem Adapter](./internals/adapters/filesystem.md)
  - [Code Splitter](./internals/splitter.md)
  - [Vector Kernel](./internals/vector-kernel.md)
  - [Resilience Patterns](./internals/resilience.md)
  - [Concurrency](./internals/concurrency.md)
  - [Request Validation](./internals/validation.md)

### Project
- **[Changelog](../CHANGELOG.md)** - Version history and changes
- **[Security Policy](../SECURITY.md)** - How to report security vulnerabilities
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community guidelines
- **[License](../LICENSE)** - Dual MIT/Apache-2.0 license

## Documentation Structure

This documentation follows the [Diataxis framework](https://diataxis.fr/), organizing content into four categories:

### 📚 Tutorials (Getting Started)
Step-by-step guides for newcomers. Start here if you're new to the project.

### 🎯 How-To Guides (Guides)
Practical instructions for specific tasks like indexing or configuration.

### 📖 Reference (Reference)
Complete API documentation, schemas, and specifications.

### 💡 Explanation (Architecture)
Conceptual documentation explaining design decisions and architecture.

## Finding Information

**I want to...**
- **...get started quickly** → [Getting Started](./GETTING_STARTED.md)
- **...understand the architecture** → [Architecture Overview](./architecture/README.md)
- **...index my codebase** → [Indexing Guide](./guides/indexing.md)
- **...search semantically** → [Searching Guide](./guides/searching.md)
- **...solve a problem** → [Troubleshooting](./TROUBLESHOOTING.md)
- **...find an API endpoint** → [REST API Reference](./reference/api-v1.md)
- **...configure the system** → [Configuration Guide](./guides/configuration.md)
- **...contribute code** → [Contributing](../CONTRIBUTING.md)
- **...understand a design decision** → [ADRs](./adrs/README.md)

## Conventions

### Code Examples
Code examples show the most common usage patterns. For complete examples, see:
- **CLI examples** in [CLI Reference](./reference/cli.md)
- **Agent workflows** in [Agent Usage](./reference/agent-usage.md)
- **API examples** in [REST API Reference](./reference/api-v1.md)
- **Configuration examples** in [Configuration Reference](./reference/config-schema.md)

### Links
- Internal documentation links use relative paths: `[Link](./path/to/doc.md)`
- External links show the full URL: `https://example.com/page`

## Feedback

- **Found a bug in the docs?** Open an issue on GitHub
- **Want to improve a guide?** See [Contributing](../CONTRIBUTING.md)
- **Have a question?** Check [FAQ](./FAQ.md) or open a discussion

## Latest Changes

See [CHANGELOG](../CHANGELOG.md) for recent updates and new features.

---

**Last updated**: February 2026
