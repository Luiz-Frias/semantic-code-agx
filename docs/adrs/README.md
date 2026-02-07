# Architecture Decision Records (ADRs)

ADRs capture key architectural decisions and the reasoning behind them. Use the template at `docs/templates/adr.template.md` when adding a new ADR.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](./0001-hexagonal-architecture.md) | Hexagonal Architecture | Accepted | 2025-01-14 |
| [0002](./0002-typestate-indexing.md) | Typestate Pattern for Indexing | Accepted | 2025-01-16 |
| [0003](./0003-gat-ports.md) | GAT-Based Port Traits | Accepted | 2025-01-16 |
| [0004](./0004-owned-requests.md) | Owned Request/Response Types | Accepted | 2025-01-16 |

## Planned

- **ADR-0005**: Async trait strategy (native vs async-trait crate)
- **ADR-0006**: Error classification (retriable vs non-retriable)
- **ADR-0007**: Adapter feature flags strategy

## Conventions

- Number ADRs sequentially (0001, 0002, ...)
- Use template at `docs/templates/adr.template.md`
- Keep ADRs short and decision-focused
- Include alternatives considered
- Link to related architecture docs
- Update this index when adding new ADRs
