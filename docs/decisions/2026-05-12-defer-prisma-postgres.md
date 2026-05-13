# ADR: Defer Prisma and PostgreSQL to Phase 3

**Date**: 2026-05-12
**Status**: Accepted

## Decision

Do not add a database in Phase 1 or Phase 2.

## Reasons

- No persistence requirement for the current calculator tools
- Database schema is clearer once saved-scenario UX is actually designed
- Premature schema design leads to rewrites

## Consequences

- All state is ephemeral (in-component or URL params)
- PostgreSQL + Prisma added in Phase 3 once saved scenarios are scoped
