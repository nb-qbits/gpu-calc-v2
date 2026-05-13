# ADR: Single repo over Turborepo monorepo

**Date**: 2026-05-12
**Status**: Accepted

## Decision

Use a single Next.js repository. Do not use Turborepo or a monorepo structure.

## Reasons

- One application, one deployment
- One engineer at a time in Phase 1 and 2
- No multiple shared packages to manage
- No remote caching or task graph needed yet

## Consequences

- `lib/gpu-math/` replaces the `packages/utils` pattern from the monorepo plan
- Refactoring to Turborepo later is straightforward if the project grows
