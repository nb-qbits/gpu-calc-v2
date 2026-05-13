# ADR: Keep GPU formulas isolated in lib/gpu-math

**Date**: 2026-05-12
**Status**: Accepted

## Decision

All GPU sizing formulas live in `lib/gpu-math/`. React components never contain
math logic directly.

## Reasons

- Formulas are testable without rendering components
- The same logic can be shared across multiple pages
- When a backend API is added, it imports from the same lib without duplication
- Prevents formula drift between pages

## Consequences

- Components call lib functions and render results — never compute inline
- New formulas go to `lib/gpu-math/` first, then get wired into components
