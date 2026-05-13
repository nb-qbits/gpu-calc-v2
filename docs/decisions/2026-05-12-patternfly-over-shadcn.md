# ADR: PatternFly over shadcn/ui + Tailwind

**Date**: 2026-05-12
**Status**: Accepted

## Decision

Use PatternFly v5 as the only UI and styling system. Do not use Tailwind CSS or shadcn/ui.

## Reasons

- gpu-calc follows Red Hat brand standards
- PatternFly is Red Hat's own design system — colors, fonts, and spacing match out of the box
- Built for data-dense, engineer-facing interfaces (exactly this use case)
- Includes Victory Charts — no separate chart library needed
- WCAG AA accessible by default

## Consequences

- No Tailwind utility classes anywhere in the codebase
- All layout uses PatternFly Grid, Flex, Stack, and Splitter components
- Consistent look without per-engineer styling decisions
