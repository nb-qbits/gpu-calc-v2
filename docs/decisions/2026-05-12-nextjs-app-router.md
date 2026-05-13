# ADR: Next.js App Router over Pages Router

**Date**: 2026-05-12
**Status**: Accepted

## Decision

Use Next.js App Router, not Pages Router.

## Reasons

- App Router is the current Next.js standard (v13+)
- Server components reduce client JavaScript
- Route Handlers replace the need for a separate backend initially
- Avoids a future migration from Pages to App Router

## Consequences

- Components are server components by default; add `"use client"` only when needed
- Data fetching uses async server components, not `getServerSideProps`
