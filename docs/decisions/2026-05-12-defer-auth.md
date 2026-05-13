# ADR: Defer NextAuth.js to a later phase

**Date**: 2026-05-12
**Status**: Accepted

## Decision

Do not add authentication in Phase 1 or Phase 2.

## Reasons

- No login requirement for the current tool set
- OAuth + sessions + user tables add significant complexity
- Saved scenarios can be anonymous (session ID or URL-encoded state) initially

## Consequences

- All tools are publicly accessible with no login
- User accounts and private scenarios deferred to Phase 5 (only if needed)
