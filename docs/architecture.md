# Architecture

## Application structure

gpu-calc is a single Next.js application using the App Router.

```
Browser → Next.js (Vercel) → lib/gpu-math (calculations)
```

All GPU sizing math runs client-side or server-side within Next.js.
There is no separate backend service in Phase 1 or 2.

## Key principle: math is isolated

All GPU sizing formulas live in `lib/gpu-math/`. React components never
contain math logic — they call lib functions and render the results.

This means:
- Formulas are testable without rendering components
- The same logic can be reused across multiple pages
- When a backend API is added later, it imports from the same lib

## Adding persistence later

When Phase 3 is reached, the plan is:
- Add PostgreSQL via Prisma
- Add Next.js Route Handlers in `app/api/`
- Route handlers import from `lib/gpu-math/` for server-side calculations
- No rewriting of existing components needed
