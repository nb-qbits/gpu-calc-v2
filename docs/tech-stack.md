# Tech stack

## Next.js App Router + TypeScript

Chosen over a plain React SPA because:
- File-based routing with no extra library
- Server components reduce client bundle size
- Route Handlers provide a backend when persistence is added
- Vercel deployment is zero-config

App Router (not Pages Router) because it is the current standard and avoids
a future migration.

## PatternFly v5

Chosen over shadcn/ui + Tailwind because:
- gpu-calc follows Red Hat brand standards
- PatternFly is Red Hat's own design system — zero theming work
- Built for data-dense, engineer-facing tools (exactly this use case)
- Includes charts (Victory Charts) — no separate chart library needed
- WCAG AA accessible out of the box

Tailwind was explicitly dropped to avoid overlapping styling systems.

## Deferred decisions

| What | Why deferred |
|---|---|
| PostgreSQL + Prisma | Schema is clearer once saved-scenario UX is designed |
| NextAuth.js | No login requirement in Phase 1 or 2 |
| Turborepo | One app, one engineer — no task graph needed yet |
