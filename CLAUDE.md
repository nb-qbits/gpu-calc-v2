# CLAUDE.md — gpu-calc

This file gives Claude Code context about this project.
Read this before making any changes.

## What this project is

gpu-calc is a web application for LLM inference sizing, GPU comparison, and
cost modeling. It is a full-stack rebuild of a static site originally hosted
at nb-qbits.github.io/gpu-calc.

## Tech stack

- **Framework**: Next.js 14 App Router + TypeScript
- **UI**: PatternFly v5 (Red Hat's design system) — no Tailwind, no shadcn
- **Charts**: PatternFly Victory Charts
- **Fonts**: Red Hat Display, Red Hat Text, Red Hat Mono (Google Fonts)
- **Deployment**: Vercel

## Project structure

```
app/                    # Next.js App Router pages
  layout.tsx            # Root layout with AppShell
  page.tsx              # Homepage
  quick-estimate/       # Quick Estimate tool
  calculator/           # Advanced Calculator tool
  gpu-explorer/         # GPU Explorer tool
  hybrid-savings/       # Hybrid Savings tool
  routing/              # Routing Economics tool
components/
  layout/
    AppShell.tsx        # Page shell with sidebar nav
lib/
  gpu-math/             # ALL GPU sizing formulas live here
    memory.ts           # Memory estimation
    throughput.ts       # Throughput estimation
    cost.ts             # Cost modeling
  utils/
    format.ts           # Number/unit formatting helpers
docs/                   # Architecture docs and ADRs
public/                 # Static assets
```

## Critical rules

1. **All GPU math belongs in `lib/gpu-math/`** — never write sizing formulas
   inside React components. Components call lib functions and display results.

2. **PatternFly only for UI** — do not install or use Tailwind, shadcn/ui,
   Material UI, or any other component library. PatternFly is the single
   source of truth for components and styling.

3. **App Router patterns** — use server components by default. Add `"use client"`
   only when you need browser APIs, state, or event handlers.

4. **TypeScript strict** — no `any` types. Define proper interfaces in the
   relevant lib file or component.

5. **Sentence case everywhere** — Red Hat brand standard. No title case in
   headings or labels.

## Commands

```bash
npm run dev        # Start dev server on localhost:3000
npm run build      # Production build
npm run lint       # ESLint
npm run type-check # TypeScript check without build
```

## What is deferred (do not add yet)

- Database / Prisma / PostgreSQL
- Authentication / NextAuth.js
- Turborepo / monorepo structure
- Tailwind CSS

These will be added in later phases when there is a real requirement for them.
