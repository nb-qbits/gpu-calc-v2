# GPUCalc

LLM inference sizing, GPU comparison, and cost modeling for engineers and infrastructure teams.

**🚀 Live at [gpu-calc-v2.vercel.app](https://gpu-calc-v2.vercel.app/quick-estimate)**

Built with Next.js + PatternFly + Red Hat design system.

## What it does

| Tool | Description |
|------|-------------|
| **Quick Estimate** | Fast GPU memory and cost estimate from model + load profile |
| **Advanced Calculator** | Detailed sizing with batching, quantization, and cost modeling |
| **GPU Explorer** | Compare GPUs across memory, throughput, cost, and availability |
| **Hybrid Savings** | Model cost savings across cloud, on-premise, and hybrid strategies |
| **Routing Economics** | Analyze request routing between model tiers |

## Getting started

### Prerequisites

- Node.js ≥ 20
- npm ≥ 10

### Setup

```bash
git clone https://github.com/nb-qbits/gpu-calc-v2.git
cd gpu-calc
npm install
npm run dev
```

App runs at **http://localhost:3000**.

### Available commands

```bash
npm run dev          # Start dev server (http://localhost:3000)
npm run build        # Production build
npm run type-check   # TypeScript check without building
npm run lint         # ESLint
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 App Router + TypeScript |
| UI | PatternFly v5 (Red Hat design system) |
| Fonts | Red Hat Display / Text / Mono |
| Deployment | Vercel |

## Project structure

```
app/                  Next.js App Router pages
  layout.tsx          Root layout, fonts, PatternFly CSS imports
  page.tsx            Homepage
  quick-estimate/     Quick Estimate tool
  calculator/         Advanced Calculator (stub)
  gpu-explorer/       GPU Explorer (stub)
  hybrid-savings/     Hybrid Savings (stub)
  routing/            Routing Economics (stub)
components/
  layout/
    AppShell.tsx      Top-nav masthead + PatternFly Page wrapper
lib/
  gpu-math/           ALL GPU sizing formulas live here
    memory.ts         Memory estimation
    throughput.ts     Throughput estimation
    cost.ts           Cost modeling
    models.ts         Model catalog
    gpus.ts           GPU catalog
    quick-estimate.ts Quick Estimate calculation engine
  utils/
    format.ts         Number / unit formatting helpers
docs/                 Architecture docs and ADRs
```

## Contributing

### Branching

- Branch from `main`: `git checkout -b feature/your-feature-name`
- Keep branches short-lived — one feature or fix per branch
- PR target is always `main`

### Before opening a PR

CI runs automatically and must pass. You can run the same checks locally:

```bash
npm run type-check   # Must be clean — no TypeScript errors
npm run lint         # Must be clean — no ESLint errors
npm run build        # Must succeed
```

Pre-commit hooks (Husky) run lint-staged automatically on `git commit` so most issues are caught before you push.

### Code conventions

1. **GPU math belongs in `lib/gpu-math/`** — never write sizing formulas inside React components. Components call lib functions and display the results.

2. **PatternFly only** — do not add Tailwind, shadcn/ui, or any other component library. PatternFly v5 is the single source of truth for UI components.

3. **Red Hat design system** — use the established CSS variables (`--rh-red`, `--rh-gray-*`, etc.) and PF spacing tokens. Do not introduce arbitrary hex colors.

4. **Sentence case everywhere** — Red Hat brand standard. No title case in headings or labels.

5. **Server components by default** — add `"use client"` only when you need browser APIs, state, or event handlers.

6. **No `any` types** — TypeScript strict mode is enforced. Define proper interfaces in the relevant lib file or component.

7. **No unnecessary comments** — only add a comment when the *why* is non-obvious. Well-named identifiers are self-documenting.

### PR checklist

- [ ] `npm run type-check` passes
- [ ] `npm run lint` passes
- [ ] `npm run build` passes
- [ ] New GPU math is in `lib/gpu-math/`, not in a component
- [ ] No new third-party UI libraries added
- [ ] Sentence case used in all UI text

## What is deferred (do not add yet)

- Database / Prisma / PostgreSQL
- Authentication / NextAuth.js
- Turborepo / monorepo structure
- Tailwind CSS

These will be added in later phases when there is a real requirement for them.
