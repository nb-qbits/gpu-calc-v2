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
- **Fonts**: Plus Jakarta Sans (display, body), JetBrains Mono (numbers, technical labels)
- **Deployment**: Vercel

## Documentation

**Essential reading:**

- **[docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md)** - Complete design system
  - Typography scale and font usage rules
  - Color palette and contrast requirements  
  - Spacing scale and layout patterns
  - Reusable component patterns (flip tiles, search, accordions, tour)
  - Animation guidelines and accessibility requirements

- **[docs/ARCHITECTURE_DETAILED.md](docs/ARCHITECTURE_DETAILED.md)** - System architecture
  - Component diagrams showing module relationships
  - Data flow visualizations
  - Inference engine internal architecture
  - API architecture and type system
  - Current integration status and roadmap

**Critical**: Follow DESIGN_SYSTEM.md for UI work and ARCHITECTURE_DETAILED.md for understanding how components interact.

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

6. **Follow the design system** — see [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md) for typography, colors, spacing, components, and patterns. Don't invent new styles or components without checking the design system first.

## Commands

```bash
npm run dev        # Start dev server on localhost:3000
npm run build      # Production build
npm run lint       # ESLint
npm run type-check # TypeScript check without build
```

Got it — your real stack is Next.js + PatternFly v5, so my raw .stat-label CSS won't map. The right move is to add a typography & color section to CLAUDE.md as a rule, expressed in PatternFly terms. Append this:

## Typography & color (legibility rules)

Goal: high-contrast, readable text on white console surfaces. No faint gray
micro-text. These rules override PatternFly defaults where they are too light
or too small.

### Color
Define these as CSS custom properties in a global stylesheet and use them for
ALL text. Also override the matching PatternFly global tokens.

- Primary text   #151515  → --pf-v5-global--Color--100
- Secondary text #3c3f42  → --pf-v5-global--Color--200  (PF default #6a6e73 is too light — override it)
- Caption (min)  #54585c  — lightest gray allowed; captions only
- Interactive    #0066cc  (links, buttons) — --pf-v5-global--primary-color--100
- Brand red      #ee0000  — LOGO ONLY. Never text, never buttons.

Contrast rule: body copy, descriptions, table/detail values, and constraint
rows must be #3c3f42 (Color--200) or darker. #54585c is only for short uppercase
captions. Never put a number, value, or explanation in a gray lighter than #54585c.

### Type scale (base 15px / line-height 1.5)
Fonts: Red Hat Display (headings + big numbers), Red Hat Text (body),
Red Hat Mono (numbers, code, eyebrow labels).

- Page title:        26px / Display / 500
- Card title:        16px / Display / 600
- Big metric number: 32px / Display / 700
- Scenario value:    26px / Display / 700
- Body / description:14px / Text / 400 / Color--200
- Detail & mono values: 12.5–13px / Mono / Color--200  (NOT the lightest gray)
- Eyebrow label:     12px / Mono / 500 / uppercase / letter-spacing 0.06em / Color--200
- Smallest caption:  11.5px / Mono  — absolute floor, nothing smaller
- Units (GB, MB, ×): 13px / Text / 500, white-space: nowrap

### Hard limits
- No font-size below 11.5px anywhere.
- No uppercase label lighter than Color--200 (#3c3f42).
- letter-spacing on uppercase labels ≤ 0.08em.
- All number displays: font-variant-numeric: tabular-nums.

When PatternFly's default component text is lighter or smaller than the above
(e.g. DescriptionList term, helper text, table caption), override it to meet
these values rather than accepting the default.

## Interaction polish (Quick Estimate specific)

The Quick Estimate page (`/quick-estimate`) has specific interaction patterns
that must be preserved:

- **Flip tiles** — result tiles flip on click/Enter to reveal formulas. Use
  opacity + rotateY (not bare backface-visibility) for robust cross-browser flip.
- **Count-up animation** — headline numbers animate from 0 on load with 1500ms
  ease-out timing. Must respect `prefers-reduced-motion`.
- **Sparkline** — GPU tile shows GPUs-vs-concurrency mini line chart (SVG).
- **Glossary popovers** — every jargon term (KV cache, max_num_seqs, tensor
  parallel, GQA, worst-case context, range drivers, etc.) has a "?" icon with
  plain-language explanation in a popover.
- **Accordions** — six assumption sections, collapsed by default. Closed rows
  MUST show current values inline so users can read state without expanding.
- **Default-first flow** — show three key numbers immediately (weights, KV/req,
  GPUs) + cost; everything else is collapsed and editable.

See `design/preview.html` for the visual reference and `design/README.md` for
complete interaction inventory.

## What is deferred (do not add yet)

- Database / Prisma / PostgreSQL
- Authentication / NextAuth.js
- Turborepo / monorepo structure
- Tailwind CSS

These will be added in later phases when there is a real requirement for them.
