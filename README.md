# gpu-calc

LLM inference sizing, GPU comparison, and cost modeling for engineers and
infrastructure teams.

## What it does

- **Quick Estimate** — Fast GPU memory and throughput estimate from model size
- **Advanced Calculator** — Detailed inference sizing with batching, quantization, and cost modeling
- **GPU Explorer** — Compare GPUs across memory, throughput, cost, and availability
- **Hybrid Savings** — Model cost savings across cloud, on-premise, and hybrid strategies
- **Routing Economics** — Analyze request routing between model tiers

## Getting started

### Prerequisites

- Node.js 20+
- npm 10+

### Setup

```bash
git clone https://github.com/YOUR_ORG/gpu-calc.git
cd gpu-calc
npm install
npm run dev
```

App runs at [http://localhost:3000](http://localhost:3000).

## Tech stack

| Layer      | Technology                     |
|------------|-------------------------------|
| Framework  | Next.js 14 App Router + TypeScript |
| UI         | PatternFly v5                 |
| Charts     | PatternFly Victory Charts     |
| Deployment | Vercel                        |

See [docs/tech-stack.md](docs/tech-stack.md) for the full rationale.

## Project structure

```
app/          Next.js pages (one folder per tool)
components/   Shared React components
lib/
  gpu-math/   GPU sizing formulas — all math lives here
  utils/      Formatting helpers
docs/         Architecture docs and decision records
```

## Contributing

1. Branch off `main` using `feature/your-feature-name`
2. Keep GPU math in `lib/gpu-math/` — never inside components
3. Use PatternFly components only — no Tailwind
4. Open a PR against `main` when ready

See [CLAUDE.md](CLAUDE.md) for Claude Code context.
