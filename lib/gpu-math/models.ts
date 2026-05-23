// Model catalog — display metadata only.
// hfId is the HuggingFace model path used to fetch config.json.
// No architecture values (layers, hidden size, heads) are stored here.
// All computation uses data fetched live from HuggingFace.

export type Vendor =
  | 'Meta'
  | 'Google'
  | 'Mistral'
  | 'NVIDIA'
  | 'Qwen'
  | 'DeepSeek'
  | 'AI21'
  | 'Falcon'
  | 'Microsoft'
  | 'Other'

export interface ModelSpec {
  id:          string   // internal UI key
  hfId:        string   // HuggingFace model path for config.json fetch
  name:        string   // display name shown in the gallery card
  vendor:      Vendor
  paramLabel:  string   // display badge — "8B", "70B", "141B-A35B" — never used in math
  tags?:       string[] // optional chips: ["MoE", "Vision", "SSM", "MLA"]
  isNew?:      boolean
}

export const MODEL_CATALOG: ModelSpec[] = [
  // ── Meta Llama ─────────────────────────────────────────────────────────────
  {
    id: 'llama-3.1-8b',
    hfId: 'meta-llama/Llama-3.1-8B-Instruct',
    name: 'Llama 3.1 8B', vendor: 'Meta', paramLabel: '8B',
  },
  {
    id: 'llama-3.1-70b',
    hfId: 'meta-llama/Llama-3.1-70B-Instruct',
    name: 'Llama 3.1 70B', vendor: 'Meta', paramLabel: '70B',
  },
  {
    id: 'llama-3.1-405b',
    hfId: 'meta-llama/Llama-3.1-405B-Instruct',
    name: 'Llama 3.1 405B', vendor: 'Meta', paramLabel: '405B',
  },
  {
    id: 'llama-3.3-70b',
    hfId: 'meta-llama/Llama-3.3-70B-Instruct',
    name: 'Llama 3.3 70B', vendor: 'Meta', paramLabel: '70B', isNew: true,
  },
  {
    id: 'llama-4-scout',
    hfId: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    name: 'Llama 4 Scout', vendor: 'Meta', paramLabel: '109B', tags: ['MoE', 'Vision'], isNew: true,
  },
  {
    id: 'llama-4-maverick',
    hfId: 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    name: 'Llama 4 Maverick', vendor: 'Meta', paramLabel: '~400B', tags: ['MoE', 'Vision'], isNew: true,
  },

  // ── Google Gemma ───────────────────────────────────────────────────────────
  {
    id: 'gemma-2-2b',
    hfId: 'google/gemma-2-2b-it',
    name: 'Gemma 2 2B', vendor: 'Google', paramLabel: '2B',
  },
  {
    id: 'gemma-2-9b',
    hfId: 'google/gemma-2-9b-it',
    name: 'Gemma 2 9B', vendor: 'Google', paramLabel: '9B',
  },
  {
    id: 'gemma-2-27b',
    hfId: 'google/gemma-2-27b-it',
    name: 'Gemma 2 27B', vendor: 'Google', paramLabel: '27B',
  },
  {
    id: 'gemma-3-4b',
    hfId: 'google/gemma-3-4b-it',
    name: 'Gemma 3 4B', vendor: 'Google', paramLabel: '4B', isNew: true,
  },
  {
    id: 'gemma-3-12b',
    hfId: 'google/gemma-3-12b-it',
    name: 'Gemma 3 12B', vendor: 'Google', paramLabel: '12B', isNew: true,
  },
  {
    id: 'gemma-3-27b',
    hfId: 'google/gemma-3-27b-it',
    name: 'Gemma 3 27B', vendor: 'Google', paramLabel: '27B', isNew: true,
  },
  {
    id: 'recurrentgemma-2b',
    hfId: 'google/recurrentgemma-2b-it',
    name: 'RecurrentGemma 2B', vendor: 'Google', paramLabel: '2B', tags: ['SSM'],
  },

  // ── Mistral / Mixtral ──────────────────────────────────────────────────────
  {
    id: 'mistral-7b',
    hfId: 'mistralai/Mistral-7B-Instruct-v0.3',
    name: 'Mistral 7B', vendor: 'Mistral', paramLabel: '7B',
  },
  {
    id: 'mistral-nemo-12b',
    hfId: 'mistralai/Mistral-Nemo-Instruct-2407',
    name: 'Mistral Nemo 12B', vendor: 'Mistral', paramLabel: '12B',
  },
  {
    id: 'mistral-small-22b',
    hfId: 'mistralai/Mistral-Small-Instruct-2409',
    name: 'Mistral Small 22B', vendor: 'Mistral', paramLabel: '22B',
  },
  {
    id: 'mixtral-8x7b',
    hfId: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    name: 'Mixtral 8×7B', vendor: 'Mistral', paramLabel: '47B', tags: ['MoE'],
  },
  {
    id: 'mixtral-8x22b',
    hfId: 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    name: 'Mixtral 8×22B', vendor: 'Mistral', paramLabel: '141B', tags: ['MoE'],
  },

  // ── DeepSeek ───────────────────────────────────────────────────────────────
  {
    id: 'deepseek-v2-lite',
    hfId: 'deepseek-ai/DeepSeek-V2-Lite-Chat',
    name: 'DeepSeek V2 Lite', vendor: 'DeepSeek', paramLabel: '16B', tags: ['MoE', 'MLA'],
  },
  {
    id: 'deepseek-v3',
    hfId: 'deepseek-ai/DeepSeek-V3',
    name: 'DeepSeek V3', vendor: 'DeepSeek', paramLabel: '671B', tags: ['MoE', 'MLA'], isNew: true,
  },
  {
    id: 'deepseek-r1',
    hfId: 'deepseek-ai/DeepSeek-R1',
    name: 'DeepSeek R1', vendor: 'DeepSeek', paramLabel: '671B', tags: ['MoE', 'MLA'], isNew: true,
  },
  {
    id: 'deepseek-r1-distill-llama-8b',
    hfId: 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    name: 'DeepSeek R1 Distill 8B', vendor: 'DeepSeek', paramLabel: '8B', isNew: true,
  },
  {
    id: 'deepseek-r1-distill-qwen-32b',
    hfId: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    name: 'DeepSeek R1 Distill 32B', vendor: 'DeepSeek', paramLabel: '32B', isNew: true,
  },

  // ── Qwen ───────────────────────────────────────────────────────────────────
  {
    id: 'qwen2.5-7b',
    hfId: 'Qwen/Qwen2.5-7B-Instruct',
    name: 'Qwen 2.5 7B', vendor: 'Qwen', paramLabel: '7B',
  },
  {
    id: 'qwen2.5-14b',
    hfId: 'Qwen/Qwen2.5-14B-Instruct',
    name: 'Qwen 2.5 14B', vendor: 'Qwen', paramLabel: '14B',
  },
  {
    id: 'qwen2.5-72b',
    hfId: 'Qwen/Qwen2.5-72B-Instruct',
    name: 'Qwen 2.5 72B', vendor: 'Qwen', paramLabel: '72B',
  },
  {
    id: 'qwen3-8b',
    hfId: 'Qwen/Qwen3-8B',
    name: 'Qwen3 8B', vendor: 'Qwen', paramLabel: '8B', isNew: true,
  },
  {
    id: 'qwen3-30b-a3b',
    hfId: 'Qwen/Qwen3-30B-A3B',
    name: 'Qwen3 30B-A3B', vendor: 'Qwen', paramLabel: '30B', tags: ['MoE'], isNew: true,
  },
  {
    id: 'qwen3-235b-a22b',
    hfId: 'Qwen/Qwen3-235B-A22B',
    name: 'Qwen3 235B-A22B', vendor: 'Qwen', paramLabel: '235B', tags: ['MoE'], isNew: true,
  },
  {
    id: 'qwen2.5-vl-7b',
    hfId: 'Qwen/Qwen2.5-VL-7B-Instruct',
    name: 'Qwen 2.5 VL 7B', vendor: 'Qwen', paramLabel: '7B', tags: ['Vision'],
  },
  {
    id: 'qwen2.5-vl-72b',
    hfId: 'Qwen/Qwen2.5-VL-72B-Instruct',
    name: 'Qwen 2.5 VL 72B', vendor: 'Qwen', paramLabel: '72B', tags: ['Vision'],
  },

  // ── NVIDIA ─────────────────────────────────────────────────────────────────
  {
    id: 'nemotron-mini-4b',
    hfId: 'nvidia/Nemotron-Mini-4B-Instruct',
    name: 'Nemotron Mini 4B', vendor: 'NVIDIA', paramLabel: '4B', tags: ['SSM'],
  },
  {
    id: 'nemotron-h-56b',
    hfId: 'nvidia/Nemotron-H-56B-Instruct',
    name: 'Nemotron-H 56B', vendor: 'NVIDIA', paramLabel: '56B', tags: ['SSM', 'MoE'], isNew: true,
  },
  {
    id: 'nemotron-340b',
    hfId: 'nvidia/Nemotron-4-340B-Instruct',
    name: 'Nemotron 340B', vendor: 'NVIDIA', paramLabel: '340B',
  },

  // ── AI21 Jamba ─────────────────────────────────────────────────────────────
  {
    id: 'jamba-1.5-mini',
    hfId: 'ai21labs/AI21-Jamba-1.5-Mini',
    name: 'Jamba 1.5 Mini', vendor: 'AI21', paramLabel: '12B', tags: ['SSM', 'MoE'],
  },
  {
    id: 'jamba-1.5-large',
    hfId: 'ai21labs/AI21-Jamba-1.5-Large',
    name: 'Jamba 1.5 Large', vendor: 'AI21', paramLabel: '94B', tags: ['SSM', 'MoE'],
  },

  // ── Microsoft Phi ──────────────────────────────────────────────────────────
  {
    id: 'phi-3.5-mini',
    hfId: 'microsoft/Phi-3.5-mini-instruct',
    name: 'Phi 3.5 Mini', vendor: 'Microsoft', paramLabel: '3.8B',
  },
  {
    id: 'phi-3.5-moe',
    hfId: 'microsoft/Phi-3.5-MoE-instruct',
    name: 'Phi 3.5 MoE', vendor: 'Microsoft', paramLabel: '42B', tags: ['MoE'],
  },
  {
    id: 'phi-4',
    hfId: 'microsoft/phi-4',
    name: 'Phi 4', vendor: 'Microsoft', paramLabel: '14B', isNew: true,
  },

  // ── Falcon ─────────────────────────────────────────────────────────────────
  {
    id: 'falcon-3-7b',
    hfId: 'tiiuae/Falcon3-7B-Instruct',
    name: 'Falcon 3 7B', vendor: 'Falcon', paramLabel: '7B',
  },
  {
    id: 'falcon-3-10b',
    hfId: 'tiiuae/Falcon3-10B-Instruct',
    name: 'Falcon 3 10B', vendor: 'Falcon', paramLabel: '10B',
  },
]
