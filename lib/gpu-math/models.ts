export type Vendor = 'Meta' | 'Google' | 'Mistral' | 'NVIDIA' | 'Qwen' | 'RedHat' | 'Other';

export interface ModelSpec {
  id: string;
  name: string;
  vendor: Vendor;
  paramsBillions: number;
  activeFraction: number; // 1.0 for dense; fraction for MoE
  numLayers: number;
  hiddenSize: number;
}

export const MODEL_CATALOG: ModelSpec[] = [
  // Google Gemma
  { id: 'gemma-2-2b',     name: 'Gemma 2 2B',     vendor: 'Google',  paramsBillions: 2,   activeFraction: 1.0, numLayers: 26, hiddenSize: 2304  },
  { id: 'gemma-2-9b',     name: 'Gemma 2 9B',     vendor: 'Google',  paramsBillions: 9,   activeFraction: 1.0, numLayers: 42, hiddenSize: 3584  },
  { id: 'gemma-2-27b',    name: 'Gemma 2 27B',    vendor: 'Google',  paramsBillions: 27,  activeFraction: 1.0, numLayers: 46, hiddenSize: 4608  },
  { id: 'gemma-3-12b',    name: 'Gemma 3 12B',    vendor: 'Google',  paramsBillions: 12,  activeFraction: 1.0, numLayers: 28, hiddenSize: 3072  },
  { id: 'gemma-3-27b',    name: 'Gemma 3 27B',    vendor: 'Google',  paramsBillions: 27,  activeFraction: 1.0, numLayers: 46, hiddenSize: 4608  },
  { id: 'gemma-4-2b',     name: 'Gemma 4 2B',     vendor: 'Google',  paramsBillions: 2,   activeFraction: 1.0, numLayers: 26, hiddenSize: 2304  },
  { id: 'gemma-4-9b',     name: 'Gemma 4 9B',     vendor: 'Google',  paramsBillions: 9,   activeFraction: 1.0, numLayers: 32, hiddenSize: 3072  },
  { id: 'gemma-4-27b',    name: 'Gemma 4 27B',    vendor: 'Google',  paramsBillions: 27,  activeFraction: 1.0, numLayers: 46, hiddenSize: 4608  },
  // Meta Llama
  { id: 'llama-3-8b',     name: 'Llama 3 8B',     vendor: 'Meta',    paramsBillions: 8,   activeFraction: 1.0, numLayers: 32, hiddenSize: 4096  },
  { id: 'llama-3-70b',    name: 'Llama 3 70B',    vendor: 'Meta',    paramsBillions: 70,  activeFraction: 1.0, numLayers: 80, hiddenSize: 8192  },
  { id: 'llama-3.1-8b',   name: 'Llama 3.1 8B',   vendor: 'Meta',    paramsBillions: 8,   activeFraction: 1.0, numLayers: 32, hiddenSize: 4096  },
  { id: 'llama-3.1-70b',  name: 'Llama 3.1 70B',  vendor: 'Meta',    paramsBillions: 70,  activeFraction: 1.0, numLayers: 80, hiddenSize: 8192  },
  // Mistral
  { id: 'mistral-7b',     name: 'Mistral 7B',     vendor: 'Mistral', paramsBillions: 7,   activeFraction: 1.0,  numLayers: 32, hiddenSize: 4096 },
  { id: 'mixtral-8x7b',   name: 'Mixtral 8x7B',   vendor: 'Mistral', paramsBillions: 47,  activeFraction: 0.25, numLayers: 32, hiddenSize: 4096 },
  // Qwen
  { id: 'qwen-2.5-7b',    name: 'Qwen 2.5 7B',    vendor: 'Qwen',    paramsBillions: 7,   activeFraction: 1.0, numLayers: 28, hiddenSize: 3584  },
  // RedHat
  { id: 'redhat-model-7b',name: 'RedHat Model 7B', vendor: 'RedHat',  paramsBillions: 7,   activeFraction: 1.0, numLayers: 32, hiddenSize: 4096  },
  // NVIDIA
  { id: 'nemotron-340b',  name: 'Nemotron 340B',  vendor: 'NVIDIA',  paramsBillions: 340, activeFraction: 1.0, numLayers: 96, hiddenSize: 18432 },
];
