export interface FrontierModel {
  id: string
  name: string
  provider: string
  pricePerMInput: number
  pricePerMOutput: number
  tier: 'fast' | 'balanced' | 'frontier'
}

export const FRONTIER_MODELS: FrontierModel[] = [
  { id: 'claude-3.5-haiku',   name: 'Claude 3.5 Haiku',   provider: 'Anthropic', pricePerMInput: 0.80,  pricePerMOutput: 4.00,  tier: 'fast' },
  { id: 'gpt-4o-mini',        name: 'GPT-4o mini',        provider: 'OpenAI',    pricePerMInput: 0.15,  pricePerMOutput: 0.60,  tier: 'fast' },
  { id: 'gemini-1.5-flash',   name: 'Gemini 1.5 Flash',   provider: 'Google',    pricePerMInput: 0.075, pricePerMOutput: 0.30,  tier: 'fast' },
  { id: 'claude-3.5-sonnet',  name: 'Claude 3.5 Sonnet',  provider: 'Anthropic', pricePerMInput: 3.00,  pricePerMOutput: 15.00, tier: 'balanced' },
  { id: 'gpt-4o',             name: 'GPT-4o',             provider: 'OpenAI',    pricePerMInput: 2.50,  pricePerMOutput: 10.00, tier: 'balanced' },
  { id: 'gemini-1.5-pro',     name: 'Gemini 1.5 Pro',     provider: 'Google',    pricePerMInput: 1.25,  pricePerMOutput: 5.00,  tier: 'balanced' },
]

export function getFrontierModel(id: string): FrontierModel | undefined {
  return FRONTIER_MODELS.find(m => m.id === id)
}

export function getFrontierModelsByTier(tier: FrontierModel['tier']): FrontierModel[] {
  return FRONTIER_MODELS.filter(m => m.tier === tier)
}
