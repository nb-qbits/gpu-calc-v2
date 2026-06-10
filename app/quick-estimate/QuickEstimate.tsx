'use client';

import * as React from 'react';
import {
  Breadcrumb, BreadcrumbItem,
  Button,
  TextInput,
  FormSelect, FormSelectOption,
  Switch,
  Label,
  Accordion, AccordionItem, AccordionToggle, AccordionContent,
} from '@patternfly/react-core';
import StarIcon from '@patternfly/react-icons/dist/esm/icons/star-icon';
import OutlinedStarIcon from '@patternfly/react-icons/dist/esm/icons/outlined-star-icon';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import MicrochipIcon from '@patternfly/react-icons/dist/esm/icons/microchip-icon';
import MemoryIcon from '@patternfly/react-icons/dist/esm/icons/memory-icon';
import DollarSignIcon from '@patternfly/react-icons/dist/esm/icons/dollar-sign-icon';
import LayerGroupIcon from '@patternfly/react-icons/dist/esm/icons/layer-group-icon';
import InfoCircleIcon from '@patternfly/react-icons/dist/esm/icons/info-circle-icon';
import EyeIcon from '@patternfly/react-icons/dist/esm/icons/eye-icon';
import EyeSlashIcon from '@patternfly/react-icons/dist/esm/icons/eye-slash-icon';
import SearchIcon from '@patternfly/react-icons/dist/esm/icons/search-icon';
import TimesIcon from '@patternfly/react-icons/dist/esm/icons/times-icon';

import styles from './QuickEstimate.module.css';
import { Term, FlipTile, Sparkline, useCountUp } from './quickEstimateHelpers';
import { ProductTour, type TourStep } from '@/components/ProductTour';
import { SaveEstimateModal } from './SaveEstimateModal';
import { computeInferenceConfig } from '@/lib/gpu-math/inference-config';
import { MODEL_CATALOG } from '@/lib/gpu-math/models';
import { GPU_CATALOG } from '@/lib/gpu-math/gpus';
import { fetchModelConfig, type HFModelConfig } from '@/lib/huggingface/fetch-config';
import { saveEstimate, getSavedEstimateCount } from '@/lib/saved-estimates';
import type { InferenceConfigResult } from '@/lib/gpu-math/inference-config';
import Link from 'next/link';

// Generate model options from actual MODEL_CATALOG
const MODEL_OPTIONS = MODEL_CATALOG.map(m => m.hfId);

const GPU_OPTIONS = [
  'NVIDIA H100 80GB', 'NVIDIA H200 141GB', 'NVIDIA A100 80GB',
  'NVIDIA A100 40GB', 'NVIDIA L40S 48GB', 'AMD MI300X 192GB',
];

const QUICK_ESTIMATE_TOUR: TourStep[] = [
  {
    target: '[data-tour="model"]',
    title: 'Start with a model',
    description: 'Type any Hugging Face model ID or pick from popular models. We\'ll auto-detect the specs and fill in smart defaults.',
    position: 'bottom'
  },
  {
    target: '[data-tour="warning"]',
    title: 'Default assumptions',
    description: 'Quick estimates start with common defaults. Click "Customize" to match your actual workload and traffic patterns.',
    position: 'bottom'
  },
  {
    target: '[data-tour="result-tile-gpus"]',
    title: 'Your results at a glance',
    description: 'These tiles show GPU count, memory requirements, and monthly cost. Click any tile to see the math behind it.',
    position: 'right'
  },
  {
    target: '[data-tour="search"]',
    title: 'Find anything instantly',
    description: 'Type "kv cache", "cost", or any term to filter and highlight matching sections. Great for focusing on specific metrics.',
    position: 'bottom'
  },
  {
    target: '[data-tour="assumptions"]',
    title: 'Fine-tune your workload',
    description: 'Expand these sections to adjust traffic patterns, sequence lengths, and hardware settings. Results update live as you edit.',
    position: 'top'
  }
];

export default function QuickEstimate() {
  console.log('🔵 QuickEstimate component mounting');
  const [model, setModel] = React.useState('nvidia/Nemotron-Mini-4B-Instruct');
  const [gpu, setGpu] = React.useState('NVIDIA H200 141GB');
  const [fav, setFav] = React.useState(false);
  const [showHf, setShowHf] = React.useState(false);
  const [hfToken, setHfToken] = React.useState('');
  const [hfReveal, setHfReveal] = React.useState(false);
  const [expanded, setExpanded] = React.useState<string[]>([]);
  const [showApi, setShowApi] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [showTour, setShowTour] = React.useState(false);
  const [tourSeen, setTourSeen] = React.useState(false);

  // 🧪 TEST: Inference config engine integration
  const [testResult, setTestResult] = React.useState<InferenceConfigResult | null>(null);
  const [testError, setTestError] = React.useState<string | null>(null);

  // HuggingFace config fetching
  const [hfConfig, setHfConfig] = React.useState<HFModelConfig | null>(null);
  const [isFetchingConfig, setIsFetchingConfig] = React.useState(false);

  // Collapsible state for "Why this GPU count?" card
  const [whyGpuExpanded, setWhyGpuExpanded] = React.useState(false);

  // Collapsible state for "Want to change assumptions?" section
  const [assumptionsExpanded, setAssumptionsExpanded] = React.useState(false);

  // Save estimate modal
  const [showSaveModal, setShowSaveModal] = React.useState(false);
  const [savedCount, setSavedCount] = React.useState(0);
  const [showToast, setShowToast] = React.useState(false);

  // Interactive controls
  const [testConcurrentUsers, setTestConcurrentUsers] = React.useState(97);
  const [testISL, setTestISL] = React.useState(1000);
  const [testOSL, setTestOSL] = React.useState(150);
  const [testWorkloadType, setTestWorkloadType] = React.useState<'chat' | 'web_search' | 'rag' | 'batch' | 'coding'>('chat');
  const [testSLAPriority, setTestSLAPriority] = React.useState<'ttft' | 'tpot' | 'throughput'>('ttft');
  const [testWeightPrecision, setTestWeightPrecision] = React.useState<'FP16' | 'FP8' | 'INT8' | 'INT4'>('FP16');
  const [testKVCachePrecision, setTestKVCachePrecision] = React.useState<'FP16' | 'FP8'>('FP16');

  // Live pricing from Cloudflare Worker
  const [livePricing, setLivePricing] = React.useState<Record<string, number>>({});

  // Map UI GPU names to catalog IDs
  const mapGpuToCatalogId = (uiGpuName: string): string => {
    // UI format: "NVIDIA H200 141GB" -> catalog format: "h200-141gb"
    const mapping: Record<string, string> = {
      'NVIDIA H100 80GB': 'h100-80gb',
      'NVIDIA H200 141GB': 'h200-141gb',
      'NVIDIA A100 80GB': 'a100-80gb',
      'NVIDIA A100 40GB': 'a100-40gb',
      'NVIDIA L40S 48GB': 'l40s-48gb',
      'AMD MI300X 192GB': 'mi300x-192gb',
    };

    return mapping[uiGpuName] || uiGpuName.toLowerCase().replace(/\s+/g, '-');
  };

  // Add loading state
  const [isCalculating, setIsCalculating] = React.useState(false);

  // Fetch HF config when model changes (if not in catalog)
  React.useEffect(() => {
    const modelInCatalog = MODEL_CATALOG.find(m =>
      m.hfId === model || m.id === model || m.name === model
    );

    if (modelInCatalog) {
      // Model in catalog - no need to fetch
      setHfConfig(null);
      setIsFetchingConfig(false);
      return;
    }

    // Model NOT in catalog - fetch from HuggingFace
    const fetchConfig = async () => {
      setIsFetchingConfig(true);
      console.log('🔄 Fetching config from HuggingFace for:', model);
      console.log('🔑 UI State - HF Token:', hfToken ? `Provided (length: ${hfToken.length}, starts with hf_: ${hfToken.startsWith('hf_')})` : 'Not provided');
      console.log('🔑 Passing token to fetchModelConfig:', hfToken ? 'Yes' : 'No');

      const result = await fetchModelConfig(model, hfToken);

      if (result.success && result.config) {
        setHfConfig(result.config);
        setTestError(null);
        console.log('✅ Fetched HF config:', result.config);
      } else {
        setHfConfig(null);
        setTestError(result.error || 'Failed to fetch model config');
        console.error('❌ Failed to fetch HF config:', result.error);
      }

      setIsFetchingConfig(false);
    };

    // Debounce to avoid fetching while user is typing
    const timer = setTimeout(fetchConfig, 500);
    return () => clearTimeout(timer);
  }, [model, hfToken]);

  // Auto-run calculation when inputs change (direct useEffect, no callback wrapper)
  React.useEffect(() => {
    // Don't calculate while fetching HF config
    if (isFetchingConfig) {
      console.log('⏸️ Skipping calculation - waiting for HF config fetch to complete');
      return;
    }

    console.log('🔄 Running calculation with model:', model);
    setIsCalculating(true);

    // Small delay to batch rapid state changes
    const timer = setTimeout(() => {
      try {
        const catalogGpuId = mapGpuToCatalogId(gpu);

        const result = computeInferenceConfig({
          model_name: model,
          precision: testWeightPrecision,
          kv_cache_precision: testKVCachePrecision,
          gpu_type: catalogGpuId,
          concurrent_users: testConcurrentUsers,
          isl: testISL,
          osl: testOSL,
          workload_type: testWorkloadType,
          sla_priority: testSLAPriority,
          hf_config: hfConfig || undefined  // Pass fetched config if available
        });
        setTestResult(result);
        setTestError(null);
        console.log('✅ Inference engine result:', result);
        console.log('   Inputs:', { model, gpu: catalogGpuId, testConcurrentUsers, testISL, testOSL, testWorkloadType, testSLAPriority, weight: testWeightPrecision, kv: testKVCachePrecision });
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        setTestError(errorMsg);
        console.error('❌ Inference engine failed:', error);
      } finally {
        setIsCalculating(false);
      }
    }, 100);

    // Cleanup timeout on unmount or dependency change
    return () => clearTimeout(timer);
  }, [model, gpu, testConcurrentUsers, testISL, testOSL, testWorkloadType, testSLAPriority, testWeightPrecision, testKVCachePrecision, hfConfig, isFetchingConfig]);

  // Fetch live pricing from Cloudflare Worker
  React.useEffect(() => {
    const fetchPricing = async () => {
      try {
        const response = await fetch('/api/v1/gpus?live_pricing=true');
        const data = await response.json();

        if (data.status === 'success' && data.data?.gpus) {
          const pricing: Record<string, number> = {};
          data.data.gpus.forEach((gpu: any) => {
            if (gpu.live_pricing?.onDemand?.median) {
              pricing[gpu.name] = gpu.live_pricing.onDemand.median;
            }
          });
          setLivePricing(pricing);
          console.log('✅ Loaded live pricing for', Object.keys(pricing).length, 'GPUs');
          console.log('📊 Live pricing data:', pricing);
        }
      } catch (error) {
        console.error('Failed to fetch live pricing:', error);
      }
    };

    fetchPricing();
    // Refresh every 5 minutes
    const interval = setInterval(fetchPricing, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Check if user has seen the tour before
  React.useEffect(() => {
    const hasSeenTour = localStorage.getItem('qe-tour-seen');
    if (hasSeenTour) {
      setTourSeen(true);
    } else {
      // Show tour after a brief delay on first visit
      const timer = setTimeout(() => setShowTour(true), 1000);
      return () => clearTimeout(timer);
    }
  }, []);

  const toggleAcc = (id: string) => {
    setExpanded((e) => (e.includes(id) ? e.filter((x) => x !== id) : [...e, id]));

    // Scroll to the accordion section after a brief delay
    setTimeout(() => {
      const element = document.getElementById(`acc-${id}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  };

  // Debounced search query
  const [debouncedQuery, setDebouncedQuery] = React.useState('');
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 120);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const matchesSearch = (keywords: string) => {
    if (!debouncedQuery.trim()) return true;
    return keywords.toLowerCase().includes(debouncedQuery.toLowerCase());
  };

  // Count matches
  const matchCount = React.useMemo(() => {
    if (!debouncedQuery.trim()) return 0;
    let count = 0;
    const sections = [
      'gpus required gpu count hardware h100 servers',
      'weight memory params parameters model size gigabytes gb',
      'kv cache key value memory tokens gqa request megabytes mb',
      'cost monthly price dollars budget pricing',
      'kv cache scenarios memory requests worst case typical prompt',
      'why gpu count constraints memory fit kv cache scheduler batch tokens tensor parallel',
      'drivers estimate range traffic peak concurrency isl osl prefix cache utilization'
    ];
    sections.forEach(keywords => {
      if (keywords.toLowerCase().includes(debouncedQuery.toLowerCase())) count++;
    });
    return count;
  }, [debouncedQuery]);

  // animated headline numbers
  // Calculate real values from inference engine
  const realGpuCount = testResult ?
    testResult.memory_analysis.tp_size * testResult.memory_analysis.replicas :
    0;

  const realWeightGB = testResult ?
    testResult.memory_analysis.weight_gb :
    0;

  const realKVPerReqMB = testResult && testResult.memory_analysis.kv_cache_used_gb ?
    (testResult.memory_analysis.kv_cache_used_gb / testConcurrentUsers) * 1000 : // Convert GB to MB
    0;

  // Use live pricing if available, fallback to estimated pricing from hardware cost
  // Map UI GPU names to catalog IDs
  const catalogGpuForPricing = GPU_CATALOG.find(g =>
    gpu.includes(g.name) || g.name.includes(gpu.replace('NVIDIA ', '').replace('AMD ', ''))
  );

  // Map UI GPU names to pricing keys (e.g., "NVIDIA H200 141GB" -> "H200")
  const gpuPricingKey = gpu.includes('H200') ? 'H200' :
                        gpu.includes('H100') ? 'H100' :
                        gpu.includes('A100 80GB') ? 'A100 80GB' :
                        gpu.includes('A100') ? 'A100' :
                        gpu.includes('L40S') ? 'L40S' :
                        gpu.includes('MI300X') ? 'MI300X' : gpu;

  // Calculate estimated hourly price from hardware cost if live pricing unavailable
  // Formula: hardware_cost_usd / (36 months * 730 hours/month) = $/hr
  const estimatedHourlyPrice = catalogGpuForPricing
    ? catalogGpuForPricing.hardware_cost_usd / (36 * 730)
    : 2.49;

  const gpuPricePerHour = livePricing[gpuPricingKey] || estimatedHourlyPrice;

  console.log('💰 Pricing lookup:', {
    gpu,
    gpuPricingKey,
    catalogGpu: catalogGpuForPricing?.name,
    hwCost: catalogGpuForPricing?.hardware_cost_usd,
    livePricing: Object.keys(livePricing),
    foundPrice: livePricing[gpuPricingKey],
    estimatedPrice: estimatedHourlyPrice.toFixed(2),
    finalPrice: gpuPricePerHour.toFixed(2)
  });

  const realMonthlyCost = testResult ?
    realGpuCount * gpuPricePerHour * 730 :
    0;

  const gpus = useCountUp(realGpuCount);
  const weight = useCountUp(realWeightGB);
  const kv = useCountUp(realKVPerReqMB);
  const cost = useCountUp(realMonthlyCost);

  const handleTourComplete = () => {
    setShowTour(false);
    setTourSeen(true);
    localStorage.setItem('qe-tour-seen', 'true');
  };

  const handleTakeTour = () => {
    setShowTour(true);
  };

  // Load saved count on mount
  React.useEffect(() => {
    setSavedCount(getSavedEstimateCount());
  }, []);

  // Generate auto name for save
  const generateAutoName = () => {
    const modelName = model.split('/').pop() || model;
    const gpuName = gpu.replace('NVIDIA ', '').replace('AMD ', '');
    return `${modelName} · ${gpuName} · ${testConcurrentUsers} users`;
  };

  const handleSaveEstimate = (data: { name: string; tags: string; notes: string }) => {
    if (!testResult || !catalogGpuForPricing) return;

    const kvPerUserGB = (testResult.memory_analysis.kv_cache_used_gb || 0) / testConcurrentUsers;
    const kvMBPerToken = (kvPerUserGB * 1000) / (testISL + testOSL);

    saveEstimate({
      name: data.name,
      tags: data.tags,
      notes: data.notes,
      model,
      gpu,
      inputs: {
        isl: testISL,
        osl: testOSL,
        concurrentUsers: testConcurrentUsers,
        workloadType: testWorkloadType,
        slaPriority: testSLAPriority,
        weightPrecision: testWeightPrecision,
        kvCachePrecision: testKVCachePrecision,
      },
      results: {
        gpusRequired: realGpuCount,
        tpSize: testResult.memory_analysis.tp_size,
        replicas: testResult.memory_analysis.replicas,
        weightMemoryGB: testResult.memory_analysis.weight_gb,
        kvCachePerUserGB: kvPerUserGB,
        kvCacheTotalGB: testResult.memory_analysis.kv_cache_used_gb || 0,
        kvCacheMBPerToken: kvMBPerToken,
        kvCategory: testResult.memory_analysis.kv_category || 'KV-1',
        kvCategoryLabel: testResult.memory_analysis.kv_category_label || 'Standard Dense',
        cloudCostMonthly: realMonthlyCost,
        cloudCost5Year: realMonthlyCost * 60,
        selfHostedCostMonthly: (catalogGpuForPricing.hardware_cost_usd * realGpuCount) / 60,
        selfHostedCost5Year: catalogGpuForPricing.hardware_cost_usd * realGpuCount,
      },
    });

    setSavedCount(getSavedEstimateCount());
    setShowToast(true);
    setTimeout(() => setShowToast(false), 5000);
  };

  // Build accordion sections dynamically from current state
  const buildAccordionSections = () => [
    {
      id: 'workload', title: 'Workload',
      summary: [
        { k: 'ISL', v: `${testISL}` },
        { k: 'OSL', v: `${testOSL}` },
        { k: 'users', v: `${testConcurrentUsers}` },
        { k: 'type', v: testWorkloadType }
      ],
      fields: [
        {
          label: 'Input sequence length (ISL)',
          value: testISL,
          term: 'isl',
          type: 'range' as const,
          min: 1,
          max: 128000,
          step: 128,
          onChange: (val: number) => setTestISL(val)
        },
        {
          label: 'Output sequence length (OSL)',
          value: testOSL,
          term: 'osl',
          type: 'range' as const,
          min: 1,
          max: 16384,
          step: 16,
          onChange: (val: number) => setTestOSL(val)
        },
        {
          label: 'Concurrent users',
          value: testConcurrentUsers,
          term: 'concurrent',
          type: 'range' as const,
          min: 1,
          max: 50000,
          step: 1,
          onChange: (val: number) => setTestConcurrentUsers(val)
        },
        {
          label: 'Workload type',
          value: testWorkloadType,
          type: 'select' as const,
          options: ['chat', 'rag', 'coding', 'batch', 'web_search'] as const,
          onChange: (val: string) => setTestWorkloadType(val as any)
        },
        {
          label: 'SLA priority',
          value: testSLAPriority,
          type: 'select' as const,
          options: ['ttft', 'tpot', 'throughput'] as const,
          onChange: (val: string) => setTestSLAPriority(val as any)
        },
      ],
    },
    {
      id: 'memory', title: 'Precision & memory',
      summary: [
        { k: 'weights', v: testWeightPrecision },
        { k: 'KV', v: testKVCachePrecision }
      ],
      fields: [
        {
          label: 'Weight precision',
          value: testWeightPrecision,
          type: 'select' as const,
          options: ['FP16', 'FP8', 'INT8', 'INT4'] as const,
          onChange: (val: string) => setTestWeightPrecision(val as any)
        },
        {
          label: 'KV cache precision',
          value: testKVCachePrecision,
          type: 'select' as const,
          options: ['FP16', 'FP8'] as const,
          onChange: (val: string) => setTestKVCachePrecision(val as any)
        },
      ],
    },
    {
      id: 'hardware', title: 'Hardware',
      summary: [{ k: 'GPU', v: gpu }],
      fields: [
        {
          label: 'GPU type',
          value: gpu,
          type: 'select' as const,
          options: GPU_OPTIONS,
          onChange: (val: string) => setGpu(val)
        },
      ],
    },
    {
      id: 'parallel', title: 'Parallelism', badge: 'Auto-computed', badgeColor: 'blue',
      summary: [
        { k: 'TP', v: testResult ? `${testResult.memory_analysis.tp_size}` : '—' },
        { k: 'replicas', v: testResult ? `${testResult.memory_analysis.replicas}` : '—' }
      ],
      fields: [
        {
          label: 'Tensor parallel size',
          value: testResult ? `${testResult.memory_analysis.tp_size}` : '—',
          term: 'tensorParallel',
          readonly: true
        },
        {
          label: 'Replica count',
          value: testResult ? `${testResult.memory_analysis.replicas}` : '—',
          readonly: true
        },
        {
          label: 'Total GPUs',
          value: testResult ? `${testResult.memory_analysis.tp_size * testResult.memory_analysis.replicas}` : '—',
          readonly: true
        },
      ],
    },
    {
      id: 'engine', title: 'vLLM config', badge: 'Auto-computed', badgeColor: 'blue',
      summary: [
        { k: 'max_num_seqs', v: testResult ? `${testResult.vllm_config.max_num_seqs}` : '—' },
        { k: 'chunked', v: testResult ? (testResult.vllm_config.enable_chunked_prefill ? 'on' : 'off') : '—' }
      ],
      fields: [
        {
          label: 'max_num_seqs',
          value: testResult ? `${testResult.vllm_config.max_num_seqs}` : '—',
          term: 'maxNumSeqs',
          readonly: true
        },
        {
          label: 'max_model_len',
          value: testResult ? `${testResult.vllm_config.max_model_len}` : '—',
          readonly: true
        },
        {
          label: 'enable_chunked_prefill',
          value: testResult ? (testResult.vllm_config.enable_chunked_prefill ? 'Yes' : 'No') : '—',
          readonly: true
        },
        {
          label: 'enable_prefix_caching',
          value: testResult ? (testResult.vllm_config.enable_prefix_caching ? 'Yes' : 'No') : '—',
          readonly: true
        },
        {
          label: 'gpu_memory_utilization',
          value: testResult ? `${(testResult.vllm_config.gpu_memory_utilization * 100).toFixed(0)}%` : '—',
          term: 'gpuUtil',
          readonly: true
        },
      ],
    },
  ];

  return (
    <div className={styles.page}>
      {showTour && (
        <ProductTour
          steps={QUICK_ESTIMATE_TOUR}
          tourId="qe"
          onComplete={handleTourComplete}
        />
      )}
      {/* ---------- header ---------- */}
      <div className={styles.header}>
        <Breadcrumb>
          <BreadcrumbItem>Estimate</BreadcrumbItem>
          <BreadcrumbItem isActive>Quick estimate</BreadcrumbItem>
        </Breadcrumb>
        <div className={styles.headRow}>
          <div>
            <h1 className={styles.pageTitle}>Quick estimate</h1>
            <p className={styles.subtitle}>Start with just a model name. We fill the rest, then let you tune every assumption.</p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <div style={{ position: 'relative' }}>
              <Button
                variant="link"
                onClick={handleTakeTour}
                style={{ fontSize: '14px' }}
              >
                Take a tour
              </Button>
              {!tourSeen && <div className={styles.tourBeacon} />}
            </div>
            <Button
              variant="plain"
              aria-label={fav ? 'Remove from favorites' : 'Add to favorites'}
              onClick={() => setFav((f) => !f)}
              icon={fav ? <StarIcon /> : <OutlinedStarIcon />}
            />
          </div>
        </div>
      </div>


      {/* ---------- input row ---------- */}
      <div className={`${styles.card} ${styles.inputCard}`} data-tour="model">
        <div className={styles.inputRow}>
          {/* Column 1: Model field */}
          <div className={styles.field}>
            <label className={styles.fieldLabel} htmlFor="qe-model">
              Model — Hugging Face ID
              <InfoCircleIcon style={{ width: 12, height: 12, opacity: 0.7 }} />
            </label>
            <div className={styles.modelInputWrapper}>
              <input
                type="text"
                id="qe-model"
                list="qe-models"
                value={model}
                onChange={(e) => {
                  console.log('📝 Model input changed to:', e.target.value);
                  setModel(e.target.value);
                }}
                placeholder="Type model name or select from dropdown..."
                aria-label="Model Hugging Face ID"
                className={styles.modelInput}
              />
              <datalist id="qe-models">
                {MODEL_OPTIONS.map((m) => <option key={m} value={m} />)}
              </datalist>
              <div className={styles.autoChipWrapper}>
                {isFetchingConfig ? (
                  <Label color="blue">🔄 Fetching from HuggingFace...</Label>
                ) : MODEL_CATALOG.find(m => m.hfId === model || m.id === model || m.name === model) ? (
                  <Label color="green" icon={<CheckCircleIcon />}>✓ In catalog</Label>
                ) : hfConfig ? (
                  <Label color="cyan" icon={<CheckCircleIcon />}>✓ Fetched from HuggingFace</Label>
                ) : testError ? (
                  <Label color="red" icon={<ExclamationTriangleIcon />}>❌ Not found</Label>
                ) : (
                  <Label color="grey">Type to search...</Label>
                )}
              </div>
            </div>
            <div className={styles.helperText}>
              Popular models: Llama 3.1, Mistral, Qwen 2.5, Gemma 2 — type to autocomplete
              {hfToken && hfToken.trim() && (
                <span style={{ marginLeft: '8px', color: '#0066cc', fontWeight: 500 }}>
                  🔑 Token active ({hfToken.startsWith('hf_') ? '✓ valid format' : '⚠️ check format'})
                </span>
              )}
            </div>
          </div>

          {/* Column 2: GPU target */}
          <div className={styles.field}>
            <label className={styles.fieldLabel} htmlFor="qe-gpu">GPU target</label>
            <select
              id="qe-gpu"
              value={gpu}
              onChange={(e) => setGpu(e.target.value)}
              aria-label="GPU target"
              className={styles.gpuSelect}
            >
              {GPU_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>

          {/* Column 3: Calculate button */}
          <div className={styles.calcBtnWrap}>
            <Button variant="primary" size="lg">Calculate</Button>
          </div>
        </div>

        <div style={{ marginTop: '18px' }}>
          <label className={styles.fieldLabel} style={{ marginBottom: '8px', display: 'block' }}>
            Hugging Face Token (optional — for gated or private models)
          </label>
          <div className={styles.hfPanel}>
            <div className={styles.fieldRow} style={{ flex: 1 }}>
              <input
                type={hfReveal ? 'text' : 'password'}
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                aria-label="Hugging Face token"
                className={styles.hfInput}
                style={{
                  flex: 1,
                  border: '1px solid #b8bbbe',
                  borderRadius: '4px',
                  padding: '8px 12px',
                  fontSize: '14px',
                  fontFamily: 'var(--sans)',
                  minHeight: '42px'
                }}
              />
              <Button
                variant="control"
                aria-label={hfReveal ? 'Hide token' : 'Show token'}
                onClick={() => {
                  console.log('Toggle clicked, current state:', hfReveal);
                  setHfReveal(!hfReveal);
                }}
                icon={hfReveal ? <EyeSlashIcon /> : <EyeIcon />}
              />
            </div>
            <Button variant="link" component="a" href="https://huggingface.co/settings/tokens" target="_blank">
              Get a token
            </Button>
          </div>
          <p className={styles.hfNote}>Stored in this browser only — never sent to our servers.</p>
        </div>
      </div>

      {/* ---------- warning strip ---------- */}
      <div className={styles.warn} data-tour="warning">
        <ExclamationTriangleIcon style={{ color: 'var(--gc-warn, #f0ab00)', flexShrink: 0 }} />
        <span>
          Based on your configuration — ISL {testISL}, OSL {testOSL}, {testKVCachePrecision} KV cache,
          {' '}{testConcurrentUsers} concurrent users. Edit settings in accordion below to customize.
        </span>
        <button className={styles.warnLink} onClick={() => toggleAcc('workload')}>Customize →</button>
      </div>

      {/* ---------- search ---------- */}
      <div className={styles.searchBox} data-tour="search">
        <SearchIcon className={styles.searchIcon} />
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Find a result — e.g. KV cache, cost, GPUs, memory, latency, scheduler"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <>
            <span className={styles.searchCount}>
              {matchCount} {matchCount === 1 ? 'match' : 'matches'}
            </span>
            <button
              className={styles.searchClear}
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
            >
              <TimesIcon style={{ width: 14, height: 14 }} />
            </button>
          </>
        )}
      </div>

      {/* ---------- error display ---------- */}
      {testError && (
        <div style={{
          padding: '16px 20px',
          marginBottom: '20px',
          background: '#fff3cd',
          border: '2px solid #ffc107',
          borderRadius: '8px',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <span style={{ fontSize: '24px', flexShrink: 0 }}>⚠️</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: '600', color: '#856404', marginBottom: '8px', fontSize: '16px' }}>
              {testError.includes('401') || testError.includes('Authentication required')
                ? '🔒 Authentication Required'
                : testError.includes('404') || testError.includes('not found')
                ? '❌ Model Not Found'
                : '⚠️ Failed to Load Model'}
            </div>
            <div style={{ fontSize: '14px', color: '#664d03', lineHeight: '1.6', marginBottom: '12px' }}>
              {testError}
            </div>

            {(testError.includes('401') || testError.includes('Authentication required')) && (
              <div style={{ fontSize: '13px', color: '#664d03', background: '#fff', padding: '12px', borderRadius: '4px', border: '1px solid #ffc107', marginBottom: '12px' }}>
                <strong>🔑 This model is gated (requires authentication):</strong>
                <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px', lineHeight: '1.6' }}>
                  <li>Add your HuggingFace token in the &ldquo;HUGGING FACE TOKEN&rdquo; field above</li>
                  <li>Get a token at: <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener" style={{ color: '#0066cc', textDecoration: 'underline' }}>https://huggingface.co/settings/tokens</a></li>
                  <li>Make sure you&apos;ve accepted the model&apos;s license on HuggingFace</li>
                </ul>
              </div>
            )}

            <div style={{ fontSize: '13px', color: '#664d03', background: '#fff', padding: '12px', borderRadius: '4px', border: '1px solid #ffc107' }}>
              <strong>✅ What to do now:</strong>
              <ol style={{ margin: '8px 0 0 0', paddingLeft: '20px', lineHeight: '1.6' }}>
                <li><strong>Use the dropdown:</strong> Select a model from the autocomplete suggestions</li>
                <li><strong>Popular models:</strong> Llama 3.1, Mistral, Qwen 2.5, DeepSeek, Gemma 2</li>
                <li><strong>Check spelling:</strong> Model names are case-sensitive (e.g., &ldquo;meta-llama&rdquo; not &ldquo;Meta-Llama&rdquo;)</li>
                <li><strong>GGUF models?</strong> Use the original base model, not the GGUF repo (e.g., &ldquo;google/gemma-2-12b-it&rdquo; not &ldquo;unsloth/gemma-...-GGUF&rdquo;)</li>
                <li><strong>Want a specific model?</strong> Let us know and we&apos;ll add it to the catalog</li>
              </ol>
            </div>
          </div>
        </div>
      )}

      {/* ---------- result tiles ---------- */}
      <div className={styles.tilesGrid}>
        <div className={!matchesSearch('gpus required gpu count hardware h100 servers') ? styles.dimmed : ''} data-tour="result-tile-gpus">
          <FlipTile
            dark
          front={
            <>
              <span className={styles.tileLabel}><MicrochipIcon /> GPUs required</span>
              <span className={styles.tileValue}>{Math.round(gpus)}<span className={styles.tileUnit}>× {gpu}</span></span>
              <span className={styles.tileSub}>
                {testResult ? (
                  <>TP={testResult.memory_analysis.tp_size} × {testResult.memory_analysis.replicas} replica{testResult.memory_analysis.replicas > 1 ? 's' : ''} · {testConcurrentUsers} concurrent users</>
                ) : (
                  <>Configure workload below to see results</>
                )}
              </span>
            </>
          }
          back={
            <>
              <div className={styles.backTitle}>How we got {Math.round(gpus)}</div>
              <div className={styles.formula}>
                {testResult ? (
                  <>
                    weight memory = <span className={styles.em}>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</span><br />
                    usable / GPU = <span className={styles.em}>{testResult.memory_analysis.usable_hbm_per_gpu.toFixed(0)} GB</span><br />
                    TP size = ⌈{testResult.memory_analysis.weight_gb.toFixed(0)} ÷ {testResult.memory_analysis.usable_hbm_per_gpu.toFixed(0)}⌉ = <span className={styles.em}>{testResult.memory_analysis.tp_size}</span><br />
                    replicas = {testResult.memory_analysis.replicas}<br />
                    total = <span className={styles.em}>{Math.round(gpus)} GPUs</span>
                  </>
                ) : (
                  <>
                    total memory = <span className={styles.em}>20 GB</span><br />
                    usable / GPU = <span className={styles.em}>72 GB</span> (90% of 80)<br />
                    ⌈20 ÷ 72⌉ = <span className={styles.em}>1 GPU</span><br />
                    peak 3× → range up to 2
                  </>
                )}
              </div>
            </>
          }
        />
        </div>

        <div className={!matchesSearch('weight memory params parameters model size gigabytes gb') ? styles.dimmed : ''}>
          <FlipTile
            front={
              <>
                <span className={styles.tileLabel}><MemoryIcon /> Weight memory <Term k="weightMemory" /></span>
                <span className={styles.tileValue}>{Math.round(weight)}<span className={styles.tileUnit}>GB</span></span>
                <span className={styles.tileSub}>
                  {testResult ? (
                    <>{model.split('/')[1] || model} · {testWeightPrecision}</>
                  ) : (
                    <>{model.split('/')[1] || model}</>
                  )}
                </span>
              </>
            }
            back={
              <>
                <div className={styles.backTitle}>Weight memory</div>
              <div className={styles.formula}>
                {testResult ? (
                  <>
                    precision = <span className={styles.em}>{testWeightPrecision}</span><br />
                    bytes/param = <span className={styles.em}>
                      {testWeightPrecision === 'FP16' ? '2' :
                       testWeightPrecision === 'FP8' ? '1' :
                       testWeightPrecision === 'INT8' ? '1' : '0.5'}
                    </span><br />
                    params × bytes/param<br />
                    = <span className={styles.em}>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</span>
                  </>
                ) : (
                  <>
                    params × bytes/param<br />
                    <span className={styles.em}>8B</span> × <span className={styles.em}>2</span> (BF16)<br />
                    = <span className={styles.em}>16 GB</span>
                  </>
                )}
              </div>
            </>
          }
        />
        </div>

        <div className={!matchesSearch('kv cache key value memory tokens gqa request megabytes mb') ? styles.dimmed : ''}>
          <FlipTile
            front={
              <>
                <span className={styles.tileLabel}><LayerGroupIcon /> KV cache / req <Term k="kvPerReq" /></span>
                <span className={styles.tileValue}>{Math.round(kv)}<span className={styles.tileUnit}>MB</span></span>
                {testResult ? (
                  <span className={styles.tileSub}>
                    {testKVCachePrecision} · {testISL + testOSL} tokens/req · {testConcurrentUsers} users
                  </span>
                ) : (
                  <span className={styles.tileSub}>
                    {testISL + testOSL} tokens/req · {testConcurrentUsers} users
                  </span>
                )}
              </>
            }
            back={
              <>
                <div className={styles.backTitle}>KV cache / request</div>
                <div className={styles.formula}>
                  {testResult && testResult.memory_analysis.kv_cache_used_gb ? (
                    <>
                      total KV used = <span className={styles.em}>{testResult.memory_analysis.kv_cache_used_gb.toFixed(1)} GB</span><br />
                      concurrent users = <span className={styles.em}>{testConcurrentUsers}</span><br />
                      KV / req = {testResult.memory_analysis.kv_cache_used_gb.toFixed(1)} ÷ {testConcurrentUsers}<br />
                      = <span className={styles.em}>{((testResult.memory_analysis.kv_cache_used_gb / testConcurrentUsers) * 1000).toFixed(0)} MB</span><br />
                      precision: <span className={styles.em}>{testKVCachePrecision}</span>
                    </>
                  ) : (
                    <>
                      2 × layers × kv_heads ×<br />
                      head_dim × bytes × tokens<br />
                      2×<span className={styles.em}>32</span>×<span className={styles.em}>8</span>×<span className={styles.em}>128</span>×2 = 128 KB/tok<br />
                      × <span className={styles.em}>150</span> tokens = 19 MB
                    </>
                  )}
                </div>
              </>
            }
          />
        </div>

        <div className={!matchesSearch('cost monthly cloud self-hosted savings price') ? styles.dimmed : ''} data-search="cost monthly cloud self-hosted savings price">
          <FlipTile
            front={
            <>
              <span className={styles.tileLabel}><DollarSignIcon /> MONTHLY COST</span>
              <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                {/* Cloud pricing tile */}
                <div style={{
                  flex: 1,
                  border: '1px solid #d2d2d2',
                  borderRadius: '6px',
                  padding: '14px',
                  transition: 'transform 200ms ease-out',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.06)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="#ee0000">
                      <path d="M8 0C3.6 0 0 3.6 0 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 14c-3.3 0-6-2.7-6-6s2.7-6 6-6 6 2.7 6 6-2.7 6-6 6z"/>
                      <circle cx="8" cy="8" r="3"/>
                    </svg>
                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em', color: '#ee0000', fontWeight: 600 }}>CLOUD</span>
                    <Term k="cloudPricing" />
                  </div>
                  <div style={{ fontSize: '28px', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#151515', marginBottom: '4px' }}>
                    ${((Math.round(gpus) * gpuPricePerHour * 730) / 1000).toFixed(1)}K/mo
                  </div>
                  <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: '#3c3f42' }}>
                    AWS · ${((Math.round(gpus) * gpuPricePerHour * 730 * 60) / 1000).toFixed(0)}K over 5yr
                  </div>
                </div>

                {/* Self-hosted pricing tile */}
                <div style={{
                  flex: 1,
                  border: '1px solid #d2d2d2',
                  borderRadius: '6px',
                  padding: '14px',
                  transition: 'transform 200ms ease-out',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.06)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="#151515">
                      <rect x="2" y="3" width="12" height="2" rx="1"/>
                      <rect x="2" y="7" width="12" height="2" rx="1"/>
                      <rect x="2" y="11" width="12" height="2" rx="1"/>
                    </svg>
                    <span style={{ fontSize: '13px', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em', color: '#151515', fontWeight: 600 }}>SELF-HOSTED</span>
                    <Term k="selfHosted" />
                  </div>
                  <div style={{ fontSize: '28px', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#151515', marginBottom: '4px' }}>
                    ${((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 60 / 1000).toFixed(1)}K/mo
                  </div>
                  <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: '#3c3f42' }}>
                    5yr amort · ${((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 1000).toFixed(0)}K total
                  </div>
                </div>
              </div>

              {/* Savings label */}
              <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '8px', height: '8px', background: '#3d7317', borderRadius: '2px' }}></div>
                <span style={{ fontSize: '13px', fontFamily: 'var(--font-display)', fontWeight: 600, color: '#3d7317' }}>
                  Self-hosted saves ${(((Math.round(gpus) * gpuPricePerHour * 730) - ((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 60)) / 1000).toFixed(1)}K/mo
                </span>
              </div>
            </>
          }
          back={
            <>
              <div className={styles.backTitle}>How we calculated this</div>
              <div className={styles.formula}>
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>Cloud:</strong><br />
                {Math.round(gpus)} GPUs × <span className={styles.em}>${gpuPricePerHour.toFixed(2)}/gpu-hr</span> × <span className={styles.em}>730 hrs</span><br />
                = <span className={styles.em}>${((Math.round(gpus) * gpuPricePerHour * 730) / 1000).toFixed(1)}K/mo</span><br />
                <br />
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>Self-hosted:</strong><br />
                ${((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 1000).toFixed(0)}K ÷ <span className={styles.em}>60 months</span><br />
                = <span className={styles.em}>${((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 60 / 1000).toFixed(1)}K/mo</span><br />
                <span style={{ fontSize: '11.5px', color: '#3c3f42' }}>(hardware amortization only)</span><br />
                <br />
                <strong style={{ color: '#3c3f42', fontSize: '12px' }}>5-year totals:</strong><br />
                Cloud: <span className={styles.em}>${((Math.round(gpus) * gpuPricePerHour * 730 * 60) / 1000).toFixed(0)}K</span><br />
                Hardware: <span className={styles.em}>${((catalogGpuForPricing ? catalogGpuForPricing.hardware_cost_usd * Math.round(gpus) : 0) / 1000).toFixed(0)}K</span><br />
                <br />
                <div style={{ background: 'rgba(255, 193, 7, 0.1)', padding: '8px', borderRadius: '4px', marginTop: '8px' }}>
                  <span style={{ fontSize: '11.5px', color: '#995c00', lineHeight: '1.5' }}>
                    ⚠️ Self-hosted excludes: power (~$X/mo), cooling, staff, networking. Typical full TCO adds 40–80% to this number.
                  </span>
                </div>
              </div>
            </>
          }
        />
        </div>
      </div>

      {/* ---------- Why this GPU count ---------- */}
      {testResult && (
        <div className={styles.card} style={{ marginBottom: 20 }}>
          <div
            className={styles.cardHead}
            onClick={() => setWhyGpuExpanded(!whyGpuExpanded)}
            style={{
              cursor: 'pointer',
              transition: 'background 150ms',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '16px 20px'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
          >
            <span className={styles.cardTitle}>Why this GPU count?</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {testResult.memory_analysis.kv_category && (
                <span style={{
                  fontSize: '11.5px',
                  fontFamily: 'var(--font-mono)',
                  background: '#f5f5f5',
                  border: '1px solid #d2d2d2',
                  borderRadius: '4px',
                  padding: '3px 8px',
                  color: '#3c3f42',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  KV cache: {testResult.memory_analysis.kv_category} · {testResult.memory_analysis.kv_category_label}
                  <Term k="kvCategory" />
                </span>
              )}
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                style={{
                  transition: 'transform 200ms',
                  transform: whyGpuExpanded ? 'rotate(0deg)' : 'rotate(-90deg)'
                }}
              >
                <path d="M4 6 L8 10 L12 6" stroke="#3c3f42" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateRows: whyGpuExpanded ? '1fr' : '0fr',
            transition: 'grid-template-rows 250ms ease-out'
          }}>
            <div style={{ overflow: 'hidden' }}>
              <div className={styles.cardBody}>
                <div style={{ display: 'grid', gap: '12px' }}>
                  <div style={{ padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                    <strong style={{ color: '#0066cc' }}>Memory Breakdown</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • Weight memory: <strong>{testResult.memory_analysis.weight_gb.toFixed(1)} GB</strong> ({testWeightPrecision})<br/>
                      • Weight per GPU: <strong>{testResult.memory_analysis.weight_gb_per_gpu.toFixed(1)} GB</strong><br/>
                      • Usable per GPU: <strong>{testResult.memory_analysis.usable_hbm_per_gpu.toFixed(0)} GB</strong> (90% of {gpu.includes('H200') ? '141' : '80'} GB)<br/>
                      • Tensor Parallel size: <strong>{testResult.memory_analysis.tp_size}</strong> {testResult.memory_analysis.weight_gb > testResult.memory_analysis.usable_hbm_per_gpu ? '(required - weights don\'t fit in 1 GPU)' : '(weights fit, but using for replicas)'}
                    </div>
                  </div>

                  <div style={{ padding: '12px', background: '#fffbf0', borderRadius: '4px' }}>
                    <strong style={{ color: '#995c00' }}>Workload Sizing</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • KV cache used: <strong>{testResult.memory_analysis.kv_cache_used_gb?.toFixed(1) || '—'} GB</strong> ({testKVCachePrecision}, {testConcurrentUsers} users)<br/>
                      • KV cache budget: <strong>{testResult.memory_analysis.kv_cache_budget_gb.toFixed(1)} GB</strong> available<br/>
                      • max_num_seqs: <strong>{testResult.vllm_config.max_num_seqs}</strong><br/>
                      • Replicas: <strong>{testResult.memory_analysis.replicas}</strong> (for throughput/redundancy)
                    </div>
                  </div>

                  <div style={{ padding: '12px', background: '#f0f9ff', borderRadius: '4px' }}>
                    <strong style={{ color: '#0066cc' }}>Bottleneck Analysis</strong>
                    <div style={{ marginTop: '8px', fontSize: '13px', lineHeight: '1.6' }}>
                      • Primary bottleneck: <strong>{testResult.bottleneck_analysis.primary}</strong><br/>
                      • Risk: {testResult.bottleneck_analysis.risk}<br/>
                      {testResult.bottleneck_analysis.fix_suggestions.length > 0 && (
                        <>• Suggestions: {testResult.bottleneck_analysis.fix_suggestions.join(', ')}</>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------- Memory Layout ---------- */}
      {testResult && (
        <div className={styles.card} style={{ marginBottom: 20 }}>
          <div className={styles.cardHead}>
            <span className={styles.cardTitle}>Memory layout per GPU</span>
            <span className={styles.cardHint}>{testResult.memory_analysis.usable_hbm_per_gpu.toFixed(0)} GB usable · {testResult.memory_analysis.tp_size} GPU{testResult.memory_analysis.tp_size > 1 ? 's' : ''} per model instance</span>
          </div>
          <div className={styles.cardBody}>
            <div style={{ marginBottom: '12px' }}>
              <div style={{
                display: 'flex',
                height: '40px',
                borderRadius: '4px',
                overflow: 'hidden',
                border: '1px solid #ddd'
              }}>
                {/* Weights */}
                <div style={{
                  width: `${(testResult.memory_analysis.weight_gb_per_gpu / testResult.memory_analysis.usable_hbm_per_gpu) * 100}%`,
                  background: '#0066cc',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  Weights
                </div>
                {/* KV Cache */}
                <div style={{
                  width: `${((testResult.memory_analysis.kv_cache_used_gb || 0) / testResult.memory_analysis.tp_size / testResult.memory_analysis.usable_hbm_per_gpu) * 100}%`,
                  background: '#f59e0b',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  KV Cache
                </div>
                {/* Reserved/Overhead */}
                <div style={{
                  flex: 1,
                  background: '#e0e0e0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666',
                  fontSize: '12px'
                }}>
                  Reserved
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '13px' }}>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#0066cc', marginRight: '6px', borderRadius: '2px' }}></span>
                Weights: <strong>{testResult.memory_analysis.weight_gb_per_gpu.toFixed(1)} GB</strong>
              </span>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#f59e0b', marginRight: '6px', borderRadius: '2px' }}></span>
                KV Cache: <strong>{((testResult.memory_analysis.kv_cache_used_gb || 0) / testResult.memory_analysis.replicas).toFixed(1)} GB</strong>
              </span>
              <span>
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#e0e0e0', marginRight: '6px', borderRadius: '2px' }}></span>
                Reserved: <strong>{(testResult.memory_analysis.usable_hbm_per_gpu - testResult.memory_analysis.weight_gb_per_gpu - ((testResult.memory_analysis.kv_cache_used_gb || 0) / testResult.memory_analysis.replicas)).toFixed(1)} GB</strong>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ---------- assumptions ---------- */}
      <div
        className={styles.assumptionsHead}
        data-tour="assumptions"
        onClick={() => setAssumptionsExpanded(!assumptionsExpanded)}
        style={{
          cursor: 'pointer',
          transition: 'background 150ms',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 0'
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
      >
        <span className={styles.assumptionsTitle}>Want to change assumptions?</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Button variant="link" isInline onClick={(e) => { e.stopPropagation(); /* Reset logic */ }}>Reset to defaults</Button>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            style={{
              transition: 'transform 200ms',
              transform: assumptionsExpanded ? 'rotate(0deg)' : 'rotate(-90deg)'
            }}
          >
            <path d="M4 6 L8 10 L12 6" stroke="#3c3f42" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateRows: assumptionsExpanded ? '1fr' : '0fr',
        transition: 'grid-template-rows 250ms ease-out'
      }}>
        <div style={{ overflow: 'hidden' }}>
          <p className={styles.assumptionsSub} style={{ marginBottom: 12 }}>
            Every number above comes from these. Open a section to tune it — closed sections show their current values.
          </p>

          <Accordion asDefinitionList={false}>
            {buildAccordionSections().map((sec) => (
              <AccordionItem key={sec.id}>
                <AccordionToggle
                  id={`acc-${sec.id}`}
                  isExpanded={expanded.includes(sec.id)}
                  onClick={() => toggleAcc(sec.id)}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    <span className={styles.cardTitle} style={{ fontSize: 16 }}>{sec.title}</span>
                    {'badge' in sec && sec.badge ? <Label isCompact color={sec.badgeColor as any}>{sec.badge}</Label> : null}
                    {!expanded.includes(sec.id) && (
                      <span className={styles.accSummary}>
                        {sec.summary.map((p) => (
                          <span key={p.k}><span className="k">{p.k}</span> {p.v}</span>
                        ))}
                      </span>
                    )}
                  </span>
                </AccordionToggle>
                <AccordionContent isHidden={!expanded.includes(sec.id)}>
                  <div className={styles.accGrid}>
                    {sec.fields.map((f: any) => (
                      <div key={f.label} className={styles.accField}>
                        <label className={styles.accFieldLabel}>
                          {f.label}{'term' in f && f.term ? <Term k={f.term as any} /> : null}
                        </label>
                        {f.readonly ? (
                          <TextInput value={String(f.value)} aria-label={f.label} isDisabled />
                        ) : f.type === 'select' ? (
                          <FormSelect
                            value={String(f.value)}
                            aria-label={f.label}
                            onChange={(_, val) => f.onChange?.(val)}
                          >
                            {(f.options || [f.value]).map((o: any) => (
                              <FormSelectOption key={o} value={o} label={o} />
                            ))}
                          </FormSelect>
                        ) : f.type === 'range' ? (
                          <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                              <span style={{ fontSize: '13px', fontWeight: '600', fontFamily: 'var(--mono)' }}>
                                {f.value}
                              </span>
                            </div>
                            <input
                              type="range"
                              min={f.min}
                              max={f.max}
                              step={f.step}
                              value={f.value}
                              onChange={(e) => f.onChange?.(Number(e.target.value))}
                              style={{ width: '100%' }}
                            />
                          </div>
                        ) : (
                          <TextInput
                            value={String(f.value)}
                            aria-label={f.label}
                            onChange={(_, val) => f.onChange?.(val)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>

      {/* ---------- footer actions ---------- */}
      <div className={styles.footerRow}>
        <Button variant="secondary">Copy API request</Button>
        <Button variant="secondary">Copy CLI command</Button>
        <Button variant="secondary">Export to Sheets</Button>
        <span className={styles.footerSpacer} />
        <Button variant="primary" onClick={() => setShowSaveModal(true)}>
          Save estimate{savedCount > 0 && ` (${savedCount})`}
        </Button>
      </div>

      {/* Save estimate modal */}
      <SaveEstimateModal
        isOpen={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        onSave={handleSaveEstimate}
        defaultName={generateAutoName()}
      />

      {/* Toast notification */}
      {showToast && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          background: '#151515',
          color: '#fff',
          padding: '16px 20px',
          borderRadius: '6px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          zIndex: 9999,
          fontFamily: 'var(--font-sans)',
          fontSize: '14px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <CheckCircleIcon style={{ color: '#3d7317' }} />
          <span>Estimate saved — <Link href="/compare" style={{ color: '#4da6ff', textDecoration: 'underline' }}>view in Compare →</Link></span>
        </div>
      )}
      <div className={styles.apiPreview}>
        <Button variant="link" isInline onClick={() => setShowApi((s) => !s)}>
          {showApi ? 'Hide' : 'Preview'} API request body
        </Button>
        {showApi && <pre className={styles.apiBody}>{API_PREVIEW}</pre>}
      </div>
    </div>
  );
}

/* ---------- constraint row (unused - kept for future implementation) ---------- */
type Status = 'ok' | 'watch' | 'bottleneck';
function ConstraintRow({ label, detail, status, term }: { label: string; detail: string; status: Status; term?: string }) {
  const dot = status === 'ok' ? styles.conOk : status === 'watch' ? styles.conWatch : styles.conBottleneck;
  const pill = status === 'ok' ? styles.pillOk : status === 'watch' ? styles.pillWatch : styles.pillBottleneck;
  const text = status === 'ok' ? 'OK' : status === 'watch' ? 'Watch' : 'Bottleneck';
  return (
    <div className={styles.constraint}>
      <span className={`${styles.conStatus} ${dot}`} />
      <span className={styles.conLabel}>{label}{term ? <Term k={term as any} /> : null}</span>
      <span className={styles.conDetail}>{detail}</span>
      <span className={`${styles.conPill} ${pill}`}>{text}</span>
    </div>
  );
}


const API_PREVIEW = `{
  "model": { "model_id": "meta-llama/Llama-3.1-8B-Instruct", "max_model_len": "auto" },
  "workload": { "isl_tokens": 100, "osl_tokens": 50, "prefix_cache_hit_rate": 0.0,
                "requests_per_day": 1000000, "peak_multiplier": 3.0 },
  "memory": { "weight_precision": "bf16", "kv_cache_precision": "fp16",
              "gpu_memory_utilization": 0.90 },
  "hardware": { "gpu_type": "H100_80GB" },
  "parallelism": { "tensor_parallel_size": "auto" },
  "engine": { "runtime": "vllm", "block_size": 16, "max_num_seqs": 256,
              "enable_prefix_caching": true, "enable_chunked_prefill": "auto" }
}`;
