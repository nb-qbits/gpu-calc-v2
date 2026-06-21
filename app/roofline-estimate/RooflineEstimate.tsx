'use client';

import { useMemo, useRef, useState } from 'react';
import {
  Grid, GridItem,
  Card, CardHeader, CardBody, CardTitle,
  Form, FormGroup, FormHelperText, FormSelect, FormSelectOption,
  TextInput,
  Label,
  DescriptionList, DescriptionListGroup, DescriptionListTerm, DescriptionListDescription,
  Alert,
  Title, Text, TextContent, TextVariants,
  Divider,
  Button,
  PageSection,
} from '@patternfly/react-core';
import { ROOFLINE_GPU_CATALOG, getRooflineGpuById } from '@/lib/gpu-math/roofline-gpu-catalog';
import { ROOFLINE_MODEL_CATALOG, getRooflineModelById } from '@/lib/gpu-math/roofline-model-catalog';
import { runRooflinePlan, CONFIDENCE_BAND } from '@/lib/gpu-math/roofline-engine';
import type { WorkloadInputs, Dtype, TrafficClass } from '@/lib/gpu-math/roofline-types';

const DEFAULT_INPUTS: WorkloadInputs = {
  model_id: 'llama-3.1-8b',
  gpu_id: 'h100_sxm',
  dtype: 'bf16',
  tp: 1,
  requests_per_day: 86_400,
  peak_multiplier: 3,
  isl: 1024,
  osl: 256,
  ttft_slo_ms: 500,
  traffic_class: 'realtime',
  gpu_mem_util: 0.90,
  runtime: 'vllm',
  // max_num_seqs, prefix_cache_len, prefix_cache_hit_rate intentionally undefined (no cap / no cache)
};

const DTYPE_OPTIONS: Dtype[] = ['bf16', 'fp16', 'fp8', 'mxfp4'];
const TRAFFIC_CLASS_OPTIONS: TrafficClass[] = ['realtime', 'mixed', 'batch'];
const TP_OPTIONS = [1, 2, 4, 8];
const RUNTIME_OPTIONS: Array<{ value: 'vllm' | 'trtllm'; label: string }> = [
  { value: 'vllm',   label: 'vLLM' },
  { value: 'trtllm', label: 'TRT-LLM (+29%)' },
];

function fmtTps(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} T tok/s`
  if (n >= 1e9)  return `${(n / 1e9).toFixed(2)} B tok/s`
  if (n >= 1e6)  return `${(n / 1e6).toFixed(2)} M tok/s`
  if (n >= 1e3)  return `${(n / 1e3).toFixed(1)} K tok/s`
  return `${n.toFixed(1)} tok/s`
}

const CONFIDENCE_COLOR: Record<string, 'green' | 'gold' | 'orange'> = {
  high: 'green', medium: 'gold', default: 'orange',
};
const CONSTRAINT_COLOR: Record<string, 'blue' | 'purple' | 'cyan'> = {
  'prefill-bound': 'blue', 'decode-bound': 'purple', 'kv-memory-bound': 'cyan',
};

const eyebrow: React.CSSProperties = {
  fontSize: 11.5,
  fontFamily: 'var(--font-mono)',
  fontWeight: 500,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: '#54585c',
  marginBottom: 4,
};

const bigNumber: React.CSSProperties = {
  fontFamily: 'var(--font-display)',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 700,
  fontSize: 52,
  lineHeight: 1,
  color: '#151515',
};

const subNote: React.CSSProperties = {
  fontSize: 12,
  color: '#54585c',
  marginTop: 4,
  fontVariantNumeric: 'tabular-nums',
};

function ReplicaBar({ low, recommended, high }: { low: number; recommended: number; high: number }) {
  const max = high * 1.25 || 1;
  const pct = (n: number) => `${Math.min(100, (n / max) * 100).toFixed(1)}%`;
  const rows: Array<{ label: string; value: number; color: string }> = [
    { label: 'Low',         value: low,         color: '#92d400' },
    { label: 'Recommended', value: recommended, color: '#0066cc' },
    { label: 'High',        value: high,        color: '#f0ab00' },
  ];
  return (
    <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#3c3f42' }}>
      {rows.map(({ label, value, color }) => (
        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ width: 92, textAlign: 'right', color: '#54585c', fontSize: 11.5 }}>{label}</span>
          <div style={{ flex: 1, height: 10, background: '#e8e8e8', borderRadius: 4, position: 'relative' }}>
            <div style={{ position: 'absolute', left: 0, width: pct(value), height: '100%', background: color, borderRadius: 4 }} />
          </div>
          <span style={{ width: 28, fontVariantNumeric: 'tabular-nums', textAlign: 'right' }}>{value}</span>
        </div>
      ))}
    </div>
  );
}

export default function RooflineEstimate() {
  const [inputs, setInputs] = useState<WorkloadInputs>(DEFAULT_INPUTS);
  const [maxNumSeqsRaw, setMaxNumSeqsRaw] = useState<string>('');
  const [prefixCacheLenRaw, setPrefixCacheLenRaw] = useState<string>('');
  const [prefixCacheRateRaw, setPrefixCacheRateRaw] = useState<string>('');
  const resultsRef = useRef<HTMLDivElement>(null);

  function set<K extends keyof WorkloadInputs>(key: K, value: WorkloadInputs[K]) {
    setInputs(prev => ({ ...prev, [key]: value }));
  }

  function scrollToResults() {
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const result = useMemo(() => {
    const model = getRooflineModelById(inputs.model_id);
    const gpu   = getRooflineGpuById(inputs.gpu_id);
    if (!model || !gpu) return null;

    const maxNumSeqs      = maxNumSeqsRaw.trim()     ? Math.max(1, parseInt(maxNumSeqsRaw, 10))                        : undefined;
    const prefixCacheLen  = prefixCacheLenRaw.trim()  ? Math.max(0, parseInt(prefixCacheLenRaw, 10))                    : undefined;
    const prefixCacheRate = prefixCacheRateRaw.trim() ? Math.min(1, Math.max(0, parseFloat(prefixCacheRateRaw)))        : undefined;

    return runRooflinePlan(
      { ...inputs, max_num_seqs: maxNumSeqs, prefix_cache_len: prefixCacheLen, prefix_cache_hit_rate: prefixCacheRate },
      model,
      gpu,
    );
  }, [inputs, maxNumSeqsRaw, prefixCacheLenRaw, prefixCacheRateRaw]);

  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{ backgroundColor: 'var(--gc-bg-2, #f5f5f5)', minHeight: '100vh' }}
    >
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '32px 24px' }}>
        <TextContent style={{ marginBottom: 24 }}>
          <Title headingLevel="h1" size="2xl" style={{ fontFamily: 'var(--font-display)', fontWeight: 600 }}>
            Roofline estimate
          </Title>
          <Text component={TextVariants.p} style={{ color: '#3c3f42', maxWidth: 640 }}>
            Physics-based capacity planning for LLM inference. Separates prefill (compute-bound) from decode (bandwidth-bound) to give a replica count with a confidence range.
          </Text>
        </TextContent>

        <Grid hasGutter>
          {/* ── Left panel: inputs ─────────────────────────────── */}
          <GridItem span={4}>
            <Card>
              <CardHeader>
                <CardTitle>Scenario</CardTitle>
              </CardHeader>
              <CardBody>
                <Form>
                  <FormGroup label="Model" fieldId="rl-model">
                    <FormSelect
                      id="rl-model"
                      value={inputs.model_id}
                      onChange={(_e, v) => set('model_id', v)}
                      aria-label="Model"
                    >
                      {ROOFLINE_MODEL_CATALOG.map(m => (
                        <FormSelectOption key={m.id} value={m.id} label={m.display_name} />
                      ))}
                    </FormSelect>
                  </FormGroup>

                  <FormGroup label="GPU" fieldId="rl-gpu">
                    <FormSelect
                      id="rl-gpu"
                      value={inputs.gpu_id}
                      onChange={(_e, v) => set('gpu_id', v)}
                      aria-label="GPU"
                    >
                      {ROOFLINE_GPU_CATALOG.map(g => (
                        <FormSelectOption key={g.id} value={g.id} label={g.display_name} />
                      ))}
                    </FormSelect>
                  </FormGroup>

                  <Grid hasGutter>
                    <GridItem span={6}>
                      <FormGroup label="Precision" fieldId="rl-dtype">
                        <FormSelect
                          id="rl-dtype"
                          value={inputs.dtype}
                          onChange={(_e, v) => set('dtype', v as Dtype)}
                          aria-label="Precision"
                        >
                          {DTYPE_OPTIONS.map(d => (
                            <FormSelectOption key={d} value={d} label={d.toUpperCase()} />
                          ))}
                        </FormSelect>
                      </FormGroup>
                    </GridItem>
                    <GridItem span={6}>
                      <FormGroup label="Tensor parallel" fieldId="rl-tp">
                        <FormSelect
                          id="rl-tp"
                          value={String(inputs.tp)}
                          onChange={(_e, v) => set('tp', Number(v))}
                          aria-label="Tensor parallel"
                        >
                          {TP_OPTIONS.map(t => (
                            <FormSelectOption key={t} value={String(t)} label={`tp=${t}`} />
                          ))}
                        </FormSelect>
                      </FormGroup>
                    </GridItem>
                  </Grid>

                  <Divider style={{ margin: '4px 0' }} />

                  <FormGroup label="Requests per day" fieldId="rl-rpd">
                    <TextInput
                      id="rl-rpd"
                      type="number"
                      value={inputs.requests_per_day}
                      onChange={(_e, v) => set('requests_per_day', Math.max(1, Number(v)))}
                    />
                  </FormGroup>

                  <FormGroup label="Peak multiplier" fieldId="rl-peak">
                    <TextInput
                      id="rl-peak"
                      type="number"
                      value={inputs.peak_multiplier}
                      onChange={(_e, v) => set('peak_multiplier', Math.max(1, Number(v)))}
                    />
                  </FormGroup>

                  <Grid hasGutter>
                    <GridItem span={6}>
                      <FormGroup label="Input length (tokens)" fieldId="rl-isl">
                        <TextInput
                          id="rl-isl"
                          type="number"
                          value={inputs.isl}
                          onChange={(_e, v) => set('isl', Math.max(1, Number(v)))}
                        />
                      </FormGroup>
                    </GridItem>
                    <GridItem span={6}>
                      <FormGroup label="Output length (tokens)" fieldId="rl-osl">
                        <TextInput
                          id="rl-osl"
                          type="number"
                          value={inputs.osl}
                          onChange={(_e, v) => set('osl', Math.max(1, Number(v)))}
                        />
                      </FormGroup>
                    </GridItem>
                  </Grid>

                  <FormGroup label="TTFT SLO (ms)" fieldId="rl-ttft">
                    <TextInput
                      id="rl-ttft"
                      type="number"
                      value={inputs.ttft_slo_ms}
                      onChange={(_e, v) => set('ttft_slo_ms', Math.max(1, Number(v)))}
                    />
                  </FormGroup>

                  <FormGroup label="Traffic class" fieldId="rl-tclass">
                    <FormSelect
                      id="rl-tclass"
                      value={inputs.traffic_class}
                      onChange={(_e, v) => set('traffic_class', v as TrafficClass)}
                      aria-label="Traffic class"
                    >
                      {TRAFFIC_CLASS_OPTIONS.map(tc => (
                        <FormSelectOption
                          key={tc}
                          value={tc}
                          label={tc.charAt(0).toUpperCase() + tc.slice(1)}
                        />
                      ))}
                    </FormSelect>
                  </FormGroup>

                  <FormGroup label="Runtime" fieldId="rl-runtime">
                    <FormSelect
                      id="rl-runtime"
                      value={inputs.runtime ?? 'vllm'}
                      onChange={(_e, v) => set('runtime', v as 'vllm' | 'trtllm')}
                      aria-label="Runtime"
                    >
                      {RUNTIME_OPTIONS.map(r => (
                        <FormSelectOption key={r.value} value={r.value} label={r.label} />
                      ))}
                    </FormSelect>
                  </FormGroup>

                  <Divider style={{ margin: '4px 0' }} />

                  <FormGroup
                    label="Max concurrent seqs"
                    fieldId="rl-maxseqs"
                  >
                    <TextInput
                      id="rl-maxseqs"
                      type="number"
                      value={maxNumSeqsRaw}
                      placeholder="no cap"
                      onChange={(_e, v) => setMaxNumSeqsRaw(v)}
                    />
                    <FormHelperText style={{ fontSize: 11.5, color: '#54585c' }}>
                      Blank = limited only by KV budget. Set to match --max-num-seqs.
                    </FormHelperText>
                  </FormGroup>

                  <Grid hasGutter>
                    <GridItem span={6}>
                      <FormGroup
                        label="Prefix cache length (tokens)"
                        fieldId="rl-plen"
                      >
                        <TextInput
                          id="rl-plen"
                          type="number"
                          value={prefixCacheLenRaw}
                          placeholder="0"
                          onChange={(_e, v) => setPrefixCacheLenRaw(v)}
                        />
                        <FormHelperText style={{ fontSize: 11.5, color: '#54585c' }}>
                          Shared prefix / system prompt length.
                        </FormHelperText>
                      </FormGroup>
                    </GridItem>
                    <GridItem span={6}>
                      <FormGroup
                        label="Prefix hit rate"
                        fieldId="rl-phit"
                      >
                        <TextInput
                          id="rl-phit"
                          type="number"
                          value={prefixCacheRateRaw}
                          placeholder="0.0"
                          onChange={(_e, v) => setPrefixCacheRateRaw(v)}
                        />
                        <FormHelperText style={{ fontSize: 11.5, color: '#54585c' }}>
                          Fraction of requests that hit the cache (0–1).
                        </FormHelperText>
                        {prefixCacheRateRaw.trim() && !prefixCacheLenRaw.trim() && (
                          <FormHelperText style={{ fontSize: 11.5, color: '#c9190b' }}>
                            Set prefix length to enable caching.
                          </FormHelperText>
                        )}
                      </FormGroup>
                    </GridItem>
                  </Grid>

                  <div style={{ paddingTop: 8 }}>
                    <Button variant="primary" isBlock onClick={scrollToResults}>
                      Run estimate
                    </Button>
                    <p style={{ fontSize: 11.5, color: '#3c3f42', marginTop: 6, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                      Results update automatically
                    </p>
                  </div>
                </Form>
              </CardBody>
            </Card>
          </GridItem>

          {/* ── Right panel: results ───────────────────────────── */}
          <GridItem span={8}>
          <div ref={resultsRef}>
            {!result && (
              <Card>
                <CardBody>
                  <Text style={{ color: '#3c3f42' }}>Configure a scenario to see the estimate.</Text>
                </CardBody>
              </Card>
            )}

            {result && !result.ok && (
              <Alert variant="danger" title="Planning error" isInline style={{ marginBottom: 16 }}>
                {result.error.message}
              </Alert>
            )}

            {result?.ok && (() => {
              const est = result.estimate;
              return (
                <Grid hasGutter>

                  {/* Sizing headline */}
                  <GridItem span={12}>
                    <Card style={{ borderLeft: '4px solid #0066cc' }}>
                      <CardBody>
                        <Grid hasGutter>
                          <GridItem span={4} style={{ textAlign: 'center', borderRight: '1px solid #e8e8e8' }}>
                            <div style={eyebrow}>Recommended replicas</div>
                            <div style={bigNumber}>{est.replicas}</div>
                            <div style={subNote}>range: {est.replicas_low} – {est.replicas_high}</div>
                          </GridItem>
                          <GridItem span={4} style={{ textAlign: 'center', borderRight: '1px solid #e8e8e8' }}>
                            <div style={eyebrow}>Total GPUs</div>
                            <div style={bigNumber}>{est.total_gpus}</div>
                            <div style={subNote}>{est.replicas} × tp={est.tp_used}</div>
                          </GridItem>
                          <GridItem span={4} style={{ textAlign: 'center' }}>
                            <div style={eyebrow}>Confidence</div>
                            <div style={{ marginTop: 10 }}>
                              <Label color={CONFIDENCE_COLOR[est.confidence]} style={{ fontSize: 15, padding: '5px 14px' }}>
                                {est.confidence.toUpperCase()}
                              </Label>
                            </div>
                            <div style={subNote}>
                              ±{Math.round((CONFIDENCE_BAND[est.confidence] ?? 0.25) * 100)}%
                            </div>
                          </GridItem>
                        </Grid>

                        <div style={{ marginTop: 20, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          <Label color={CONSTRAINT_COLOR[est.binding_constraint]}>
                            {est.binding_constraint}
                          </Label>
                          <Label color={est.ttft_estimate.slo_met ? 'green' : 'red'}>
                            TTFT {est.ttft_estimate.ttft_ms.toFixed(0)} ms
                            {est.ttft_estimate.slo_met ? ' ✓ SLO met' : ' ✗ SLO breach'}
                          </Label>
                        </div>

                        <div style={{ marginTop: 20 }}>
                          <div style={eyebrow}>Replica range</div>
                          <ReplicaBar
                            low={est.replicas_low}
                            recommended={est.replicas}
                            high={est.replicas_high}
                          />
                        </div>
                      </CardBody>
                    </Card>
                  </GridItem>

                  {/* Breakdown + ceilings */}
                  <GridItem span={6}>
                    <Card>
                      <CardHeader><CardTitle>Replica breakdown</CardTitle></CardHeader>
                      <CardBody>
                        <DescriptionList isCompact columnModifier={{ default: '1Col' }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Prefill-driven</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.replicas_prefill}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Decode-driven</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.replicas_decode}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Concurrency-driven</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.replicas_concurrency}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Headroom factor</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.headroom_factor.toFixed(2)}×
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>

                  <GridItem span={6}>
                    <Card>
                      <CardHeader><CardTitle>Per-GPU ceilings</CardTitle></CardHeader>
                      <CardBody>
                        <DescriptionList isCompact columnModifier={{ default: '1Col' }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Prefill ceiling</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {fmtTps(est.prefill_tps_gpu)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Decode ceiling</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {fmtTps(est.decode_tps_gpu)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Max concurrent seqs</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.kv_budget.max_concurrent_seqs}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          {est.kv_budget.effective_context_tokens !== (inputs.isl + inputs.osl) && (
                            <DescriptionListGroup>
                              <DescriptionListTerm>Effective context (sliding window)</DescriptionListTerm>
                              <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                                {est.kv_budget.effective_context_tokens.toFixed(0)} tokens/layer
                              </DescriptionListDescription>
                            </DescriptionListGroup>
                          )}
                          <DescriptionListGroup>
                            <DescriptionListTerm>MFU (prefill)</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {(est.mfu_used * 100).toFixed(0)}%
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>TPOT</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.tpot_ms.toFixed(1)} ms/tok
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>

                  {/* Traffic + KV budget */}
                  <GridItem span={6}>
                    <Card>
                      <CardHeader><CardTitle>Traffic</CardTitle></CardHeader>
                      <CardBody>
                        <DescriptionList isCompact columnModifier={{ default: '1Col' }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Avg RPS</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.traffic.avg_rps.toFixed(2)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Peak RPS</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.traffic.peak_rps.toFixed(2)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Peak input tok/s</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {fmtTps(est.traffic.input_tps_peak)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Peak output tok/s</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {fmtTps(est.traffic.output_tps_peak)}
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>

                  <GridItem span={6}>
                    <Card>
                      <CardHeader><CardTitle>KV budget (per GPU)</CardTitle></CardHeader>
                      <CardBody>
                        <DescriptionList isCompact columnModifier={{ default: '1Col' }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Weights</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {(est.kv_budget.weights_resident_bytes / 1e9).toFixed(1)} GB
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>KV cache budget</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {(est.kv_budget.kv_cache_budget_bytes / 1e9).toFixed(1)} GB
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>KV ratio</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.kv_ratio.toFixed(1)}×
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Eff. batch size</DescriptionListTerm>
                            <DescriptionListDescription style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {est.eff_batch_used} seqs
                            </DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>

                  {/* Warnings */}
                  {est.warnings.length > 0 && (
                    <GridItem span={12}>
                      {est.warnings.map((w, i) => (
                        <Alert key={i} variant="warning" title={w} isInline isPlain style={{ marginBottom: 8 }} />
                      ))}
                    </GridItem>
                  )}

                  {/* Assumptions */}
                  <GridItem span={12}>
                    <Card isPlain>
                      <CardHeader>
                        <CardTitle style={{ fontSize: 13, color: '#54585c', fontFamily: 'var(--font-mono)' }}>
                          Assumptions
                        </CardTitle>
                      </CardHeader>
                      <CardBody>
                        <ul style={{
                          margin: 0, paddingLeft: 18,
                          fontSize: 12, color: '#54585c',
                          fontFamily: 'var(--font-mono)', lineHeight: 1.7,
                        }}>
                          {est.assumptions.map((a, i) => <li key={i}>{a}</li>)}
                        </ul>
                      </CardBody>
                    </Card>
                  </GridItem>

                </Grid>
              );
            })()}
          </div>
          </GridItem>
        </Grid>
      </div>
    </PageSection>
  );
}
