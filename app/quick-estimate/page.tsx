"use client";

import React, { useState, useMemo } from "react";
import {
  Alert,
  Button,
  Card, CardBody, CardHeader, CardTitle,
  DescriptionList, DescriptionListDescription, DescriptionListGroup, DescriptionListTerm,
  ExpandableSection,
  Flex, FlexItem,
  FormSelect, FormSelectOption,
  Grid, GridItem,
  PageSection,
  Progress,
  Slider,
  Split, SplitItem,
  Stack, StackItem,
  Text, TextContent, TextVariants,
  TextInput,
  Tile,
  Title,
  ToggleGroup, ToggleGroupItem,
} from "@patternfly/react-core";
import {
  ArrowLeftIcon,
  BoltIcon,
  ChartBarIcon,
  CogIcon,
  CubeIcon,
  DollarSignIcon,
  MemoryIcon,
  MicrochipIcon,
  ServerIcon,
  TrendUpIcon,
} from "@patternfly/react-icons";

import {
  MODEL_CATALOG,
  GPU_CATALOG,
  DEFAULT_GPU,
  runQuickEstimate,
  PRECISION_BYTES,
  CONTEXT_TOKENS,
  CONTEXT_LABELS,
  CONTEXT_SUBLABELS,
  type Precision,
  type ContextLength,
  type DeploymentType,
  type TensorParallelism,
  type QuickEstimateResult,
} from "@/lib/gpu-math";
import { type Vendor } from "@/lib/gpu-math/models";

// ─── helpers ──────────────────────────────────────────────────────────────────

function fmtGb(gb: number): string {
  if (gb >= 1000) return `${(gb / 1000).toFixed(1)} TB`;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(gb * 1024).toFixed(0)} MB`;
}

function fmtMoney(usd: number): string {
  if (usd >= 1_000_000) return `$${(usd / 1_000_000).toFixed(1)}M`;
  if (usd >= 1_000) return `$${(usd / 1_000).toFixed(1)}k`;
  return `$${usd.toFixed(0)}`;
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

function fmtBytes(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(2)} MB`;
  if (bytes >= 1_024) return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${bytes.toFixed(0)} B`;
}

function fmtMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms.toFixed(0)}ms`;
}

const USER_PRESETS = [
  { label: "<10",  value: 9,    sub: "Small team"   },
  { label: "30",   value: 30,   sub: "Department"   },
  { label: "100",  value: 100,  sub: "Organization" },
  { label: "500",  value: 500,  sub: "Platform"     },
  { label: "1K+",  value: 1000, sub: "Enterprise"   },
] as const;

const VENDORS: Array<{ label: string; value: Vendor | "all" }> = [
  { label: "All",     value: "all"     },
  { label: "Meta",    value: "Meta"    },
  { label: "Google",  value: "Google"  },
  { label: "Mistral", value: "Mistral" },
  { label: "NVIDIA",  value: "NVIDIA"  },
  { label: "Qwen",    value: "Qwen"    },
  { label: "RedHat",  value: "RedHat"  },
  { label: "Other",   value: "Other"   },
];

const DEPLOYMENT_OPTIONS = [
  { key: "cloud",  label: "Cloud only",   sub: "Pay per hour, no hardware" },
  { key: "onprem", label: "On-prem only", sub: "Own hardware, capex model" },
  { key: "hybrid", label: "Hybrid",       sub: "Own base, rent burst"      },
] as const;

// ─── page state ───────────────────────────────────────────────────────────────

interface PageState {
  view: "input" | "results";
  selectedModelId: string;
  concurrentUsers: number;
  customUsersText: string;
  contextLength: ContextLength;
  deploymentType: DeploymentType;
  precision: Precision;
  tensorParallelism: TensorParallelism;
  memUtilization: number;
  vendorFilter: Vendor | "all";
  advancedOpen: boolean;
  controlsVisible: boolean;
  expandedSections: Record<string, boolean>;
  gpuId: string;
}

const DEFAULT_STATE: PageState = {
  view: "input",
  selectedModelId: "gemma-2-2b",
  concurrentUsers: 100,
  customUsersText: "",
  contextLength: "medium",
  deploymentType: "onprem",
  precision: "fp16",
  tensorParallelism: "auto",
  memUtilization: 0.9,
  vendorFilter: "all",
  advancedOpen: false,
  controlsVisible: true,
  expandedSections: {
    modelWeights: true,
    kvCache: true,
    vramBudget: true,
    precision: false,
    gpuCapability: false,
    throughput: false,
    cost: false,
    sensitivity: true,
  },
  gpuId: "h100",
};

// ─── InputView ────────────────────────────────────────────────────────────────

function InputView({ state, setState }: {
  state: PageState;
  setState: React.Dispatch<React.SetStateAction<PageState>>;
}) {
  const filteredModels = MODEL_CATALOG.filter(
    (m) => state.vendorFilter === "all" || m.vendor === state.vendorFilter
  );

  const presetMatch = USER_PRESETS.find((p) => p.value === state.concurrentUsers);
  const contextToken = CONTEXT_TOKENS[state.contextLength];
  const deployLabel = DEPLOYMENT_OPTIONS.find((d) => d.key === state.deploymentType)?.label ?? "Cloud only";

  return (
    <PageSection>
      <div className="rh-content-narrow">
      <Stack hasGutter>
        {/* title */}
        <StackItem>
          <TextContent className="pf-v5-u-text-align-center">
            <Title headingLevel="h1" size="2xl">Configure your LLM deployment</Title>
            <Text component={TextVariants.p} className="pf-v5-u-color-200">
              We&apos;ll estimate GPU requirements based on your key parameters
            </Text>
          </TextContent>
        </StackItem>

        {/* model selection */}
        <StackItem>
          <Stack hasGutter>
            <StackItem>
              <Title headingLevel="h2" size="lg">What model are you serving?</Title>
            </StackItem>

            <StackItem>
              <Flex spaceItems={{ default: "spaceItemsSm" }} flexWrap={{ default: "wrap" }}>
                {VENDORS.map((v) => {
                  const active = state.vendorFilter === v.value;
                  return (
                    <FlexItem key={v.value}>
                      <Button
                        variant={active ? "primary" : "control"}
                        onClick={() => setState((s) => ({ ...s, vendorFilter: v.value as Vendor | "all" }))}
                      >
                        {v.label}
                      </Button>
                    </FlexItem>
                  );
                })}
              </Flex>
            </StackItem>

            <StackItem>
              <Grid hasGutter>
                {filteredModels.map((model) => {
                  const selected = model.id === state.selectedModelId;
                  return (
                    <GridItem span={3} key={model.id}>
                      <Card
                        isSelectable
                        isSelected={selected}
                        isCompact
                        onClick={() => setState((s) => ({ ...s, selectedModelId: model.id }))}
                      >
                        <CardBody>
                          <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                            {model.name}
                          </Text>
                          <Text component={TextVariants.small} className="pf-v5-u-color-200">
                            {model.paramsBillions}B · {model.vendor}
                            {model.activeFraction < 1 ? " · MoE" : ""}
                          </Text>
                        </CardBody>
                      </Card>
                    </GridItem>
                  );
                })}
              </Grid>
            </StackItem>

            <StackItem>
              <TextInput
                placeholder="Or enter HuggingFace model ID..."
                aria-label="HuggingFace model ID"
              />
            </StackItem>
          </Stack>
        </StackItem>

        {/* concurrent users */}
        <StackItem>
          <Stack hasGutter>
            <StackItem>
              <Flex alignItems={{ default: "alignItemsBaseline" }} spaceItems={{ default: "spaceItemsSm" }}>
                <FlexItem>
                  <Title headingLevel="h2" size="lg">How many people at the same time?</Title>
                </FlexItem>
                <FlexItem>
                  <Text component={TextVariants.small} className="pf-v5-u-color-200">
                    Peak concurrent users – not total users
                  </Text>
                </FlexItem>
              </Flex>
            </StackItem>

            <StackItem>
              <Flex spaceItems={{ default: "spaceItemsMd" }} flexWrap={{ default: "wrap" }}>
                {USER_PRESETS.map((p) => (
                  <FlexItem key={p.value}>
                    <Tile
                      title={p.label}
                      isSelected={state.concurrentUsers === p.value}
                      onClick={() => setState((s) => ({ ...s, concurrentUsers: p.value, customUsersText: "" }))}
                      isStacked
                    >
                      {p.sub}
                    </Tile>
                  </FlexItem>
                ))}
              </Flex>
            </StackItem>

            <StackItem>
              <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                <FlexItem>
                  <Text component={TextVariants.small}>Or type exact:</Text>
                </FlexItem>
                <FlexItem>
                  <TextInput
                    type="number"
                    placeholder="e.g. 250"
                    value={state.customUsersText}
                    onChange={(_e, v) => {
                      const n = parseInt(v, 10);
                      setState((s) => ({
                        ...s,
                        customUsersText: v,
                        concurrentUsers: isNaN(n) ? s.concurrentUsers : n,
                      }));
                    }}
                    style={{ width: 140 }}
                    aria-label="Custom concurrent users"
                  />
                </FlexItem>
              </Flex>
            </StackItem>
          </Stack>
        </StackItem>

        {/* defaults info */}
        <StackItem>
          <Alert
            variant="info"
            isInline
            title="Using default parameters"
          >
            <ul>
              <li>
                Conversation length: {CONTEXT_LABELS[state.contextLength]} ({(contextToken / 1024).toFixed(0)}K tokens)
              </li>
              <li>Deployment type: {deployLabel}</li>
            </ul>
          </Alert>
        </StackItem>

        {/* advanced options */}
        <StackItem>
          <ExpandableSection
            toggleText="Advanced options"
            isExpanded={state.advancedOpen}
            onToggle={(_e, open) => setState((s) => ({ ...s, advancedOpen: open }))}
          >
            <Stack hasGutter>
              <StackItem>
                <Stack hasGutter>
                  <StackItem>
                    <Flex alignItems={{ default: "alignItemsBaseline" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                          Conversation length
                        </Text>
                      </FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.small} className="pf-v5-u-color-200">
                          How much memory each user needs
                        </Text>
                      </FlexItem>
                    </Flex>
                  </StackItem>
                  <StackItem>
                    <Flex spaceItems={{ default: "spaceItemsMd" }}>
                      {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                        <FlexItem key={cl}>
                          <Tile
                            title={CONTEXT_LABELS[cl]}
                            isSelected={state.contextLength === cl}
                            onClick={() => setState((s) => ({ ...s, contextLength: cl }))}
                            isStacked
                          >
                            {CONTEXT_SUBLABELS[cl]}
                          </Tile>
                        </FlexItem>
                      ))}
                    </Flex>
                  </StackItem>
                </Stack>
              </StackItem>

              <StackItem>
                <Stack hasGutter>
                  <StackItem>
                    <Flex alignItems={{ default: "alignItemsBaseline" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                          Where do you want to run this?
                        </Text>
                      </FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.small} className="pf-v5-u-color-200">
                          Sets your cost model default
                        </Text>
                      </FlexItem>
                    </Flex>
                  </StackItem>
                  <StackItem>
                    <Flex spaceItems={{ default: "spaceItemsMd" }}>
                      {DEPLOYMENT_OPTIONS.map((dt) => (
                        <FlexItem key={dt.key}>
                          <Tile
                            title={dt.label}
                            isSelected={state.deploymentType === dt.key}
                            onClick={() => setState((s) => ({ ...s, deploymentType: dt.key }))}
                            isStacked
                          >
                            {dt.sub}
                          </Tile>
                        </FlexItem>
                      ))}
                    </Flex>
                  </StackItem>
                </Stack>
              </StackItem>
            </Stack>
          </ExpandableSection>
        </StackItem>

        {/* calculate */}
        <StackItem>
          <Flex justifyContent={{ default: "justifyContentFlexEnd" }}>
            <FlexItem>
              <Button
                variant="primary"
                size="lg"
                onClick={() => setState((s) => ({ ...s, view: "results" }))}
              >
                Calculate GPU requirements
              </Button>
            </FlexItem>
          </Flex>
        </StackItem>
      </Stack>
      </div>
    </PageSection>
  );
}

// ─── ResultsView helpers ──────────────────────────────────────────────────────

function SummaryCard({
  icon, title, value, sub, isDanger,
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  sub: string;
  isDanger?: boolean;
}) {
  return (
    <Card isFlat isFullHeight>
      <CardBody>
        <Stack>
          <StackItem>
            <Text component={TextVariants.small} className="pf-v5-u-color-200">
              {icon} {title}
            </Text>
          </StackItem>
          <StackItem>
            <Title
              headingLevel="h3"
              size="2xl"
              className={isDanger ? "pf-v5-u-primary-color-100" : ""}
            >
              {value}
            </Title>
          </StackItem>
          <StackItem>
            <Text component={TextVariants.small} className="pf-v5-u-color-200">{sub}</Text>
          </StackItem>
        </Stack>
      </CardBody>
    </Card>
  );
}

function ResultSection({
  id, icon, title, subtitle, expanded, onToggle, children,
}: {
  id: string;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  expanded: boolean;
  onToggle: (id: string, open: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <Card isFlat className="pf-v5-u-mb-sm">
      <ExpandableSection
        toggleContent={
          <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
            <FlexItem><span className="pf-v5-u-primary-color-100">{icon}</span></FlexItem>
            <FlexItem>
              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>
                {title}
              </Text>
              <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>
                {subtitle}
              </Text>
            </FlexItem>
          </Flex>
        }
        isExpanded={expanded}
        onToggle={(_e, open) => onToggle(id, open)}
      >
        <CardBody>{children}</CardBody>
      </ExpandableSection>
    </Card>
  );
}

// ─── ResultsView ──────────────────────────────────────────────────────────────

function ResultsView({ state, setState }: {
  state: PageState;
  setState: React.Dispatch<React.SetStateAction<PageState>>;
}) {
  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId)!;
  const selectedGpu = GPU_CATALOG.find((g) => g.id === state.gpuId) ?? DEFAULT_GPU;

  const result: QuickEstimateResult = useMemo(
    () =>
      runQuickEstimate({
        model: selectedModel,
        concurrentUsers: state.concurrentUsers,
        contextLength: state.contextLength,
        precision: state.precision,
        deploymentType: state.deploymentType,
        gpu: selectedGpu,
        tensorParallelism: state.tensorParallelism,
        memUtilization: state.memUtilization,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedModel, state.concurrentUsers, state.contextLength, state.precision,
      state.deploymentType, selectedGpu, state.tensorParallelism, state.memUtilization]
  );

  const { precision } = state;
  const { weight: bytesPerWeight } = PRECISION_BYTES[precision];
  const deployLabel = state.deploymentType === "cloud" ? "cloud" : state.deploymentType === "onprem" ? "on-prem" : "hybrid";

  function toggleSection(id: string, open: boolean) {
    setState((s) => ({ ...s, expandedSections: { ...s.expandedSections, [id]: open } }));
  }

  // Precision variants for comparison
  const precisionVariants = (["fp16", "int8", "int4", "mixed"] as Precision[]).map((p) => {
    const { weight: bw, kv: bkv, label } = PRECISION_BYTES[p];
    const wGb = selectedModel.paramsBillions * selectedModel.activeFraction * bw;
    const kvGb = (2 * selectedModel.numLayers * selectedModel.hiddenSize * bkv *
      state.concurrentUsers * CONTEXT_TOKENS[state.contextLength]) / 1e9;
    const actGb = wGb * 0.2;
    const kvKb = (2 * selectedModel.numLayers * selectedModel.hiddenSize * bkv) / 1024;
    return { key: p, label, total: wGb + kvGb + actGb, kvKb };
  });

  const maxSensBar = Math.max(...result.sensitivity.map((s) => s.usersCapacity), state.concurrentUsers, 1);

  return (
    <PageSection>
      {/* header row */}
      <Flex
        justifyContent={{ default: "justifyContentSpaceBetween" }}
        alignItems={{ default: "alignItemsCenter" }}
        className="pf-v5-u-mb-md"
      >
        <FlexItem>
          <Button
            variant="link"
            icon={<ArrowLeftIcon />}
            onClick={() => setState((s) => ({ ...s, view: "input" }))}
          >
            Back to inputs
          </Button>
        </FlexItem>
        <FlexItem>
          <Button
            variant="secondary"
            icon={<CogIcon />}
            onClick={() => setState((s) => ({ ...s, controlsVisible: !s.controlsVisible }))}
          >
            {state.controlsVisible ? "Hide controls" : "Show controls"}
          </Button>
        </FlexItem>
      </Flex>

      <Title headingLevel="h1" size="xl" className="pf-v5-u-mb-lg">
        GPU requirements for {selectedModel.name}
      </Title>

      <Grid hasGutter>
        {/* ── main content ── */}
        <GridItem span={state.controlsVisible ? 8 : 12}>
          <Stack hasGutter>
            {/* summary cards */}
            <StackItem>
              <Grid hasGutter>
                <GridItem span={3}>
                  <SummaryCard icon={<ServerIcon />} title="Replicas needed"
                    value={`${result.replicasNeeded}+`} sub={`${result.gpusPerReplica} × GPU`} isDanger />
                </GridItem>
                <GridItem span={3}>
                  <SummaryCard icon={<MicrochipIcon />} title="GPU availability"
                    value={`${result.cloudAvailabilityPct}%`} sub={`TPU availability: ${result.tpuAvailabilityPct}%`} />
                </GridItem>
                <GridItem span={3}>
                  <SummaryCard icon={<DollarSignIcon />} title="Server costs"
                    value={fmtMoney(result.costPerMonth)} sub={`/mo ${deployLabel}`} />
                </GridItem>
                <GridItem span={3}>
                  <SummaryCard icon={<TrendUpIcon />} title="Idle server costs"
                    value={fmtMoney(result.idleServerCostsPerMonth)} sub="30-min+ idle" />
                </GridItem>
              </Grid>
            </StackItem>

            {/* Model weights */}
            <StackItem>
              <ResultSection id="modelWeights" icon={<ServerIcon />}
                title="Model weights" subtitle="How much memory each user needs"
                expanded={state.expandedSections.modelWeights} onToggle={toggleSection}>
                <DescriptionList columnModifier={{ default: "3Col" }} className="pf-v5-u-mb-md">
                  <DescriptionListGroup>
                    <DescriptionListTerm>Memory string</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">{selectedModel.paramsBillions}B params</span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>{PRECISION_BYTES[precision].label} ({bytesPerWeight} bytes)</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">{fmtGb(result.weightsGb)}</span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Active params</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">
                        {(selectedModel.paramsBillions * selectedModel.activeFraction).toFixed(1)}B
                        ({(selectedModel.activeFraction * 100).toFixed(0)}%)
                      </span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
                <Alert variant="custom" isInline isPlain title={
                  `${selectedModel.paramsBillions}B params × ${bytesPerWeight} bytes (${PRECISION_BYTES[precision].label}) = ${fmtGb(result.weightsGb)}`
                }>
                  <Text component={TextVariants.small} className="pf-v5-u-font-family-mono">
                    Calculation: {selectedModel.paramsBillions}B × {bytesPerWeight} bytes = {fmtGb(result.weightsGb)}
                  </Text>
                </Alert>
              </ResultSection>
            </StackItem>

            {/* KV cache */}
            <StackItem>
              <ResultSection id="kvCache" icon={<MemoryIcon />}
                title="KV cache" subtitle="Memory for attention mechanism"
                expanded={state.expandedSections.kvCache} onToggle={toggleSection}>
                <DescriptionList columnModifier={{ default: "3Col" }} className="pf-v5-u-mb-md">
                  <DescriptionListGroup>
                    <DescriptionListTerm>KV cache</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">{fmtGb(result.kvTotalGb)}</span>
                      <br />
                      <Text component={TextVariants.small} className="pf-v5-u-color-200">
                        {fmtBytes(result.kvBytesPerToken)} per token
                      </Text>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Total context</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">{fmtTokens(result.totalContextTokens)} tokens</span>
                      <br />
                      <Text component={TextVariants.small} className="pf-v5-u-color-200">
                        {state.concurrentUsers} users × {fmtTokens(CONTEXT_TOKENS[state.contextLength])} tokens
                      </Text>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Layers</DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold">{selectedModel.numLayers}</span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
                <Alert variant="custom" isInline isPlain title="Formula">
                  <Text component={TextVariants.small} className="pf-v5-u-font-family-mono">
                    2 × {selectedModel.numLayers} layers × {selectedModel.hiddenSize} hidden × {PRECISION_BYTES[precision].kv} bytes
                    = {fmtBytes(result.kvBytesPerToken)} per token
                    <br />
                    {state.concurrentUsers} users × {fmtTokens(CONTEXT_TOKENS[state.contextLength])} tokens
                    = {fmtTokens(result.totalContextTokens)} tokens total
                    <br />
                    Total KV cache: {fmtGb(result.kvTotalGb)}
                  </Text>
                </Alert>
              </ResultSection>
            </StackItem>

            {/* VRAM budget */}
            <StackItem>
              <ResultSection id="vramBudget" icon={<CubeIcon />}
                title="VRAM budget breakdown" subtitle="Total per-replica memory"
                expanded={state.expandedSections.vramBudget} onToggle={toggleSection}>
                <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "20ch" }}>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Model weights ({PRECISION_BYTES[precision].label})</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.weightsGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>KV cache ({state.concurrentUsers} users)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.kvTotalGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Activation overhead</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.activationGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>
                      <span className="pf-v5-u-font-weight-bold">TOTAL (≈) per replica</span>
                    </DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">{fmtGb(result.totalVramGb)}</span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </ResultSection>
            </StackItem>

            {/* Precision comparison */}
            <StackItem>
              <ResultSection id="precision" icon={<BoltIcon />}
                title="Precision & memory per replica" subtitle="Quantization options"
                expanded={state.expandedSections.precision} onToggle={toggleSection}>
                <Flex spaceItems={{ default: "spaceItemsMd" }}>
                  {precisionVariants.map((pv) => {
                    const selected = precision === pv.key;
                    return (
                      <FlexItem key={pv.key}>
                        <Tile
                          title={pv.label}
                          isSelected={selected}
                          onClick={() => setState((s) => ({ ...s, precision: pv.key }))}
                          isStacked
                        >
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.small} className="pf-v5-u-color-200">
                                KV: {pv.kvKb >= 1 ? `${pv.kvKb.toFixed(1)} KB` : `${(pv.kvKb * 1024).toFixed(0)} B`}
                              </Text>
                            </StackItem>
                            <StackItem>
                              <span className="pf-v5-u-font-weight-bold">{fmtGb(pv.total)}</span>
                            </StackItem>
                            {selected && (
                              <StackItem>
                                <Text component={TextVariants.small} className="pf-v5-u-primary-color-100">
                                  ✓ Recommended
                                </Text>
                              </StackItem>
                            )}
                          </Stack>
                        </Tile>
                      </FlexItem>
                    );
                  })}
                </Flex>
              </ResultSection>
            </StackItem>

            {/* GPU capability */}
            <StackItem>
              <ResultSection id="gpuCapability" icon={<MicrochipIcon />}
                title="GPU capability" subtitle="Hardware specifications"
                expanded={state.expandedSections.gpuCapability} onToggle={toggleSection}>
                <Card isFlat>
                  <CardTitle>{selectedGpu.name}</CardTitle>
                  <CardBody>
                    <DescriptionList columnModifier={{ default: "3Col" }}>
                      <DescriptionListGroup>
                        <DescriptionListTerm>VRAM</DescriptionListTerm>
                        <DescriptionListDescription>{selectedGpu.vramGb} GB</DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Bandwidth</DescriptionListTerm>
                        <DescriptionListDescription>{selectedGpu.bandwidthTbps} TB/s</DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>TFLOPs</DescriptionListTerm>
                        <DescriptionListDescription>{selectedGpu.tflops}</DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Price/hr</DescriptionListTerm>
                        <DescriptionListDescription>${selectedGpu.pricePerHour.toFixed(2)}</DescriptionListDescription>
                      </DescriptionListGroup>
                    </DescriptionList>
                  </CardBody>
                </Card>
              </ResultSection>
            </StackItem>

            {/* Throughput */}
            <StackItem>
              <ResultSection id="throughput" icon={<TrendUpIcon />}
                title="Throughput" subtitle="Estimated serving performance"
                expanded={state.expandedSections.throughput} onToggle={toggleSection}>
                <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "20ch" }}>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Prefill (TTFT)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtMs(result.ttftMs)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Decode (TPOT)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtMs(result.tpotMs)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Tokens/sec (system)</DescriptionListTerm>
                    <DescriptionListDescription>{result.systemTps.toLocaleString()}</DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </ResultSection>
            </StackItem>

            {/* Cost */}
            <StackItem>
              <ResultSection id="cost" icon={<DollarSignIcon />}
                title="Cost" subtitle="Detailed cost breakdown"
                expanded={state.expandedSections.cost} onToggle={toggleSection}>
                <Grid hasGutter>
                  <GridItem span={6}>
                    <Card isFlat>
                      <CardTitle>GPU costs</CardTitle>
                      <CardBody>
                        <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "12ch" }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Per hour</DescriptionListTerm>
                            <DescriptionListDescription>${result.costPerHour.toFixed(2)}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Per day (24h)</DescriptionListTerm>
                            <DescriptionListDescription>${Math.round(result.costPerDay).toLocaleString()}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Per month</DescriptionListTerm>
                            <DescriptionListDescription>${Math.round(result.costPerMonth).toLocaleString()}</DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>
                  <GridItem span={6}>
                    <Card isFlat>
                      <CardTitle>On-prem estimate</CardTitle>
                      <CardBody>
                        <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "14ch" }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Hardware</DescriptionListTerm>
                            <DescriptionListDescription>{fmtMoney(result.hardwareCost)}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Electricity/mo</DescriptionListTerm>
                            <DescriptionListDescription>{fmtMoney(result.electricityPerMonth)}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Break-even</DescriptionListTerm>
                            <DescriptionListDescription>{result.breakEvenMonths} months</DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>
                </Grid>
              </ResultSection>
            </StackItem>

            {/* Sensitivity analysis */}
            <StackItem>
              <ResultSection id="sensitivity" icon={<ChartBarIcon />}
                title="What changes things" subtitle="Sensitivity analysis"
                expanded={state.expandedSections.sensitivity} onToggle={toggleSection}>
                <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-mb-md">
                  Using {PRECISION_BYTES[precision].label} for current {result.replicasNeeded} replicas
                </Text>
                <Stack hasGutter>
                  {result.sensitivity.map((item) => (
                    <StackItem key={item.label}>
                      <Progress
                        id={`sens-${item.label.replace(/\s/g, "-")}`}
                        title={item.label}
                        value={Math.min(100, (item.usersCapacity / maxSensBar) * 100)}
                        label={`${item.usersCapacity} users (${item.changePct >= 0 ? "+" : ""}${item.changePct}%)`}
                        measureLocation="outside"
                        size="sm"
                      />
                    </StackItem>
                  ))}
                </Stack>
              </ResultSection>
            </StackItem>
          </Stack>
        </GridItem>

        {/* ── quick adjustments sidebar ── */}
        {state.controlsVisible && (
          <GridItem span={4}>
            <Card isFlat style={{ position: "sticky", top: "1rem" }}>
              <CardTitle>
                <Stack>
                  <StackItem>
                    <Title headingLevel="h2" size="lg">Quick adjustments</Title>
                  </StackItem>
                  <StackItem>
                    <Text component={TextVariants.small} className="pf-v5-u-color-200">
                      Results update in real-time
                    </Text>
                  </StackItem>
                </Stack>
              </CardTitle>
              <CardBody>
                <Stack hasGutter>

                  {/* Model */}
                  <StackItem>
                    <Stack>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">
                          Model
                        </Text>
                      </StackItem>
                      <StackItem>
                        <FormSelect
                          value={state.selectedModelId}
                          onChange={(_e, v) => setState((s) => ({ ...s, selectedModelId: v }))}
                          aria-label="Select model"
                        >
                          {MODEL_CATALOG.map((m) => (
                            <FormSelectOption key={m.id} value={m.id} label={m.name} />
                          ))}
                        </FormSelect>
                      </StackItem>
                    </Stack>
                  </StackItem>

                  {/* Concurrent users */}
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                          Concurrent users
                        </Text>
                      </StackItem>
                      <StackItem>
                        <TextInput
                          type="number"
                          value={state.concurrentUsers}
                          onChange={(_e, v) => {
                            const n = parseInt(v, 10);
                            if (!isNaN(n) && n > 0) setState((s) => ({ ...s, concurrentUsers: n }));
                          }}
                          aria-label="Concurrent users"
                        />
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Concurrent user presets" isCompact>
                          {[10, 30, 100, 500, 1000].map((n) => (
                            <ToggleGroupItem
                              key={n}
                              text={n >= 1000 ? "1K" : String(n)}
                              buttonId={`users-${n}`}
                              isSelected={state.concurrentUsers === n}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, concurrentUsers: n }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>

                  {/* Context length */}
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                          Context length
                        </Text>
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Context length" isCompact>
                          {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                            <ToggleGroupItem
                              key={cl}
                              text={`${CONTEXT_LABELS[cl]} · ${CONTEXT_SUBLABELS[cl]}`}
                              buttonId={`ctx-${cl}`}
                              isSelected={state.contextLength === cl}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, contextLength: cl }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>

                  {/* Deployment type */}
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">
                          Deployment type
                        </Text>
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Deployment type" isCompact>
                          {DEPLOYMENT_OPTIONS.map((dt) => (
                            <ToggleGroupItem
                              key={dt.key}
                              text={dt.label}
                              buttonId={`deploy-${dt.key}`}
                              isSelected={state.deploymentType === dt.key}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, deploymentType: dt.key }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>

                  {/* Advanced */}
                  <StackItem>
                    <ExpandableSection toggleText="Advanced">
                      <Stack hasGutter>

                        {/* Precision */}
                        <StackItem>
                          <Stack hasGutter>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Precision</Text>
                            </StackItem>
                            <StackItem>
                              <ToggleGroup aria-label="Precision" isCompact>
                                {(["fp16", "int8", "int4", "mixed"] as Precision[]).map((p) => (
                                  <ToggleGroupItem
                                    key={p}
                                    text={PRECISION_BYTES[p].label}
                                    buttonId={`prec-${p}`}
                                    isSelected={state.precision === p}
                                    onChange={(_e, sel) => sel && setState((s) => ({ ...s, precision: p }))}
                                  />
                                ))}
                              </ToggleGroup>
                            </StackItem>
                          </Stack>
                        </StackItem>

                        {/* GPU */}
                        <StackItem>
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">GPU</Text>
                            </StackItem>
                            <StackItem>
                              <FormSelect
                                value={state.gpuId}
                                onChange={(_e, v) => setState((s) => ({ ...s, gpuId: v }))}
                                aria-label="Select GPU"
                              >
                                {GPU_CATALOG.map((g) => (
                                  <FormSelectOption key={g.id} value={g.id} label={g.name} />
                                ))}
                              </FormSelect>
                            </StackItem>
                          </Stack>
                        </StackItem>

                        {/* Tensor parallelism */}
                        <StackItem>
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">
                                Tensor parallelism
                              </Text>
                            </StackItem>
                            <StackItem>
                              <FormSelect
                                value={String(state.tensorParallelism)}
                                onChange={(_e, v) =>
                                  setState((s) => ({
                                    ...s,
                                    tensorParallelism: v === "auto" ? "auto" : (parseInt(v, 10) as 1 | 2 | 4 | 8),
                                  }))
                                }
                                aria-label="Tensor parallelism"
                              >
                                {["auto", "1", "2", "4", "8"].map((v) => (
                                  <FormSelectOption key={v} value={v} label={v === "auto" ? "Auto" : `TP-${v}`} />
                                ))}
                              </FormSelect>
                            </StackItem>
                          </Stack>
                        </StackItem>

                        {/* Memory utilization */}
                        <StackItem>
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">
                                Memory utilization target
                              </Text>
                            </StackItem>
                            <StackItem>
                              <Slider
                                value={Math.round(state.memUtilization * 100)}
                                min={50}
                                max={95}
                                step={5}
                                onChange={(_e, v) => setState((s) => ({ ...s, memUtilization: v / 100 }))}
                                aria-label="Memory utilization target"
                                showTicks
                              />
                            </StackItem>
                          </Stack>
                        </StackItem>

                      </Stack>
                    </ExpandableSection>
                  </StackItem>

                  <StackItem>
                    <Alert variant="info" isInline isPlain
                      title="Adjust any parameter to see instant updates in calculations"
                    />
                  </StackItem>

                </Stack>
              </CardBody>
            </Card>
          </GridItem>
        )}
      </Grid>
    </PageSection>
  );
}

// ─── main page ────────────────────────────────────────────────────────────────

export default function QuickEstimatePage() {
  const [state, setState] = useState<PageState>(DEFAULT_STATE);

  if (state.view === "results") {
    return <ResultsView state={state} setState={setState} />;
  }
  return <InputView state={state} setState={setState} />;
}
