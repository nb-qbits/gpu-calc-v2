"use client";

import * as React from 'react';
import {
  PageSection,
  Title,
  TextContent,
  ToggleGroup,
  ToggleGroupItem,
  Card,
  CardBody,
  Flex,
  FlexItem,
  Label,
  Text
} from "@patternfly/react-core";
import { GPU_CATALOG } from '@/lib/gpu-math/gpus';
import { GpuBubbleChart } from './GpuBubbleChart';

type Preset = 'balanced' | 'cost-efficiency' | 'performance';
type XAxis = 'vram' | 'price' | 'throughput-index' | 'mem-bw';
type YAxis = 'vram' | 'price' | 'throughput-index' | 'mem-bw';

export default function GpuExplorerPage() {
  const [mounted, setMounted] = React.useState(false);
  const [preset, setPreset] = React.useState<Preset>('balanced');
  const [xAxis, setXAxis] = React.useState<XAxis>('vram');
  const [yAxis, setYAxis] = React.useState<YAxis>('throughput-index');
  const [vendorFilter, setVendorFilter] = React.useState<'all' | 'nvidia' | 'amd'>('all');

  React.useEffect(() => {
    setMounted(true);
  }, []);

  // Filter GPUs by vendor
  const filteredGPUs = GPU_CATALOG.filter(gpu => {
    if (vendorFilter === 'all') return true;
    return gpu.vendor === vendorFilter;
  });

  // Calculate throughput index
  const calculateThroughputIndex = (gpu: typeof GPU_CATALOG[0]) => {
    const archMultiplier = {
      'blackwell': 1.3,
      'hopper': 1.2,
      'ada': 1.0,
      'ampere': 0.85,
      'cdna4': 1.25,
      'cdna3': 1.15,
      'cdna2': 0.9
    }[gpu.architecture] || 1.0;

    return (gpu.memory_bandwidth_tbps * 1000) * archMultiplier * (gpu.vram_gb / 80);
  };

  // Get axis value
  const getAxisValue = (gpu: typeof GPU_CATALOG[0], axis: XAxis | YAxis): number => {
    switch (axis) {
      case 'vram':
        return gpu.vram_gb;
      case 'price':
        return gpu.hardware_cost_usd; // One-time GPU purchase price
      case 'throughput-index':
        return calculateThroughputIndex(gpu);
      case 'mem-bw':
        return gpu.memory_bandwidth_tbps * 1000;
      default:
        return 0;
    }
  };

  // Get axis label
  const getAxisLabel = (axis: XAxis | YAxis): string => {
    switch (axis) {
      case 'vram':
        return 'VRAM (GB)';
      case 'price':
        return 'Hardware Cost (USD)';
      case 'throughput-index':
        return 'Throughput Index';
      case 'mem-bw':
        return 'Memory Bandwidth (GB/s)';
      default:
        return '';
    }
  };

  // Apply preset
  React.useEffect(() => {
    switch (preset) {
      case 'balanced':
        setXAxis('vram');
        setYAxis('throughput-index');
        break;
      case 'cost-efficiency':
        setXAxis('price'); // Hardware cost
        setYAxis('throughput-index');
        break;
      case 'performance':
        setXAxis('mem-bw');
        setYAxis('throughput-index');
        break;
    }
  }, [preset]);

  // Prepare all data for bubble chart with full specs
  const allData = filteredGPUs.map(gpu => ({
    x: getAxisValue(gpu, xAxis),
    y: getAxisValue(gpu, yAxis),
    size: gpu.tokens_per_dollar, // Use actual tokens_per_dollar for bubble size
    name: gpu.name.replace(/NVIDIA |AMD /, ''),
    fullName: gpu.name,
    color: gpu.vendor === 'nvidia' ? '#5b9bd5' : '#c55a5a',
    // Extra specs for tooltip
    vram: gpu.vram_gb,
    hwCost: gpu.hardware_cost_usd,
    memBW: gpu.memory_bandwidth_tbps * 1000,
    tokensPerDollar: gpu.tokens_per_dollar,
    architecture: gpu.architecture.charAt(0).toUpperCase() + gpu.architecture.slice(1),
    tflops: gpu.tflops_bf16
  }));

  return (
    <>
      <PageSection variant="light">
        <TextContent>
          <Title headingLevel="h1" size="2xl">GPU Explorer</Title>
          <Text>
            LLM inference planning — compare GPU generations by memory, bandwidth, and cost efficiency
          </Text>
        </TextContent>
      </PageSection>

      <PageSection>
        <Card>
          <CardBody>
            <Flex direction={{ default: 'column' }} spaceItems={{ default: 'spaceItemsMd' }}>
              {/* Presets */}
              <FlexItem>
                <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                  Preset:
                </Text>
                <ToggleGroup>
                  <ToggleGroupItem text="Balanced" isSelected={preset === 'balanced'} onChange={() => setPreset('balanced')} />
                  <ToggleGroupItem text="Cost Efficiency" isSelected={preset === 'cost-efficiency'} onChange={() => setPreset('cost-efficiency')} />
                  <ToggleGroupItem text="Performance" isSelected={preset === 'performance'} onChange={() => setPreset('performance')} />
                </ToggleGroup>
              </FlexItem>

              {/* Vendor Filter */}
              <FlexItem>
                <Flex spaceItems={{ default: 'spaceItemsLg' }}>
                  <FlexItem>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', fontSize: '14px' }}>
                      Vendor:
                    </Text>
                  </FlexItem>
                  <FlexItem>
                    <Label
                      color={vendorFilter === 'nvidia' ? 'blue' : 'grey'}
                      onClick={() => setVendorFilter(vendorFilter === 'nvidia' ? 'all' : 'nvidia')}
                      style={{ cursor: 'pointer' }}
                    >
                      NVIDIA
                    </Label>
                  </FlexItem>
                  <FlexItem>
                    <Label
                      color={vendorFilter === 'amd' ? 'red' : 'grey'}
                      onClick={() => setVendorFilter(vendorFilter === 'amd' ? 'all' : 'amd')}
                      style={{ cursor: 'pointer' }}
                    >
                      AMD
                    </Label>
                  </FlexItem>
                </Flex>
              </FlexItem>

              {/* Axis Selectors */}
              <FlexItem>
                <Flex direction={{ default: 'row' }} spaceItems={{ default: 'spaceItemsLg' }}>
                  <FlexItem flex={{ default: 'flex_1' }}>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                      X Axis:
                    </Text>
                    <ToggleGroup>
                      <ToggleGroupItem text="VRAM" isSelected={xAxis === 'vram'} onChange={() => setXAxis('vram')} />
                      <ToggleGroupItem text="HW Cost" isSelected={xAxis === 'price'} onChange={() => setXAxis('price')} />
                      <ToggleGroupItem text="Throughput Index" isSelected={xAxis === 'throughput-index'} onChange={() => setXAxis('throughput-index')} />
                      <ToggleGroupItem text="Mem BW" isSelected={xAxis === 'mem-bw'} onChange={() => setXAxis('mem-bw')} />
                    </ToggleGroup>
                  </FlexItem>

                  <FlexItem flex={{ default: 'flex_1' }}>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                      Y Axis:
                    </Text>
                    <ToggleGroup>
                      <ToggleGroupItem text="VRAM" isSelected={yAxis === 'vram'} onChange={() => setYAxis('vram')} />
                      <ToggleGroupItem text="HW Cost" isSelected={yAxis === 'price'} onChange={() => setYAxis('price')} />
                      <ToggleGroupItem text="Throughput Index" isSelected={yAxis === 'throughput-index'} onChange={() => setYAxis('throughput-index')} />
                      <ToggleGroupItem text="Mem BW" isSelected={yAxis === 'mem-bw'} onChange={() => setYAxis('mem-bw')} />
                    </ToggleGroup>
                  </FlexItem>
                </Flex>
              </FlexItem>

              {/* Chart */}
              <FlexItem style={{ marginTop: '24px' }}>
                {!mounted ? (
                  <div style={{ padding: '60px', textAlign: 'center', background: '#f5f5f5', borderRadius: '8px' }}>
                    <Text component="p" style={{ color: '#6a6e73' }}>Loading chart...</Text>
                  </div>
                ) : allData.length === 0 ? (
                  <div style={{ padding: '60px', textAlign: 'center', background: '#f5f5f5', borderRadius: '8px' }}>
                    <Text component="h3" style={{ color: '#6a6e73', marginBottom: '8px' }}>No data available</Text>
                    <Text component="p" style={{ color: '#6a6e73', fontSize: '14px' }}>Try selecting a different vendor filter</Text>
                  </div>
                ) : (
                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                    <GpuBubbleChart
                      data={allData}
                      width={1100}
                      height={550}
                      xLabel={getAxisLabel(xAxis)}
                      yLabel={getAxisLabel(yAxis)}
                    />
                  </div>
                )}
              </FlexItem>

              {/* Legend */}
              <FlexItem>
                <Card isCompact>
                  <CardBody>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      💡 <strong>Top-right = high VRAM and throughput.</strong> These GPUs handle larger models and longer contexts.
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Bubble size</strong> represents tokens-per-dollar efficiency. Larger bubbles = better cost efficiency (more tokens generated per dollar spent).
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Throughput Index</strong> is a planning metric derived from memory bandwidth, VRAM, and architecture generation.
                      It enables relative GPU comparison — not exact model throughput.
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Inference performance</strong> depends on model architecture (GQA vs MHA), sequence length, batching, and inference backend (vLLM, TensorRT-LLM, etc.).
                    </Text>
                    <Text component="p" style={{ display: 'block', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Hardware cost</strong> shown in data is GPU purchase price (USD, one-time). Hourly cloud pricing varies by provider and region.
                    </Text>
                  </CardBody>
                </Card>
              </FlexItem>
            </Flex>
          </CardBody>
        </Card>
      </PageSection>
    </>
  );
}
