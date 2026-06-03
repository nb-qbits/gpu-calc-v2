"use client";
import {
  PageSection,
  Title,
  Text,
  TextContent,
  Grid,
  GridItem,
  Card,
  CardBody,
  CardTitle,
  Button,
} from "@patternfly/react-core";
import {
  MigrationIcon,
  ChartBarIcon,
  CpuIcon,
  CalculatorIcon,
  RouteIcon,
} from "@patternfly/react-icons";
import Link from "next/link";

const tools = [
  {
    title: "Quick Estimate",
    description:
      "Fast GPU memory and throughput estimate from model size and serving parameters.",
    href: "/quick-estimate",
    icon: <CalculatorIcon />,
  },
  {
    title: "Advanced Calculator",
    description:
      "Detailed inference sizing with batching, quantization, KV cache, and cost modeling.",
    href: "/calculator",
    icon: <CpuIcon />,
  },
  {
    title: "GPU Explorer",
    description:
      "Compare GPUs across memory, throughput, cost, and availability tiers.",
    href: "/gpu-explorer",
    icon: <ChartBarIcon />,
  },
  {
    title: "Hybrid Savings",
    description:
      "Model cost savings between cloud, on-premise, and hybrid GPU deployment strategies.",
    href: "/hybrid-savings",
    icon: <MigrationIcon />,
  },
  {
    title: "Routing Economics",
    description:
      "Analyze request routing between model tiers to optimize cost vs quality tradeoffs.",
    href: "/routing",
    icon: <RouteIcon />,
  },
];

export default function HomePage() {
  return (
    <>
      <PageSection variant="light">
        <div style={{ position: 'relative' }}>
          <div style={{ position: 'absolute', top: 0, right: 0, fontSize: '0.875rem', color: '#6A6E73' }}>
            Work in progress
          </div>
          <TextContent>
            <Title headingLevel="h1" size="2xl">
              GPU Calc
            </Title>
            <Text component="p">
              LLM inference sizing, GPU comparison, and cost modeling for
              engineers and infrastructure teams.
            </Text>
          </TextContent>
        </div>
      </PageSection>

      <PageSection>
        <Grid hasGutter md={6} xl={4}>
          {tools.map((tool) => (
            <GridItem key={tool.href}>
              <Card isFullHeight isClickable>
                <CardTitle>
                  <span style={{ marginRight: "0.5rem" }}>{tool.icon}</span>
                  {tool.title}
                </CardTitle>
                <CardBody>
                  <TextContent>
                    <Text component="p">{tool.description}</Text>
                  </TextContent>
                  <br />
                  <Button
                    variant="link"
                    isInline
                    component={(props) => <Link href={tool.href} {...props} />}
                  >
                    Open tool →
                  </Button>
                </CardBody>
              </Card>
            </GridItem>
          ))}
        </Grid>
      </PageSection>
    </>
  );
}
