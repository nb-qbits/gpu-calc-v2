"use client";
import { PageSection, Title, TextContent, EmptyState, EmptyStateBody } from "@patternfly/react-core";
import { CubesIcon } from "@patternfly/react-icons";

export default function RoutingPage() {
  return (
    <>
      <PageSection variant="light">
        <TextContent>
          <Title headingLevel="h1" size="2xl">Routing Economics</Title>
        </TextContent>
      </PageSection>
      <PageSection>
        <EmptyState>
          <CubesIcon />
          <Title headingLevel="h2" size="lg">Coming soon</Title>
          <EmptyStateBody>
            This tool is being ported from the original gpu-calc static site.
          </EmptyStateBody>
        </EmptyState>
      </PageSection>
    </>
  );
}
