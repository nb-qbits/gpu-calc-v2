'use client';

import { PageSection } from '@patternfly/react-core';
import QuickEstimate from './QuickEstimate';

export default function Page() {
  return (
    <PageSection
      padding={{ default: 'noPadding' }}
      style={{
        backgroundColor: 'var(--gc-bg-2, #f5f5f5)',
        minHeight: '100vh',
        padding: 0,
        paddingLeft: 0,
        paddingRight: 0
      }}
    >
      <QuickEstimate />
    </PageSection>
  );
}
