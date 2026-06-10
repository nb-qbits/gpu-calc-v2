'use client';

import * as React from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@patternfly/react-core';
import styles from './ProductTour.module.css';

export interface TourStep {
  target: string; // CSS selector or data-tour attribute
  title: string;
  description: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

interface ProductTourProps {
  steps: TourStep[];
  tourId: string; // Used for localStorage key: `${tourId}-tour-seen`
  onComplete: () => void;
}

export function ProductTour({ steps, tourId, onComplete }: ProductTourProps) {
  const [currentStep, setCurrentStep] = React.useState(0);
  const [tooltipPosition, setTooltipPosition] = React.useState({ top: 0, left: 0 });
  const [spotlightRect, setSpotlightRect] = React.useState({ top: 0, left: 0, width: 0, height: 0 });
  const [mounted, setMounted] = React.useState(false);

  const step = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  // Only render portal on client-side
  React.useEffect(() => {
    setMounted(true);
  }, []);

  // Handle Esc key to skip tour
  React.useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onComplete();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onComplete]);

  React.useEffect(() => {
    const updatePositions = () => {
      const targetEl = document.querySelector(step.target);
      if (!targetEl) {
        console.warn(`ProductTour: target not found: ${step.target}`);
        return;
      }

      // Use getBoundingClientRect() directly for FIXED positioning
      // (no scroll offset needed)
      const rect = targetEl.getBoundingClientRect();

      // Update spotlight position with padding
      const padding = 8;
      const spotlightTop = rect.top - padding;
      const spotlightLeft = rect.left - padding;
      const spotlightWidth = rect.width + padding * 2;
      const spotlightHeight = rect.height + padding * 2;

      setSpotlightRect({
        top: spotlightTop,
        left: spotlightLeft,
        width: spotlightWidth,
        height: spotlightHeight
      });

      // Sanity check: log all measurements
      const targetCenterX = rect.left + rect.width / 2;
      const targetCenterY = rect.top + rect.height / 2;
      const spotlightCenterX = spotlightLeft + spotlightWidth / 2;
      const spotlightCenterY = spotlightTop + spotlightHeight / 2;

      console.log(`\n=== Step ${currentStep + 1}: "${step.title}" ===`);
      console.log(`Target selector: ${step.target}`);
      console.log(`Element found:`, targetEl);
      console.log(`Target rect:`, {
        top: rect.top.toFixed(1),
        left: rect.left.toFixed(1),
        width: rect.width.toFixed(1),
        height: rect.height.toFixed(1),
        bottom: rect.bottom.toFixed(1),
        right: rect.right.toFixed(1)
      });
      console.log(`Spotlight rect:`, {
        top: spotlightTop.toFixed(1),
        left: spotlightLeft.toFixed(1),
        width: spotlightWidth.toFixed(1),
        height: spotlightHeight.toFixed(1)
      });
      console.log(`Target center: (${targetCenterX.toFixed(1)}, ${targetCenterY.toFixed(1)})`);
      console.log(`Spotlight center: (${spotlightCenterX.toFixed(1)}, ${spotlightCenterY.toFixed(1)})`);
      console.log(`Offset: ${(spotlightCenterY - targetCenterY).toFixed(1)}px vertical, ${(spotlightCenterX - targetCenterX).toFixed(1)}px horizontal`);

      // Position tooltip based on step.position (FIXED positioning, no scroll offset)
      let top = 0;
      let left = 0;

      switch (step.position) {
        case 'bottom':
          top = rect.bottom + 16;
          left = rect.left + rect.width / 2;
          break;
        case 'top':
          top = rect.top - 16;
          left = rect.left + rect.width / 2;
          break;
        case 'left':
          top = rect.top + rect.height / 2;
          left = rect.left - 16;
          break;
        case 'right':
          top = rect.top + rect.height / 2;
          left = rect.right + 16;
          break;
        default:
          top = rect.bottom + 16;
          left = rect.left + rect.width / 2;
      }

      setTooltipPosition({ top, left });
    };

    // Scroll element into view first, then measure after scroll settles
    const targetEl = document.querySelector(step.target);
    if (!targetEl) {
      console.warn(`ProductTour: target element not found on mount: ${step.target}`);
      return;
    }

    // Scroll into view with instant scrolling for more predictable behavior
    targetEl.scrollIntoView({ behavior: 'auto', block: 'center' });

    // Wait longer for layout to settle (300ms)
    const timer = setTimeout(() => {
      updatePositions();
    }, 300);

    // Update on scroll/resize
    const handleScroll = () => {
      updatePositions();
    };

    const handleResize = () => {
      updatePositions();
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
    };
  }, [currentStep, step]);

  const handleNext = () => {
    if (isLastStep) {
      onComplete();
    } else {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handleBack = () => {
    if (!isFirstStep) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleSkip = () => {
    onComplete();
  };

  if (!mounted) return null;

  const tourContent = (
    <>
      {/* Invisible click target to skip tour */}
      <div className={styles.overlay} onClick={handleSkip} />

      {/* DEBUG: Red outline showing exact target position */}
      <div
        style={{
          position: 'fixed',
          top: `${spotlightRect.top + 8}px`,
          left: `${spotlightRect.left + 8}px`,
          width: `${spotlightRect.width - 16}px`,
          height: `${spotlightRect.height - 16}px`,
          border: '2px solid red',
          pointerEvents: 'none',
          zIndex: 10001,
        }}
      />

      {/* Spotlight on target element */}
      <div
        className={styles.spotlight}
        style={{
          top: `${spotlightRect.top}px`,
          left: `${spotlightRect.left}px`,
          width: `${spotlightRect.width}px`,
          height: `${spotlightRect.height}px`,
        }}
      />

      {/* Tooltip */}
      <div
        className={`${styles.tooltip} ${styles[`tooltip-${step.position || 'bottom'}`]}`}
        style={{
          top: `${tooltipPosition.top}px`,
          left: `${tooltipPosition.left}px`,
        }}
      >
        <div className={styles.tooltipHeader}>
          <h3 className={styles.tooltipTitle}>{step.title}</h3>
          <span className={styles.tooltipProgress}>
            {currentStep + 1} of {steps.length}
          </span>
        </div>
        <p className={styles.tooltipDescription}>{step.description}</p>

        {/* Progress dots */}
        <div className={styles.progressDots}>
          {steps.map((_, i) => (
            <div
              key={i}
              className={`${styles.dot} ${i === currentStep ? styles.dotActive : ''}`}
            />
          ))}
        </div>

        <div className={styles.tooltipActions}>
          <div style={{ display: 'flex', gap: '12px' }}>
            {!isFirstStep && (
              <button className={styles.backButton} onClick={handleBack}>
                Back
              </button>
            )}
            <button className={styles.skipButton} onClick={handleSkip}>
              Skip tour
            </button>
          </div>
          <Button
            variant="primary"
            onClick={handleNext}
            style={{
              backgroundColor: '#0066cc',
              borderColor: '#0066cc'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#004d99';
              e.currentTarget.style.borderColor = '#004d99';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#0066cc';
              e.currentTarget.style.borderColor = '#0066cc';
            }}
          >
            {isLastStep ? 'Got it!' : 'Next'}
          </Button>
        </div>
      </div>
    </>
  );

  // Render portal to document.body to avoid transform/positioning issues
  return createPortal(tourContent, document.body);
}
