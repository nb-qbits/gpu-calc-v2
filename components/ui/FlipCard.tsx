"use client";

import React, { useState } from "react";
import { Card, CardBody } from "@patternfly/react-core";

interface FlipCardProps {
  front: React.ReactNode;
  back: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export function FlipCard({ front, back, className, style }: FlipCardProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div
      className={`flip-card ${className || ""}`}
      onClick={() => setIsFlipped(!isFlipped)}
      style={{ cursor: "pointer", ...style }}
    >
      <div className={`flip-card-inner ${isFlipped ? "flip-card-flipped" : ""}`}>
        {/* Front */}
        <div className="flip-card-face flip-card-front">
          {front}
        </div>

        {/* Back */}
        <div className="flip-card-face flip-card-back">
          {back}
        </div>
      </div>

      <style jsx>{`
        .flip-card {
          perspective: 1000px;
          position: relative;
        }

        .flip-card-inner {
          position: relative;
          width: 100%;
          height: 100%;
          transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1),
                      opacity 0.6s ease;
          transform-style: preserve-3d;
        }

        .flip-card-flipped {
          transform: rotateY(180deg);
        }

        .flip-card-face {
          position: absolute;
          width: 100%;
          height: 100%;
          backface-visibility: hidden;
          -webkit-backface-visibility: hidden;
        }

        .flip-card-front {
          opacity: 1;
          transform: rotateY(0deg);
          transition: opacity 0.3s ease;
        }

        .flip-card-back {
          opacity: 0;
          transform: rotateY(180deg);
          transition: opacity 0.3s ease 0.3s;
        }

        .flip-card-flipped .flip-card-front {
          opacity: 0;
        }

        .flip-card-flipped .flip-card-back {
          opacity: 1;
        }
      `}</style>
    </div>
  );
}
