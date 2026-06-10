"use client";

import React, { useRef, useState, useEffect } from "react";
import { Card } from "@patternfly/react-core";

interface HoverCardProps {
  children: React.ReactNode;
  effectConfig?: EffectConfig;
  className?: string;
  style?: React.CSSProperties;
}

export interface EffectConfig {
  // Perspective tilt
  tiltEnabled: boolean;
  tiltIntensity: number; // 0-50 degrees
  tiltSmooth: number; // 0-1 (transition smoothness)

  // Glow
  glowEnabled: boolean;
  glowColor: string;
  glowIntensity: number; // 0-100px blur
  glowSpread: number; // 0-50px spread

  // Iridescence
  iridescenceEnabled: boolean;
  iridescenceIntensity: number; // 0-1 opacity
  iridescenceHueShift: number; // 0-360 degrees

  // Noise texture
  noiseEnabled: boolean;
  noiseOpacity: number; // 0-1
  noiseScale: number; // 1-10 (grain size)

  // Specular highlight
  specularEnabled: boolean;
  specularSize: number; // 50-500px radius
  specularIntensity: number; // 0-1 opacity
  specularColor: string;

  // Border glow
  borderGlowEnabled: boolean;
  borderGlowColor: string;
  borderGlowWidth: number; // 1-10px
}

export const defaultEffectConfig: EffectConfig = {
  tiltEnabled: true,
  tiltIntensity: 15,
  tiltSmooth: 0.1,

  glowEnabled: true,
  glowColor: "#ee0000",
  glowIntensity: 25,
  glowSpread: 8,

  iridescenceEnabled: true,
  iridescenceIntensity: 0.2,
  iridescenceHueShift: 180,

  noiseEnabled: true,
  noiseOpacity: 0.03,
  noiseScale: 2,

  specularEnabled: true,
  specularSize: 400,
  specularIntensity: 0.8,
  specularColor: "#ffffff",

  borderGlowEnabled: true,
  borderGlowColor: "#ee0000",
  borderGlowWidth: 2,
};

export function HoverCard({ children, effectConfig = defaultEffectConfig, className, style }: HoverCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 }); // Normalized 0-1
  const [isHovered, setIsHovered] = useState(false);
  const [smoothPos, setSmoothPos] = useState({ x: 0.5, y: 0.5 });

  // Smooth mouse tracking
  useEffect(() => {
    let rafId: number;
    const animate = () => {
      setSmoothPos((prev) => ({
        x: prev.x + (mousePos.x - prev.x) * effectConfig.tiltSmooth,
        y: prev.y + (mousePos.y - prev.y) * effectConfig.tiltSmooth,
      }));
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [mousePos, effectConfig.tiltSmooth]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setMousePos({ x, y });
  };

  const handleMouseEnter = () => setIsHovered(true);
  const handleMouseLeave = () => {
    setIsHovered(false);
    setMousePos({ x: 0.5, y: 0.5 });
  };

  // Calculate transforms
  const rotateX = effectConfig.tiltEnabled && isHovered
    ? (smoothPos.y - 0.5) * effectConfig.tiltIntensity
    : 0;
  const rotateY = effectConfig.tiltEnabled && isHovered
    ? (0.5 - smoothPos.x) * effectConfig.tiltIntensity
    : 0;

  // Iridescent gradient based on mouse position - Red Hat brand colors
  const iridescenceGradient = effectConfig.iridescenceEnabled
    ? `linear-gradient(
        ${smoothPos.x * 360}deg,
        hsl(0, 100%, ${50 + smoothPos.x * 20}%) 0%,
        hsl(${15 + smoothPos.y * 30}, 100%, ${60 + smoothPos.y * 15}%) 50%,
        hsl(${340 + smoothPos.x * 20}, 90%, ${55 + smoothPos.x * 10}%) 100%
      )`
    : "none";

  // Specular highlight position
  const specularX = smoothPos.x * 100;
  const specularY = smoothPos.y * 100;

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        perspective: "1000px",
        position: "relative",
        ...style,
      }}
      className={className}
    >
      <div
        style={{
          transform: `rotateX(${-rotateX}deg) rotateY(${rotateY}deg) scale(${isHovered ? 1.02 : 1})`,
          transition: effectConfig.tiltEnabled
            ? `transform ${effectConfig.tiltSmooth * 300}ms ease-out`
            : "none",
          transformStyle: "preserve-3d",
          position: "relative",
          boxShadow: effectConfig.glowEnabled && isHovered
            ? `0 0 ${effectConfig.glowIntensity}px ${effectConfig.glowSpread}px ${effectConfig.glowColor}`
            : "none",
          borderRadius: "4px",
          overflow: "hidden",
        }}
      >
        {/* Main card content */}
        <div style={{ position: "relative", zIndex: 1 }}>
          {children}
        </div>

        {/* Iridescent overlay */}
        {effectConfig.iridescenceEnabled && isHovered && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: iridescenceGradient,
              opacity: effectConfig.iridescenceIntensity,
              mixBlendMode: "overlay",
              pointerEvents: "none",
              zIndex: 2,
            }}
          />
        )}

        {/* Specular highlight */}
        {effectConfig.specularEnabled && isHovered && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `radial-gradient(
                ${effectConfig.specularSize}px circle at ${specularX}% ${specularY}%,
                ${effectConfig.specularColor} 0%,
                rgba(255, 255, 255, 0.3) 30%,
                transparent 60%
              )`,
              opacity: effectConfig.specularIntensity,
              mixBlendMode: "soft-light",
              pointerEvents: "none",
              zIndex: 3,
              transition: "opacity 0.2s ease-out",
            }}
          />
        )}

        {/* Noise texture overlay */}
        {effectConfig.noiseEnabled && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='${effectConfig.noiseScale / 10}' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
              opacity: effectConfig.noiseOpacity,
              mixBlendMode: "overlay",
              pointerEvents: "none",
              zIndex: 4,
            }}
          />
        )}

        {/* Border glow */}
        {effectConfig.borderGlowEnabled && isHovered && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              border: `${effectConfig.borderGlowWidth}px solid ${effectConfig.borderGlowColor}`,
              borderRadius: "4px",
              opacity: 0.6,
              pointerEvents: "none",
              zIndex: 5,
            }}
          />
        )}
      </div>
    </div>
  );
}
