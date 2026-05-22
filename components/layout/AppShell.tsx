"use client";

import React from "react";
import {
  Page,
  Masthead,
  MastheadMain,
  MastheadBrand,
  MastheadContent,
  SkipToContent,
} from "@patternfly/react-core";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { label: "Home",          href: "/"              },
  { label: "Quick Estimate",href: "/quick-estimate"},
  { label: "Advanced",      href: "/calculator"    },
  { label: "Explorer",      href: "/gpu-explorer"  },
  { label: "Hybrid Savings",href: "/hybrid-savings"},
  { label: "Routing",       href: "/routing"       },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const masthead = (
    <Masthead>
      <MastheadMain>
        <MastheadBrand>
          <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="10" cy="10" r="10" fill="#ee0000" />
              <path d="M5 13 L10 6 L15 13" stroke="white" strokeWidth="2" fill="none" strokeLinejoin="round" />
            </svg>
            <span style={{
              fontFamily: "var(--font-display)",
              fontWeight: 700,
              fontSize: "1.1rem",
              color: "white",
              letterSpacing: "-0.01em",
            }}>
              GPU<span style={{ color: "#ee0000" }}>Calc</span>
            </span>
          </Link>
        </MastheadBrand>
      </MastheadMain>

      <MastheadContent>
        <nav
          aria-label="Global navigation"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            flex: 1,
            justifyContent: "center",
          }}
        >
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                style={{
                  textDecoration: "none",
                  color: active ? "white" : "rgba(255,255,255,0.65)",
                  fontFamily: "var(--font-body)",
                  fontSize: "0.875rem",
                  fontWeight: active ? 600 : 400,
                  padding: "6px 12px",
                  borderRadius: 4,
                  background: active ? "rgba(255,255,255,0.12)" : "transparent",
                  whiteSpace: "nowrap",
                  transition: "color 0.15s, background 0.15s",
                }}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <span style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.62rem",
          color: "rgba(255,255,255,0.35)",
          letterSpacing: "0.05em",
          whiteSpace: "nowrap",
          flexShrink: 0,
        }}>
          v0.5 · redesign
        </span>
      </MastheadContent>
    </Masthead>
  );

  return (
    <Page
      header={masthead}
      skipToContent={
        <SkipToContent href="#main-content">Skip to content</SkipToContent>
      }
      mainContainerId="main-content"
    >
      {children}
    </Page>
  );
}
