"use client";

import React, { useState } from "react";
import {
  Page,
  Masthead,
  MastheadMain,
  MastheadBrand,
  MastheadContent,
  Nav,
  NavList,
  NavItem,
  PageSidebar,
  PageSidebarBody,
  SkipToContent,
  Brand,
} from "@patternfly/react-core";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { label: "Home",               href: "/" },
  { label: "Quick Estimate",     href: "/quick-estimate" },
  { label: "Advanced Calculator",href: "/calculator" },
  { label: "GPU Explorer",       href: "/gpu-explorer" },
  { label: "Hybrid Savings",     href: "/hybrid-savings" },
  { label: "Routing Economics",  href: "/routing" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const masthead = (
    <Masthead>
      <MastheadMain>
        <MastheadBrand>
          <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontWeight: 700,
                fontSize: "1.25rem",
                color: "#ee0000",
              }}
            >
              GPU Calc
            </span>
          </Link>
        </MastheadBrand>
      </MastheadMain>
      <MastheadContent />
    </Masthead>
  );

  const sidebar = (
    <PageSidebar isSidebarOpen={isSidebarOpen}>
      <PageSidebarBody>
        <Nav aria-label="Global navigation">
          <NavList>
            {navItems.map((item) => (
              <NavItem
                key={item.href}
                isActive={pathname === item.href}
                component="div"
              >
                <Link
                  href={item.href}
                  style={{ textDecoration: "none", color: "inherit" }}
                >
                  {item.label}
                </Link>
              </NavItem>
            ))}
          </NavList>
        </Nav>
      </PageSidebarBody>
    </PageSidebar>
  );

  return (
    <Page
      header={masthead}
      sidebar={sidebar}
      skipToContent={
        <SkipToContent href="#main-content">Skip to content</SkipToContent>
      }
      mainContainerId="main-content"
    >
      {children}
    </Page>
  );
}
