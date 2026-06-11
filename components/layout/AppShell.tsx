"use client";

import React from "react";
import {
  Page,
  PageSidebar,
  PageSidebarBody,
  Masthead,
  MastheadMain,
  MastheadBrand,
  Nav,
  NavList,
  NavItem,
} from "@patternfly/react-core";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  HomeIcon,
  BoltIcon,
  SlidersHIcon,
  CubesIcon,
  HistoryIcon,
  DollarSignIcon,
  RouteIcon,
  CogIcon,
  ListIcon,
} from "@patternfly/react-icons";
import { getVersionString, getBuildTimeString, getShortCommit } from "@/lib/version";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const masthead = (
    <Masthead style={{ backgroundColor: "#1a1a1a", borderBottom: "1px solid #2d2d2d" }}>
      <MastheadMain>
        <MastheadBrand>
          <Link
            href="/"
            style={{
              textDecoration: "none",
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "0 1rem",
            }}
          >
            <svg width="24" height="24" viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="10" cy="10" r="10" fill="#ee0000" />
              <path
                d="M5 13 L10 6 L15 13"
                stroke="white"
                strokeWidth="2"
                fill="none"
                strokeLinejoin="round"
              />
            </svg>
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontWeight: 700,
                fontSize: "1.5rem",
                color: "white",
                letterSpacing: "-0.02em",
              }}
            >
              gpu<span style={{ color: "#ee0000" }}>calc</span>
            </span>
          </Link>
        </MastheadBrand>
      </MastheadMain>
    </Masthead>
  );

  const navItemStyle = (isActive: boolean) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '14.5px',
    fontWeight: 500,
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
    color: isActive ? '#ffffff' : 'rgba(255,255,255,0.85)',
    padding: '9px 16px',
    textDecoration: 'none',
    borderBottom: 'none',
  });

  const iconStyle = { width: '15px', height: '15px' };

  const NavItemWithIcon = ({
    icon: Icon,
    label,
    href,
    isActive
  }: {
    icon: any;
    label: string;
    href: string;
    isActive: boolean;
  }) => (
    <NavItem
      isActive={isActive}
      to={href}
      component={(props: any) => (
        <Link {...props} style={navItemStyle(isActive)}>
          <Icon style={iconStyle} />
          <span>{label}</span>
        </Link>
      )}
    />
  );

  const groupLabelStyle = {
    padding: '12px 16px 3px',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
    color: 'rgba(255,255,255,0.55)',
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
  };

  const sidebar = (
    <PageSidebar
      style={{
        backgroundColor: "#1a1a1a",
        borderRight: "1px solid #2d2d2d",
      }}
    >
      <PageSidebarBody>
        <Nav theme="dark" aria-label="Main navigation" style={{
          '--pf-v5-c-nav__link--after--BorderColor': 'transparent',
          '--pf-v5-c-nav__link--after--BorderWidth': '0'
        } as React.CSSProperties}>
          <NavList style={{
            '--pf-v5-c-nav__item--after--BorderColor': 'transparent',
            '--pf-v5-c-nav__item--after--BorderWidth': '0'
          } as React.CSSProperties}>
            <NavItemWithIcon
              icon={HomeIcon}
              label="Home"
              href="/"
              isActive={pathname === "/"}
            />

            <div style={groupLabelStyle}>ESTIMATE</div>

            <NavItemWithIcon
              icon={BoltIcon}
              label="Quick estimate"
              href="/quick-estimate"
              isActive={pathname === "/quick-estimate"}
            />
            <NavItemWithIcon
              icon={SlidersHIcon}
              label="Advanced estimate"
              href="/calculator"
              isActive={pathname === "/calculator"}
            />
            <NavItemWithIcon
              icon={CubesIcon}
              label="GPU explorer"
              href="/gpu-explorer"
              isActive={pathname === "/gpu-explorer"}
            />
            <NavItemWithIcon
              icon={ListIcon}
              label="Compare"
              href="/compare"
              isActive={pathname === "/compare"}
            />
            <NavItemWithIcon
              icon={HistoryIcon}
              label="History"
              href="#"
              isActive={false}
            />

            <div style={groupLabelStyle}>OPTIMIZE</div>

            <NavItemWithIcon
              icon={DollarSignIcon}
              label="Hybrid savings"
              href="/hybrid-savings"
              isActive={pathname === "/hybrid-savings"}
            />
            <NavItemWithIcon
              icon={RouteIcon}
              label="Routing"
              href="/routing"
              isActive={pathname === "/routing"}
            />

            <div style={groupLabelStyle}>SYSTEM</div>

            <NavItemWithIcon
              icon={CogIcon}
              label="Settings"
              href="#"
              isActive={false}
            />
          </NavList>
        </Nav>

        {/* Version Footer */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          padding: '12px 16px',
          borderTop: '1px solid #2d2d2d',
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          color: 'rgba(255,255,255,0.4)',
          lineHeight: '1.4'
        }}>
          <div>{getVersionString()} · {getShortCommit()}</div>
          <div style={{ fontSize: '10px', marginTop: '2px' }}>
            {getBuildTimeString()}
          </div>
        </div>
      </PageSidebarBody>
    </PageSidebar>
  );

  return (
    <Page
      header={masthead}
      sidebar={sidebar}
      isManagedSidebar
      style={{ backgroundColor: "#f5f5f5" }}
    >
      {children}
    </Page>
  );
}
