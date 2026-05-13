import type { Metadata } from "next";
import { Red_Hat_Display, Red_Hat_Text, Red_Hat_Mono } from "next/font/google";
import "@patternfly/react-core/dist/styles/base.css";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";

const redHatDisplay = Red_Hat_Display({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600", "700"],
});

const redHatText = Red_Hat_Text({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});

const redHatMono = Red_Hat_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "GPU Calc — LLM Inference Sizing & Cost Calculator",
  description:
    "Estimate GPU requirements, compare costs, and model LLM inference economics across cloud and on-premise deployments.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${redHatDisplay.variable} ${redHatText.variable} ${redHatMono.variable}`}
    >
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
