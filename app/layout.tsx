import type { Metadata } from "next";
import { Red_Hat_Display, Red_Hat_Text, Red_Hat_Mono } from "next/font/google";
import "@patternfly/react-core/dist/styles/base.css";
// Layouts
import "@patternfly/react-styles/css/layouts/Grid/grid.css";
import "@patternfly/react-styles/css/layouts/Flex/flex.css";
import "@patternfly/react-styles/css/layouts/Stack/stack.css";
import "@patternfly/react-styles/css/layouts/Split/split.css";
import "@patternfly/react-styles/css/layouts/Gallery/gallery.css";
import "@patternfly/react-styles/css/layouts/Bullseye/bullseye.css";
// Components used across the app
import "@patternfly/react-styles/css/components/Card/card.css";
import "@patternfly/react-styles/css/components/Alert/alert.css";
import "@patternfly/react-styles/css/components/ExpandableSection/expandable-section.css";
import "@patternfly/react-styles/css/components/Progress/progress.css";
import "@patternfly/react-styles/css/components/Slider/slider.css";
import "@patternfly/react-styles/css/components/Tile/tile.css";
import "@patternfly/react-styles/css/components/ToggleGroup/toggle-group.css";
import "@patternfly/react-styles/css/components/DescriptionList/description-list.css";
import "@patternfly/react-styles/css/components/Form/form.css";
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
