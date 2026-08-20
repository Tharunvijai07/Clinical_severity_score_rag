import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17212b",
        clinical: {
          50: "#f6faf9",
          100: "#e5f0ee",
          200: "#c7ddda",
          500: "#4f8f8a",
          700: "#2e605e",
          900: "#193836",
        },
        severity: {
          low: "#4f8f6a",
          moderate: "#b89b46",
          high: "#c97942",
          critical: "#b65f5f",
        },
      },
      boxShadow: {
        panel: "0 1px 2px rgba(15, 23, 42, 0.06)",
      },
    },
  },
  plugins: [],
} satisfies Config;
