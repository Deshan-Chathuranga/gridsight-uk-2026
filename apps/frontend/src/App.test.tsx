import { render, screen, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import "@testing-library/jest-dom";
import App from "./App";

// Mock Recharts ResponsiveContainer to avoid JSDOM sizing errors
vi.mock("recharts", async (importOriginal) => {
  const original = await importOriginal<typeof import("recharts")>();
  return {
    ...original,
    ResponsiveContainer: ({ children }: any) => <div data-testid="mock-responsive-container">{children}</div>,
  };
});

describe("GridSight UK Dashboard App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();

    // Mock fetch for all endpoints app calls on load
    globalThis.fetch = vi.fn().mockImplementation((url) => {
      if (url.includes("/api/forecasts")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              status: "success",
              is_mock: true,
              metrics: {
                mean_pinball: 0.12,
                coverage_80: 0.81,
                skill_vs_neso: 0.35,
                crossing_rate: 0.0,
              },
              data: [
                {
                  timestamp_utc: "2026-07-15T00:00:00Z",
                  y_true_mw: 0.0,
                  q10_mw: 0.0,
                  q50_mw: 0.0,
                  q90_mw: 0.0,
                  neso_mw: 0.0,
                  capacity_mwp: 14500,
                  is_daylight: false,
                },
                {
                  timestamp_utc: "2026-07-15T12:00:00Z",
                  y_true_mw: 5000.0,
                  q10_mw: 4000.0,
                  q50_mw: 4800.0,
                  q90_mw: 5800.0,
                  neso_mw: 4600.0,
                  capacity_mwp: 14500,
                  is_daylight: true,
                },
              ],
            }),
        });
      }

      if (url.includes("/api/xai/global")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              status: "success",
              is_mock: true,
              importances: [
                { feature: "ssrd_uk", importance: 450, description: "Solar Driver" },
                { feature: "tcc_uk", importance: 280, description: "Cloud Cover" },
              ],
            }),
        });
      }

      if (url.includes("/api/xai/meta")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              status: "success",
              is_mock: true,
              features: [
                "TCN q10", "TCN q50", "TCN q90",
                "LGBM q10", "LGBM q50", "LGBM q90",
                "Clear-Sky GHI"
              ],
              weights: {
                0.5: [0.05, 0.35, 0.05, 0.05, 0.40, 0.05, 0.05],
              },
            }),
        });
      }

      if (url.includes("/api/xai/local")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              status: "success",
              is_mock: true,
              base_value: 0.15,
              prediction: 0.70,
              contributions: [
                { feature: "ssrd_uk", contribution: 0.35, value: 420.0, description: "Solar Driver" },
                { feature: "tcc_uk", contribution: -0.15, value: 0.65, description: "Cloud Cover" },
              ],
            }),
        });
      }

      if (url.includes("/api/pipeline/status")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              status: "success",
              pipeline_state: { status: "IDLE", last_run_timestamp: "2026-07-15 01:00:00" },
              storage_stats: {
                bronze: { exists: true, size_mb: 120.5, file_count: 10 },
                silver: { exists: true, size_mb: 45.2, file_count: 8 },
                gold: { exists: true, size_mb: 8.9, file_count: 4 },
              },
            }),
        });
      }

      // Default fallback empty response
      return Promise.resolve({
        json: () => Promise.resolve({ status: "success", data: [] }),
      });
    });
  });

  it("renders the dashboard title and layout grids", async () => {
    await act(async () => {
      render(<App />);
    });

    // Verify main header title
    expect(screen.getByText(/GridSight/)).toBeInTheDocument();
    expect(screen.getByText(/Probabilistic Solar Forecasting/)).toBeInTheDocument();

    // Verify critical status metrics card names
    expect(screen.getByText(/Pinball Loss/)).toBeInTheDocument();
    expect(screen.getByText(/Empirical Coverage/)).toBeInTheDocument();
    expect(screen.getByText(/Skill vs NESO/)).toBeInTheDocument();
  });

  it("displays the correct fetched metrics on load", async () => {
    await act(async () => {
      render(<App />);
    });

    await waitFor(() => {
      // 81.0% coverage should be rendered (0.81 from mock)
      expect(screen.getByText("81.0%")).toBeInTheDocument();
      // 35.0% skill vs baseline should be rendered (0.35 from mock)
      expect(screen.getByText("35.0%")).toBeInTheDocument();
    });
  });
});
