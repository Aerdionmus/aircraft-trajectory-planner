import type { PlanResponse } from "../types/api";
import { profileSeries } from "./metrics";
import type * as PlotlyTypes from "plotly.js-dist-min";

type PlotlyApi = typeof import("plotly.js-dist-min").default;
let plotly: PlotlyApi | undefined;
let plotlyPromise: Promise<PlotlyApi> | undefined;

async function loadPlotly(): Promise<PlotlyApi> {
  if (plotly) return plotly;
  plotlyPromise ??= import("plotly.js-dist-min").then((module) => module.default);
  plotly = await plotlyPromise;
  return plotly;
}

const layoutBase: Partial<PlotlyTypes.Layout> = {
  paper_bgcolor: "#27343d",
  plot_bgcolor: "#202b32",
  font: { color: "#d5e2e7", family: "Segoe UI, Arial, sans-serif", size: 11 },
  margin: { l: 52, r: 18, t: 34, b: 42 },
  hovermode: "x unified",
  legend: { orientation: "h", y: 1.12 }
};

const config: Partial<PlotlyTypes.Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"]
};

async function plot(id: string, title: string, yTitle: string, x: number[], y: number[], color: string): Promise<void> {
  const Plotly = await loadPlotly();
  Plotly.newPlot(
    id,
    [{ x, y, type: "scatter", mode: "lines+markers", name: yTitle, line: { color, width: 2 }, marker: { size: 4 } }],
    { ...layoutBase, title: { text: title, font: { size: 12 } }, xaxis: { title: { text: "Mission time (h)" }, gridcolor: "#384952" }, yaxis: { title: { text: yTitle }, gridcolor: "#384952" } },
    config
  );
}

export async function renderFlightCharts(result: PlanResponse): Promise<void> {
  const series = profileSeries(result);
  const message = result.environment.mode === "static"
    ? "Static routes do not provide time-series analytics."
    : "No timed trajectory data available.";
  if (!series) {
    document.querySelectorAll<HTMLElement>(".profile-chart").forEach((element) => {
      element.innerHTML = `<div class="chart-empty">${message}</div>`;
    });
    return;
  }
  const Plotly = await loadPlotly();
  await plot("altitude-chart", "Altitude vs Time", "Altitude (ft)", series.time, series.altitude, "#55b9d0");
  if (series.tas.length) {
    await plot("tas-chart", "TAS vs Time", "TAS (kt)", series.tasTime, series.tas, "#8fcf9c");
  } else {
    document.querySelector("#tas-chart")!.innerHTML = `<div class="chart-empty">TAS unavailable.</div>`;
  }
  if (series.heading.length) {
    await plot("heading-chart", "Heading vs Time", "Heading (deg)", series.headingTime, series.heading, "#d7b06d");
  } else {
    document.querySelector("#heading-chart")!.innerHTML = `<div class="chart-empty">Heading unavailable.</div>`;
  }
  if (series.buckets.length) {
    await plot("mission-chart", "Mission Time / Time Bucket", "Time bucket", series.bucketTime, series.buckets, "#c3a4d8");
  } else {
    document.querySelector("#mission-chart")!.innerHTML = `<div class="chart-empty">Time bucket unavailable.</div>`;
  }
  if (series.groundSpeed.length) {
    const Plotly = await loadPlotly();
    await plot("ground-speed-chart", "Ground Speed vs Time", "Ground speed (kt)", series.groundSpeedTime, series.groundSpeed, "#e0a15d");
  } else {
    document.querySelector("#ground-speed-chart")!.innerHTML = `<div class="chart-empty">Ground speed unavailable.</div>`;
  }
  if (series.fuelFlow.length) {
    await plot("fuel-flow-chart", "Fuel Flow vs Time", "Fuel flow (kg/h)", series.fuelFlowTime, series.fuelFlow, "#d88f9b");
  } else {
    document.querySelector("#fuel-flow-chart")!.innerHTML = `<div class="chart-empty">Fuel flow unavailable.</div>`;
  }
  if (series.fuelRemaining.length) {
    await plot("fuel-remaining-chart", "Fuel Remaining vs Time", "Fuel remaining (kg)", series.fuelRemainingTime, series.fuelRemaining, "#b8d38f");
  } else {
    document.querySelector("#fuel-remaining-chart")!.innerHTML = `<div class="chart-empty">Fuel remaining unavailable.</div>`;
  }
}

export function updateFlightChartCursor(timeH: number | null): void {
  if (timeH === null) return;
  if (!plotly) return;
  document.querySelectorAll<HTMLElement>(".profile-chart").forEach((element) => {
    if (!element.querySelector(".js-plotly-plot")) return;
    void plotly?.relayout(element, { shapes: [{ type: "line", x0: timeH, x1: timeH, y0: 0, y1: 1, yref: "paper", line: { color: "#e0a15d", width: 1, dash: "dot" } }] });
  });
}

export function clearFlightCharts(): void {
  if (!plotly) return;
  document.querySelectorAll<HTMLElement>(".profile-chart").forEach((element) => {
    void plotly?.purge(element);
    element.innerHTML = "";
  });
}
