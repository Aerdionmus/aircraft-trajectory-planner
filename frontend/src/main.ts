import "cesium/Build/Cesium/Widgets/widgets.css";
import { createViewer, setInitialCamera } from "./cesium/viewer";
import { playbackControls, renderTrajectory } from "./cesium/trajectory";
import { api } from "./api/client";
import type { PlanRequest, PlanResponse, ScenarioSummary } from "./types/api";
import "./styles.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Application root is missing");

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div><div class="eyebrow">RESEARCH / SYNTHETIC ENVIRONMENT</div><h1>AIRCRAFT TRAJECTORY PLANNER</h1></div>
      <div class="topbar-context">Flight planning workstation <span>·</span> M5.2.2</div>
    </header>
    <section class="workspace">
      <section class="viewer"><div class="badge">SYNTHETIC PLANNING COORDINATE SYSTEM · LOCAL ENU<br><span>SYNTHETIC WIND VISUALIZATION · ORANGE VECTORS</span></div><div id="cesium-container"></div><div class="legend"><span><b class="dot origin"></b> Origin</span><span><b class="dot destination"></b> Destination</span><span><b class="line completed"></b> Completed route</span><span><b class="line remaining"></b> Remaining route</span><span><b class="arrow"></b> Synthetic wind</span></div><div class="playback"><button id="play-button" type="button" disabled>PLAY</button><button id="pause-button" type="button" disabled>PAUSE</button><button id="restart-button" type="button" disabled>RESTART</button><input id="timeline" type="range" min="0" max="0" step="0.01" value="0" disabled><span id="timeline-label">T+0.00 h</span></div></section>
      <aside class="panel panel-right"><h2>FLIGHT / PLAN INFORMATION</h2><div id="current" class="current-readout">No trajectory loaded.</div><div id="metrics" class="subtle"></div></aside>
    </section>
    <section class="panel controls">
      <div class="controls-heading"><h2>MISSION CONTROLS</h2><div id="status" class="status"></div></div>
      <form id="plan-form">
        <div class="field"><label for="scenario">Scenario</label><select id="scenario"></select></div>
        <div class="field"><label for="mode">Mode</label><select id="mode"><option value="static">Static</option><option value="dynamic">Dynamic</option></select></div>
        <div class="field"><label for="algorithm">Algorithm</label><select id="algorithm"><option value="astar">A*</option><option value="dijkstra">Dijkstra</option></select></div>
        <div class="field"><label for="heuristic">Heuristic</label><select id="heuristic"><option value="zero">Zero</option><option value="dynamic-optimistic">Dynamic optimistic</option></select></div>
        <div class="field"><label for="speeds">Speed set (TAS kt, optional)</label><input id="speeds" placeholder="e.g. 330" inputmode="decimal"></div>
        <div class="field dynamic-control"><label for="time-step">Temporal resolution (h)</label><input id="time-step" type="number" value="0.5" min="0.01" step="0.1"></div>
        <div class="field dynamic-control"><label for="wind-amplitude">Wind amplitude (kt)</label><input id="wind-amplitude" type="number" value="0" min="0" step="1"></div>
        <div class="field dynamic-control"><label for="wind-period">Wind period (h)</label><input id="wind-period" type="number" value="4" min="0.01" step="0.5"></div>
        <div class="field"><label for="turn-model">Turn model</label><select id="turn-model"><option value="none">None</option><option value="gate">Gate</option><option value="gate+cost">Gate + cost</option></select></div>
        <button id="plan-button" type="submit">PLAN TRAJECTORY</button>
      </form>
    </section>
  </main>
`;

const viewer = createViewer(document.querySelector("#cesium-container") as HTMLElement);
setInitialCamera(viewer);
const scenarioSelect = document.querySelector<HTMLSelectElement>("#scenario")!;
const modeSelect = document.querySelector<HTMLSelectElement>("#mode")!;
const algorithmSelect = document.querySelector<HTMLSelectElement>("#algorithm")!;
const heuristicSelect = document.querySelector<HTMLSelectElement>("#heuristic")!;
const status = document.querySelector<HTMLDivElement>("#status")!;
const metrics = document.querySelector<HTMLDivElement>("#metrics")!;
const current = document.querySelector<HTMLDivElement>("#current")!;
const button = document.querySelector<HTMLButtonElement>("#plan-button")!;
const timeline = document.querySelector<HTMLInputElement>("#timeline")!;
const timelineLabel = document.querySelector<HTMLSpanElement>("#timeline-label")!;
const playButton = document.querySelector<HTMLButtonElement>("#play-button")!;
const pauseButton = document.querySelector<HTMLButtonElement>("#pause-button")!;
const restartButton = document.querySelector<HTMLButtonElement>("#restart-button")!;

function updateControls(): void {
  const dynamic = modeSelect.value === "dynamic";
  document.querySelectorAll<HTMLElement>(".dynamic-control").forEach((element) => {
    element.hidden = !dynamic;
  });
  const dynamicHeuristic = heuristicSelect.querySelector<HTMLOptionElement>('option[value="dynamic-optimistic"]')!;
  dynamicHeuristic.disabled = !dynamic || algorithmSelect.value === "dijkstra";
  if (!dynamic) heuristicSelect.value = "zero";
  if (algorithmSelect.value === "dijkstra") heuristicSelect.value = "zero";
}

function value(id: string): string {
  return (document.querySelector<HTMLInputElement>(`#${id}`)!).value;
}

function makeRequest(): PlanRequest {
  const mode = modeSelect.value as PlanRequest["mode"];
  const speedText = value("speeds").trim();
  return {
    scenario: scenarioSelect.value,
    mode,
    algorithm: (document.querySelector<HTMLSelectElement>("#algorithm")!).value as PlanRequest["algorithm"],
    heuristic: heuristicSelect.value as PlanRequest["heuristic"],
    speed_set_kt: speedText ? speedText.split(",").map(Number) : undefined,
    temporal_resolution_h: Number(value("time-step")),
    wind_amplitude_kt: Number(value("wind-amplitude")),
    wind_period_h: Number(value("wind-period")),
    turn_model: (document.querySelector<HTMLSelectElement>("#turn-model")!).value as PlanRequest["turn_model"]
  };
}

function formatMetric(value: number | null): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

function showMetrics(result: PlanResponse): void {
  const metric = (label: string, item: string) => `<div class="metric"><span>${label}</span><span>${item}</span></div>`;
  metrics.innerHTML = `
    <h3>TRAJECTORY</h3>
    ${metric("Status", result.status)}${metric("Algorithm", result.algorithm)}${metric("Heuristic", result.heuristic)}
    <h3>ROUTE</h3>
    ${metric("Origin", result.trajectory.length ? `${result.trajectory[0].x.toFixed(2)}, ${result.trajectory[0].y.toFixed(2)}` : "—")}
    ${metric("Destination", result.trajectory.length ? `${result.trajectory[result.trajectory.length - 1].x.toFixed(2)}, ${result.trajectory[result.trajectory.length - 1].y.toFixed(2)}` : "—")}
    ${metric("Distance", `${formatMetric(result.metrics.distance_nm)} NM`)}
    <h3>FLIGHT</h3>
    ${metric("Altitude", result.trajectory.length ? `${formatMetric(result.trajectory[0].altitude_ft)} ft` : "—")}
    ${metric("TAS", result.trajectory.length ? `${formatMetric(result.trajectory[0].speed_tas_kt)} kt` : "—")}
    ${metric("Heading", result.trajectory.length && result.trajectory[0].heading_deg !== null ? `${result.trajectory[0].heading_deg.toFixed(1)}°` : "—")}
    <h3>PERFORMANCE</h3>
    ${metric("Objective cost", formatMetric(result.metrics.objective_cost))}${metric("Elapsed time", `${formatMetric(result.metrics.elapsed_time_h)} h`)}${metric("Planned time", `${formatMetric(result.metrics.planned_time_h)} h`)}${metric("Fuel", `${formatMetric(result.metrics.fuel_kg)} kg`)}${metric("Risk", formatMetric(result.metrics.risk_exposure))}${metric("Expansions", formatMetric(result.metrics.expansions))}${metric("Generated states", formatMetric(result.metrics.generated_states))}
  `;
}

function showCurrent(point: PlanResponse["trajectory"][number] | null, timeH: number | null): void {
  if (!point) {
    current.textContent = "No timed trajectory available.";
    return;
  }
  if (timeH !== null) timeline.value = String(timeH);
  timelineLabel.textContent = `T+${(timeH ?? 0).toFixed(2)} h`;
  current.innerHTML = [
    ["Time", timeH === null ? "—" : `${timeH.toFixed(2)} h`],
    ["Time bucket", point.time_bucket === null ? "—" : String(point.time_bucket)],
    ["Altitude", `${formatMetric(point.altitude_ft)} ft`],
    ["TAS", `${formatMetric(point.speed_tas_kt)} kt`],
    ["Heading", point.heading_deg === null ? "—" : `${point.heading_deg.toFixed(1)}°`]
  ].map(([label, value]) => `<div class="metric"><span>${label}</span><span>${value}</span></div>`).join("");
}

function setPlaybackEnabled(enabled: boolean, durationH = 0): void {
  [playButton, pauseButton, restartButton].forEach((control) => { control.disabled = !enabled; });
  timeline.disabled = !enabled;
  timeline.max = String(durationH);
  timeline.value = "0";
}

async function loadScenarios(): Promise<void> {
  const scenarios: ScenarioSummary[] = await api.getScenarios();
  scenarioSelect.innerHTML = scenarios.map((scenario) => `<option value="${scenario.name}">${scenario.name}</option>`).join("");
}

document.querySelector("#mode")!.addEventListener("change", updateControls);
document.querySelector("#heuristic")!.addEventListener("change", updateControls);
algorithmSelect.addEventListener("change", updateControls);
document.querySelector("#plan-form")!.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  status.textContent = "Planning trajectory…";
  status.className = "status";
  try {
    const result = await api.plan(makeRequest());
    if (!result.trajectory.length) throw new Error(`Planner returned ${result.status} with no trajectory`);
    const durationH = Math.max(0, ...result.trajectory.map((point) => point.time_h ?? 0));
    setPlaybackEnabled(result.trajectory.some((point) => point.time_h !== null), durationH);
    renderTrajectory(viewer, result.trajectory, result, showCurrent);
    showMetrics(result);
    status.textContent = "Trajectory loaded";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unexpected planning error";
    status.className = "status error";
  } finally {
    button.disabled = false;
  }
});

playButton.addEventListener("click", () => playbackControls().play());
pauseButton.addEventListener("click", () => playbackControls().pause());
restartButton.addEventListener("click", () => playbackControls().restart());
timeline.addEventListener("input", () => playbackControls().scrub(Number(timeline.value)));

updateControls();
loadScenarios().catch((error: unknown) => {
  status.textContent = error instanceof Error ? error.message : "Unable to load scenarios";
  status.className = "status error";
});
