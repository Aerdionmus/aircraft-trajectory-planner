import "cesium/Build/Cesium/Widgets/widgets.css";
import { createViewer, setInitialCamera } from "./cesium/viewer";
import { renderTrajectory } from "./cesium/trajectory";
import { api } from "./api/client";
import type { PlanRequest, PlanResponse, ScenarioSummary } from "./types/api";
import "./styles.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Application root is missing");

app.innerHTML = `
  <main class="shell">
    <section class="panel panel-left">
      <div class="eyebrow">RESEARCH / EDUCATIONAL TOOL</div>
      <h1>AIRCRAFT TRAJECTORY PLANNER</h1>
      <div class="subtle">Mission control foundation · M5.2.1</div>
      <h2>MISSION CONTROLS</h2>
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
        <div id="status" class="status"></div>
      </form>
    </section>
    <section class="viewer"><div class="badge">SYNTHETIC PLANNING COORDINATE SYSTEM · LOCAL ENU</div><div id="cesium-container"></div></section>
    <aside class="panel panel-right"><h2>TRAJECTORY READOUT</h2><div id="metrics" class="subtle">No trajectory planned.</div></aside>
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
const button = document.querySelector<HTMLButtonElement>("#plan-button")!;

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
  const items: [string, string][] = [
    ["Status", result.status],
    ["Algorithm", result.algorithm],
    ["Heuristic", result.heuristic],
    ["Distance", `${formatMetric(result.metrics.distance_nm)} NM`],
    ["Elapsed time", `${formatMetric(result.metrics.elapsed_time_h)} h`],
    ["Planned time", `${formatMetric(result.metrics.planned_time_h)} h`],
    ["Fuel", `${formatMetric(result.metrics.fuel_kg)} kg`],
    ["Risk", formatMetric(result.metrics.risk_exposure)],
    ["Objective cost", formatMetric(result.metrics.objective_cost)],
    ["Expansions", formatMetric(result.metrics.expansions)],
    ["Generated states", formatMetric(result.metrics.generated_states)],
    ["Path length", formatMetric(result.metrics.path_length)]
  ];
  metrics.innerHTML = items.map(([label, item]) => `<div class="metric"><span>${label}</span><span>${item}</span></div>`).join("");
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
    renderTrajectory(viewer, result.trajectory);
    showMetrics(result);
    status.textContent = "Trajectory loaded";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unexpected planning error";
    status.className = "status error";
  } finally {
    button.disabled = false;
  }
});

updateControls();
loadScenarios().catch((error: unknown) => {
  status.textContent = error instanceof Error ? error.message : "Unable to load scenarios";
  status.className = "status error";
});
