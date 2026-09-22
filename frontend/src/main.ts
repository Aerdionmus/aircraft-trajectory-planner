import "cesium/Build/Cesium/Widgets/widgets.css";
import { createViewer, setInitialCamera } from "./cesium/viewer";
import { playbackControls, renderTrajectory } from "./cesium/trajectory";
import { clearOverlays, renderOverlays, setGridVisible, setRestrictionsVisible } from "./cesium/overlays";
import { api } from "./api/client";
import { clearFlightCharts, renderFlightCharts, updateFlightChartCursor } from "./analytics/charts";
import { renderComparisonTable } from "./analytics/comparison";
import type { Airport, ExperimentRequest, PlanRequest, PlanResponse, ScenarioSummary } from "./types/api";
import "./styles.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Application root is missing");

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div><div class="eyebrow">RESEARCH / GEOGRAPHIC AIRPORT GROUNDING</div><h1>AIRCRAFT TRAJECTORY PLANNER</h1></div>
      <div class="topbar-context">Flight planning workstation <span>·</span> M6.1</div>
    </header>
    <section class="workspace">
      <section class="viewer"><div id="map-badge" class="badge">REAL AIRPORT ROUTE · WGS84<br><span>SYNTHETIC WIND VISUALIZATION · ORANGE VECTORS</span></div><button id="fullscreen-button" class="fullscreen-button" type="button" title="Show map fullscreen">FULLSCREEN MAP</button><div id="cesium-container"></div><div class="legend"><span><b class="dot origin"></b> Origin</span><span><b class="dot destination"></b> Destination</span><span><b class="line completed"></b> Completed route</span><span><b class="line remaining"></b> Remaining route</span><span><b class="arrow"></b> Synthetic wind</span><span><b class="box restriction"></b> Restriction</span><span><b class="box grid-key"></b> Planning grid</span></div><div class="playback"><button id="play-button" type="button" disabled>PLAY</button><button id="pause-button" type="button" disabled>PAUSE</button><button id="restart-button" type="button" disabled>RESTART</button><input id="timeline" type="range" min="0" max="0" step="0.01" value="0" disabled><span id="timeline-label">T+0.00 h</span></div></section>
      <aside class="panel panel-right"><h2>FLIGHT / PLAN INFORMATION</h2><div id="current" class="current-readout">No trajectory loaded.</div><div id="metrics" class="subtle"></div></aside>
    </section>
    <section class="panel controls">
      <div class="controls-heading"><h2>MISSION CONTROLS</h2><div class="layer-controls"><label id="restrictions-control"><input id="restrictions-layer" type="checkbox" checked> <span id="restrictions-label">Restrictions</span></label><label><input id="grid-layer" type="checkbox"> Grid</label><span class="layer-disabled">Risk: aggregate only</span></div><div id="status" class="status"></div></div>
      <form id="plan-form">
        <div class="field"><label for="departure-airport">Departure airport</label><select id="departure-airport"></select></div>
        <div class="field"><label for="arrival-airport">Arrival airport</label><select id="arrival-airport"></select></div>
        <div class="field"><label for="route-source">Route source</label><select id="route-source"><option value="airports">Real airports</option><option value="synthetic">Synthetic scenario</option></select></div>
        <div class="field"><label for="scenario">Synthetic scenario</label><select id="scenario"></select></div>
        <div class="field"><label for="mode">Mode</label><select id="mode"><option value="static">Static</option><option value="dynamic">Dynamic</option></select></div>
        <div class="field"><label for="algorithm">Algorithm</label><select id="algorithm"><option value="astar">A*</option><option value="dijkstra">Dijkstra</option></select></div>
        <div class="field"><label for="heuristic">Heuristic</label><select id="heuristic"><option value="zero">Zero</option><option value="dynamic-optimistic">Dynamic optimistic</option></select></div>
        <div class="field"><label for="speeds">Speed set (TAS kt, optional)</label><input id="speeds" placeholder="e.g. 330" inputmode="decimal"></div>
        <div class="field dynamic-control"><label for="time-step">Temporal resolution (h)</label><input id="time-step" type="number" value="0.5" min="0.1" step="0.1"></div>
        <div class="field dynamic-control"><label for="wind-amplitude">Wind amplitude (kt)</label><input id="wind-amplitude" type="number" value="0" min="0" step="1"></div>
        <div class="field dynamic-control"><label for="wind-period">Wind period (h)</label><input id="wind-period" type="number" value="4" min="0.01" step="0.5"></div>
        <div class="field"><label for="turn-model">Turn model</label><select id="turn-model"><option value="none">None</option><option value="gate">Gate</option><option value="gate+cost">Gate + cost</option></select></div>
        <div class="field"><label for="aircraft-source">Aircraft source</label><select id="aircraft-source"><option value="synthetic">Synthetic</option><option value="openap">OpenAP</option></select></div>
        <div class="field"><label for="aircraft-model">Aircraft</label><select id="aircraft-model"><option value="A320">A320</option></select></div>
        <div class="field"><label for="weather-source">Weather source</label><select id="weather-source"><option value="synthetic">Synthetic</option><option value="snapshot">Snapshot</option></select></div>
        <button id="plan-button" type="submit">PLAN TRAJECTORY</button>
      </form>
    </section>
    <section class="panel analytics">
      <div class="analytics-heading"><div><h2>FLIGHT ANALYTICS</h2><p class="subtle">Presentation of API trajectory and experiment data; not operational aviation analysis.</p></div><div class="comparison-controls"><label for="comparison-family">Comparison</label><select id="comparison-family"><option value="DYNAMIC_REFERENCE">Dijkstra vs A*</option><option value="DYNAMIC_HEURISTIC">Zero vs Dynamic optimistic heuristic</option><option value="STATIC_REGRESSION">Static vs Dynamic</option><option value="TIME_STEP_SENSITIVITY">Temporal resolution sensitivity</option><option value="WIND_SENSITIVITY">Wind amplitude sensitivity</option><option value="SPEED_INTERACTION">Speed-set sensitivity</option><option value="TURN_INTERACTION">Turn-model sensitivity</option><option value="SCALING">Scaling by grid size</option></select><button id="comparison-button" type="button">RUN COMPARISON</button></div></div>
      <div class="profile-charts"><div id="altitude-chart" class="profile-chart"></div><div id="tas-chart" class="profile-chart"></div><div id="heading-chart" class="profile-chart"></div><div id="mission-chart" class="profile-chart"></div><div id="ground-speed-chart" class="profile-chart"></div><div id="fuel-flow-chart" class="profile-chart"></div><div id="fuel-remaining-chart" class="profile-chart"></div></div>
      <div class="analytics-summary"><h3>PERFORMANCE METRICS</h3><div id="analytics-metrics" class="metric-grid">Plan a trajectory to populate exact API metrics.</div><h3>EXPERIMENT COMPARISON</h3><div id="comparison-status" class="subtle">Runtime is machine-dependent.</div><div id="comparison-table" class="comparison-table"></div></div>
    </section>
  </main>
`;

const scenarioSelect = document.querySelector<HTMLSelectElement>("#scenario")!;
const departureSelect = document.querySelector<HTMLSelectElement>("#departure-airport")!;
const arrivalSelect = document.querySelector<HTMLSelectElement>("#arrival-airport")!;
const routeSourceSelect = document.querySelector<HTMLSelectElement>("#route-source")!;
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
const restrictionsLayer = document.querySelector<HTMLInputElement>("#restrictions-layer")!;
const restrictionsControl = document.querySelector<HTMLLabelElement>("#restrictions-control")!;
const restrictionsLabel = document.querySelector<HTMLSpanElement>("#restrictions-label")!;
const gridLayer = document.querySelector<HTMLInputElement>("#grid-layer")!;
const comparisonButton = document.querySelector<HTMLButtonElement>("#comparison-button")!;
const comparisonFamily = document.querySelector<HTMLSelectElement>("#comparison-family")!;
const comparisonStatus = document.querySelector<HTMLDivElement>("#comparison-status")!;
const analyticsMetrics = document.querySelector<HTMLDivElement>("#analytics-metrics")!;
const viewerElement = document.querySelector<HTMLElement>(".viewer")!;
const fullscreenButton = document.querySelector<HTMLButtonElement>("#fullscreen-button")!;
const mapBadge = document.querySelector<HTMLDivElement>("#map-badge")!;
let loadedRestrictionCount = 0;
let viewer: ReturnType<typeof createViewer> | null = null;

function updateControls(): void {
  const dynamic = modeSelect.value === "dynamic";
  document.querySelectorAll<HTMLElement>(".dynamic-control").forEach((element) => {
    element.hidden = !dynamic;
  });
  loadAirports().catch((error: unknown) => {
    status.textContent = error instanceof Error ? error.message : "Unable to load airports";
    status.className = "status error";
  });
  const dynamicHeuristic = heuristicSelect.querySelector<HTMLOptionElement>('option[value="dynamic-optimistic"]')!;
  dynamicHeuristic.disabled = !dynamic || algorithmSelect.value === "dijkstra";
  if (!dynamic) heuristicSelect.value = "zero";
  if (algorithmSelect.value === "dijkstra") heuristicSelect.value = "zero";
  const synthetic = routeSourceSelect.value === "synthetic";
  departureSelect.disabled = synthetic;
  arrivalSelect.disabled = synthetic;
  scenarioSelect.disabled = !synthetic;
}

function value(id: string): string {
  return (document.querySelector<HTMLInputElement>(`#${id}`)!).value;
}

async function loadAirports(): Promise<void> {
  const airports: Airport[] = await api.getAirports();
  const options = airports
    .map((airport) => `<option value="${airport.icao}">${airport.icao} · ${airport.iata ?? "—"} · ${airport.name}</option>`)
    .join("");
  departureSelect.innerHTML = options;
  arrivalSelect.innerHTML = options;
  departureSelect.value = airports.some((airport) => airport.icao === "VOMM") ? "VOMM" : airports[0]?.icao ?? "";
  arrivalSelect.value = airports.some((airport) => airport.icao === "VABB") ? "VABB" : airports[1]?.icao ?? "";
}

function makeRequest(): PlanRequest {
  const mode = modeSelect.value as PlanRequest["mode"];
  const speedText = value("speeds").trim();
  const request: PlanRequest = {
    mode,
    algorithm: (document.querySelector<HTMLSelectElement>("#algorithm")!).value as PlanRequest["algorithm"],
    heuristic: heuristicSelect.value as PlanRequest["heuristic"],
    speed_set_kt: speedText ? speedText.split(",").map(Number) : undefined,
    temporal_resolution_h: Number(value("time-step")),
    wind_amplitude_kt: Number(value("wind-amplitude")),
    wind_period_h: Number(value("wind-period")),
    turn_model: (document.querySelector<HTMLSelectElement>("#turn-model")!).value as PlanRequest["turn_model"],
    aircraft_source: (document.querySelector<HTMLSelectElement>("#aircraft-source")!).value as PlanRequest["aircraft_source"],
    aircraft_model: (document.querySelector<HTMLSelectElement>("#aircraft-model")!).value,
    weather_source: (document.querySelector<HTMLSelectElement>("#weather-source")!).value as PlanRequest["weather_source"]
  };
  if (routeSourceSelect.value === "airports") {
    request.start_airport = departureSelect.value;
    request.goal_airport = arrivalSelect.value;
  } else {
    request.scenario = scenarioSelect.value;
  }
  return request;
}

function formatMetric(value: number | null): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

function showMetrics(result: PlanResponse): void {
  const metric = (label: string, item: string) => `<div class="metric"><span>${label}</span><span>${item}</span></div>`;
  const geography = result.geography
    ? `<h3>GEOGRAPHIC GROUNDING</h3>
       ${metric("Departure grid snap", `${result.geography.departure_snap_error_nm.toFixed(1)} NM`)}
       ${metric("Arrival grid snap", `${result.geography.arrival_snap_error_nm.toFixed(1)} NM`)}`
    : "";
  metrics.innerHTML = `
    <h3>TRAJECTORY</h3>
    ${metric("Aircraft", result.aircraft ? `${result.aircraft.model} (${result.aircraft.source})` : "—")}
    ${metric("Weather", result.weather ? `${result.weather.source} (${result.weather.mode})` : "—")}
    ${metric("Status", result.status)}${metric("Algorithm", result.algorithm)}${metric("Heuristic", result.heuristic)}
    <h3>ROUTE</h3>
    ${metric("Origin", result.trajectory.length ? `${result.trajectory[0].x.toFixed(2)}, ${result.trajectory[0].y.toFixed(2)}` : "—")}
    ${metric("Destination", result.trajectory.length ? `${result.trajectory[result.trajectory.length - 1].x.toFixed(2)}, ${result.trajectory[result.trajectory.length - 1].y.toFixed(2)}` : "—")}
    ${metric("Distance", `${formatMetric(result.metrics.distance_nm)} NM`)}
    ${geography}
    <h3>FLIGHT</h3>
    ${metric("Altitude", result.trajectory.length ? `${formatMetric(result.trajectory[0].altitude_ft)} ft` : "—")}
    ${metric("TAS", result.trajectory.length ? `${formatMetric(result.trajectory[0].speed_tas_kt)} kt` : "—")}
    ${metric("Phase", result.trajectory.length ? result.trajectory[0].flight_phase ?? "—" : "—")}
    ${metric("Fuel flow", result.trajectory.length ? `${formatMetric(result.trajectory[0].fuel_flow_kg_h)} kg/h` : "—")}
    ${metric("Heading", result.trajectory.length && result.trajectory[0].heading_deg !== null ? `${result.trajectory[0].heading_deg.toFixed(1)}°` : "—")}
    <h3>PERFORMANCE</h3>
    ${metric("Objective cost", formatMetric(result.metrics.objective_cost))}${metric("Elapsed time", `${formatMetric(result.metrics.elapsed_time_h)} h`)}${metric("Planned time", `${formatMetric(result.metrics.planned_time_h)} h`)}${metric("Fuel", `${formatMetric(result.metrics.fuel_kg)} kg`)}${metric("Risk", formatMetric(result.metrics.risk_exposure))}${metric("Expansions", formatMetric(result.metrics.expansions))}${metric("Generated states", formatMetric(result.metrics.generated_states))}
  `;
  analyticsMetrics.innerHTML = [
    ["Distance", `${formatMetric(result.metrics.distance_nm)} NM`],
    ["Planned time", `${formatMetric(result.metrics.planned_time_h)} h`],
    ["Elapsed time", `${formatMetric(result.metrics.elapsed_time_h)} h`],
    ["Fuel", `${formatMetric(result.metrics.fuel_kg)} kg`],
    ["Risk", formatMetric(result.metrics.risk_exposure)],
    ["Objective cost", formatMetric(result.metrics.objective_cost)],
    ["Expansions", formatMetric(result.metrics.expansions)],
    ["Generated states", formatMetric(result.metrics.generated_states)],
    ["Path length", formatMetric(result.metrics.path_length)]
  ].map(([label, item]) => `<div class="metric"><span>${label}</span><span>${item}</span></div>`).join("");
}

function showCurrent(point: PlanResponse["trajectory"][number] | null, timeH: number | null): void {
  if (!point) {
    current.textContent = "No timed trajectory available.";
    return;
  }
  if (timeH !== null) timeline.value = String(timeH);
  timelineLabel.textContent = `T+${(timeH ?? 0).toFixed(2)} h`;
  updateFlightChartCursor(timeH);
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

function updateRestrictionControl(): void {
  const dynamic = modeSelect.value === "dynamic";
  const unavailable = dynamic;
  restrictionsLayer.disabled = unavailable || loadedRestrictionCount === 0;
  restrictionsControl.title = unavailable
    ? "Scenario restrictions are not part of the dynamic planner model"
    : loadedRestrictionCount === 0
      ? "No scenario restrictions"
      : "";
  restrictionsLabel.textContent = unavailable
    ? "Restrictions unavailable in dynamic mode"
    : "Restrictions";
}

async function loadScenarios(): Promise<void> {
  const scenarios: ScenarioSummary[] = await api.getScenarios();
  scenarioSelect.innerHTML = scenarios.map((scenario) => `<option value="${scenario.name}">${scenario.name}</option>`).join("");
}

document.querySelector("#mode")!.addEventListener("change", updateControls);
routeSourceSelect.addEventListener("change", updateControls);
document.querySelector("#mode")!.addEventListener("change", updateRestrictionControl);
document.querySelector("#heuristic")!.addEventListener("change", updateControls);
algorithmSelect.addEventListener("change", updateControls);
document.querySelector("#plan-form")!.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  if (viewer) clearOverlays(viewer);
  clearFlightCharts();
  status.textContent = "Planning trajectory…";
  status.className = "status";
  try {
    const result = await api.plan(makeRequest());
    if (!result.trajectory.length) throw new Error(`Planner returned ${result.status} with no trajectory`);
    const durationH = Math.max(0, ...result.trajectory.map((point) => point.time_h ?? 0));
    setPlaybackEnabled(result.trajectory.some((point) => point.time_h !== null), durationH);
    if (viewer) renderTrajectory(viewer, result.trajectory, result, showCurrent);
    else showCurrent(result.trajectory[0] ?? null, result.trajectory[0]?.time_h ?? null);
    const geographic = result.geographic === true;
    mapBadge.innerHTML = geographic
      ? "REAL AIRPORT ROUTE · WGS84<br><span>SYNTHETIC DYNAMIC WIND MODEL</span>"
      : "SYNTHETIC PLANNING COORDINATE SYSTEM · LOCAL ENU<br><span>SYNTHETIC WIND VISUALIZATION · ORANGE VECTORS</span>";
    const scenario = geographic ? null : await api.getScenario(scenarioSelect.value);
    loadedRestrictionCount = scenario?.restrictions.length ?? 0;
    if (viewer) {
      if (scenario) {
        renderOverlays(viewer, modeSelect.value === "static" ? scenario.restrictions : [], scenario.grid);
      } else {
        clearOverlays(viewer);
      }
    }
    updateRestrictionControl();
    if (viewer) {
      setRestrictionsVisible(viewer, restrictionsLayer.checked);
      setGridVisible(viewer, gridLayer.checked);
    }
    showMetrics(result);
    void renderFlightCharts(result);
    updateFlightChartCursor(result.trajectory[0]?.time_h ?? null);
    status.textContent = "Trajectory loaded";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unexpected planning error";
    status.className = "status error";
  } finally {
    button.disabled = false;
  }
});

function comparisonRequest(family: string): ExperimentRequest {
  const base: ExperimentRequest = {
    name: `frontend-${family.toLowerCase()}`,
    cells: [3],
    time_steps_h: [0.5],
    wind_amplitudes_kt: [0, 20],
    wind_periods_h: [4],
    speed_sets_kt: [[330]],
    turn_models: ["none"],
    modes: ["dynamic"],
    heuristics: ["zero"],
    seed: 0,
    family
  };
  if (family === "DYNAMIC_HEURISTIC") base.heuristics = ["zero", "dynamic-optimistic"];
  if (family === "STATIC_REGRESSION") base.modes = ["static", "dynamic"];
  if (family === "TIME_STEP_SENSITIVITY") base.time_steps_h = [0.25, 0.5, 1];
  if (family === "WIND_SENSITIVITY") base.wind_amplitudes_kt = [0, 10, 20];
  if (family === "SPEED_INTERACTION") base.speed_sets_kt = [[330], [300, 330]];
  if (family === "TURN_INTERACTION") base.turn_models = ["none", "gate", "gate+cost"];
  if (family === "SCALING") {
    base.family = "SCALING";
    base.cells = [2, 3, 4];
  }
  if (family === "DYNAMIC_REFERENCE") {
    base.wind_amplitudes_kt = [0];
    base.heuristics = ["zero"];
  }
  return base;
}

comparisonButton.addEventListener("click", async () => {
  comparisonButton.disabled = true;
  comparisonStatus.textContent = "Running comparison…";
  try {
    const response = await api.runExperiment(comparisonRequest(comparisonFamily.value));
    renderComparisonTable(response.results);
    comparisonStatus.textContent = `${response.results.length} measurements loaded. Runtime is machine-dependent.`;
  } catch (error) {
    comparisonStatus.textContent = error instanceof Error ? error.message : "Comparison failed";
    comparisonStatus.className = "subtle error";
  } finally {
    comparisonButton.disabled = false;
  }
});

playButton.addEventListener("click", () => playbackControls().play());
pauseButton.addEventListener("click", () => playbackControls().pause());
restartButton.addEventListener("click", () => playbackControls().restart());
timeline.addEventListener("input", () => playbackControls().scrub(Number(timeline.value)));
restrictionsLayer.addEventListener("change", () => {
  if (viewer) setRestrictionsVisible(viewer, restrictionsLayer.checked);
});
gridLayer.addEventListener("change", () => {
  if (viewer) setGridVisible(viewer, gridLayer.checked);
});
fullscreenButton.addEventListener("click", async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await viewerElement.requestFullscreen();
  }
});
document.addEventListener("fullscreenchange", () => {
  fullscreenButton.textContent = document.fullscreenElement === viewerElement ? "EXIT FULLSCREEN" : "FULLSCREEN MAP";
  fullscreenButton.title = document.fullscreenElement === viewerElement ? "Exit map fullscreen" : "Show map fullscreen";
});

updateControls();
updateRestrictionControl();
clearFlightCharts();
try {
  viewer = createViewer(document.querySelector("#cesium-container") as HTMLElement);
  setInitialCamera(viewer);
} catch (error) {
  const message = error instanceof Error ? error.message : "Cesium viewer initialization failed";
  status.textContent = `Map unavailable: ${message}`;
  status.className = "status error";
}
loadScenarios().catch((error: unknown) => {
  status.textContent = error instanceof Error ? error.message : "Unable to load scenarios";
  status.className = "status error";
});
