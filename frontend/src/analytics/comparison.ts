import type { ExperimentRecord } from "../types/api";
import { experimentLabel } from "./metrics";

function display(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

export function renderComparisonTable(records: ExperimentRecord[]): void {
  const target = document.querySelector<HTMLDivElement>("#comparison-table");
  if (!target) return;
  if (!records.length) {
    target.textContent = "No comparison results.";
    return;
  }
  target.innerHTML = `
    <table>
      <thead><tr><th>Configuration</th><th>Algorithm</th><th>Heuristic</th><th>Distance</th><th>Time</th><th>Fuel</th><th>Risk</th><th>Cost</th><th>Expansions</th><th>Runtime (machine-dependent)</th></tr></thead>
      <tbody>${records.map((record) => `
        <tr>
          <td>${experimentLabel(record)}</td><td>${record.algorithm}</td><td>${record.heuristic}</td>
          <td>${display(record.distance_nm)} NM</td><td>${display(record.planned_time_h)} h</td>
          <td>${display(record.fuel_kg)} kg</td><td>${display(record.risk_exposure)}</td>
          <td>${display(record.objective_cost)}</td><td>${display(record.expansions)}</td><td>${display(record.runtime_s)} s</td>
        </tr>`).join("")}</tbody>
    </table>`;
}
