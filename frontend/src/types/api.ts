export interface HealthResponse {
  status: string;
  service: string;
}

export interface ScenarioSummary {
  name: string;
  description: string;
}

export interface ScenarioDetail extends ScenarioSummary {
  grid: GridConfig;
  start: [number, number, number];
  goal: [number, number, number];
  restrictions: unknown[];
  aircraft: {
    name: string;
    cruise_tas_kt: number;
  };
  speed_options_kt: number[];
  supported_modes: string[];
}

export interface GridConfig {
  cells_x: number;
  cells_y: number;
  cell_size_nm: number;
  connectivity: number;
  flight_levels: number[];
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  altitude_ft: number;
  heading_deg: number | null;
  speed_tas_kt: number | null;
  time_h: number | null;
  time_bucket: number | null;
}

export interface PlanMetrics {
  objective_cost: number | null;
  distance_nm: number | null;
  elapsed_time_h: number | null;
  planned_time_h: number | null;
  fuel_kg: number | null;
  risk_exposure: number | null;
  expansions: number | null;
  generated_states: number | null;
  runtime_s: number | null;
  path_length: number | null;
}

export interface PlanResponse {
  status: string;
  algorithm: string;
  heuristic: string;
  trajectory: TrajectoryPoint[];
  metrics: PlanMetrics;
  environment: {
    mode: string;
    temporal_resolution_h: number | null;
    wind: { amplitude_kt: number | null; period_h: number | null };
  };
  grid: GridConfig;
}

export interface PlanRequest {
  scenario?: string;
  mode: "static" | "dynamic";
  algorithm: "astar" | "dijkstra";
  heuristic: "zero" | "dynamic-optimistic";
  speed_set_kt?: number[];
  temporal_resolution_h?: number;
  wind_amplitude_kt?: number;
  wind_period_h?: number;
  turn_model?: "none" | "gate" | "gate+cost";
}

export interface ApiError {
  error?: { code?: string; message?: string; details?: unknown };
}
