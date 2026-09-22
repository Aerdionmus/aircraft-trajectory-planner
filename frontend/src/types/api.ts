export interface HealthResponse {
  status: string;
  service: string;
}

export interface ScenarioSummary {
  name: string;
  description: string;
}

export interface Airport {
  icao: string;
  iata: string | null;
  name: string;
  municipality: string | null;
  country: string | null;
  latitude_deg: number;
  longitude_deg: number;
  elevation_ft: number | null;
  airport_type: string | null;
  source: string;
  source_url: string;
  source_date: string;
}

export interface ScenarioDetail extends ScenarioSummary {
  grid: GridConfig;
  start: [number, number, number];
  goal: [number, number, number];
  restrictions: RestrictionDto[];
  aircraft: {
    name: string;
    cruise_tas_kt: number;
  };
  speed_options_kt: number[];
  supported_modes: string[];
}

export interface RestrictionDto {
  type: "circle" | "polygon" | "corridor";
  id: string;
  centre_nm?: [number, number];
  radius_nm?: number;
  vertices_nm?: [number, number][];
  start_nm?: [number, number];
  end_nm?: [number, number];
  half_width_nm?: number;
  lower_ft?: number;
  upper_ft?: number;
  hard?: boolean;
  penalty_per_nm?: number;
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
  latitude_deg?: number;
  longitude_deg?: number;
  altitude_ft: number;
  heading_deg: number | null;
  speed_tas_kt: number | null;
  time_h: number | null;
  time_bucket: number | null;
  flight_phase: "climb" | "cruise" | "descent" | null;
  ground_speed_kt: number | null;
  fuel_flow_kg_h: number | null;
  fuel_remaining_kg: number | null;
  mass_kg: number | null;
  wind_speed_kt: number | null;
  wind_direction_deg: number | null;
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
  planner_path_length?: number;
  telemetry_sample_count?: number;
}

export interface PlanResponse {
  status: string;
  algorithm: string;
  heuristic: string;
  trajectory: TrajectoryPoint[];
  planner_path_length?: number;
  telemetry_sample_count?: number;
  metrics: PlanMetrics;
  environment: {
    mode: string;
    temporal_resolution_h: number | null;
    wind: { amplitude_kt: number | null; period_h: number | null };
  };
  grid: GridConfig;
  geographic?: boolean;
  reference?: {
    latitude_deg: number;
    longitude_deg: number;
    name: string;
  };
  airports?: {
    origin: Airport;
    destination: Airport;
  };
  geography?: {
    departure_snap_error_nm: number;
    arrival_snap_error_nm: number;
  };
  aircraft?: {
    source: "synthetic" | "openap";
    model: string;
  };
  weather?: {
    source: string;
    valid_time: string | null;
    mode: "synthetic" | "snapshot";
    spatial_resolution: string | null;
    vertical_levels: number[] | null;
    temporal_resolution: string | null;
    provenance: string;
    interpolation_method?: string;
  };
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
  start_airport?: string;
  goal_airport?: string;
  reference_airport?: string;
  aircraft_source?: "synthetic" | "openap";
  aircraft_model?: string;
  weather_source?: "synthetic" | "snapshot";
}

export interface ExperimentRequest {
  name: string;
  cells: number[];
  time_steps_h: number[];
  wind_amplitudes_kt: number[];
  wind_periods_h: number[];
  speed_sets_kt: number[][];
  turn_models: string[];
  modes: string[];
  heuristics: string[];
  seed: number;
  max_expansions?: number;
  family: string;
}

export interface ExperimentRecord {
  family: string;
  scenario_name: string;
  mode: string;
  heuristic: string;
  algorithm: string;
  temporal_resolution_h: number | null;
  wind_amplitude_kt: number;
  wind_period_h: number | null;
  speed_set_kt: number[];
  turn_model: string;
  solved: boolean;
  status: string;
  objective_cost: number | null;
  distance_nm: number | null;
  elapsed_time_h: number | null;
  planned_time_h: number | null;
  fuel_kg: number | null;
  risk_exposure: number | null;
  expansions: number | null;
  generated_states: number | null;
  runtime_s: number | null;
  path_length: number;
}

export interface ExperimentResponse {
  results: ExperimentRecord[];
}

export interface ApiError {
  error?: { code?: string; message?: string; details?: unknown };
}
