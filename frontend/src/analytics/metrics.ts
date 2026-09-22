import type { ExperimentRecord, PlanResponse, TrajectoryPoint } from "../types/api";

export interface ProfileSeries {
  time: number[];
  altitude: number[];
  tasTime: number[];
  tas: number[];
  headingTime: number[];
  heading: number[];
  bucketTime: number[];
  buckets: number[];
  groundSpeedTime: number[];
  groundSpeed: number[];
  fuelFlowTime: number[];
  fuelFlow: number[];
  fuelRemainingTime: number[];
  fuelRemaining: number[];
}

export function profileSeries(result: PlanResponse): ProfileSeries | null {
  if (result.environment.mode !== "dynamic") return null;
  const timed = result.trajectory.filter(
    (point): point is TrajectoryPoint & { time_h: number; time_bucket: number | null } =>
      point.time_h !== null
  );
  if (!timed.length) return null;
  const tasPoints = timed.filter(
    (point): point is typeof point & { speed_tas_kt: number } => point.speed_tas_kt !== null
  );
  const headingPoints = timed.filter(
    (point): point is typeof point & { heading_deg: number } => point.heading_deg !== null
  );
  const bucketPoints = timed.filter(
    (point): point is typeof point & { time_bucket: number } => point.time_bucket !== null
  );
  const groundSpeedPoints = timed.filter(
    (point): point is typeof point & { ground_speed_kt: number } => point.ground_speed_kt !== null
  );
  const fuelFlowPoints = timed.filter(
    (point): point is typeof point & { fuel_flow_kg_h: number } => point.fuel_flow_kg_h !== null
  );
  const fuelRemainingPoints = timed.filter(
    (point): point is typeof point & { fuel_remaining_kg: number } => point.fuel_remaining_kg !== null
  );
  return {
    time: timed.map((point) => point.time_h),
    altitude: timed.map((point) => point.altitude_ft),
    tasTime: tasPoints.map((point) => point.time_h),
    tas: tasPoints.map((point) => point.speed_tas_kt),
    headingTime: headingPoints.map((point) => point.time_h),
    heading: headingPoints.map((point) => point.heading_deg),
    bucketTime: bucketPoints.map((point) => point.time_h),
    buckets: bucketPoints.map((point) => point.time_bucket),
    groundSpeedTime: groundSpeedPoints.map((point) => point.time_h),
    groundSpeed: groundSpeedPoints.map((point) => point.ground_speed_kt),
    fuelFlowTime: fuelFlowPoints.map((point) => point.time_h),
    fuelFlow: fuelFlowPoints.map((point) => point.fuel_flow_kg_h),
    fuelRemainingTime: fuelRemainingPoints.map((point) => point.time_h),
    fuelRemaining: fuelRemainingPoints.map((point) => point.fuel_remaining_kg)
  };
}

export function experimentLabel(record: ExperimentRecord): string {
  const speed = record.speed_set_kt.length ? record.speed_set_kt.join(",") : "—";
  return `${record.mode} / ${record.algorithm} / ${record.heuristic} / ${speed} kt`;
}
