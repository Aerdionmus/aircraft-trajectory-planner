import {
  Cartesian3,
  Color,
  ConstantProperty,
  Entity,
  JulianDate,
  LabelStyle,
  Matrix4,
  PolylineCollection,
  SampledPositionProperty,
  PointPrimitiveCollection,
  Transforms,
  Viewer
} from "cesium";
import type { PlanResponse, TrajectoryPoint } from "../types/api";

const REFERENCE_LONGITUDE = -30;
const REFERENCE_LATITUDE = 35;
const NM_TO_METERS = 1852;
const FT_TO_METERS = 0.3048;
let activeMarkers: PointPrimitiveCollection | undefined;
let activePlayback: Playback | undefined;

export function syntheticCartesian(
  point: TrajectoryPoint
): Cartesian3 {
  const reference = Cartesian3.fromDegrees(
    REFERENCE_LONGITUDE,
    REFERENCE_LATITUDE,
    0
  );
  const enu = Transforms.eastNorthUpToFixedFrame(reference);
  const local = new Cartesian3(
    point.x * NM_TO_METERS,
    point.y * NM_TO_METERS,
    point.altitude_ft * FT_TO_METERS
  );
  return Matrix4.multiplyByPoint(enu, local, new Cartesian3());
}

export function geographicCartesian(point: TrajectoryPoint): Cartesian3 {
  if (point.latitude_deg === undefined || point.longitude_deg === undefined) {
    throw new Error("Geographic trajectory point is missing latitude/longitude");
  }
  return Cartesian3.fromDegrees(
    point.longitude_deg,
    point.latitude_deg,
    point.altitude_ft * FT_TO_METERS
  );
}

export function renderTrajectory(
  viewer: Viewer,
  trajectory: TrajectoryPoint[],
  result: PlanResponse,
  onTime: (point: TrajectoryPoint | null, timeH: number | null) => void
): void {
  activePlayback?.destroy();
  activePlayback = undefined;
  viewer.entities.removeAll();
  if (activeMarkers) {
    viewer.scene.primitives.remove(activeMarkers);
    activeMarkers = undefined;
  }
  if (trajectory.length === 0) {
    onTime(null, null);
    return;
  }

  const positions = trajectory.map((point) =>
    result.geographic ? geographicCartesian(point) : syntheticCartesian(point)
  );
  const markerPositions = result.geographic && result.airports
    ? [
        Cartesian3.fromDegrees(
          result.airports.origin.longitude_deg,
          result.airports.origin.latitude_deg,
          (result.airports.origin.elevation_ft ?? 0) * FT_TO_METERS
        ),
        Cartesian3.fromDegrees(
          result.airports.destination.longitude_deg,
          result.airports.destination.latitude_deg,
          (result.airports.destination.elevation_ft ?? 0) * FT_TO_METERS
        )
      ]
    : [positions[0], positions[positions.length - 1]];
  const markers = viewer.scene.primitives.add(
    new PointPrimitiveCollection()
  );
  activeMarkers = markers;
  markers.add({ position: markerPositions[0], color: Color.LIME, pixelSize: 12 });
  markers.add({
    position: markerPositions[1],
    color: Color.ORANGE,
    pixelSize: 12
  });
  viewer.entities.add({
    position: markerPositions[0],
    label: {
      text: result.airports?.origin.icao ?? "ORIGIN",
      fillColor: Color.LIME,
      style: LabelStyle.FILL,
      pixelOffset: new Cartesian3(0, -18, 0)
    }
  });
  viewer.entities.add({
    position: markerPositions[1],
    label: {
      text: result.airports?.destination.icao ?? "DESTINATION",
      fillColor: Color.ORANGE,
      style: LabelStyle.FILL,
      pixelOffset: new Cartesian3(0, -18, 0)
    }
  });
  if (!trajectory.some((point) => point.time_h !== null)) {
    viewer.entities.add({
      polyline: {
        positions,
        width: 4,
        material: Color.CYAN
      }
    });
    void viewer.flyTo(viewer.entities);
    onTime(trajectory[0], null);
    return;
  }
  activePlayback = new Playback(viewer, trajectory, result, positions, onTime);
  activePlayback.start();
  void viewer.flyTo(viewer.entities);
}

export function playbackControls(): {
  play(): void;
  pause(): void;
  restart(): void;
  scrub(timeH: number): void;
} {
  return {
    play: () => activePlayback?.play(),
    pause: () => activePlayback?.pause(),
    restart: () => activePlayback?.restart(),
    scrub: (timeH: number) => activePlayback?.scrub(timeH)
  };
}

class Playback {
  private readonly startTime = JulianDate.fromIso8601("2000-01-01T00:00:00Z");
  private readonly stop: JulianDate;
  private readonly aircraft: SampledPositionProperty;
  private readonly completedRoute: Entity;
  private readonly remainingRoute: Entity;
  private readonly wind: PolylineCollection;
  private readonly windGlyphs: ReturnType<PolylineCollection["add"]>[] = [];
  private readonly route: Cartesian3[];
  private readonly trajectory: TrajectoryPoint[];
  private readonly result: PlanResponse;
  private readonly onTime: (point: TrajectoryPoint | null, timeH: number | null) => void;
  private readonly tick = (): void => this.update();
  private readonly clock: Viewer["clock"];

  constructor(
    private readonly viewer: Viewer,
    trajectory: TrajectoryPoint[],
    result: PlanResponse,
    route: Cartesian3[],
    onTime: (point: TrajectoryPoint | null, timeH: number | null) => void
  ) {
    this.trajectory = trajectory;
    this.result = result;
    this.route = route;
    this.onTime = onTime;
    this.clock = viewer.clock;
    const duration = Math.max(
      0,
      ...trajectory.map((point) => point.time_h ?? 0)
    );
    this.stop = JulianDate.addSeconds(this.startTime, duration * 3600, new JulianDate());
    this.aircraft = new SampledPositionProperty();
    trajectory.forEach((point, index) => {
      const time = JulianDate.addSeconds(
        this.startTime,
        (point.time_h ?? 0) * 3600,
        new JulianDate()
      );
      this.aircraft.addSample(time, route[index]);
    });
    this.wind = viewer.scene.primitives.add(new PolylineCollection());
    this.createWindGlyphs();
    this.completedRoute = viewer.entities.add({
      polyline: { positions: [], width: 5, material: Color.CYAN }
    });
    this.remainingRoute = viewer.entities.add({
      polyline: { positions: [], width: 3, material: Color.CYAN.withAlpha(0.35) }
    });
    viewer.entities.add({
      position: this.aircraft,
      point: { pixelSize: 14, color: Color.YELLOW },
      label: {
        text: "AIRCRAFT",
        fillColor: Color.YELLOW,
        style: LabelStyle.FILL,
        pixelOffset: new Cartesian3(0, 20, 0)
      }
    });
    this.clock.onTick.addEventListener(this.tick);
    this.clock.startTime = this.startTime.clone();
    this.clock.stopTime = this.stop.clone();
    this.clock.currentTime = this.startTime.clone();
    this.clock.clockRange = 2;
    this.clock.multiplier = 1800;
    this.clock.shouldAnimate = false;
  }

  start(): void {
    this.update();
  }

  play(): void {
    this.clock.shouldAnimate = true;
  }

  pause(): void {
    this.clock.shouldAnimate = false;
  }

  restart(): void {
    this.clock.currentTime = this.startTime.clone();
    this.clock.shouldAnimate = false;
    this.update();
  }

  scrub(timeH: number): void {
    const bounded = Math.max(0, Math.min(timeH, this.durationHours()));
    this.clock.currentTime = JulianDate.addSeconds(
      this.startTime,
      bounded * 3600,
      new JulianDate()
    );
    this.clock.shouldAnimate = false;
    this.update();
  }

  destroy(): void {
    this.clock.onTick.removeEventListener(this.tick);
    this.viewer.scene.primitives.remove(this.wind);
    this.clock.shouldAnimate = false;
  }

  private durationHours(): number {
    return Math.max(0, ...this.trajectory.map((point) => point.time_h ?? 0));
  }

  private update(): void {
    const timeH = JulianDate.secondsDifference(this.clock.currentTime, this.startTime) / 3600;
    const point = this.pointAt(timeH);
    this.onTime(point, point?.time_h ?? (this.hasTimedTrajectory() ? timeH : null));
    this.updateRoute(point);
    this.updateWind(timeH);
  }

  private updateRoute(point: TrajectoryPoint | null): void {
    const index = point ? this.trajectory.indexOf(point) : 0;
    this.completedRoute.polyline!.positions = new ConstantProperty(
      this.route.slice(0, index + 1)
    );
    this.remainingRoute.polyline!.positions = new ConstantProperty(
      this.route.slice(index + 1)
    );
  }

  private hasTimedTrajectory(): boolean {
    return this.trajectory.some((point) => point.time_h !== null);
  }

  private pointAt(timeH: number): TrajectoryPoint | null {
    if (!this.trajectory.length) return null;
    let current = this.trajectory[0];
    for (const point of this.trajectory) {
      if ((point.time_h ?? 0) <= timeH) current = point;
      else break;
    }
    return current;
  }

  private updateWind(timeH: number): void {
    const amplitude = this.result.environment.wind.amplitude_kt ?? 0;
    const period = this.result.environment.wind.period_h;
    if (amplitude <= 0 || period === null || period <= 0) {
      this.windGlyphs.forEach((glyph) => { glyph.show = false; });
      return;
    }
    if (this.result.geographic) {
      this.windGlyphs.forEach((glyph) => { glyph.show = false; });
      return;
    }
    const grid = this.result.grid;
    const altitudeFt = grid.flight_levels[0];
    const centerX = 20;
    const centerY = 20;
    let glyphIndex = 0;
    for (let x = 0; x < grid.cells_x; x++) {
      for (let y = 0; y < grid.cells_y; y++) {
        const xNm = x * grid.cell_size_nm;
        const yNm = y * grid.cell_size_nm;
        const dx = xNm - centerX;
        const dy = yNm - centerY;
        const length = Math.hypot(dx, dy);
        if (length === 0) continue;
        const directionX = -dy / length;
        const directionY = dx / length;
        const phase =
          (2 * Math.PI * timeH) / period +
          Math.hypot(dx, dy) +
          altitudeFt / 1000;
        const scale = (Math.sin(phase) * amplitude) / 80;
        const tail = syntheticCartesian({
          x: xNm,
          y: yNm,
          altitude_ft: altitudeFt,
          heading_deg: null,
          speed_tas_kt: null,
          time_h: null,
          time_bucket: null,
          flight_phase: null,
          ground_speed_kt: null,
          fuel_flow_kg_h: null,
          fuel_remaining_kg: null,
          mass_kg: null,
          wind_speed_kt: null,
          wind_direction_deg: null
        });
        const head = syntheticCartesian({
          x: xNm + directionX * scale,
          y: yNm + directionY * scale,
          altitude_ft: altitudeFt,
          heading_deg: null,
          speed_tas_kt: null,
          time_h: null,
          time_bucket: null,
          flight_phase: null,
          ground_speed_kt: null,
          fuel_flow_kg_h: null,
          fuel_remaining_kg: null,
          mass_kg: null,
          wind_speed_kt: null,
          wind_direction_deg: null
        });
        const glyph = this.windGlyphs[glyphIndex++];
        glyph.show = true;
        glyph.positions = [tail, head];
      }
    }
    this.windGlyphs.slice(glyphIndex).forEach((glyph) => { glyph.show = false; });
  }

  private createWindGlyphs(): void {
    const grid = this.result.grid;
    for (let x = 0; x < grid.cells_x; x++) {
      for (let y = 0; y < grid.cells_y; y++) {
        const xNm = x * grid.cell_size_nm;
        const yNm = y * grid.cell_size_nm;
        if (Math.hypot(xNm - 20, yNm - 20) === 0) continue;
        this.windGlyphs.push(this.wind.add({
          positions: [Cartesian3.ZERO, Cartesian3.ZERO],
          width: 2,
          material: Color.ORANGE.withAlpha(0.65),
          show: false
        }));
      }
    }
  }
}
