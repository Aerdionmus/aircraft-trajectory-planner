import {
  Cartesian3,
  Color,
  ConstantProperty,
  Entity,
  PolygonHierarchy,
  Matrix4,
  Transforms,
  Viewer
} from "cesium";
import type { GridConfig, RestrictionDto } from "../types/api";

const NM_TO_METERS = 1852;
const FT_TO_METERS = 0.3048;
const ORIGIN_LON = -30;
const ORIGIN_LAT = 35;

let activeEntities: Entity[] = [];

function position(xNm: number, yNm: number, altitudeFt: number): Cartesian3 {
  const origin = Cartesian3.fromDegrees(ORIGIN_LON, ORIGIN_LAT, 0);
  const enu = Transforms.eastNorthUpToFixedFrame(origin);
  return Matrix4.multiplyByPoint(
    enu,
    new Cartesian3(xNm * NM_TO_METERS, yNm * NM_TO_METERS, altitudeFt * FT_TO_METERS),
    new Cartesian3()
  );
}

function regionAltitude(region: RestrictionDto): number {
  return ((region.lower_ft ?? 0) + (region.upper_ft ?? 60000)) / 2;
}

export function clearOverlays(viewer: Viewer): void {
  activeEntities.forEach((entity) => viewer.entities.remove(entity));
  activeEntities = [];
}

export function renderOverlays(
  viewer: Viewer,
  restrictions: RestrictionDto[],
  grid: GridConfig
): void {
  clearOverlays(viewer);
  restrictions.forEach((region) => {
    const altitude = regionAltitude(region);
    const color = region.hard ? Color.RED.withAlpha(0.24) : Color.ORANGE.withAlpha(0.2);
    let entity: Entity | undefined;
    if (region.type === "circle" && region.centre_nm && region.radius_nm) {
      entity = viewer.entities.add({
        position: position(region.centre_nm[0], region.centre_nm[1], altitude),
        ellipse: {
          semiMajorAxis: region.radius_nm * NM_TO_METERS,
          semiMinorAxis: region.radius_nm * NM_TO_METERS,
          material: color,
          outline: true,
          outlineColor: region.hard ? Color.RED : Color.ORANGE
        },
        name: region.id,
        properties: { layer: "restriction" },
        description: `Restriction ${region.id}; altitude ${region.lower_ft ?? 0}-${region.upper_ft ?? 60000} ft`
      });
    } else if (region.type === "polygon" && region.vertices_nm) {
      const points = region.vertices_nm.map(([x, y]) => position(x, y, altitude));
      entity = viewer.entities.add({
        polygon: {
          hierarchy: new ConstantProperty(new PolygonHierarchy(points)),
          material: color,
          outline: true,
          outlineColor: region.hard ? Color.RED : Color.ORANGE
        },
        name: region.id,
        properties: { layer: "restriction" }
      });
    } else if (region.type === "corridor" && region.start_nm && region.end_nm) {
      entity = viewer.entities.add({
        polyline: {
          positions: [
            position(region.start_nm[0], region.start_nm[1], altitude),
            position(region.end_nm[0], region.end_nm[1], altitude)
          ],
          width: Math.max(3, (region.half_width_nm ?? 1) * 2),
          material: color
        },
        name: region.id,
        properties: { layer: "restriction" }
      });
    }
    if (entity) activeEntities.push(entity);
  });

  for (let x = 0; x <= grid.cells_x; x++) {
    const xNm = x * grid.cell_size_nm;
    activeEntities.push(viewer.entities.add({
      polyline: {
        positions: [position(xNm, 0, 0), position(xNm, grid.cells_y * grid.cell_size_nm, 0)],
        width: 1,
        material: Color.WHITE.withAlpha(0.08)
      },
      properties: { layer: "grid" }
    }));
  }
  for (let y = 0; y <= grid.cells_y; y++) {
    const yNm = y * grid.cell_size_nm;
    activeEntities.push(viewer.entities.add({
      polyline: {
        positions: [position(0, yNm, 0), position(grid.cells_x * grid.cell_size_nm, yNm, 0)],
        width: 1,
        material: Color.WHITE.withAlpha(0.08)
      },
      properties: { layer: "grid" }
    }));
  }
}

export function setRestrictionsVisible(viewer: Viewer, visible: boolean): void {
  activeEntities.forEach((entity) => {
    if (entity.properties?.getValue()?.layer === "restriction") entity.show = visible;
  });
}

export function setGridVisible(viewer: Viewer, visible: boolean): void {
  activeEntities.forEach((entity) => {
    if (entity.properties?.getValue()?.layer === "grid") entity.show = visible;
  });
}
