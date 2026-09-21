import {
  Cartesian3,
  Color,
  LabelStyle,
  Matrix4,
  PointPrimitiveCollection,
  Transforms,
  Viewer
} from "cesium";
import type { TrajectoryPoint } from "../types/api";

const REFERENCE_LONGITUDE = -30;
const REFERENCE_LATITUDE = 35;
const NM_TO_METERS = 1852;
const FT_TO_METERS = 0.3048;
let activeMarkers: PointPrimitiveCollection | undefined;

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

export function renderTrajectory(
  viewer: Viewer,
  trajectory: TrajectoryPoint[]
): void {
  viewer.entities.removeAll();
  if (activeMarkers) {
    viewer.scene.primitives.remove(activeMarkers);
    activeMarkers = undefined;
  }
  if (trajectory.length === 0) return;

  const positions = trajectory.map((point) => syntheticCartesian(point));
  viewer.entities.add({
    polyline: {
      positions,
      width: 4,
      material: Color.CYAN
    }
  });
  const markers = viewer.scene.primitives.add(
    new PointPrimitiveCollection()
  );
  activeMarkers = markers;
  markers.add({ position: positions[0], color: Color.LIME, pixelSize: 12 });
  markers.add({
    position: positions[positions.length - 1],
    color: Color.ORANGE,
    pixelSize: 12
  });
  viewer.entities.add({
    position: positions[0],
    label: {
      text: "ORIGIN",
      fillColor: Color.LIME,
      style: LabelStyle.FILL,
      pixelOffset: new Cartesian3(0, -18, 0)
    }
  });
  viewer.entities.add({
    position: positions[positions.length - 1],
    label: {
      text: "DESTINATION",
      fillColor: Color.ORANGE,
      style: LabelStyle.FILL,
      pixelOffset: new Cartesian3(0, -18, 0)
    }
  });
  viewer.flyTo(viewer.entities);
}
