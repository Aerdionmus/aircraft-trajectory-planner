import {
  Cartesian3,
  EllipsoidTerrainProvider,
  Viewer
} from "cesium";

export function createViewer(container: HTMLElement): Viewer {
  return new Viewer(container, {
    terrainProvider: new EllipsoidTerrainProvider(),
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    infoBox: false,
    selectionIndicator: false,
    creditContainer: document.createElement("div")
  });
}

export function setInitialCamera(viewer: Viewer): void {
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(78, 20, 4_500_000)
  });
}
