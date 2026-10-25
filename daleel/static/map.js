"use strict";

(() => {
  const target = document.getElementById("map");
  const data = document.getElementById("map-data");
  if (!target || !data) return;
  const notice = document.getElementById("map-notice");
  const status = document.getElementById("map-status");
  const retry = document.querySelector("[data-map-retry]");
  const markers = new Map();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let missionMap, currentLayer;
  const message = text => { notice.textContent = text; status.hidden = false; };
  try {
    if (typeof L === "undefined") throw new Error("Map assets unavailable");
    const observations = JSON.parse(data.textContent);
    missionMap = L.map(target, {scrollWheelZoom: true, maxZoom: 19}).setView([24.7, 46.6], 6);
    const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      // Desert coverage at higher native levels may return HTTP-200 placeholder
      // images instead of tileerror. Enlarge available level-16 imagery on close zoom.
      maxNativeZoom: 16, maxZoom: 19,
      attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics and the GIS User Community"
    });
    const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    });
    function bindTiles(layer, name) {
      let failed = false;
      layer.on("loading", () => { failed = false; });
      layer.on("tileerror", () => {
        if (currentLayer !== layer) return;
        failed = true;
        message(name + " imagery unavailable. Check your connection, retry, or switch map view.");
        retry.hidden = false;
      });
      layer.on("load", () => {
        if (currentLayer === layer && !failed) { status.hidden = true; retry.hidden = true; }
      });
    }
    bindTiles(satellite, "Satellite");
    bindTiles(streets, "Map");
    currentLayer = satellite;
    satellite.addTo(missionMap);
    L.control.layers({Satellite: satellite, Map: streets}, {}, {position: "topright"}).addTo(missionMap);
    missionMap.on("baselayerchange", event => {
      currentLayer = event.layer;
      message("Loading " + event.name.toLowerCase() + " view…");
      currentLayer.redraw();
    });
    retry.addEventListener("click", () => {
      message("Loading map view…");
      currentLayer.redraw();
    });
    const colors = {Unreviewed: "#B5832E", Confirmed: "#4F7A4A", Rejected: "#8B3A2E"};
    const bounds = [];
    observations.forEach(observation => {
      const card = document.getElementById(observation.anchor);
      if (!card) return;
      const position = [observation.latitude, observation.longitude];
      const marker = L.circleMarker(position, {
        radius: 8, color: "#FBF8F1", weight: 2.5,
        fillColor: colors[observation.review_status], fillOpacity: 1,
        className: "observation-marker"
      }).addTo(missionMap);
      marker.bindTooltip("Observation " + observation.observation_number, {direction: "top", offset: [0, -8]});
      marker.on("click", () => {
        focus(String(observation.observation_number));
        card.scrollIntoView({block: "nearest"});
      });
      markers.set(String(observation.observation_number), marker);
      bounds.push(position);
    });
    if (bounds.length > 1) missionMap.fitBounds(bounds, {padding: [40, 40], maxZoom: 16, animate: false});
    else if (bounds.length === 1) missionMap.setView(bounds[0], 16);
    else message("No observations have a map point.");
    function focus(number, zoom = true) {
      const selected = markers.get(String(number));
      if (!selected) return false;
      markers.forEach((marker, id) => {
        const active = id === String(number);
        marker.setRadius(active ? 12 : 8);
        marker.setStyle({weight: active ? 4 : 2.5, color: active ? "#C9A96E" : "#FBF8F1"});
        marker.getElement()?.classList.toggle("is-highlighted", active);
        const card = document.querySelector('[data-observation-number="' + id + '"]');
        card?.querySelector(".detection").classList.toggle("is-selected", active);
      });
      const level = zoom ? Math.max(18, missionMap.getZoom()) : missionMap.getZoom();
      if (reducedMotion.matches) missionMap.setView(selected.getLatLng(), level, {animate: false});
      else missionMap.flyTo(selected.getLatLng(), level, {duration: 0.5});
      selected.openTooltip();
      return true;
    }
    function filter() {
      markers.forEach((marker, number) => {
        const card = document.querySelector('[data-observation-number="' + number + '"]');
        if (card.hidden) marker.removeFrom(missionMap);
        else marker.addTo(missionMap);
      });
    }
    window.daleelMap = {focus, filter, map: missionMap, markers};
    new ResizeObserver(() => missionMap.invalidateSize({pan: false})).observe(target);
  } catch (error) {
    message("Map unavailable. Observation review remains available.");
  }
})();
