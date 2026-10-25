// Opt-in rendered inspection with installed Edge + Node, not part of pytest.
// Uses synthetic temporary Flask data; external map tiles require explicit opt-in.
import {spawn} from "node:child_process";
import {existsSync, mkdirSync, mkdtempSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {once} from "node:events";

const output = process.env.DALEEL_UI_OUTPUT ? resolve(process.env.DALEEL_UI_OUTPUT) :
  mkdtempSync(join(tmpdir(), "daleel-interface-render-"));
mkdirSync(output, {recursive: true});
const python = process.env.DALEEL_PYTHON || resolve(".venv", "Scripts", "python.exe");
const edgeCandidates = [
  process.env.DALEEL_EDGE,
  process.env["PROGRAMFILES(X86)"] && join(process.env["PROGRAMFILES(X86)"], "Microsoft", "Edge", "Application", "msedge.exe"),
  process.env.PROGRAMFILES && join(process.env.PROGRAMFILES, "Microsoft", "Edge", "Application", "msedge.exe")
].filter(Boolean);
const edge = edgeCandidates.find(existsSync);
if (!existsSync(python)) throw new Error("Project Python environment not found. Install the documented requirements first.");
if (!edge) throw new Error("Microsoft Edge not found. Set DALEEL_EDGE to its executable path.");
const checks = [];
const allowMap = process.env.DALEEL_UI_ALLOW_MAP === "1";
const referenceStyles = new Map();
const styleSelectors = {
  login: [".login", ".login__brand", ".login__form-wrap", ".login__form", ".login__title", ".field__input"],
  upload: [".shell", ".sidebar", ".topbar", ".content", ".upload-grid", ".dropzone"],
  processing: [".shell", ".sidebar", ".topbar", ".processing-grid", ".processing-stages", ".processing-monitor", ".gauge"],
  results: [".shell", ".sidebar", ".topbar", ".results-shell", ".results-list", ".results-list__head", ".detection", ".detection__icon", ".results-toolbar"],
  history: [".shell", ".sidebar", ".topbar", ".content", ".history-search", ".table-wrap", ".table"],
  admin: [".shell", ".sidebar", ".topbar", ".content", ".admin-grid", ".account-list", ".create-form"]
};
let fixture, browser, socket;
let nextId = 0;
const pending = new Map();
const listeners = new Map();
let session;
function check(name, passed, detail = null) {
  checks.push({name, passed: Boolean(passed), detail});
}
function send(method, params = {}, targetSession = session) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {pending.delete(id); reject(new Error("CDP timeout: " + method));}, 15000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params, ...(targetSession ? {sessionId: targetSession} : {})}));
  });
}
function on(method, callback) { listeners.set(method, callback); }
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error("Browser evaluation failed: " +
    (result.exceptionDetails.exception?.description || result.exceptionDetails.text) + " in " + expression.slice(0, 180));
  return result.result.value;
}
function waitForOutput(child, stream, pattern) {
  return new Promise((resolve, reject) => {
    let text = "";
    const timer = setTimeout(() => reject(new Error("Process startup timed out")), 20000);
    child[stream].on("data", data => {
      text += data;
      const match = text.match(pattern);
      if (match) {clearTimeout(timer); resolve(match[1]);}
    });
    child.on("error", reject);
  });
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  fixture = spawn(python,
    ["-B", "tests/test_interface.py"], {windowsHide: true, env: {...process.env, PYTHONPATH: process.cwd()}});
  const base = await waitForOutput(fixture, "stdout", /UI_FIXTURE=(http:\/\/[^\s]+)/);
  browser = spawn(edge,
    ["--headless=new", "--disable-gpu", "--no-first-run", "--disable-background-networking",
     "--disable-component-update", "--remote-debugging-port=0",
     "--user-data-dir=" + join(output, "profile"), "about:blank"], {windowsHide: true});
  const address = await waitForOutput(browser, "stderr", /DevTools listening on (ws:\/\/[^\s]+)/);
  socket = new WebSocket(address);
  await once(socket, "open");
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const item = pending.get(message.id);
      if (item) {
        clearTimeout(item.timer);
        pending.delete(message.id);
        message.error ? item.reject(new Error(message.error.message)) : item.resolve(message.result);
      }
    } else listeners.get(message.method)?.(message.params);
  });
  const version = await send("Browser.getVersion", {}, null);
  const target = await send("Target.createTarget", {url: "about:blank"}, null);
  session = (await send("Target.attachToTarget", {targetId: target.targetId, flatten: true}, null)).sessionId;
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setFocusEmulationEnabled", {enabled: true});
  await send("Network.enable");
  await send("Network.setCacheDisabled", {cacheDisabled: true});
  let blocked = 0;
  let blockStatus = false;
  const browserErrors = [];
  on("Runtime.exceptionThrown", event => browserErrors.push(event.exceptionDetails.text));
  on("Fetch.requestPaused", event => {
    const mapTile = /^https:\/\/server\.arcgisonline\.com\/ArcGIS\/rest\/services\/World_Imagery\/MapServer\/tile\/\d+\/\d+\/\d+$/.test(event.request.url)
      || /^https:\/\/[abc]\.tile\.openstreetmap\.org\/\d+\/\d+\/\d+\.png$/.test(event.request.url);
    const allowed = (event.request.url.startsWith(base + "/") &&
      !(blockStatus && event.request.url.includes("/api/status/"))) || (allowMap && mapTile);
    if (!allowed) blocked++;
    send(allowed ? "Fetch.continueRequest" : "Fetch.failRequest",
      allowed ? {requestId: event.requestId} : {requestId: event.requestId, errorReason: "BlockedByClient"}).catch(() => {});
  });
  await send("Fetch.enable", {patterns: [{urlPattern: "*"}]});
  async function navigate(path) {
    await send("Page.navigate", {url: base + path});
    for (let i = 0; i < 60; i++) {
      if (await evaluate('document.readyState === "complete" && (!!document.getElementById("main-content") || !!document.querySelector(".screen.is-active"))')) break;
      await pause(50);
    }
    await pause(200);
  }
  async function viewport(width, height = 900) {
    await send("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false});
  }
  async function snapshot(name) {
    const image = await send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
    writeFileSync(join(output, name + ".png"), Buffer.from(image.data, "base64"));
  }
  async function designStyles(name, reference = false) {
    return evaluate(`(() => {
      const scope = ${reference ? 'document.querySelector(".screen.is-active")' : 'document'};
      const selectors = ${JSON.stringify(styleSelectors)}[${JSON.stringify(name)}];
      // Compare data-independent styles. Resolved grid tracks vary with real names,
      // support labels and scrollbar presence; their declared rules are tested by pytest.
      const properties = ["fontFamily", "fontSize", "fontWeight", "lineHeight", "backgroundColor", "color", "padding", "gap", "borderRadius"];
      return Object.fromEntries(selectors.map(selector => {
        const element = scope.querySelector(selector);
        if (!element) return [selector, null];
        const css = getComputedStyle(element);
        return [selector, Object.fromEntries(properties.map(property => [property, css[property]]))];
      }));
    })()`);
  }
  async function matchReference(name, width) {
    if (width < 1024) return;
    const expected = referenceStyles.get(name + "-" + width);
    const actual = await designStyles(name);
    const differences = [];
    for (const [selector, properties] of Object.entries(expected)) {
      if (!actual[selector]) { differences.push({selector, missing: true}); continue; }
      for (const [property, value] of Object.entries(properties)) {
        // The compact top bar accommodates the navigation toggle.
        if (selector === ".topbar" && property === "padding") continue;
        // The icon-rail sidebar adds internal horizontal padding.
        if (selector === ".sidebar" && property === "padding") continue;
        // Version 2 shows model agreement in neutral ink rather than the prototype's green,
        // which is reserved for Confirmed, so agreement is not read as a review decision.
        if (selector === ".detection__icon" && (property === "color" || property === "backgroundColor")) continue;
        if (actual[selector][property] !== value) differences.push({selector, property, expected: value, actual: actual[selector][property]});
      }
    }
    check(name + "-" + width + ": reference component styles", differences.length === 0, differences);
    const dimensions = await evaluate(`({
      sidebar: document.querySelector(".sidebar")?.getBoundingClientRect().width,
      list: document.querySelector(".results-list")?.getBoundingClientRect().width,
      login: document.querySelector(".login__brand")?.getBoundingClientRect().width,
      form: document.querySelector(".login__form")?.getBoundingClientRect().width,
      content: document.querySelector(".content") && getComputedStyle(document.querySelector(".content")).padding
    })`);
    check(name + "-" + width + ": reference layout dimensions",
      (!dimensions.sidebar || dimensions.sidebar === 220) &&
      (!dimensions.list || dimensions.list === 360) &&
      (!dimensions.login || dimensions.login === width / 2) &&
      (!dimensions.form || dimensions.form === 360) &&
      (!dimensions.content || dimensions.content === "32px"), dimensions);
  }
  async function layout(name) {
    const result = await evaluate(`({
      width: innerWidth, scroll: document.documentElement.scrollWidth,
      small: [...document.querySelectorAll(".btn,.password-toggle,.nav-links a,.nav-logout,.sidebar-toggle")]
        .filter(e => e.getClientRects().length && e.getBoundingClientRect().height < (innerWidth <= 900 && !e.classList.contains("password-toggle") ? 43 : 13)).length,
      images: [...document.querySelectorAll(".observation-evidence")].filter(e=>!e.hidden)
        .every(e=>e.getBoundingClientRect().width <= e.parentElement.getBoundingClientRect().width),
      labels: [...document.querySelectorAll("input:not([type=hidden])")]
        .every(e=>e.closest("label") || (e.id && document.querySelector('label[for="'+e.id+'"]'))),
      overflowElements: [...document.querySelectorAll("body *")].filter(e=>{
        const r=e.getBoundingClientRect();
        return r.right > innerWidth && getComputedStyle(e).position==="absolute";
      }).map(e=>e.tagName+"."+e.className)
    })`);
    check(name + ": no page overflow", result.scroll <= result.width, result);
    check(name + ": touch targets", result.small === 0);
    check(name + ": labels and image containment", result.labels && result.images);
  }
  for (const width of [375, 768, 1024, 1366, 1920]) {
    await viewport(width);
    await navigate("/_ui/reference");
    for (const page of ["login", "upload", "processing", "results", "history", "admin"]) {
      await evaluate(`showScreen(${JSON.stringify(page)})`);
      await pause(250);
      referenceStyles.set(page + "-" + width, await designStyles(page, true));
      await snapshot("reference-" + page + "-" + width);
    }
  }
  browserErrors.length = 0; // Prototype errors are not production errors.
  for (const width of [375, 768, 1024, 1366, 1920]) {
    await viewport(width);
    // Sign out by normal CSRF form when a previous screen has a session.
    await navigate("/_ui/analyst");
    await evaluate('document.querySelector(".nav-form").requestSubmit()');
    await pause(250);
    await navigate("/login");
    await layout("login-" + width);
    await matchReference("login", width);
    await snapshot("login-" + width);
    await navigate("/_ui/analyst");
    for (const [name, path] of [["upload","/upload"],["history","/history"],["processing","/status/2"],
                                ["failed","/status/3"],["results","/results/1"],["zero","/results/4"]]) {
      await navigate(path);
      await layout(name + "-" + width);
      if (styleSelectors[name]) await matchReference(name, width);
      await snapshot(name + "-" + width);
    }
    await navigate("/_ui/admin");
    await layout("admin-" + width);
    await matchReference("admin", width);
    await snapshot("admin-" + width);
    check("admin account search has a visible icon", await evaluate('!!document.querySelector(".account-search svg circle")'));
    check("admin form offers administrator role without an employee password", await evaluate('!!document.querySelector("option[value=administrator]") && !document.querySelector("#create-account-form input[name=password]")'));
    await navigate("/admin/accounts/1/edit");
    check("account edit uses the correct breadcrumb", await evaluate('document.querySelector(".topbar__crumb strong").textContent==="Edit account"'));
    await layout("edit-account-" + width);
    await snapshot("edit-account-" + width);
    await navigate("/_ui/password-setup");
    await layout("set-password-" + width);
    check("password setup has confirmation and 15-character minimum", await evaluate('document.querySelector("#set-password").minLength===15 && !!document.querySelector("#confirm-password")'));
    await snapshot("set-password-" + width);
    // A temporary-password session may only change its password or sign out, so sign out first.
    await evaluate('document.querySelector(".account-signout").requestSubmit()');
    await pause(250);
    await navigate("/_ui/empty");
    await layout("empty-history-" + width);
    if (width === 375) check("empty mobile navigation stays compact", await evaluate('document.querySelector(".sidebar").getBoundingClientRect().height < 250'));
    await snapshot("empty-history-" + width);
  }
  await viewport(375);
  await navigate("/_ui/analyst");
  await navigate("/results/1");
  check("mobile cards precede map", await evaluate('document.querySelector(".observation-list").getBoundingClientRect().bottom <= document.querySelector(".map-panel").getBoundingClientRect().top'));
  await evaluate('document.getElementById("observation-3").scrollIntoView()');
  await evaluate('document.querySelector("#observation-3 [data-open-observation]").click()');
  await pause(300);
  check("missing evidence is truthful", await evaluate('document.getElementById("evidence-dialog").open && !document.querySelector("#evidence-dialog .evidence-notice").hidden'));
  await snapshot("missing-evidence-375");
  await evaluate('document.getElementById("evidence-dialog").close()');
  await pause(100); // Native dialog close dispatches its focus-restoration event asynchronously.
  check("modal closes and restores row focus", await evaluate('!document.getElementById("evidence-dialog").open && document.activeElement.matches("[data-open-observation]")'));
  check("local map works without external script", await evaluate('!!window.daleelMap && window.daleelMap.markers.size===2'));
  if (!allowMap) check("map unavailable fallback", await evaluate('document.getElementById("map-notice").textContent.includes("unavailable")'));
  check("persisted review decision displayed", await evaluate('document.querySelector("#observation-2 button[value=Confirmed]").getAttribute("aria-pressed")==="true"'));
  await evaluate('document.querySelector("[data-agreement-filter=\'One model\']").click()');
  check("agreement filter selects matching observations", await evaluate('[...document.querySelectorAll(".observation-card")].filter(card=>!card.hidden).every(card=>card.dataset.agreement==="One model") && document.getElementById("visible-observation-count").textContent==="1 observations"'));
  check("agreement filter also hides unrelated map markers", await evaluate('[...window.daleelMap.markers.values()].every(marker=>!window.daleelMap.map.hasLayer(marker))'));
  await evaluate('document.querySelector("[data-agreement-filter=all]").click()');
  await viewport(1366);
  await navigate("/results/1");
  await evaluate('document.querySelector("#observation-1 [data-open-observation]").click()');
  await pause(300);
  check("evidence modal shows real image and review form", await evaluate('document.getElementById("evidence-dialog").open && document.querySelector("#evidence-dialog .observation-evidence").naturalWidth===960 && document.querySelector("#evidence-dialog .review-actions").getAttribute("action")==="/observations/1/review"'));
  check("evidence has no redundant approximation or open-image action", await evaluate('!document.getElementById("evidence-dialog").textContent.includes("Drone-Position Approximation") && !document.getElementById("evidence-dialog").textContent.includes("Open image")'));
  check("evidence uses formal timestamp and supporting-frame labels", await evaluate('document.getElementById("evidence-dialog").textContent.includes("Video timestamp") && document.getElementById("evidence-dialog").textContent.includes("Supporting frames") && document.getElementById("evidence-dialog").textContent.includes("00:00:01.500") && !document.getElementById("evidence-dialog").textContent.includes("Visibility")'));
  check("map has no redundant information paragraph", await evaluate('!document.querySelector(".map-help") && !document.querySelector(".map-panel").textContent.includes("Markers show the UAV position")'));
  await snapshot("evidence-modal-1366");
  await evaluate('document.querySelector("#evidence-dialog [data-map-observation]").click()');
  await pause(800);
  check("show on map closes evidence and zooms to highlighted location", await evaluate('!document.getElementById("evidence-dialog").open && window.daleelMap.map.getZoom()>=18 && window.daleelMap.markers.get("1").getRadius()===12 && window.daleelMap.map.getCenter().distanceTo(window.daleelMap.markers.get("1").getLatLng())<1 && !document.querySelector(".leaflet-popup")'));
  await snapshot("selected-map-point-1366");
  await evaluate('window.daleelMap.map.setZoom(15); window.daleelMap.markers.get("2").fire("click"); void 0');
  await pause(800);
  check("map marker click zooms to its location without image popup", await evaluate('window.daleelMap.map.getZoom()>=18 && window.daleelMap.markers.get("2").getRadius()===12 && window.daleelMap.markers.get("1").getRadius()===8 && !document.querySelector(".leaflet-popup")'));
  if (allowMap) {
    for (let attempt = 0; attempt < 40; attempt++) {
      if (await evaluate('[...document.querySelectorAll("img.leaflet-tile")].some(tile=>tile.naturalWidth>0) && document.getElementById("map-status").hidden')) break;
      await pause(250);
    }
    check("real satellite imagery loads over synthetic Saudi desert coordinates", await evaluate('[...document.querySelectorAll("img.leaflet-tile")].some(tile=>tile.naturalWidth>0) && document.getElementById("map-status").hidden'));
    check("close satellite zoom uses available imagery rather than empty high-zoom tiles", await evaluate('window.daleelMap.map.getZoom()>=18 && [...document.querySelectorAll("img.leaflet-tile")].filter(tile=>tile.classList.contains("leaflet-tile-loaded")).every(tile=>tile.src.includes("/tile/16/"))'));
    await snapshot("live-satellite-desert-1366");
    await evaluate('document.querySelectorAll(".leaflet-control-layers input")[1].click()');
    await pause(1000);
    check("satellite and standard map views can switch", await evaluate('document.querySelectorAll(".leaflet-control-layers input")[1].checked'));
  }
  await evaluate('document.querySelector("#observation-1 [data-open-observation]").click()');
  await pause(200);
  await send("Input.dispatchKeyEvent", {type:"keyDown",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  await send("Input.dispatchKeyEvent", {type:"keyUp",key:"Escape",code:"Escape",windowsVirtualKeyCode:27});
  await pause(100);
  check("Escape closes evidence modal", await evaluate('!document.getElementById("evidence-dialog").open'));
  await evaluate('document.querySelector("#observation-2 [data-open-observation]").click()');
  await pause(100);
  check("COCO car has formal Vehicle display label", await evaluate('document.querySelector("#evidence-dialog .frame-info__class").textContent==="Vehicle" && document.querySelector("#observation-2 [data-open-observation]").getAttribute("aria-label").includes("Vehicle")'));
  await evaluate('document.getElementById("evidence-dialog").close()');
  await pause(100);
  await navigate("/history");
  check("history displays original video names and AST", await evaluate('document.querySelector(".mission-table").textContent.includes("desert_search_") && document.querySelector(".mission-table").textContent.includes("Submitted (AST)") && !document.querySelector(".mission-table").textContent.includes("Saudi time") && !document.querySelector(".mission-table").textContent.includes("MP4 footage")'));
  check("history omits redundant heading and view actions", await evaluate('!document.querySelector(".page-header").textContent.includes("Past missions") && !document.querySelector(".mission-table").textContent.includes("View results") && !document.querySelector(".mission-table").textContent.includes("View status")'));
  check("history times include AM or PM", await evaluate('[...document.querySelectorAll(".mission-table .date-abs")].every(node=>/\\d{1,2}:\\d{2} (AM|PM)/.test(node.textContent))'));
  check("delete buttons remain close to status column", await evaluate('[...document.querySelectorAll(".mission-table .delete-link")].every(link=>link.getBoundingClientRect().left-link.closest("tr").querySelector(".status-badge").getBoundingClientRect().right<80 && link.closest("td").getBoundingClientRect().width===64)'));
  check("sidebar owns its collapse control", await evaluate('!!document.querySelector(".sidebar [data-sidebar-toggle]") && !document.querySelector(".topbar [data-sidebar-toggle]")'));
  check("sidebar omits unnecessary D badge", await evaluate('!document.querySelector(".sidebar__brand-mark")'));
  await evaluate('[...document.querySelectorAll("[data-history-row]")].find(row=>row.dataset.missionHref==="/results/1").querySelector(".video-name").click()');
  await pause(400);
  check("mission row opens results without view-results button", await evaluate('location.pathname==="/results/1"'));
  check("observation rows have readable review state, not unexplained symbols", await evaluate('[...document.querySelectorAll(".detection__status")].map(node=>node.textContent).join(",")==="Unreviewed,Confirmed,Rejected" && !document.querySelector(".review-pending-dot,.detection__review,.detection__support")'));
  await navigate("/history");
  const expandedToggleTop = await evaluate('document.querySelector("[data-sidebar-toggle]").getBoundingClientRect().top');
  await evaluate('document.querySelector("[data-sidebar-toggle]").click()');
  await pause(300);
  check("desktop sidebar toggle stays at same vertical level", Math.abs((await evaluate('document.querySelector("[data-sidebar-toggle]").getBoundingClientRect().top'))-expandedToggleTop)<1);
  check("sidebar collapses to a usable 72px icon rail", await evaluate('document.body.classList.contains("sidebar-collapsed") && document.getElementById("primary-sidebar").getBoundingClientRect().width===72 && !document.getElementById("primary-sidebar").inert && document.querySelector("[data-sidebar-toggle]").getAttribute("aria-expanded")==="false" && getComputedStyle(document.querySelector(".sidebar__link-label")).display==="none" && document.querySelector(".sidebar__link svg").getBoundingClientRect().width===20'));
  await evaluate('document.querySelector(".sidebar__link").focus()');
  check("collapsed icon labels appear on keyboard focus", await evaluate('getComputedStyle(document.querySelector(".sidebar__link"),"::after").content.includes("New Mission")'));
  await snapshot("sidebar-collapsed-1366");
  await navigate("/upload");
  check("sidebar preference survives navigation", await evaluate('document.body.classList.contains("sidebar-collapsed")'));
  await evaluate('[...document.querySelectorAll(".sidebar__link")].find(link=>link.getAttribute("href")==="/history").click()');
  await pause(400);
  check("collapsed rail navigation still works", await evaluate('location.pathname==="/history" && document.body.classList.contains("sidebar-collapsed")'));
  await navigate("/upload");
  await evaluate('document.querySelector("[data-sidebar-toggle]").click()');
  await pause(300);
  check("sidebar reopens", await evaluate('!document.body.classList.contains("sidebar-collapsed") && !document.getElementById("primary-sidebar").inert'));
  check("upload cards fill removed brief space", await evaluate('!document.querySelector(".mission-brief") && document.querySelector(".dropzone").getBoundingClientRect().height>=260'));
  check("upload heading has no Mission setup subtitle", await evaluate('!document.querySelector(".page-header").textContent.includes("Mission setup")'));
  await navigate("/history");
  await evaluate('document.getElementById("history-search").value="nonexistent-mission"; document.getElementById("history-search").dispatchEvent(new Event("input"))');
  check("history search shows truthful empty state", await evaluate('!document.querySelector("[data-history-empty]").hidden && [...document.querySelectorAll("[data-history-row]")].every(row=>row.hidden)'));
  await viewport(375);

  await navigate("/upload");
  const mobileToggleTop = await evaluate('document.querySelector("[data-sidebar-toggle]").getBoundingClientRect().top');
  await evaluate('document.querySelector("[data-sidebar-toggle]").click()');
  await pause(300);
  check("mobile sidebar toggle stays at same vertical level", Math.abs((await evaluate('document.querySelector("[data-sidebar-toggle]").getBoundingClientRect().top'))-mobileToggleTop)<1);
  await layout("mobile-sidebar-collapsed");
  await snapshot("sidebar-collapsed-375");
  await evaluate('document.querySelector("[data-sidebar-toggle]").click()');
  await pause(300);
  await evaluate('document.getElementById("submit-mission").disabled=true; window.dispatchEvent(new PageTransitionEvent("pageshow",{persisted:true}))');
  check("back-navigation resets submit", await evaluate('!document.getElementById("submit-mission").disabled'));
  const fakeFile = join(output, "desert_search_" + "long_" .repeat(20) + ".mp4");
  writeFileSync(fakeFile, "synthetic filename selection only");
  const dom = await send("DOM.getDocument");
  const input = await send("DOM.querySelector", {nodeId: dom.root.nodeId, selector: "#video"});
  await send("DOM.setFileInputFiles", {nodeId: input.nodeId, files: [fakeFile]});
  check("file-selection feedback", await evaluate('document.querySelector(".file-panel").classList.contains("has-file") && !document.getElementById("video-selection").hidden'));
  await layout("selected-long-filename-375");
  await snapshot("selected-long-filename-375");

  await navigate("/_ui/admin");
  await send("Input.dispatchKeyEvent", {type:"keyDown", key:"Tab", code:"Tab", windowsVirtualKeyCode:9});
  await send("Input.dispatchKeyEvent", {type:"keyUp", key:"Tab", code:"Tab", windowsVirtualKeyCode:9});
  let visibleFocus = await evaluate('getComputedStyle(document.activeElement).outlineStyle !== "none"');
  check("keyboard focus visible", visibleFocus);
  await snapshot("keyboard-focus");
  await navigate("/_ui/password-setup");
  await evaluate('document.querySelector("[data-password-toggle]").focus()');
  await send("Input.dispatchKeyEvent", {type:"keyDown", key:" ", code:"Space", windowsVirtualKeyCode:32});
  await send("Input.dispatchKeyEvent", {type:"keyUp", key:" ", code:"Space", windowsVirtualKeyCode:32});
  await pause(100);
  const passwordState = await evaluate('({type:document.getElementById("set-password").type,label:document.querySelector("[data-password-toggle]").getAttribute("aria-label"),active:document.activeElement.className})');
  check("keyboard password visibility state", passwordState.type==="text" && passwordState.label==="Hide password", passwordState);
  await snapshot("keyboard-password");
  await evaluate('document.querySelector(".account-signout").requestSubmit()');
  await pause(250);

  await navigate("/_ui/analyst");
  await send("Emulation.setEmulatedMedia", {features:[{name:"prefers-reduced-motion",value:"reduce"}]});
  await navigate("/status/2");
  check("reduced motion disables the progress gauge and indicator", await evaluate('getComputedStyle(document.querySelector(".gauge__bar")).animationName==="none" && getComputedStyle(document.querySelector(".processing-indicator")).animationName==="none"'));
  await snapshot("reduced-motion");
  blockStatus = true;
  await navigate("/status/2");
  await pause(300);
  check("polling error keeps truthful state", await evaluate('document.getElementById("message").textContent.includes("Status unavailable") && document.getElementById("state").textContent==="Processing"'));
  await snapshot("status-unavailable");
  blockStatus = false;

  await navigate("/missions/4/delete");
  check("delete mission requires a confirmation page", await evaluate('document.querySelector(".delete-confirmation") && document.querySelector("button[name=confirm]").value==="delete"'));
  check("delete confirmation has web-product copy without device or trash references", await evaluate('!document.querySelector(".delete-confirmation").textContent.includes("this computer") && !document.querySelector(".delete-confirmation").textContent.includes("local trash")'));
  await snapshot("delete-confirmation-375");
  await evaluate('document.querySelector("button[name=confirm]").click()');
  await pause(500);
  check("confirmed synthetic mission deletion removes it from history", await evaluate('location.pathname==="/history" && ![...document.querySelectorAll("[data-history-row]")].some(row=>row.dataset.missionHref==="/results/4") && document.querySelector(".flash-success").textContent.trim()==="Mission deleted."'));

  // 200% CSS zoom with the actual layout engine: viewport remains 1366x900.
  await viewport(1366);
  for (const path of ["/upload","/history","/results/1"]) {
    await navigate(path);
    await evaluate('document.documentElement.style.zoom="2"');
    await layout("200-percent-" + path);
    await snapshot("zoom-" + path.replaceAll("/","-"));
  }
  await send("Emulation.setScriptExecutionDisabled", {value:true});
  await navigate("/results/1");
  check("no-JS results and native review remain", await evaluate('document.querySelectorAll(".observation-card").length===3 && document.querySelectorAll(".review-actions button[type=submit]").length===6'));
  await snapshot("no-javascript-results");
  await evaluate('document.querySelector("#observation-1 button[value=Confirmed]").click()');
  await pause(500);
  check("no-JS review POST persists", await evaluate('document.querySelector("#observation-1 button[value=Confirmed]").getAttribute("aria-pressed")==="true"'));
  await evaluate('document.querySelector(".nav-form button").click()');
  await pause(500);
  await navigate("/login");
  await evaluate('document.querySelector("input[name=username]").value="example_analyst_"+"a".repeat(84); document.querySelector("input[name=password]").value="temporary-fixture-only"; document.querySelector("form").requestSubmit()');
  await pause(500);
  check("no-JS login POST succeeds", await evaluate('location.pathname==="/upload"'));
  await send("Emulation.setScriptExecutionDisabled", {value:false});
  check("no uncaught application script errors", browserErrors.length === 0, browserErrors);
  check("external resources blocked", blocked > 0, blocked);
  const report = {browser: version.product, liveMap: allowMap, output, checks, failed: checks.filter(c=>!c.passed)};
  writeFileSync(join(output, "inspection.json"), JSON.stringify(report,null,2));
  console.log(JSON.stringify({browser: report.browser, liveMap: allowMap, output,
    passed: checks.filter(c=>c.passed).length, total: checks.length, failed: report.failed}));
  if (report.failed.length) process.exitCode = 1;
} catch (error) {
  console.error(String(error));
  process.exitCode = 1;
} finally {
  if (socket?.readyState === 1) {
    await send("Browser.close", {}, null).catch(()=>{});
    socket.close();
  } else browser?.kill();
  fixture?.stdin.end("stop\n");
}
