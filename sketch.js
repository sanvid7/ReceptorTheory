// ===============================
// sketch.js
// ===============================

// ─────────────────────────────────────────────
// Global Constants and Variables
// ─────────────────────────────────────────────
const TARGET_CONC_MAX = 400;
const sliderMin = 1;
const sliderMax = TARGET_CONC_MAX;
const yMin = 0;
const yMax = 14; //adjust graph height
const MIN_POINTS = 5; // or whatever minimum you want


let scene = 'intro'; // start on achGraph again
let currentSceneIndex = 0;
const scenes = ['intro', 'achGraph', 'inhibitor', 'partialAgonists', 'spareReceptors', 'heartGraph', 'compareGraphs'];

let lastBallCountChangeTime = 0;
let totalAttachmentsSinceLastChange = 0;
let pointCounter = 0;
let conc = 0;
let concentration = 1;
let inhibitorConcentration = 0;
let graphPlotted = false;
let lastConc = 0;
let horizontalShift = 0;
let inhibitorButtonClicked = false;
let maxObservedAvgFreq = 1;
let frameCounter = 0;
let startTime = 0;

let pointList = [];
let attachmentTimes = [];
let attachmentCount = 0;

// Fit-button + fitted curve storage
let fitButton;
let fittedCurve = null; // { points: [{x,y}], params: {Emax, EC50, n, mse} }


// Snapshots for comparison scene
let achSnapshot = null;   // { label, color, points, curve }
let heartSnapshot = null; // { label, color, points, curve }

// UI Elements and Buttons
let slider, inhibitorSlider;
let pointButton, graphButton, continueButton, inhibitorButton;
let clarkButton, mrtButton, compareButton;

// Canvas/graphics variables
let w = 640;
let h = 500;
let unit = 20;
let muscleStrip, sarcolemma, receptor, gpcr, vessel, heart;
let AchBackgroundX = 0;
let fade = 0;

// Ball simulation variables
let particles = [];
let radius = 5;
let diameter = radius * 2;
let separator = 1;
let ballColor = "white";
let follow = false;

// Ghost point used in graph scenes
let ghostPoint = { x: 0, y: 0, alpha: 100 };

// --- Receptor occupancy tracking tied to the existing timer window ---
let occSlots = []; // [{bound:boolean, lastChangeMs:number, cumBoundMs:number}]

function initOccupancySlots(count) {
  const now = millis();
  occSlots = new Array(count).fill(null).map((_, i) => ({
    bound:
      (typeof attachedLigands !== 'undefined' && attachedLigands[i] != null),
    lastChangeMs: now,
    cumBoundMs: 0
  }));
}

// Reset accumulation for the current timing window, preserving current bound state
function resetOccupancyWindow() {
  const now = millis();
  for (let i = 0; i < occSlots.length; i++) {
    if (!occSlots[i]) continue;
    occSlots[i].cumBoundMs = 0;
    occSlots[i].lastChangeMs = now; // start counting from window start
  }
}

// Hooks called by ball.js on state changes
function markReceptorBound(i, nowMs) {
  if (!occSlots[i]) return;
  if (!occSlots[i].bound) {
    occSlots[i].bound = true;
    occSlots[i].lastChangeMs = nowMs;
  }
}

function markReceptorUnbound(i, nowMs) {
  if (!occSlots[i]) return;
  if (occSlots[i].bound) {
    occSlots[i].cumBoundMs += (nowMs - occSlots[i].lastChangeMs);
    occSlots[i].bound = false;
    occSlots[i].lastChangeMs = nowMs;
  }
}

// Sum of bound ms across all receptors in the current window
function getTotalBoundMs(nowMs) {
  let total = 0;
  for (const s of occSlots) {
    if (!s) continue;
    total += s.cumBoundMs + (s.bound ? (nowMs - s.lastChangeMs) : 0);
  }
  return total;
}

// Always normalize as if 8 receptors is the max (for spare receptor demo)
function getNormalizedOccupancyFraction() {
  const now = millis();
  const elapsedMs = now - lastBallCountChangeTime; // timer unchanged (your code)
  if (elapsedMs <= 0) return 0;
  const totalBound = getTotalBoundMs(now); // ms across all receptors
  const maxPossible = elapsedMs * 8;       // normalize to 8 receptors
  return totalBound / maxPossible;         // 0..1
}


// ─────────────────────────────────────────────
// p5.js Preload and Setup
// ─────────────────────────────────────────────
function preload() {
  muscleStrip = loadImage('musclestrip.png');
  sarcolemma = loadImage('sarcolemma.png');
  receptor = loadImage('receptor.png');
  gpcr = loadImage('gpcr.png');
  vessel = loadImage('vessel.png');
  heart = loadImage('heart.png');
}

function setup() {
  createCanvas(1280, 720);
  startTime = millis();
  createNavigationButtons();
  createSliders();
  createActionButtons();
  hideUIElements();
  reset();

  // Respect the chosen scene at startup
  initializeScene(scene);

}

// ─────────────────────────────────────────────
// UI Creation Functions
// ─────────────────────────────────────────────
function createNavigationButtons() {

}


function createSliders() {
  // Main ligand slider (logarithmic-ish display)
  let sliderContainer = createDiv('');
  slider = createCustomSlider(sliderContainer, 1, TARGET_CONC_MAX, 1, 1);


  // Inhibitor slider (linear)
  let inhibitorSliderContainer = createDiv('');
  inhibitorSlider = createCustomSlider(inhibitorSliderContainer, 0, 150, 0, 1, true);
}

function createActionButtons() {

    // Intro menu buttons
  clarkButton = createButton('Clark');
  clarkButton.id('clarkButton');
  clarkButton.class('button-base button-blue');
  clarkButton.position(width/2 - 90, 200); 
  clarkButton.mousePressed(() => {
    scene = 'achGraph';
    initializeScene('achGraph');
  });

  mrtButton = createButton('MRT');
  mrtButton.id('mrtButton');
  mrtButton.class('button-base button-blue');
  mrtButton.position(width/2 - 90, 300);
  mrtButton.mousePressed(() => {
    // do nothing for now
  });

  compareButton = createButton('Clark vs MRT');
  compareButton.id('compareButton');
  compareButton.class('button-base button-blue');
  compareButton.position(width/2 - 90, 400);
  compareButton.mousePressed(() => {
    // do nothing for now
  });

  pointButton = createButton('Plot Point');
  pointButton.id('plotPointButton');
  pointButton.mousePressed(handlePointButtonClick);
  pointButton.class('button-base button-red');

  continueButton = createButton('Continue');
  continueButton.id('continueButton');
  continueButton.mousePressed(handleContinueButtonClick);
  continueButton.class('button-base button-green');

  inhibitorButton = createButton('Plot Inhibitor');
  inhibitorButton.id('plotInhibitorButton');
  inhibitorButton.mousePressed(handleInhibitorButtonClick);
  inhibitorButton.class('button-base button-orange');
  
  // New: Fit Sigmoid button
  fitButton = createButton('Fit Sigmoid');
  fitButton.id('fitSigmoidButton');
  fitButton.mousePressed(handleFitSigmoidClick);
  fitButton.class('button-base button-blue');
  
  

  
}
//test

// ─────────────────────────────────────────────
// Custom Slider Creation (with Log Scale for main slider)
// ─────────────────────────────────────────────
function createCustomSlider(container, min, max, value, step, isInhibitor = false) {
  const sliderContainer = createDiv('');
  sliderContainer.class(isInhibitor ? 'inhibitor-slider-container' : 'slider-container');

  const sliderElement = createSlider(min, max, value, step);
  sliderElement.class(isInhibitor ? 'inhibitor-slider' : 'slider');
  sliderElement.parent(sliderContainer);

  const valueDisplay = createDiv('0');
  valueDisplay.class('slider-value');
  valueDisplay.parent(sliderContainer);

  // cache the range you passed in
  const minVal = min;
  const maxVal = max;

  sliderElement.input(() => {
    const raw = sliderElement.value();

    if (isInhibitor) {
      // Linear: 0 → max
      inhibitorConcentration = raw;
      valueDisplay.html(`${Math.round(inhibitorConcentration)}`);
      updateInhibitorCount(Math.floor(inhibitorConcentration));
      horizontalShift = map(inhibitorConcentration, 0, 150, -4, 4);
      inhibitorButtonClicked = true;
      detachAllLigands();
      resetLigandProperties();
    } else {
      // Logarithmic: map [minVal, maxVal] → [0,1]
      const fraction = (raw - minVal) / (maxVal - minVal); // now 0 at raw=minVal, 1 at raw=maxVal
      const logMin = Math.log10(1);                       // = 0
      const logMax = Math.log10(TARGET_CONC_MAX);         // e.g., log10(400)
      const logValue = logMin + fraction * (logMax - logMin);
      concentration = Math.pow(10, logValue);

      valueDisplay.html(`${Math.round(concentration)}`);
      detachAllLigands();
      resetLigandProperties();
      updateBallCount(); // no need to pass raw
    }
  });

  sliderContainer.parent(container);
  return sliderElement;
}


// ─────────────────────────────────────────────
// Receptor Layout Helpers
// ─────────────────────────────────────────────
function getAchRectangles() {
  return [
    { x: 925,  y: 580, w: 10, h: 20 },
    { x: 755,  y: 585, w: 10, h: 20 },
    { x: 1095, y: 555, w: 10, h: 20 },
    { x: 1225, y: 555, w: 10, h: 20 },
  ];
}

// 6 GPCRs in a single horizontal line for heartGraph.
// Sprites are smaller so all six fit; binding rectangles remain 10×20.
function getHeartLayout() {
  const left = 660;   // leftmost sprite x
  const right = 1240; // rightmost boundary to stay inside the membrane
  const y = 560;      // vertical position of the GPCR row
  const count = 6;
  const size = 90;    // smaller sprite size (only the sprite changes size)
  const widthAvail = right - left;
  const totalGpcrWidth = count * size;
  const gaps = count - 1;
  const gap = (widthAvail - totalGpcrWidth) / gaps; // even spacing

  const gpcrPos = [];
  const rects = [];
  for (let i = 0; i < count; i++) {
    const x = left + i * (size + gap);
    gpcrPos.push({ x, y });
    rects.push({
      x: x + size / 2 - 5,   // centerX - 5
      y: y + size / 2 - 10,  // centerY - 10
      w: 10,
      h: 20
    });
  }
  return { gpcrPos, rects, size };
}

// ─────────────────────────────────────────────
// UI Visibility Helpers
// ─────────────────────────────────────────────
function hideUIElements() {
  slider.hide();
  pointButton.hide();
  continueButton.hide();
  inhibitorSlider.hide();
  inhibitorButton.hide();
  fitButton.hide(); // NEW
  if (clarkButton) clarkButton.hide();
  if (mrtButton) mrtButton.hide();
  if (compareButton) compareButton.hide();

}


function showUIElements() {
  slider.show();
  pointButton.show();
  continueButton.show();
  fitButton.show(); // NEW
}


// ─────────────────────────────────────────────
function initializeScene(sceneName) {
  // Reset simulation variables for a new scene
  particles = [];
  pointList = [];
  pointCounter = 0;
  frameCounter = 0;
  attachmentTimes = [];
  attachmentCount = 0;
  horizontalShift = 0;
  inhibitorButtonClicked = false;
  fittedCurve = null;       // NEW: reset the fitted curve on scene change
  lastBallCountChangeTime = millis(); // keep the rate window fresh

  // NEW: reset occupancy accumulation window to match the timer
  if (typeof resetOccupancyWindow === 'function') resetOccupancyWindow();

  
  hideUIElements();

  switch (sceneName) {
    case 'achGraph': {
      if (typeof setReceptorLayout === 'function') {
        setReceptorLayout(getAchRectangles());
      }
      slider.show();
      pointButton.show();
      continueButton.show();
      inhibitorSlider.hide();
      inhibitorButton.hide();
      break;
    }
    case 'heartGraph': {
      if (typeof setReceptorLayout === 'function') {
        const { rects } = getHeartLayout();
        setReceptorLayout(rects);
      }
      slider.show();
      pointButton.show();
      continueButton.show();
      inhibitorSlider.hide();
      inhibitorButton.hide();
      break;
    }
    case 'compareGraphs': {
      // comparison: graph only on left, right pane whited out
      hideUIElements();
      break;
    }

    case 'intro': {
      hideUIElements();
      if (clarkButton) clarkButton.show();
      if (mrtButton) mrtButton.show();
      if (compareButton) compareButton.show();
      break;
    }
    
    default:
      break;
  }

  // Force an initial spawn when entering graph scenes
  if (sceneName === 'achGraph' || sceneName === 'heartGraph') {
    concentration = 1;
    updateBallCount();   // this creates the first ball
  }

  scene = sceneName;
}


function drawIntroScene() {
  background(173, 216, 230);
  textAlign(CENTER);
  textSize(32);
  fill(0);
  text("Choose a Mode", width/2, 150);

}


// ─────────────────────────────────────────────
// Button and Slider Event Handlers
// ─────────────────────────────────────────────
function handlePointButtonClick() {
  detachAllLigands();
  resetLigandProperties();

  const currentTime = millis();
  const elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;

  if (scene === 'achGraph' || scene === 'heartGraph') {
    // x: current concentration (keep as before)
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);

    // y: normalized occupancy fraction scaled to axis
    const respFraction = getNormalizedOccupancyFraction(); // 0..1
    const yLock = constrain(respFraction * yMax, yMin, yMax);

    ghostPoint.y = yLock;
    pointList.push({ x: ghostPoint.x, y: yLock, alpha: 255 });

    // Keep your existing downstream behavior
    pointCounter = pointList.length;
    graphPlotted = false;
    if (typeof fittedCurve !== 'undefined') fittedCurve = null;
  }
}




function handleContinueButtonClick() {
  lastConc = particles.length;

  if (scene === 'achGraph') {
    achSnapshot = snapshotCurrent('ACH', color(220, 20, 60)); // crimson
    scene = 'heartGraph';
    initializeScene('heartGraph');
  } else if (scene === 'heartGraph') {
    heartSnapshot = snapshotCurrent('HEART', color(30, 144, 255)); // dodgerblue
    scene = 'compareGraphs';
    initializeScene('compareGraphs');
  }
}

function handleInhibitorButtonClick() {
  if (scene === 'inhibitor') {
    let mappedInhibitorValue = mapInhibitorSliderValue(inhibitorSlider.value());
    let targetInhibitorCount = Math.floor(map(mappedInhibitorValue, -12, 12, 0, 150));
    attachmentCount = 0;
    updateInhibitorCount(targetInhibitorCount);
    detachAllLigands();
    horizontalShift = map(mappedInhibitorValue, -12, 12, -4, 4);
    inhibitorButtonClicked = true;
  }
}

// ─────────────────────────────────────────────
// Ball Simulation Functions
// ─────────────────────────────────────────────

function desiredLigandCount() {
  return Math.round(concentration); // or Math.floor(...) if you prefer
}

// Recreate ALL ligand balls on every slider move (preserve inhibitors)
function updateBallCount() {
  // Desired ligand count from current concentration
  const targetLigands = desiredLigandCount();

  // 1) Keep inhibitors only; drop all ligands
  const inhibitorsOnly = particles.filter(p => p.isInhibitor);
  particles = inhibitorsOnly;

  // 2) Reset the attachments/sec window
  lastBallCountChangeTime = millis();
  attachmentTimes = [];
  totalAttachmentsSinceLastChange = 0;

    // NEW: reset occupancy accumulation window to match the timer
  if (typeof resetOccupancyWindow === 'function') resetOccupancyWindow();


  // 3) Spawn fresh ligands
  const spots = placeBalls()[0];
  const startIndex = particles.length; // continue ids after inhibitors
  for (let i = 0; i < targetLigands; i++) {
    const position = spots[(startIndex + i) % spots.length];
    const velocity = p5.Vector.random2D();
    velocity.setMag(3); // base; Ball constructor multiplies by 5 => 7.5 px/frame
    particles.push(new Ball(
      position,
      velocity,
      radius,
      startIndex + i,
      particles,
      ballColor,
      follow
    ));
  }
}


function reset(count = 1) {
  let possiblePlaces = placeBalls()[0];
  particles = [];
  for (let i = 0; i < count; i++) {
    let position = possiblePlaces[i % possiblePlaces.length];
    let velocity = p5.Vector.random2D();
    velocity.setMag(3);
    particles[i] = new Ball(
      position,
      velocity,
      radius,
      i,
      particles,
      ballColor,
      follow
    );
  }
}

function placeBalls() {
  let positions = [];
  let place = createVector(640 + radius, 300 + radius);
  let gridDim = createVector(0, 0);

  while (place.x <= 1280 - radius && place.y <= 605 - radius) {
    positions.push(place.copy());
    place.x += diameter + separator;
    gridDim.x++;
    if (place.x > 1280 - radius) {
      place.x = 640 + radius;
      place.y += diameter + separator;
      gridDim.y++;
    }
  }
  gridDim.x = gridDim.x / gridDim.y;
  return [positions, gridDim];
}

function addInhibitorBalls(count) {
  let possiblePlaces = placeBalls()[0];
  let startIndex = particles.length;
  for (let i = 0; i < count; i++) {
    let position = possiblePlaces[(startIndex + i) % possiblePlaces.length];
    let velocity = p5.Vector.random2D();
    velocity.setMag(3);
    particles.push(new Ball(
      position,
      velocity,
      radius,
      startIndex + i,
      particles,
      "red",
      false,
      true
    ));
  }
}

function updateInhibitorCount(targetCount) {
  let currentInhibitorCount = particles.filter(p => p.isInhibitor).length;
  if (targetCount > currentInhibitorCount) {
    addInhibitorBalls(targetCount - currentInhibitorCount);
  } else if (targetCount < currentInhibitorCount) {
    removeInhibitorBalls(currentInhibitorCount - targetCount);
  }
}

function removeInhibitorBalls(count) {
  let removedCount = 0;
  for (let i = particles.length - 1; i >= 0 && removedCount < count; i--) {
    if (particles[i].isInhibitor) {
      particles.splice(i, 1);
      removedCount++;
    }
  }
}

// Mapping helper for inhibitor slider
function mapInhibitorSliderValue(value) {
  return map(value, 3, 100, -12, 12);
}

// Ligand helpers (preserved behavior)
function detachAllLigands() {
  for (let i = 0; i < particles.length; i++) {
    if (particles[i].isAttached) {
      particles[i].detachFromRectangle();
    }
  }
  attachmentCount = 0;
}

function resetLigandProperties() {
  // Note: attachedLigands is defined in ball.js
  for (let i = 0; i < (attachedLigands?.length || 0); i++) {
    attachedLigands[i] = null;
  }
  for (let particle of particles) {
    particle.attachedRectIndex = -1;
    particle.gracePeriod = 0;
  }
}

// ─────────────────────────────────────────────
// Graph & Compare Helpers
// ─────────────────────────────────────────────
function displayFunction(fn, type) {
  stroke(type === 'ligand' ? color(101, 100, 250) : color(250, 0, 0));
  strokeWeight(3);
  let output = [];
  for (let x = -12; x <= 12; x += 0.01) {
    let y = fn(x);
    if (y <= h / (1 * unit) && y >= -h / (1.9 * unit)) {
      output.push([x, y]);
    }
  }
  for (let i = 1; i < output.length - 1; i++) {
    let x1 = w / 2 + unit * output[i][0];
    let y1 = 420 - unit * output[i][1];
    let x2 = w / 2 + unit * output[i + 1][0];
    let y2 = 420 - unit * output[i + 1][1];
    line(x1, y1, x2, y2);
  }
}

function handleFitSigmoidClick() {
  const pts = (pointList || []).filter(p => p && isFinite(p.x) && isFinite(p.y) && p.x > 0);
  if (pts.length < 3) {
    alert('Plot at least 5 points before fitting.');
    return;
  }

  const fit = fitHill(pts);
  if (!fit) {
    alert('Could not fit a curve to the current points.');
    return;
  }

  fittedCurve = { points: sampleFittedCurve(fit, 1), params: fit };
  graphPlotted = true
}

// Hill model pieces
function hillG(x, EC50, n) {
  if (x <= 0) x = 1e-6;
  return 1 / (1 + Math.pow(EC50 / x, n));
}

// For fixed (EC50, n), optimal Emax in least squares sense
function bestEmax(points, EC50, n) {
  let num = 0, den = 0;
  for (const p of points) {
    const g = hillG(p.x, EC50, n);
    num += p.y * g;
    den += g * g;
  }
  return den > 0 ? num / den : 0;
}

function fitHill(points) {
  const xMin = 1, xMax = TARGET_CONC_MAX;
  const nMin = 0.5, nMax = 3.0;
  const nSteps = 26;   // ~0.1 steps
  const ecSteps = 30;  // log-spaced EC50

  // NEW: check if user pinned Emax at x ≈ max
  const pinnedEmax = getPinnedEmax(points, 1); // tol=1 unit around 400
  let best = null;

  for (let i = 0; i < ecSteps; i++) {
    const t = i / (ecSteps - 1);
    const EC50 = Math.pow(10, Math.log10(xMin) + t * (Math.log10(xMax) - Math.log10(xMin)));

    for (let j = 0; j < nSteps; j++) {
      const n = nMin + (nMax - nMin) * (j / (nSteps - 1));

      // If pinned, use it; else compute LS-optimal Emax for this (EC50,n)
      const Emax = (pinnedEmax !== null) ? pinnedEmax : bestEmax(points, EC50, n);

      // Evaluate MSE
      let sse = 0;
      for (const p of points) {
        const yhat = Emax * hillG(p.x, EC50, n);
        const e = p.y - yhat;
        sse += e * e;
      }
      const mse = sse / points.length;

      if (!best || mse < best.mse) best = { Emax, EC50, n, mse };
    }
  }
  return best; // {Emax, EC50, n, mse}
}


// Build a smooth drawable polyline in SCREEN coords (left plot)
function sampleFittedCurve(params, stepX = 1) {
  const out = [];
  for (let x = sliderMin; x <= sliderMax; x += stepX) {
    const y = params.Emax * hillG(x, params.EC50, params.n);
    const sx = map(x, sliderMin, sliderMax, 80, w - 80);
    const sy = map(y, yMin, yMax, h - 80, 80);
    out.push({ x: sx, y: sy });
  }
  return out;
}


function redrawGraph() {
  background(173, 216, 230);
  drawGridAndAxes();
  // Redraw locked-in points
  fill(255, 0, 0);
  for (let point of pointList) {
    let xCoord = map(point.x, sliderMin, sliderMax, 80, w - 80);
    let yCoord = map(point.y, yMin, yMax, h - 80, 80);
    ellipse(xCoord, yCoord, 10, 10);
  }
  drawBalls();
}

function drawGridAndAxes() {
  stroke(255, 0, 0);
  strokeWeight(3);
  line(640, 0, 640, 720);
  fill(255);
  noStroke();
  rect(0, 0, 640, 720);

  stroke(180);
  strokeWeight(1);
  for (let i = 4; i <= h / (unit + 3); i++) {
    line(80, 20 * i, w - 80, 20 * i);
  }
  for (let i = 4; i <= w / (unit + 2.5); i++) {
    line(20 * i, 80, 20 * i, h - 80);
  }

  strokeWeight(2);
  stroke(0);
  line(80, h - 80, w - 80, h - 80);
  line(80, 80, 80, h - 80);
}

function drawBalls() {
  for (let a of particles) {
    a.bounceOthers();
    a.update();
    a.display();
  }
}

// snapshot of current points + smooth reference curve
function snapshotCurrent(label, colorVal) {
  const pointsCopy = pointList.map(p => ({ x: p.x, y: p.y, alpha: p.alpha ?? 255 }));
  let curve = null;
  if (fittedCurve && fittedCurve.params) {
    curve = {
      params: { ...fittedCurve.params },
      polyline: (fittedCurve.points || []).map(pt => ({ x: pt.x, y: pt.y }))
    };
  }
  return { label, color: colorVal, points: pointsCopy, curve };
}



function computeCurve(fn) {
  const out = [];
  for (let x = sliderMin; x <= sliderMax; x += 1) {
    const scrX = map(x, sliderMin, sliderMax, 80, w - 80);
    const fx = map(x, 0, TARGET_CONC_MAX, -12, 12);
    const yVal = fn(fx);
    if (yVal === undefined) continue;
    const scrY = map(yVal, yMin, yMax, h - 80, 80);
    out.push({ x: scrX, y: scrY });
  }
  return out;
}

function drawSnapshot(snapshot) {
  if (!snapshot) return;

  // 1) Points
  if (snapshot.points?.length) {
    noStroke();
    fill(snapshot.color);
    for (const pt of snapshot.points) {
      const xCoord = map(pt.x, sliderMin, sliderMax, 80, w - 80);
      const yCoord = map(pt.y, yMin, yMax, h - 80, 80);
      ellipse(xCoord, yCoord, 8, 8);
    }
  }

  // 2) Curve — resample from params for consistent resolution
  if (snapshot.curve?.params) {
    const poly = sampleFittedCurve(snapshot.curve.params, 1); // screen coords
    noFill();
    stroke(snapshot.color);
    strokeWeight(3);
    beginShape();
    for (const p of poly) vertex(p.x, p.y);
    endShape();
  } else if (snapshot.curve?.polyline?.length) {
    // fallback: draw stored polyline as-is
    noFill();
    stroke(snapshot.color);
    strokeWeight(3);
    beginShape();
    for (const p of snapshot.curve.polyline) vertex(p.x, p.y);
    endShape();
  }
}



// ─────────────────────────────────────────────
// Mathematical Functions for the Graph
// ─────────────────────────────────────────────
function f(x) {
  if (x > -12) {
    return 10 * (10 ** (0.25 * (x + 4)) ** 1) / (1 + (10 ** (0.25 * (x + 4)) ** 1));
  }
}

function u(x) {
  if (x > (-60 - horizontalShift)) {
    return f(x - horizontalShift - 4);
  }
  return 0;
}

// ─────────────────────────────────────────────
// p5.js Draw Loop
// ─────────────────────────────────────────────
function draw() {
  // INTRO
  if (scene === 'intro') {
    drawIntroScene();
    return;
  }

  // ACH GRAPH — vessel + 4 receptors
  if (scene === 'achGraph') {
    showUIElements();

    // ── one-time per-frame timing + response (keep timer usage the same)
    const currentTime = millis();
    const elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000; // unchanged window
    const respFraction = getNormalizedOccupancyFraction(); // 0..1 vs 8 receptors
    const respScaled   = constrain(respFraction * yMax, yMin, yMax);

    // Left grid / axes
    background(173, 216, 230);
    stroke(255, 0, 0); strokeWeight(3);
    line(640, 0, 640, 720);

    fill(255); noStroke();
    rect(0, 0, 640, 720);

    stroke(180); strokeWeight(1);
    for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
    for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

    strokeWeight(2); stroke(0);
    line(80, h - 80, w - 80, h - 80);
    line(80, 80, 80, h - 80);

    // HUD text (unchanged positions; new metric)
    noStroke(); textSize(20); fill(180); textAlign(CENTER);
    text(`Ligand Concentration: ${Math.round(concentration)}`, 320, 650);
    text(`Response (occupancy vs 8): ${(respFraction * 100).toFixed(1)}%`, 320, 680);

    // Right visuals
    image(vessel, 660, 10, 600, 400);

    noFill(); stroke(255, 255, 102);
    rect(945, 80, 30, 30);
    line(945, 110, 640, 300);
    line(975, 110, 1280, 300);
    line(640, 300, 1280, 300);

    image(sarcolemma, 640, 300, 640, 640);

    // 4 GPCRs + binding rectangles
    image(gpcr, 630, 560, 200, 200);
    image(gpcr, 800, 555, 200, 200);
    image(gpcr, 970, 530, 200, 200);
    image(gpcr, 1100, 530, 200, 200);

    fill(255, 0, 0);
    rect(925, 580, 10, 20);
    rect(755, 585, 10, 20);
    rect(1095, 555, 10, 20);
    rect(1225, 555, 10, 20);

    // Particles
    stroke(0); noFill();
    for (let a of particles) { a.bounceOthers(); a.update(); a.display(); }

    // Ghost point (deduped calc)
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);
    ghostPoint.y = respScaled;
    const ghostXCoord = map(ghostPoint.x, sliderMin, sliderMax, 80, w - 80);
    const ghostYCoord = map(ghostPoint.y, yMin, yMax, h - 80, 80);
    fill(255, 0, 0, ghostPoint.alpha);
    noStroke();
    ellipse(ghostXCoord, ghostYCoord, 10, 10);

    // Saved points
    if (pointList.length > 0) {
      for (let point of pointList) {
        const xCoord = map(point.x, sliderMin, sliderMax, 80, w - 80);
        const yCoord = map(point.y, yMin, yMax, h - 80, 80);
        fill(255, 0, 0, point.alpha);
        noStroke();
        ellipse(xCoord, yCoord, 10, 10);
      }
    }

    // Fitted curve (if any)
    if (fittedCurve && fittedCurve.points?.length) {
      stroke(0, 0, 255);
      strokeWeight(3);
      noFill();
      beginShape();
      for (const p of fittedCurve.points) vertex(p.x, p.y);
      endShape();
    }

    frameCounter++;

    // Continue button visibility
    if (graphPlotted === true) continueButton.show();
    else continueButton.hide();

    return;
  }

  // HEART GRAPH — heart + 6 receptors in a row
  if (scene === 'heartGraph') {
    showUIElements();

    // ── one-time per-frame timing + response (keep timer usage the same)
    const currentTime = millis();
    const elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000; // unchanged window
    const respFraction = getNormalizedOccupancyFraction(); // 0..1 vs 8 receptors
    const respScaled   = constrain(respFraction * yMax, yMin, yMax);

    // Left grid / axes
    background(173, 216, 230);
    stroke(255, 0, 0); strokeWeight(3);
    line(640, 0, 640, 720);

    fill(255); noStroke();
    rect(0, 0, 640, 720);

    stroke(180); strokeWeight(1);
    for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
    for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

    strokeWeight(2); stroke(0);
    line(80, h - 80, w - 80, h - 80);
    line(80, 80, 80, h - 80);

    // HUD text (unchanged positions; new metric)
    noStroke(); textSize(20); fill(180); textAlign(CENTER);
    text(`Ligand Concentration: ${Math.round(concentration)}`, 320, 650);
    text(`Response (occupancy vs 8): ${(respFraction * 100).toFixed(1)}%`, 320, 680);

    // Right visuals (heart)
    image(heart, 660, 10, 600, 400);

    noFill(); stroke(255, 255, 102);
    rect(945, 80, 30, 30);
    line(945, 110, 640, 300);
    line(975, 110, 1280, 300);
    line(640, 300, 1280, 300);

    image(sarcolemma, 640, 300, 640, 640);

    // 6 GPCRs (smaller) + binding rectangles
    {
      const { gpcrPos, rects, size } = getHeartLayout();
      for (const p of gpcrPos) image(gpcr, p.x, p.y, size, size);
      fill(255, 0, 0); noStroke();
      for (const r of rects) rect(r.x, r.y, r.w, r.h);
    }

    // Particles
    stroke(0); noFill();
    for (let a of particles) { a.bounceOthers(); a.update(); a.display(); }

    // Ghost point (deduped calc)
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);
    ghostPoint.y = respScaled;
    const ghostXCoord = map(ghostPoint.x, sliderMin, sliderMax, 80, w - 80);
    const ghostYCoord = map(ghostPoint.y, yMin, yMax, h - 80, 80);
    fill(255, 0, 0, ghostPoint.alpha);
    noStroke();
    ellipse(ghostXCoord, ghostYCoord, 10, 10);

    // Saved points
    if (pointList.length > 0) {
      for (let point of pointList) {
        const xCoord = map(point.x, sliderMin, sliderMax, 80, w - 80);
        const yCoord = map(point.y, yMin, yMax, h - 80, 80);
        fill(255, 0, 0, point.alpha);
        noStroke();
        ellipse(xCoord, yCoord, 10, 10);
      }
    }

    // Fitted curve (if any)
    if (fittedCurve && fittedCurve.points?.length) {
      stroke(0, 0, 255);
      strokeWeight(3);
      noFill();
      beginShape();
      for (const p of fittedCurve.points) vertex(p.x, p.y);
      endShape();
    }

    frameCounter++;

    // Continue button visibility
    if (graphPlotted === true) continueButton.show();
    else continueButton.hide();

    return;
  }

  // COMPARE GRAPHS — overlay ACH vs HEART on left plot
  if (scene === 'compareGraphs') {
    background(173, 216, 230);
    stroke(255, 0, 0); strokeWeight(3);
    line(640, 0, 640, 720);

    // Left plot area
    fill(255); noStroke();
    rect(0, 0, 640, 720);
    stroke(180); strokeWeight(1);
    for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
    for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);
    strokeWeight(2); stroke(0);
    line(80, h - 80, w - 80, h - 80);
    line(80, 80, 80, h - 80);

    // Right side whited out
    noStroke(); fill(255);
    rect(640, 0, 640, 720);

    // Title + legend
    noStroke(); fill(100); textAlign(CENTER); textSize(22);
    text('ACH vs HEART — Overlaid Graphs', 320, 40);

    textAlign(LEFT); const lx = 100, ly = 70;
    fill(220, 20, 60); rect(lx, ly - 12, 18, 8);  // ACH swatch
    fill(30, 144, 255); rect(lx, ly + 8, 18, 8);  // HEART swatch
    fill(60); textSize(16);
    text('ACH', lx + 26, ly - 4);
    text('HEART', lx + 26, ly + 16);

    // Draw saved snapshots
    drawSnapshot(achSnapshot);
    drawSnapshot(heartSnapshot);

    return;
  }

  // (other scenes if you add them later)
}


// ─────────────────────────────────────────────
// Additional Scene Draw Functions
// ─────────────────────────────────────────────
function drawChooseLigandScene() {
  background(173, 216, 230);
  stroke(0);
  fill(222, 184, 135);
  rect(0, 550, 1200, 720);

  beginShape();
  vertex(1280, 450);
  vertex(1280, 720);
  vertex(1200, 720);
  vertex(1200, 550);
  endShape(CLOSE);

  fill(50, 50, 50);
  beginShape();
  vertex(0, 550);
  vertex(1200, 550);
  vertex(1280, 450);
  vertex(80, 450);
  endShape(CLOSE);

  fill(0);
  textSize(50);
  noStroke();
  text('Pick Ligand:', 500, 100);

  strokeWeight(3);
  fill(220, 220, 220);
  rect(30, 150, 300, 100);
  rect(490, 150, 300, 100);
  rect(950, 150, 300, 100);

  fill(0);
  textSize(45);
  noStroke();
  text('Acetylcholine', 48, 215);
  text('Epinephrine', 520, 215);
  text('FILLER12345', -200, -200);
  strokeWeight(3);
}

function drawAchStartScene() {
  fill(72, 61, 139);
  if (AchBackgroundX > -1280) {
    AchBackgroundX -= 12;
    rect(1280, 0, AchBackgroundX, 720);
  } else if (fade < 255) {
    noStroke();
    fill(220, 220, 220, fade);
    rect(390, 260, 500, 200);
    fill(0, 0, 0, fade);
    stroke(0);
    textSize(75);
    text('Start', 560, 385);
    fade += 1;
  }
}

// ─────────────────────────────────────────────
// p5.js Mouse Click Handler
// ─────────────────────────────────────────────
function mouseClicked() {
  if (scene === 'choose ligand') {
    if (mouseX > 30 && mouseX < 330 && mouseY > 150 && mouseY < 250) {
      scene = 'achStart';
    }
  } else if (scene === 'achStart') {
    if (mouseX > 390 && mouseX < (390 + 500) && mouseY > 260 && mouseY < (260 + 200)) {
      scene = 'achGraph';
      initializeScene('achGraph');
    }
  }
}

// If there's a point at x ≈ TARGET_CONC_MAX, return its y (or the mean if multiple)
function getPinnedEmax(points, tol = 1) {
  const nearMax = points.filter(p => Math.abs(p.x - TARGET_CONC_MAX) <= tol);
  if (nearMax.length === 0) return null;
  const sum = nearMax.reduce((s, p) => s + p.y, 0);
  return sum / nearMax.length;
}
