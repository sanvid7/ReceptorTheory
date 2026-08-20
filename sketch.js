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


let scene = 'title';
let currentSceneIndex = 0;
const scenes = ['title', 'clarkIntro', 'intro', 'achGraph', 'tissueTransition', 'inhibitor', 'heartGraph', 'dataCollected', 'compareGraphs', 'limitationTitle'];

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

let beginButton;
let slideButton;
let gotItButton;
let devButton;
let devAnalysisButton;
let devPartialButton;
let devSpareButton;
let devAnalysisBtn2;
let clarkIntroPage = 1;
let problemPage = 1;

// Spotlight system (freeze + darken + erase hole)
let activeSpotlight = null;   // { num, x, y, radius, heading, caption } or null
let pendingSpotlight = null;  // queued to show after active is dismissed
let spotlightOverlayG = null; // off-screen graphics buffer for dark overlay + hole
let clarkPostulateShown = new Set();
let spotlightShownAtMillis = -1; // guard: prevent same-click dismiss
let mrtDrugType = 'none'; // 'none' | 'fullAgonist' | 'antagonist' | 'partialAgonist'
let mrtPrinciplesPage = 1;
let showGraphOverlay = true;

// ── Chapter 2: Partial Agonism lab ───────────────────────────────────────────
let showPartialOverlay  = true;
let partialPointList    = [];
let partialPointCounter = 0;
let partialFittedCurve  = null;
let partialGraphPlotted = false;
let partialGhostY       = 0;           // in binding-rate units (same as yMin/yMax)
const PARTIAL_EMAX_RATE = 7.0;        // ceiling in binding-rate units (~50% of yMax=14)
const PARTIAL_EC50      = 80;         // concentration at half-max
const PARTIAL_N         = 2.5;        // Hill steepness

// ── Chapter 3: MRT lab ───────────────────────────────────────────────────────
let showMrtBasalOverlay = true;
let showMrtDrugOverlay  = false;

// ── Chapter 2: Spare Receptors lab ───────────────────────────────────────────
let showSpareOverlay   = true;
let sparePointList     = [];
let sparePointCounter  = 0;
let spareFittedCurve   = null;
let spareGraphPlotted  = false;
let spareGhostY        = 0;
const SPARE_EC50_FACTOR  = 0.2;   // Ligand C EC50 = 20% of Ch1 EC50 (shifted left)
const SPARE_SLIDER_MAX   = 120;   // restrict slider so students can't saturate all 8 receptors

// Snapshots for comparison scene
let achSnapshot = null;   // { label, color, points, curve }
let heartSnapshot = null; // { label, color, points, curve }

// UI Elements and Buttons
let slider, inhibitorSlider;
let pointButton, graphButton, continueButton, inhibitorButton;
let clarkButton, mrtButton, compareButton;
let compareP5Button;
let postulateButton;
let postulatePage = 0; // 0 = none, 1 = first two, 2 = next two, 3 = last one
const postulates = [
  "1. Drug action depends on receptor binding.",
  "2. Binding is reversible and saturable.",
  "3. Maximum response occurs at full receptor occupancy.",
  "4. Drug effect is proportional to receptor occupancy.",
  "5. Drugs act through a single type of receptor."
];

// Canvas/graphics variables
let w = 640;
let h = 500;
let unit = 20;
let muscleStrip, membrane, receptor, gpcr, Diaphragm, Smallintestine;
let AchBackgroundX = 0;
let fade = 0;

// MRT module state
const MRT_N = 10;
let mrtBasalStates = [];
let mrtFlipTimer = 0;
let mrtActivityHistory = []; // rolling EMA samples (0–1)
let mrtEma = 0.5;            // exponential moving average of R* fraction
let mrtPhase = 'basal';      // 'basal' | 'selecting' | 'drug'
let mrtDrugButtons = [];     // p5 button refs for the 5 drug choices
let mrtInstructionEl = null; // DOM element for drug-selection instruction text
let mrtDrugColor = [255, 255, 255]; // rgb of current drug balls
let mrtWashButton = null;
let mrtChooseButton = null;
let mrtLigandSlider = null;

// MRT graph state
const MRT_SLIDER_MIN = 1;
const MRT_SLIDER_MAX = 80;
const MRT_Y_MIN = 0;    // 0% R* activity
const MRT_Y_MAX = 1;    // 100% R* activity
let mrtPlotData = {};          // { drugType: [{x: sliderVal, y: ema}, ...] }
let mrtFittedCurves = {};      // { drugType: {Emax, EC50, n, mse} }
let mrtCompletedDrugs = new Set();
let mrtPlotButton = null;
let mrtFitButton = null;
let mrtDoneButton = null;

// Ball simulation variables
let particles = [];
let radius = 5;
let diameter = radius * 2;
let separator = 1;
let ballColor = "white";
let follow = false;

// Ghost point used in graph scenes
let ghostPoint = { x: 0, y: 0, alpha: 100 };

// ─────────────────────────────────────────────
// p5.js Preload and Setup
// ─────────────────────────────────────────────
function preload() {
  muscleStrip = loadImage('musclestrip.png');
  membrane = loadImage('membrane.png');
  receptor = loadImage('receptor.png');
  gpcr = loadImage('gpcr.png');
  Diaphragm = loadImage('Diaphragm.png');
  Smallintestine = loadImage('Smallintestine.png');
}

function setup() {
  let cnv = createCanvas(1280, 720);
  cnv.parent('wrapper');
  textFont('Nunito');
  startTime = millis();
  createNavigationButtons();
  createSliders();
  createActionButtons();
  hideUIElements();
  reset();
  fitToWindow();

  // Respect the chosen scene at startup
  initializeScene(scene);

}

function fitToWindow() {
  const sf = Math.min(windowWidth / 1280, windowHeight / 720);
  const left = (windowWidth - 1280 * sf) / 2;
  const top  = (windowHeight - 720 * sf) / 2;
  select('#wrapper').style('transform', `translate(${left}px, ${top}px) scale(${sf})`);
}

function windowResized() {
  fitToWindow();
}

let evidenceParticles = [];      // mini ligand balls for flashback
let evidenceReceptor;            // position of the mini receptor
let evidenceConcentration = 5;   // number of ligands
let showEvidence = false;        // whether to display flashback

// ─────────────────────────────────────────────
// UI Creation Functions
// ─────────────────────────────────────────────
function createNavigationButtons() {

}


function createSliders() {
  // Main ligand slider (logarithmic-ish display)
  let sliderContainer = createDiv('');
  sliderContainer.parent('wrapper');
  slider = createCustomSlider(sliderContainer, 1, TARGET_CONC_MAX, 1, 1);

  // Inhibitor slider (linear)
  let inhibitorSliderContainer = createDiv('');
  inhibitorSliderContainer.parent('wrapper');
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
    clarkIntroPage = 1;
    initializeScene('mrtTitle');
  });

  compareButton = createButton('Chapter II →');
  compareButton.id('compareButton');
  compareButton.class('button-base button-blue');
  compareButton.position(width/2 - 90, 400);
  compareButton.mousePressed(() => {
    initializeScene('limitationTitle');
  });

  pointButton = createButton('Record It!');
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
  fitButton = createButton('Draw the Curve!');
  fitButton.id('fitSigmoidButton');
  fitButton.mousePressed(handleFitSigmoidClick);
  fitButton.class('button-base button-blue');
  
  postulateButton = createButton("Clark's Postulates");
  postulateButton.id("postulateButton");
  postulateButton.mousePressed(handlePostulateButtonClick);
  postulateButton.class("button-base button-green");
  postulateButton.hide();

  beginButton = createButton("Let's Go! →");
  beginButton.id('beginButton');
  beginButton.class('button-base button-green');
  beginButton.mousePressed(() => {
    clarkIntroPage = 1;
    initializeScene('clarkIntro');
  });
  beginButton.hide();

  slideButton = createButton('Continue →');
  slideButton.id('slideNextButton');
  slideButton.class('button-base button-green');
  slideButton.mousePressed(handleSlideButtonClick);
  slideButton.hide();

  gotItButton = createButton('Continue to lab tour →');
  gotItButton.id('gotItButton');
  gotItButton.class('button-base button-green');
  gotItButton.mousePressed(() => {
    if (scene === 'partialGraph') {
      showPartialOverlay = false;
      gotItButton.hide();
      showSpotlight(20, 330, 303, { w: 580, h: 460, r: 10 },
        "Stop 1 of 4  —  The Original Graph",
        [{ label: "Your Chapter 1 Baseline",
           text: "This dashed curve is the dose-response relationship you built in Chapter 1 with acetylcholine. Clark's assumptions — proportional occupancy, maximal response when all receptors are filled — predict that any drug on this same tissue should land here. This is our benchmark." }],
        { num: 94, x: 960, y: 500, shape: 'rect', rw: 590, rh: 380, rr: 14,
          heading: "Stop 2 of 4  —  The Tissue Bath",
          postulates: [{ label: "Same Molecular View, Different Drug",
                         text: "You're looking at the same diaphragm tissue with the same 4 GPCRs. Ligand B molecules are bouncing around and binding to the receptors — just like ACh did. Watch the binding rate on the left as you adjust concentration." }],
          pending: {
            num: 95, x: 960, y: 640, shape: 'rect', rw: 590, rh: 100, rr: 8,
            heading: "Stop 3 of 4  —  The Readout",
            postulates: [{ label: "Tissue Response",
                           text: "The display shows how much the tissue is contracting relative to its maximum possible contraction. Watch it update live as you adjust the concentration." }],
            pending: {
              num: 96, x: 320, y: 640, shape: 'rect', rw: 330, rh: 48, rr: 8,
              heading: "Stop 4 of 4  —  Your Controls",
              postulates: [{ label: "Concentration Slider",
                             text: "Same as before — drag to set the drug concentration. Set a level, wait for the readout to stabilise, then hit Record It! to log the point." }]
            }
          }
        }
      );
    } else if (scene === 'spareGraph') {
      showSpareOverlay = false;
      gotItButton.hide();
      showSpotlight(25, 330, 303, { w: 580, h: 460, r: 10 },
        "Stop 1 of 4  —  The Original Graph",
        [{ label: "Your Chapter 1 Baseline",
           text: "The dashed curve is your Chapter 1 result with ACh on these same 4 receptors. Clark predicts any drug that fills all 4 should land here. See whether Ligand C needs to reach full occupancy to hit the same maximum." }],
        { num: 97, x: 960, y: 500, shape: 'rect', rw: 590, rh: 380, rr: 14,
          heading: "Stop 2 of 4  —  Eight Receptors",
          postulates: [{ label: "Same 4 Receptors — Different Drug",
                         text: "Same tissue, same 4 GPCRs as before. Clark's model says you need to fill all 4 to reach maximum response. Notice what concentration Ligand C needs to get there compared to ACh." }],
          pending: {
            num: 98, x: 960, y: 640, shape: 'rect', rw: 590, rh: 100, rr: 8,
            heading: "Stop 3 of 4  —  The Readout",
            postulates: [{ label: "Binding Rate",
                           text: "The left panel tracks binding rate as you sweep concentration. Once you have enough points, fit the curve and compare it to the Original Graph." }],
            pending: {
              num: 99, x: 320, y: 640, shape: 'rect', rw: 330, rh: 48, rr: 8,
              heading: "Stop 4 of 4  —  Your Controls",
              postulates: [{ label: "Concentration Slider",
                             text: "Drag to set Ligand C concentration. Set a level, wait for the readout to stabilise, then hit Record It! to log the point." }]
            }
          }
        }
      );
    } else if (scene === 'mrtBasal' && showMrtDrugOverlay) {
      showMrtDrugOverlay = false;
      gotItButton.hide();
      showSpotlight(105, 960, 400, { w: 590, h: 420, r: 12 },
        "Pick a Drug Type",
        [{ label: "Five Drugs. Five Stories.",
           text: "Each drug type interacts with the R⇄R* equilibrium differently. Full agonists push receptors toward R*. Full inverse agonists push them toward R. Antagonists just block — no shift. Test all five, build a curve for each, and see the difference." }]
      );
    } else if (scene === 'mrtBasal') {
      showMrtBasalOverlay = false;
      gotItButton.hide();
      // After tour ends, reveal the Continue → button
      showSpotlight(101, 320, 90, { w: 560, h: 110, r: 8 },
        "Stop 1 of 4  —  The Two-State Model",
        [{ label: "Receptors Are Never Fully Off",
           text: "Even without any drug, receptors spontaneously toggle between an inactive state (R) and an active state (R*). This happens constantly. Clark's model assumed receptors were inert until a drug bound — that turns out to be wrong." }],
        { num: 102, x: 340, y: 325, shape: 'rect', rw: 500, rh: 280, rr: 8,
          heading: "Stop 2 of 4  —  Basal Activity",
          postulates: [{ label: "The Time-Series Graph",
                         text: "This graph shows the fraction of receptors in R* over time. Without any drug, it hovers around 50% — that's the natural equilibrium. Your job is to see how different drugs shift this balance." }],
          pending: {
            num: 103, x: 960, y: 500, shape: 'rect', rw: 590, rh: 420, rr: 14,
            heading: "Stop 3 of 4  —  The Receptors",
            postulates: [{ label: "Green = R*  (Active)    Red = R  (Inactive)",
                           text: "Each receptor is color-coded by its current state. Watch them flicker. When a drug ball binds, it locks the receptor and influences which state it stays in — that's the key difference between drug types." }],
            pending: {
              num: 104, x: 320, y: 655, shape: 'rect', rw: 360, rh: 36, rr: 8,
              heading: "Stop 4 of 4  —  Start the Experiment",
              postulates: [{ label: "Click Continue When You're Ready",
                             text: "You'll test 5 different drug types on these receptors. Each one shifts the R⇄R* equilibrium differently. Watch the graph — the shift tells you everything about what the drug is doing." }]
            }
          }
        }
      );
      // Show slideButton once tour is fully dismissed — handled in dismissSpotlight chain end
      // For now we reveal it via the overlay-clear path in drawMrtBasalScene
    } else {
      // Chapter 1 lab tour — MSB magnification
      showGraphOverlay = false;
      gotItButton.hide();
      showSpotlight(91, 960, 420, { w: 570, h: 220, r: 14 },
        "Stop 1 of 3  —  Drug Molecules",
        [{ label: "Acetylcholine (ACh) Molecules",
           text: "Each ball is a single ACh molecule diffusing through the muscle tissue at molecular scale. The slider controls how many are present — that's your concentration variable." }],
        { num: 92, x: 990, y: 578, shape: 'rect', rw: 540, rh: 85, rr: 8,
          heading: "Stop 2 of 3  —  Receptors",
          postulates: [{ label: "G-Protein Coupled Receptors (GPCRs)",
                         text: "The red slots in the cell membrane are your receptors. When an ACh molecule collides with one and binds, the receptor becomes occupied and generates a physiological signal downstream." }],
          pending: {
            num: 93, x: 320, y: 612, shape: 'rect', rw: 330, rh: 48, rr: 8,
            heading: "Stop 3 of 3  —  Your Controls",
            postulates: [{ label: "Ligand Concentration Slider",
                           text: "Drag this to add or remove ACh molecules from the tissue. Set a concentration, let the binding rate stabilise, then hit Record It! — each click plots one data point." }]
          }
        }
      );
    }
  });
  gotItButton.hide();

  devButton = createButton('⚡ MRT Basal');
  devButton.position(10, 676);
  devButton.style('position', 'absolute');
  devButton.style('font-size', '11px');
  devButton.style('padding', '4px 10px');
  devButton.style('opacity', '0.55');
  devButton.style('cursor', 'pointer');
  devButton.mousePressed(() => initializeScene('mrtBasal'));

  devAnalysisButton = createButton('⚡ MRT Analysis');
  devAnalysisButton.position(155, 676);
  devAnalysisButton.style('position', 'absolute');
  devAnalysisButton.style('font-size', '11px');
  devAnalysisButton.style('padding', '4px 10px');
  devAnalysisButton.style('opacity', '0.55');
  devAnalysisButton.style('cursor', 'pointer');
  devAnalysisButton.mousePressed(() => {
    seedMrtDevData();
    initializeScene('mrtAnalysis');
  });

  devPartialButton = createButton('⚡ Partial Lab');
  devPartialButton.position(300, 676);
  devPartialButton.style('position', 'absolute');
  devPartialButton.style('font-size', '11px');
  devPartialButton.style('padding', '4px 10px');
  devPartialButton.style('opacity', '0.55');
  devPartialButton.style('cursor', 'pointer');
  devPartialButton.mousePressed(() => initializeScene('partialGraph'));

  devSpareButton = createButton('⚡ Spare Lab');
  devSpareButton.position(415, 676);
  devSpareButton.style('position', 'absolute');
  devSpareButton.style('font-size', '11px');
  devSpareButton.style('padding', '4px 10px');
  devSpareButton.style('opacity', '0.55');
  devSpareButton.style('cursor', 'pointer');
  devSpareButton.mousePressed(() => initializeScene('spareGraph'));

  devAnalysisBtn2 = createButton('⚡ Ch2 Analysis');
  devAnalysisBtn2.position(530, 676);
  devAnalysisBtn2.style('position', 'absolute');
  devAnalysisBtn2.style('font-size', '11px');
  devAnalysisBtn2.style('padding', '4px 10px');
  devAnalysisBtn2.style('opacity', '0.55');
  devAnalysisBtn2.style('cursor', 'pointer');
  devAnalysisBtn2.mousePressed(() => initializeScene('ch2Analysis'));

  // "Click me!" P5 badge for compareGraphs scene
  compareP5Button = createButton('Click me!');
  compareP5Button.position(490, 300);
  compareP5Button.size(120, 32);
  compareP5Button.style('background-color', '#FFD500');
  compareP5Button.style('color', '#0f1627');
  compareP5Button.style('border', '1.5px solid rgba(15,22,55,0.7)');
  compareP5Button.style('border-radius', '7px');
  compareP5Button.style('font-family', "'Nunito', sans-serif");
  compareP5Button.style('font-weight', '800');
  compareP5Button.style('font-size', '12px');
  compareP5Button.style('cursor', 'pointer');
  compareP5Button.style('animation', 'badgePulse 1.8s ease-in-out infinite');
  compareP5Button.mousePressed(() => {
    if (activeSpotlight) return;
    // Spotlight the sigmoid curves (graph panel: gx=50, gy=88, gw=560, gh=430)
    activeSpotlight = {
      num: 5, x: 330, y: 303,
      heading: "Clark's Assumption 5",
      postulates: [{ label: "Drug molecules are in excess of receptor availability",
        text: "There are far more drug molecules in solution than there are receptors to bind them. This means the concentration you set drives the response — the receptors are never 'running out' of ligand to bind." }],
      shape: 'rect', rw: 580, rh: 460, rr: 10, radius: null
    };
    spotlightShownAtMillis = millis();
    _buildSpotlightOverlay(activeSpotlight);
  });
  compareP5Button.hide();

  // Move all buttons inside the scaled wrapper so they stay aligned with the canvas
  [clarkButton, mrtButton, compareButton, pointButton, continueButton,
   inhibitorButton, fitButton, postulateButton, beginButton, slideButton, gotItButton,
   devButton, devAnalysisButton, compareP5Button].forEach(b => b.parent('wrapper'));
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
      updateBallCount();
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
      x: x + size - 32,
      y: y + 5,
      w: 10,
      h: 20
    });
  }
  return { gpcrPos, rects, size };
}

// 10 GPCRs for MRT scenes — smaller sprites, same membrane bounds.
function getMrtLayout() {
  const left = 660, right = 1240, y = 565;
  const count = 10, size = 55;
  const gap = (right - left - count * size) / (count - 1); // ~3.3px
  const gpcrPos = [], rects = [];
  for (let i = 0; i < count; i++) {
    const x = left + i * (size + gap);
    gpcrPos.push({ x, y });
    rects.push({ x: x + size - 22, y: y + 5, w: 6, h: 12 });
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
  if (postulateButton) postulateButton.hide();
  if (beginButton) beginButton.hide();
  if (slideButton) slideButton.hide();
  if (gotItButton) gotItButton.hide();
  if (compareP5Button) compareP5Button.hide();
}


function showUIElements() {
  slider.show();
  pointButton.show();
  continueButton.show();
  fitButton.show(); // NEW
}


// ─────────────────────────────────────────────
function initializeScene(sceneName) {
  // Restore default button theme when leaving Chapter 2 or Chapter 3 labs
  if (scene === 'partialGraph' || scene === 'spareGraph') {
    const _yellowBtns = [pointButton, fitButton, continueButton, gotItButton];
    for (const b of _yellowBtns) if (b) {
      b.style('background-color', '#FFD700');
      b.style('color', '#15163a');
    }
    if (scene === 'spareGraph' && slider) slider.elt.max = sliderMax;
  }
  if (scene === 'mrtBasal') {
    const _yellowBtns2 = [gotItButton, slideButton];
    for (const b of _yellowBtns2) if (b) {
      b.style('background-color', '#FFD700');
      b.style('color', '#15163a');
    }
  }

  // Reset simulation variables for a new scene
  particles = [];
  pointList = [];
  pointCounter = 0;
  frameCounter = 0;
  attachmentTimes = [];
  attachmentCount = 0;
  horizontalShift = 0;
  inhibitorButtonClicked = false;
  fittedCurve = null;
  lastBallCountChangeTime = millis();

  hideUIElements();

  switch (sceneName) {
    case 'title': {
      hideUIElements();
      beginButton.show();
      break;
    }
    case 'clarkIntro': {
      hideUIElements();
      clarkPostulateShown = new Set();
      if (spotlightOverlayG) { spotlightOverlayG.remove(); spotlightOverlayG = null; }
      activeSpotlight = null;
      pendingSpotlight = null;
      slideButton.show();
      break;
    }
    case 'tissueTransition': {
      hideUIElements();
      slideButton.html('Next Stop! →');
      slideButton.show();
      break;
    }
    case 'achGraph': {
      if (typeof setReceptorLayout === 'function') {
        setReceptorLayout(getAchRectangles());
      }
      showGraphOverlay = true;
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
    case 'dataCollected': {
      hideUIElements();
      if (slideButton) {
        slideButton.html("Let's Compare! →");
        slideButton.position(550, 610);
        slideButton.style('width', '200px');
        slideButton.style('height', '48px');
        slideButton.style('line-height', '28px');
        slideButton.style('font-size', '15px');
        slideButton.show();
      }
      break;
    }
    case 'compareGraphs': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Next Chapter →');
        slideButton.position(1060, 655);
        slideButton.style('width', '180px');
        slideButton.style('height', '40px');
        slideButton.style('line-height', '20px');
        slideButton.style('font-size', '14px');
        slideButton.show();
      }
      break;
    }

    case 'clarkSummary': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('But wait... →');
        slideButton.position(1060, 655);
        slideButton.show();
      }
      break;
    }

    case 'clarkProblems': {
      hideUIElements();
      problemPage = 1;
      if (slideButton) {
        slideButton.html('Next Limitation →');
        slideButton.position(1046, 655);
        slideButton.style('width', '194px');
        slideButton.show();
      }
      break;
    }

    case 'limitationTitle': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Enter the Lab →');
        slideButton.position(550, 555);
        slideButton.show();
      }
      break;
    }

    case 'partialGraph': {
      showPartialOverlay = true;
      partialPointList   = [];
      partialPointCounter = 0;
      partialFittedCurve  = null;
      partialGraphPlotted = false;
      partialGhostY       = 0;
      hideUIElements();
      // Red button theme for Chapter 2
      const _redBtns = [pointButton, fitButton, continueButton, gotItButton];
      for (const b of _redBtns) if (b) {
        b.style('background-color', '#cc3c3c');
        b.style('color', '#fff5f5');
      }
      if (gotItButton) {
        gotItButton.html('Continue to lab tour →');
        gotItButton.show();
      }
      break;
    }

    case 'spareGraph': {
      showSpareOverlay  = true;
      sparePointList    = [];
      sparePointCounter = 0;
      spareFittedCurve  = null;
      spareGraphPlotted = false;
      spareGhostY       = 0;
      hideUIElements();
      // Restrict slider to low-concentration range — students can't saturate all receptors
      if (slider) { slider.elt.max = SPARE_SLIDER_MAX; slider.elt.value = 1; }
      // Same 4-receptor layout as partialGraph — keeps the comparison concrete
      if (typeof setReceptorLayout === 'function') {
        setReceptorLayout([
          { x: 925,  y: 580, w: 10, h: 20 },
          { x: 755,  y: 585, w: 10, h: 20 },
          { x: 1095, y: 555, w: 10, h: 20 },
          { x: 1225, y: 555, w: 10, h: 20 },
        ]);
      }
      const _redBtns2 = [pointButton, fitButton, continueButton, gotItButton];
      for (const b of _redBtns2) if (b) {
        b.style('background-color', '#cc3c3c');
        b.style('color', '#fff5f5');
      }
      if (gotItButton) {
        gotItButton.html('Continue to lab tour →');
        gotItButton.show();
      }
      break;
    }

    case 'mrtTitle': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Begin →');
        slideButton.position(550, 555);
        slideButton.show();
      }
      break;
    }

    case 'ch2Analysis': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Next Chapter →');
        slideButton.style('width', '194px');
        slideButton.position(550, 640);
        slideButton.show();
      }
      break;
    }

    case 'mrtBasal': {
      hideUIElements();
      showMrtBasalOverlay = true;
      showMrtDrugOverlay  = false;
      mrtBasalStates = [];
      for (let i = 0; i < MRT_N; i++) {
        mrtBasalStates[i] = (i % 2 === 0);
      }
      mrtFlipTimer = 0;
      mrtActivityHistory = [];
      mrtEma = 0.5;
      mrtDrugType = 'none';
      mrtPhase = 'basal';
      mrtDrugButtons.forEach(b => b.remove());
      mrtDrugButtons = [];
      if (mrtInstructionEl) { mrtInstructionEl.remove(); mrtInstructionEl = null; }
      hideMrtActionButtons();
      hideMrtGraphButtons();
      mrtPlotData = {};
      mrtFittedCurves = {};
      mrtCompletedDrugs = new Set();
      particles = [];
      detachmentTime = 10;
      const { rects: mrtRects } = getMrtLayout();
      setReceptorLayout(mrtRects);
      // Teal button theme for Chapter 3
      const _tealBtns = [gotItButton, slideButton];
      for (const b of _tealBtns) if (b) {
        b.style('background-color', '#2EC4A0');
        b.style('color', '#0f2e28');
      }
      if (gotItButton) {
        gotItButton.html('Continue to lab tour →');
        gotItButton.show();
      }
      // slideButton configured but hidden until tour completes
      if (slideButton) {
        slideButton.html('Continue →');
        slideButton.position(280, 645);
        slideButton.style('width', '110px');
        slideButton.style('height', '34px');
        slideButton.style('line-height', '14px');
        slideButton.style('font-size', '11px');
        slideButton.hide();
      }
      break;
    }

    case 'mrtAnalysis': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Summary →');
        slideButton.position(540, 678);
        slideButton.style('width', '200px');
        slideButton.style('height', '36px');
        slideButton.style('line-height', '16px');
        slideButton.style('font-size', '13px');
        slideButton.style('background-color', '#2EC4A0');
        slideButton.style('color', '#0f2e28');
        slideButton.show();
      }
      break;
    }

    case 'mrtPrinciples': {
      hideUIElements();
      if (slideButton) {
        slideButton.html('Finish →');
        slideButton.position(560, 672);
        slideButton.style('width', '160px');
        slideButton.style('height', '36px');
        slideButton.style('line-height', '16px');
        slideButton.style('font-size', '13px');
        slideButton.style('background-color', '#2EC4A0');
        slideButton.style('color', '#0f2e28');
        slideButton.show();
      }
      break;
    }

    case 'intro': {
      hideUIElements();
      // Chapter I card button
      if (clarkButton) {
        clarkButton.html('Chapter I →');
        clarkButton.style('width', '160px');
        clarkButton.style('background-color', '#FFD700');
        clarkButton.style('color', '#0f1637');
        clarkButton.style('opacity', '1');
        clarkButton.style('cursor', 'pointer');
        clarkButton.position(147, 512);
        clarkButton.show();
      }
      // Chapter II card button
      if (compareButton) {
        compareButton.html('Chapter II →');
        compareButton.style('width', '160px');
        compareButton.style('background-color', '#FFD700');
        compareButton.style('color', '#0f1637');
        compareButton.style('opacity', '1');
        compareButton.style('cursor', 'pointer');
        compareButton.position(560, 512);
        compareButton.show();
      }
      // Chapter III card button
      if (mrtButton) {
        mrtButton.html('Chapter III →');
        mrtButton.style('width', '160px');
        mrtButton.style('background-color', '#2EC4A0');
        mrtButton.style('color', '#0f1637');
        mrtButton.style('opacity', '1');
        mrtButton.style('cursor', 'pointer');
        mrtButton.position(973, 512);
        mrtButton.show();
      }
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


// ─────────────────────────────────────────────
// Magic School Bus — p5.js bus drawing helper
// x,y = top-left of bus body; w,h = body dimensions
// ─────────────────────────────────────────────
function drawMSBus(x, y, w, h) {
  push();

  // Drop shadow
  noStroke(); fill(0, 0, 0, 28);
  ellipse(x + w * 0.45, y + h + h * 0.22 + 6, w * 0.75, h * 0.18);

  // Main body — bus yellow
  fill(255, 205, 0); stroke(25, 15, 0); strokeWeight(3);
  rect(x, y, w * 0.82, h, 10, 10, 4, 4);

  // Cab (front section)
  fill(245, 175, 0); stroke(25, 15, 0); strokeWeight(3);
  rect(x + w * 0.82, y + h * 0.15, w * 0.18, h * 0.85, 2, 14, 14, 2);

  // Black stripe along bus body bottom
  fill(20, 15, 0); noStroke();
  rect(x + 4, y + h * 0.70, w * 0.82 - 4, h * 0.30, 0, 0, 4, 4);

  // Red bumper at cab front
  fill(210, 35, 35); noStroke();
  rect(x + w * 0.82 + 2, y + h * 0.72, w * 0.18 - 4, h * 0.28, 0, 12, 12, 0);

  // Side windows
  fill(65, 115, 190); stroke(25, 15, 0); strokeWeight(1.5);
  const nWin = 4;
  const winW = (w * 0.60) / nWin - 5;
  const winH = h * 0.30;
  const winY = y + h * 0.10;
  for (let i = 0; i < nWin; i++) {
    rect(x + w * 0.05 + i * (winW + 5), winY, winW, winH, 4);
  }

  // Windshield
  fill(80, 150, 215); stroke(25, 15, 0); strokeWeight(1.5);
  rect(x + w * 0.84, y + h * 0.18, w * 0.12, h * 0.33, 3);

  // Wheels
  fill(18, 18, 18); stroke(50); strokeWeight(2);
  circle(x + w * 0.18, y + h + h * 0.18, h * 0.36);
  circle(x + w * 0.66, y + h + h * 0.18, h * 0.36);
  // Rims
  fill(175, 175, 175); noStroke();
  circle(x + w * 0.18, y + h + h * 0.18, h * 0.16);
  circle(x + w * 0.66, y + h + h * 0.18, h * 0.16);

  // "MAGIC SCHOOL BUS" text on body
  noStroke(); fill(25, 15, 0);
  textAlign(LEFT); textStyle(BOLD); textSize(h * 0.13);
  text('MAGIC SCHOOL BUS', x + w * 0.06, y + h * 0.61);
  textStyle(NORMAL);

  pop();
}

function drawTitleScene() {
  // MSB yellow background
  background(255, 210, 0);

  // Dark navy top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 80);

  // Top bar label
  fill(255, 210, 0); textAlign(CENTER); textSize(13); textStyle(BOLD);
  text('THE MAGIC SCHOOL BUS  ·  MODULE #1', width / 2, 30);
  textStyle(NORMAL);
  fill(255, 210, 100); textSize(12);
  text('Made for Students by Students  ·  HTHSCI 1I06', width / 2, 56);

  // Divider
  stroke(15, 22, 55); strokeWeight(2);
  line(200, 160, 1080, 160);
  noStroke();

  // Subtitle
  fill(40, 30, 0); textSize(17); textAlign(CENTER);
  text('You may have just learned the two types of pharmacological models that describe receptors and their response.', width / 2, 248);
  text('Many students have stumbled upon this hurdle. We decided to illustrate the models in live action for better visualization.', width / 2, 274);
  fill(15, 22, 55); textStyle(BOLD); textSize(19);
  text('Welcome aboard!', width / 2, 318);
  textStyle(NORMAL);

  // Bus — lower centre
  drawMSBus(390, 390, 500, 150);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter 2 — Partial Agonism Lab
// ─────────────────────────────────────────────────────────────────────────────

function drawPartialGraphOverlay() {
  fill(0, 0, 0, 165); noStroke();
  rect(0, 0, width, height);

  const px = 190, py = 90, pw = 900, ph = 540;
  fill(204, 60, 60); noStroke(); rect(px, py, pw, 58, 10, 10, 0, 0);
  fill(250, 248, 240); stroke(200, 192, 175); strokeWeight(1);
  rect(px, py + 58, pw, ph - 58, 0, 0, 10, 10);

  noStroke(); fill(255, 240, 240);
  textAlign(CENTER); textStyle(BOLD); textSize(14);
  text("MS. FRIZZLE'S FIELD NOTES  \xB7  CHAPTER II  \xB7  FIELD STOP", px + pw / 2, py + 26);
  textStyle(NORMAL); fill(255, 210, 210); textSize(11);
  text('"Different drug. Different story. Same graph."', px + pw / 2, py + 46);

  const tx = px + 52, lh = 22;
  let ty = py + 88;

  noStroke(); fill(30, 38, 80);
  textAlign(CENTER); textStyle(BOLD); textSize(20);
  text('Welcome back to the lab', px + pw / 2, ty);
  textStyle(NORMAL); ty += 16;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 22;

  fill(55, 60, 80); textAlign(LEFT); textSize(14);
  text("Same tissue. Same magnification. Same 4 receptors. But this time we're using a different drug.", tx, ty); ty += lh;
  text("Watch what happens at the molecular level as Ligand B interacts with the same receptors.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text("Clark's Prediction", tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("According to Clark's assumptions, any drug that fills all receptors should produce the same", tx, ty); ty += lh;
  text("maximum response. The dashed curve is your Chapter 1 result — that's the benchmark.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text('Your Goal', tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("Build the dose-response curve for Ligand B. The dashed line is your Original Graph from Chapter 1 —", tx, ty); ty += lh;
  text("Clark's prediction for what you should see with any drug on this tissue. See if Ligand B agrees.", tx, ty); ty += lh * 1.8;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty - 8, px + pw - 40, ty - 8); noStroke();

  fill(180, 50, 50); textStyle(BOLD); textSize(12);
  text('UP NEXT', tx, ty + 4);
  fill(55, 60, 80); textStyle(NORMAL); textSize(13);
  text('A quick tour of the field site.', tx + 76, ty + 4);
}

function drawPartialGraphScene() {
  const conc = slider.value();

  // ── Right panel drawn first (background), left panel draws on top ─────────
  background(173, 216, 230);
  stroke(180, 140, 0); strokeWeight(2);
  line(640, 0, 640, 720);

  // Diaphragm sprite (identical position to achGraph)
  image(Diaphragm, 735, 10, 450, 250);

  // Yellow magnification zoom box + lines
  noFill(); stroke(255, 255, 102);
  rect(945, 80, 30, 30);
  line(945, 110, 640, 300);
  line(975, 110, 1280, 300);
  line(640, 300, 1280, 300);

  // Membrane sprite
  image(membrane, 640, 580, 750, 105);

  // 4 GPCRs + red binding rectangles
  image(gpcr, 630, 530, 200, 200);
  image(gpcr, 790, 530, 200, 200);
  image(gpcr, 950, 530, 200, 200);
  image(gpcr, 1100, 530, 200, 200);
  fill(255, 0, 0); noStroke();
  rect(925, 580, 10, 20); rect(755, 585, 10, 20);
  rect(1095, 555, 10, 20); rect(1225, 555, 10, 20);

  // Balls run underneath (will be hidden by black box)
  stroke(0); noFill();
  for (let a of particles) {
    if (!activeSpotlight) { a.bounceOthers(); a.update(); }
    a.display();
  }

  // Label the right panel
  noStroke(); fill(204, 60, 60); textAlign(CENTER); textStyle(BOLD); textSize(12);
  text('Ligand B  \xB7  Synthetic Compound  \xB7  Chapter II', 960, 320);
  textStyle(NORMAL);

  // ── Left panel: white graph (drawn AFTER background so it sits on top) ────
  fill(255); noStroke(); rect(0, 0, 640, 720);

  // Grid
  stroke(180); strokeWeight(1);
  for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
  for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

  // Axes
  strokeWeight(2); stroke(0);
  line(80, h - 80, w - 80, h - 80);
  line(80, 80, 80, h - 80);

  // Labels
  noStroke(); textAlign(CENTER);
  fill(40, 45, 70); textStyle(BOLD); textSize(16);
  text("Field Stop 3  —  Partial Agonism", 320, 38);
  textStyle(NORMAL);
  fill(120, 50, 50); textSize(13);
  text('Concentration — Response Curve', 320, 68);
  fill(60, 65, 90); textSize(13);
  text('Ligand B Concentration (units)', 320, 452);
  push(); translate(18, 250); rotate(-HALF_PI);
  textAlign(CENTER); fill(60, 65, 90); textSize(13);
  text('Binding Rate (binds / sec)', 0, 0); pop();

  // "Original Graph" — Chapter 1 fitted curve, used as Clark's baseline prediction
  if (fittedCurve && fittedCurve.points.length) {
    stroke(30, 38, 80); strokeWeight(2); noFill();
    drawingContext.setLineDash([8, 6]);
    beginShape();
    for (const p of fittedCurve.points) vertex(p.x, p.y);
    endShape();
    drawingContext.setLineDash([]);
    const ogEnd = fittedCurve.points[fittedCurve.points.length - 1];
    noStroke(); fill(30, 38, 80); textAlign(RIGHT); textSize(10);
    text('Original Graph', ogEnd.x, ogEnd.y - 8);
  } else {
    // Chapter 1 not yet completed — show a placeholder
    noStroke(); fill(160, 160, 160); textAlign(CENTER); textSize(11);
    text('(Complete Chapter 1 to see the Original Graph here)', 320, 260);
  }

  // Fitted actual curve (red)
  if (partialFittedCurve && partialFittedCurve.points.length) {
    stroke(204, 60, 60); strokeWeight(3); noFill();
    beginShape();
    for (const p of partialFittedCurve.points) vertex(p.x, p.y);
    endShape();
  }

  // Plotted points
  noStroke();
  for (const pt of partialPointList) {
    fill(204, 60, 60, pt.alpha);
    ellipse(map(pt.x, sliderMin, sliderMax, 80, w - 80),
            map(pt.y, yMin, yMax, h - 80, 80), 10, 10);
  }

  // Ghost point — 50% of Chapter 1 Emax, same EC50/n (falls short of Original Graph)
  const _ch1p = fittedCurve ? fittedCurve.params : { Emax: yMax, EC50: PARTIAL_EC50, n: PARTIAL_N };
  const baseResp = (_ch1p.Emax * 0.5) * hillG(conc, _ch1p.EC50, _ch1p.n);
  partialGhostY  = constrain(baseResp + (noise(frameCount * 0.04) - 0.5) * 0.8, yMin, yMax);
  const gx = map(conc, sliderMin, sliderMax, 80, w - 80);
  const gy = map(partialGhostY, yMin, yMax, h - 80, 80);
  fill(204, 60, 60, 180); noStroke(); ellipse(gx, gy, 10, 10);

  // Stats
  fill(120, 125, 150); textSize(12); textAlign(LEFT);
  text('Concentration: ' + Math.round(conc), 85, 478);
  textAlign(RIGHT);
  text('Rate: ' + partialGhostY.toFixed(2) + ' /sec', 555, 478);

  // Overlay + spotlight
  if (showPartialOverlay) drawPartialGraphOverlay();
  if (showPartialOverlay || activeSpotlight) {
    slider.hide(); pointButton.hide(); fitButton.hide(); continueButton.hide();
  } else {
    slider.show(); pointButton.show(); fitButton.show();
    if (partialGraphPlotted) continueButton.show(); else continueButton.hide();
  }
  drawSpotlight();
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter 2 — Spare Receptors Lab
// ─────────────────────────────────────────────────────────────────────────────

function drawSpareGraphOverlay() {
  fill(0, 0, 0, 165); noStroke();
  rect(0, 0, width, height);

  const px = 190, py = 90, pw = 900, ph = 540;
  fill(204, 60, 60); noStroke(); rect(px, py, pw, 58, 10, 10, 0, 0);
  fill(250, 248, 240); stroke(200, 192, 175); strokeWeight(1);
  rect(px, py + 58, pw, ph - 58, 0, 0, 10, 10);

  noStroke(); fill(255, 240, 240);
  textAlign(CENTER); textStyle(BOLD); textSize(14);
  text("MS. FRIZZLE'S FIELD NOTES  \xB7  CHAPTER II  \xB7  FIELD STOP 4", px + pw / 2, py + 26);
  textStyle(NORMAL); fill(255, 210, 210); textSize(11);
  text('"More receptors. Same story?"', px + pw / 2, py + 46);

  const tx = px + 52, lh = 22;
  let ty = py + 88;

  noStroke(); fill(30, 38, 80);
  textAlign(CENTER); textStyle(BOLD); textSize(20);
  text('Back in the lab', px + pw / 2, ty);
  textStyle(NORMAL); ty += 16;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 22;

  fill(55, 60, 80); textAlign(LEFT); textSize(14);
  text("Same tissue. Same 4 receptors. Same setup as before — but this time we're using Ligand C,", tx, ty); ty += lh;
  text("a different synthetic compound on the exact same system.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text("Clark's Prediction", tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("Same tissue, same 4 receptors — Clark says any drug that fills all 4 should produce the same", tx, ty); ty += lh;
  text("maximum response. The Original Graph is still the benchmark.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text('Your Goal', tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("Build the dose-response curve for Ligand C. Compare it to the Original Graph.", tx, ty); ty += lh;
  text("See if the curve lands where Clark would predict.", tx, ty); ty += lh * 1.8;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty - 8, px + pw - 40, ty - 8); noStroke();

  fill(180, 50, 50); textStyle(BOLD); textSize(12);
  text('UP NEXT', tx, ty + 4);
  fill(55, 60, 80); textStyle(NORMAL); textSize(13);
  text('A quick tour of the field site.', tx + 76, ty + 4);
}

function drawSpareGraphScene() {
  const conc = slider.value();

  // Right panel first
  background(173, 216, 230);
  stroke(180, 140, 0); strokeWeight(2);
  line(640, 0, 640, 720);

  image(Diaphragm, 735, 10, 450, 250);

  noFill(); stroke(255, 255, 102);
  rect(945, 80, 30, 30);
  line(945, 110, 640, 300);
  line(975, 110, 1280, 300);
  line(640, 300, 1280, 300);

  image(membrane, 640, 580, 750, 105);

  // Same 4 GPCRs as Chapter 1 — keeps the comparison concrete
  image(gpcr, 630, 530, 200, 200);
  image(gpcr, 790, 530, 200, 200);
  image(gpcr, 950, 530, 200, 200);
  image(gpcr, 1100, 530, 200, 200);
  fill(255, 0, 0); noStroke();
  rect(925, 580, 10, 20); rect(755, 585, 10, 20);
  rect(1095, 555, 10, 20); rect(1225, 555, 10, 20);

  stroke(0); noFill();
  for (let a of particles) {
    if (!activeSpotlight) { a.bounceOthers(); a.update(); }
    a.display();
  }

  noStroke(); fill(204, 60, 60); textAlign(CENTER); textStyle(BOLD); textSize(12);
  text('Ligand C  \xB7  Spare Receptors  \xB7  Chapter II', 960, 320);
  textStyle(NORMAL);

  // Left panel
  fill(255); noStroke(); rect(0, 0, 640, 720);

  stroke(180); strokeWeight(1);
  for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
  for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

  strokeWeight(2); stroke(0);
  line(80, h - 80, w - 80, h - 80);
  line(80, 80, 80, h - 80);

  noStroke(); textAlign(CENTER);
  fill(40, 45, 70); textStyle(BOLD); textSize(16);
  text("Field Stop 4  —  Spare Receptors", 320, 38);
  textStyle(NORMAL);
  fill(120, 50, 50); textSize(13);
  text('Concentration — Response Curve', 320, 68);
  fill(60, 65, 90); textSize(13);
  text('Ligand C Concentration (units)', 320, 452);
  push(); translate(18, 250); rotate(-HALF_PI);
  textAlign(CENTER); fill(60, 65, 90); textSize(13);
  text('Binding Rate (binds / sec)', 0, 0); pop();

  // Original Graph (Chapter 1 curve) as dashed navy line
  if (fittedCurve && fittedCurve.points.length) {
    stroke(30, 38, 80); strokeWeight(2); noFill();
    drawingContext.setLineDash([8, 6]);
    beginShape();
    for (const p of fittedCurve.points) vertex(p.x, p.y);
    endShape();
    drawingContext.setLineDash([]);
    const ogEnd = fittedCurve.points[fittedCurve.points.length - 1];
    noStroke(); fill(30, 38, 80); textAlign(RIGHT); textSize(10);
    text('Original Graph', ogEnd.x, ogEnd.y - 8);
  } else {
    noStroke(); fill(160, 160, 160); textAlign(CENTER); textSize(11);
    text('(Complete Chapter 1 to see the Original Graph here)', 320, 260);
  }

  // Fitted Ligand C curve (red)
  if (spareFittedCurve && spareFittedCurve.points.length) {
    stroke(204, 60, 60); strokeWeight(3); noFill();
    beginShape();
    for (const p of spareFittedCurve.points) vertex(p.x, p.y);
    endShape();
  }

  // Plotted points
  noStroke();
  for (const pt of sparePointList) {
    fill(204, 60, 60, pt.alpha);
    ellipse(map(pt.x, sliderMin, sliderMax, 80, w - 80),
            map(pt.y, yMin, yMax, h - 80, 80), 10, 10);
  }

  // Ghost point — same Emax as Ch1, EC50 shifted left
  const _sp = fittedCurve ? fittedCurve.params : { Emax: yMax, EC50: 80, n: 2 };
  const spareBase = _sp.Emax * hillG(conc, _sp.EC50 * SPARE_EC50_FACTOR, _sp.n);
  spareGhostY = constrain(spareBase + (noise(frameCount * 0.04 + 100) - 0.5) * 0.8, yMin, yMax);
  const sgx = map(conc, sliderMin, sliderMax, 80, w - 80);
  const sgy = map(spareGhostY, yMin, yMax, h - 80, 80);
  fill(204, 60, 60, 180); noStroke(); ellipse(sgx, sgy, 10, 10);

  // Vertical marker showing the slider's concentration ceiling
  const ceilX = map(SPARE_SLIDER_MAX, sliderMin, sliderMax, 80, w - 80);
  stroke(200, 100, 100, 120); strokeWeight(1); drawingContext.setLineDash([4, 4]);
  line(ceilX, 80, ceilX, h - 80);
  drawingContext.setLineDash([]);
  noStroke(); fill(180, 80, 80); textAlign(CENTER); textSize(9);
  text('max conc.', ceilX, 76);

  fill(120, 125, 150); noStroke(); textSize(12); textAlign(LEFT);
  text('Concentration: ' + Math.round(conc), 85, 478);
  textAlign(RIGHT);
  text('Rate: ' + spareGhostY.toFixed(2) + ' /sec', 555, 478);

  if (showSpareOverlay) drawSpareGraphOverlay();
  if (showSpareOverlay || activeSpotlight) {
    slider.hide(); pointButton.hide(); fitButton.hide(); continueButton.hide();
  } else {
    slider.show(); pointButton.show(); fitButton.show();
    if (spareGraphPlotted) continueButton.show(); else continueButton.hide();
  }
  drawSpotlight();
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter 2 — Analysis Page
// ─────────────────────────────────────────────────────────────────────────────
function drawCh2AnalysisScene() {
  const R = 204, G = 60, B = 60;
  background(R, G, B);

  // Top bar
  fill(15, 22, 55); noStroke(); rect(0, 0, width, 72);
  fill(255, 200, 200); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  \xB7  CHAPTER II  \xB7  ANALYSIS', width / 2, 30);
  textStyle(NORMAL); fill(255, 170, 170); textSize(12);
  text('Two experiments. Two contradictions. One broken model.', width / 2, 54);

  // ── LEFT PANEL — combined graph ───────────────────────────────────────────
  const gx = 52, gy = 100, gw = 530, gh = 470;
  fill(255); noStroke(); rect(gx, gy, gw, gh, 8);

  // Grid
  stroke(220); strokeWeight(0.8);
  for (let i = 1; i < 5; i++) {
    line(gx, gy + gh * i / 5, gx + gw, gy + gh * i / 5);
    line(gx + gw * i / 5, gy, gx + gw * i / 5, gy + gh);
  }

  // Axes
  stroke(60); strokeWeight(1.5);
  line(gx, gy + gh, gx + gw, gy + gh);
  line(gx, gy, gx, gy + gh);

  // Axis labels
  noStroke(); fill(60, 65, 90); textAlign(CENTER); textStyle(NORMAL); textSize(11);
  text('Concentration', gx + gw / 2, gy + gh + 22);
  push(); translate(gx - 22, gy + gh / 2); rotate(-HALF_PI);
  textAlign(CENTER); text('Response', 0, 0); pop();

  // Graph title
  fill(40, 45, 70); textStyle(BOLD); textSize(14); textAlign(CENTER);
  text('All Three Curves — Same Tissue', gx + gw / 2, gy - 14);

  // Helper: map data coords to this mini-graph
  const mx = (xv) => map(xv, sliderMin, sliderMax, gx, gx + gw);
  const my = (yv) => map(yv, yMin, yMax, gy + gh, gy);

  // 1. Original Graph (ACh) — dashed dark navy
  if (fittedCurve && fittedCurve.points.length) {
    stroke(30, 38, 80); strokeWeight(2); noFill();
    drawingContext.setLineDash([7, 5]);
    beginShape();
    for (const p of fittedCurve.points) {
      // fittedCurve.points are in sketch screen coords — remap to mini-graph
      const xv = map(p.x, 80, w - 80, sliderMin, sliderMax);
      const yv = map(p.y, h - 80, 80, yMin, yMax);
      vertex(mx(xv), my(yv));
    }
    endShape();
    drawingContext.setLineDash([]);
    // Label
    const ogTip = fittedCurve.points[fittedCurve.points.length - 1];
    const ogYv  = map(ogTip.y, h - 80, 80, yMin, yMax);
    noStroke(); fill(30, 38, 80); textAlign(LEFT); textSize(10); textStyle(BOLD);
    text('ACh (Original)', mx(sliderMax) - 5, my(ogYv) - 6);
  }

  // 2. Ligand B (partial) — solid red, lower plateau
  if (partialFittedCurve && partialFittedCurve.points.length) {
    stroke(204, 60, 60); strokeWeight(2.5); noFill();
    beginShape();
    for (const p of partialFittedCurve.points) {
      const xv = map(p.x, 80, w - 80, sliderMin, sliderMax);
      const yv = map(p.y, h - 80, 80, yMin, yMax);
      vertex(mx(xv), my(yv));
    }
    endShape();
    const bTip = partialFittedCurve.points[partialFittedCurve.points.length - 1];
    const bYv  = map(bTip.y, h - 80, 80, yMin, yMax);
    noStroke(); fill(204, 60, 60); textAlign(LEFT); textSize(10); textStyle(BOLD);
    text('Ligand B', mx(sliderMax) - 5, my(bYv) - 6);
  } else {
    // Draw a representative Ligand B curve if student skipped
    const repEmax = yMax * 0.5;
    const repEC50 = 80;
    stroke(204, 60, 60); strokeWeight(2); noFill();
    beginShape();
    for (let x = sliderMin; x <= sliderMax; x += 3) {
      vertex(mx(x), my(repEmax * hillG(x, repEC50, 2.5)));
    }
    endShape();
    noStroke(); fill(204, 60, 60); textAlign(LEFT); textSize(10); textStyle(BOLD);
    text('Ligand B', mx(sliderMax * 0.92), my(repEmax) - 6);
  }

  // 3. Ligand C (spare) — solid orange-red, same Emax but shifted left
  if (spareFittedCurve && spareFittedCurve.points.length) {
    stroke(230, 120, 40); strokeWeight(2.5); noFill();
    beginShape();
    for (const p of spareFittedCurve.points) {
      const xv = map(p.x, 80, w - 80, sliderMin, sliderMax);
      const yv = map(p.y, h - 80, 80, yMin, yMax);
      vertex(mx(xv), my(yv));
    }
    endShape();
    const cTip = spareFittedCurve.points[spareFittedCurve.points.length - 1];
    const cYv  = map(cTip.y, h - 80, 80, yMin, yMax);
    noStroke(); fill(230, 120, 40); textAlign(LEFT); textSize(10); textStyle(BOLD);
    text('Ligand C', mx(SPARE_SLIDER_MAX) + 4, my(cYv) - 6);
  } else {
    // Representative Ligand C
    const ch1p = fittedCurve ? fittedCurve.params : { Emax: yMax, EC50: 80, n: 2 };
    stroke(230, 120, 40); strokeWeight(2); noFill();
    beginShape();
    for (let x = sliderMin; x <= SPARE_SLIDER_MAX; x += 2) {
      vertex(mx(x), my(ch1p.Emax * hillG(x, ch1p.EC50 * SPARE_EC50_FACTOR, ch1p.n)));
    }
    endShape();
    noStroke(); fill(230, 120, 40); textAlign(LEFT); textSize(10); textStyle(BOLD);
    text('Ligand C', mx(SPARE_SLIDER_MAX) + 4, my(ch1p.Emax * 0.93));
  }

  textStyle(NORMAL);

  // ── RIGHT PANEL — findings ────────────────────────────────────────────────
  const rx = 620, ry = 88, rw = 630, rh = 580;
  const cardW = rw - 24, cardX = rx + 12;

  // Finding 1 — Partial Agonism
  const f1y = ry + 10, f1h = 222;
  fill(15, 22, 55, 230); noStroke(); rect(cardX, f1y, cardW, f1h, 8);
  stroke(204, 60, 60); strokeWeight(1.5); noFill(); rect(cardX, f1y, cardW, f1h, 8);

  noStroke(); fill(204, 60, 60); textAlign(LEFT); textStyle(BOLD); textSize(11);
  text('FINDING 1  —  PARTIAL AGONISM', cardX + 14, f1y + 20);
  stroke(204, 60, 60, 60); strokeWeight(0.75);
  line(cardX + 14, f1y + 28, cardX + cardW - 14, f1y + 28);
  noStroke();

  fill(255, 220, 220); textStyle(BOLD); textSize(13);
  text('Ligand B hit the same 4 receptors — but couldn\'t', cardX + 14, f1y + 48);
  text('reach the same maximum response.', cardX + 14, f1y + 65);

  fill(200, 210, 240); textStyle(NORMAL); textSize(11.5);
  text('Clark assumed: full occupancy = full response.', cardX + 14, f1y + 92);
  text('Ligand B occupied all 4 receptors but plateaued', cardX + 14, f1y + 110);
  text('at ~50% Emax. Binding happened — activation didn\'t.', cardX + 14, f1y + 128);

  fill(255, 180, 100); textStyle(BOLD); textSize(11);
  text('CONCLUSION:', cardX + 14, f1y + 155);
  fill(255, 220, 180); textStyle(NORMAL); textSize(11);
  text('Occupying a receptor and activating it are not the same', cardX + 14, f1y + 172);
  text('thing. Clark\'s model has no way to distinguish between them.', cardX + 14, f1y + 189);

  // Finding 2 — Spare Receptors
  const f2y = f1y + f1h + 14, f2h = 222;
  fill(15, 22, 55, 230); noStroke(); rect(cardX, f2y, cardW, f2h, 8);
  stroke(230, 120, 40); strokeWeight(1.5); noFill(); rect(cardX, f2y, cardW, f2h, 8);

  noStroke(); fill(230, 120, 40); textAlign(LEFT); textStyle(BOLD); textSize(11);
  text('FINDING 2  —  SPARE RECEPTORS', cardX + 14, f2y + 20);
  stroke(230, 120, 40, 60); strokeWeight(0.75);
  line(cardX + 14, f2y + 28, cardX + cardW - 14, f2y + 28);
  noStroke();

  fill(255, 230, 200); textStyle(BOLD); textSize(13);
  text('Ligand C reached the same Emax — but at a much', cardX + 14, f2y + 48);
  text('lower concentration than ACh.', cardX + 14, f2y + 65);

  fill(200, 210, 240); textStyle(NORMAL); textSize(11.5);
  text('Clark assumed: response is proportional to occupancy.', cardX + 14, f2y + 92);
  text('Ligand C hit maximum response before all receptors', cardX + 14, f2y + 110);
  text('were occupied. The rest were never needed.', cardX + 14, f2y + 128);

  fill(255, 180, 100); textStyle(BOLD); textSize(11);
  text('CONCLUSION:', cardX + 14, f2y + 155);
  fill(255, 220, 180); textStyle(NORMAL); textSize(11);
  text('You don\'t need 100% occupancy to get 100% response.', cardX + 14, f2y + 172);
  text('Spare receptors exist — and Clark\'s model can\'t account for them.', cardX + 14, f2y + 189);

  // Bottom transition strip
  const bsy = f2y + f2h + 14;
  fill(15, 22, 55, 200); noStroke(); rect(cardX, bsy, cardW, 68, 8);
  fill(255, 200, 200); textAlign(CENTER); textStyle(BOLD); textSize(12);
  text('So what comes next?', cardX + cardW / 2, bsy + 22);
  fill(200, 210, 240); textStyle(NORMAL); textSize(11);
  text('Ariens, Stephenson, and Black built a new model that could explain both.', cardX + cardW / 2, bsy + 42);
  text('That\'s Chapter III.', cardX + cardW / 2, bsy + 58);
}

function drawLimitationTitleScene() {
  // Chapter 2 colour — red [204, 60, 60]
  const R = 204, G = 60, B = 60;

  background(R, G, B);

  // Dark top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 72);

  fill(255, 200, 200); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  \xB7  CHAPTER II', width / 2, 30);
  textStyle(NORMAL);
  fill(255, 180, 180); textSize(12);
  text('Something\'s off — Clark\'s model has limits', width / 2, 54);

  // Horizontal rules flanking the title block
  stroke(15, 22, 55); strokeWeight(2);
  line(160, 195, 1120, 195);
  line(160, 490, 1120, 490);
  noStroke();

  // Big chapter title
  textAlign(CENTER);
  fill(15, 22, 55); textStyle(BOLD); textSize(72);
  text('Where Clark Fell Short', width / 2, 356);
  textStyle(NORMAL);

  // Subtitle
  fill(255, 240, 240); textSize(22);
  text('Spare receptors, partial agonism, and the cracks in occupancy theory', width / 2, 416);

  // Scholar credits line
  fill(15, 22, 55); textSize(15);
  text('Ariëns  \xB7  Stephenson  \xB7  1950 – 1956', width / 2, 458);

  // Small bus bottom-right
  drawMSBus(1020, 560, 220, 68);
}

function drawMrtTitleScene() {
  const MG = [46, 196, 160]; // MRT teal green
  background(MG[0], MG[1], MG[2]);

  // Dark top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 72);

  fill(MG[0], MG[1], MG[2]); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  \xB7  CHAPTER III', width / 2, 30);
  textStyle(NORMAL);
  fill(180, 240, 225); textSize(12);
  text('The model that fixed what Clark couldn\'t explain', width / 2, 54);

  stroke(15, 22, 55); strokeWeight(2);
  line(160, 195, 1120, 195);
  line(160, 490, 1120, 490);
  noStroke();

  textAlign(CENTER);
  fill(15, 22, 55); textStyle(BOLD); textSize(78);
  text('Modern Receptor Theory', width / 2, 356);
  textStyle(NORMAL);

  fill(10, 55, 45); textSize(22);
  text('Efficacy, intrinsic activity, and the two-state model', width / 2, 416);

  fill(15, 70, 58); textSize(15);
  text('Ari\xEBns  \xB7  Stephenson  \xB7  Black & Leff  \xB7  1954–1983', width / 2, 458);

  // Small bus bottom-right
  drawMSBus(1020, 560, 220, 68);
}

// ─────────────────────────────────────────────────────────────────────────────
// MRT — Two-State Equilibrium Slide
// Left: R⇌R* theory + 3 drug-type sections
// Right: membrane + tinted GPCR sprites + drug balls (3 stacked molecular rows)
// ─────────────────────────────────────────────────────────────────────────────

const MRT_DRUGS = [
  { label: 'Full Agonist',         type: 'fullAgonist',    bg: '#2EC4A0', r: 46,  g: 196, b: 160 },
  { label: 'Partial Agonist',      type: 'partialAgonist', bg: '#F0A030', r: 240, g: 160, b: 48  },
  { label: 'Antagonist',           type: 'antagonist',     bg: '#7878C0', r: 120, g: 120, b: 192 },
  { label: 'Partial Inv. Agonist', type: 'partialInverse', bg: '#C05888', r: 192, g: 88,  b: 136 },
  { label: 'Full Inv. Agonist',    type: 'fullInverse',    bg: '#D03535', r: 208, g: 53,  b: 53  },
];

function showMrtDrugButtons() {
  if (slideButton) slideButton.hide();
  // DOM instruction text — same coordinate space as buttons, no canvas/DOM mismatch
  if (mrtInstructionEl) { mrtInstructionEl.remove(); mrtInstructionEl = null; }
  mrtInstructionEl = createDiv('');
  mrtInstructionEl.parent('wrapper');
  mrtInstructionEl.style('position', 'absolute');
  mrtInstructionEl.style('left', '0px');
  mrtInstructionEl.style('width', '640px');
  mrtInstructionEl.style('text-align', 'center');
  mrtInstructionEl.style('pointer-events', 'none');

  const heading = createP('What happens when we add a drug?');
  heading.parent(mrtInstructionEl);
  heading.style('margin', '0 0 4px 0');
  heading.style('font-family', 'Arial, sans-serif');
  heading.style('font-size', '13px');
  heading.style('font-weight', '700');
  heading.style('color', '#28293e');

  const sub = createP('Select a ligand below to drop it into the environment');
  sub.parent(mrtInstructionEl);
  sub.style('margin', '0');
  sub.style('font-family', 'Arial, sans-serif');
  sub.style('font-size', '11px');
  sub.style('color', '#646882');

  const bw = 110, bh = 40, gap = 6;
  const totalW = MRT_DRUGS.length * bw + (MRT_DRUGS.length - 1) * gap;
  const startX = Math.round((640 - totalW) / 2);
  const btnY = 620;
  MRT_DRUGS.forEach((drug, i) => {
    const btn = createButton(drug.label);
    btn.position(startX + i * (bw + gap), btnY);
    btn.size(bw, bh);
    btn.style('background-color', drug.bg);
    btn.style('color', '#fff');
    btn.style('border', 'none');
    btn.style('border-radius', '7px');
    btn.style('font-size', '11px');
    btn.style('font-weight', '700');
    btn.style('cursor', 'pointer');
    btn.style('letter-spacing', '0.3px');
    btn.parent('wrapper');
    btn.mousePressed(() => selectMrtDrug(drug.type, drug.r, drug.g, drug.b));
    mrtDrugButtons.push(btn);
  });

  // Position the instruction div above the buttons
  mrtInstructionEl.style('top', (btnY - 46) + 'px');
}

function selectMrtDrug(type, r, g, b) {
  mrtDrugType = type;
  mrtDrugColor = [r, g, b];
  mrtPhase = 'drug';
  mrtActivityHistory = [];
  mrtEma = mrtBasalStates.filter(s => s).length / MRT_N;

  mrtDrugButtons.forEach(btn => btn.remove());
  mrtDrugButtons = [];
  if (mrtInstructionEl) { mrtInstructionEl.remove(); mrtInstructionEl = null; }

  detachmentTime = 600;

  if (!mrtPlotData[type]) mrtPlotData[type] = [];

  // Spawn 1 ligand ball — student uses slider to add more
  particles = [];
  attachedLigands = new Array(rectangles.length).fill(null);
  for (let i = 0; i < MRT_N; i++) mrtBasalStates[i] = (i % 2 === 0);
  const bounds = { x: 645, y: 305, w: 630, h: 260 };
  const pos = createVector(
    random(bounds.x + 20, bounds.x + bounds.w - 20),
    random(bounds.y + 20, bounds.y + bounds.h - 20)
  );
  particles.push(new Ball(pos, p5.Vector.random2D().mult(3), 3, 0, particles, color(r, g, b), false, bounds));

  if (slideButton) slideButton.hide();
  showMrtActionButtons();
  showMrtGraphButtons();

  // Drug-type educational spotlights (fire once per session)
  if (type === 'fullAgonist') {
    showSpotlight(106, 320, 390, { w: 580, h: 340, r: 12 },
      "Full Agonist — This Solves the Spare Receptors Problem",
      [
        { label: "Why spare receptors make sense now",
          text: "Remember Chapter II? The tissue with spare receptors reached 100% response even when only a fraction of receptors were occupied. Clark had no explanation." },
        { label: "MRT's answer",
          text: "Full agonists bind selectively to R* (the active state). Spare receptors are just R sitting in the background — but spontaneous R⇄R* toggling means some always flip to R* naturally. The drug catches them when they do. You only need to occupy a small fraction because the equilibrium does the rest." }
      ]
    );
  } else if (type === 'partialAgonist') {
    showSpotlight(107, 320, 390, { w: 580, h: 340, r: 12 },
      "Partial Agonist — Where Clark's Model Breaks Down",
      [
        { label: "Clark's prediction",
          text: "Clark's occupancy model said: saturate all receptors → maximum response. A partial agonist saturates all receptors — and still can't reach Emax. Clark had no way to explain this." },
        { label: "MRT's answer",
          text: "Partial agonists bind both R and R* with intermediate affinity. They push the equilibrium toward R* — but not all the way. Even at 100% occupancy, the population is split between R and R*. That split caps the response below Emax. It's not about how many receptors are occupied; it's about which state they're in." }
      ]
    );
  }
}

function showMrtActionButtons() {
  hideMrtActionButtons();
  const btnStyle = btn => {
    btn.style('color', '#fff');
    btn.style('border', 'none');
    btn.style('border-radius', '7px');
    btn.style('font-size', '11px');
    btn.style('font-weight', '700');
    btn.style('cursor', 'pointer');
    btn.parent('wrapper');
  };

  mrtWashButton = createButton('Wash ligand away');
  mrtWashButton.position(160, 590);
  mrtWashButton.size(140, 34);
  mrtWashButton.style('background-color', '#5B8DD9');
  btnStyle(mrtWashButton);
  mrtWashButton.mousePressed(washMrtLigand);

  mrtChooseButton = createButton('Choose another ligand');
  mrtChooseButton.position(310, 590);
  mrtChooseButton.size(160, 34);
  mrtChooseButton.style('background-color', '#7878C0');
  btnStyle(mrtChooseButton);
  mrtChooseButton.mousePressed(chooseMrtDrug);

  mrtLigandSlider = createSlider(1, 80, 1, 1);
  mrtLigandSlider.position(110, 518);
  mrtLigandSlider.style('width', '420px');
  mrtLigandSlider.parent('wrapper');

  // Reset receptors to 50:50 R⇌R* whenever concentration changes
  mrtLigandSlider.input(() => {
    for (let i = 0; i < MRT_N; i++) mrtBasalStates[i] = (i % 2 === 0);
  });

}

function hideMrtActionButtons() {
  if (mrtWashButton)   { mrtWashButton.remove();   mrtWashButton   = null; }
  if (mrtChooseButton) { mrtChooseButton.remove();  mrtChooseButton = null; }
  if (mrtLigandSlider) { mrtLigandSlider.remove();  mrtLigandSlider = null; }
}

function showMrtGraphButtons() {
  hideMrtGraphButtons();
  const s = btn => {
    btn.style('color', '#fff');
    btn.style('border', 'none');
    btn.style('border-radius', '7px');
    btn.style('font-size', '11px');
    btn.style('font-weight', '700');
    btn.style('cursor', 'pointer');
    btn.parent('wrapper');
  };

  mrtPlotButton = createButton('Record It!');
  mrtPlotButton.position(80, 548);
  mrtPlotButton.size(120, 32);
  mrtPlotButton.style('background-color', '#3a9e6e');
  s(mrtPlotButton);
  mrtPlotButton.mousePressed(handleMrtPlotPoint);

  mrtFitButton = createButton('Draw the Curve!');
  mrtFitButton.position(210, 548);
  mrtFitButton.size(120, 32);
  mrtFitButton.style('background-color', '#3a6ea5');
  s(mrtFitButton);
  mrtFitButton.mousePressed(handleMrtFitSigmoid);

  mrtDoneButton = createButton('Done →');
  mrtDoneButton.position(340, 548);
  mrtDoneButton.size(100, 32);
  mrtDoneButton.style('background-color', '#888');
  s(mrtDoneButton);
  mrtDoneButton.mousePressed(handleMrtDoneDrug);
}

function hideMrtGraphButtons() {
  if (mrtPlotButton) { mrtPlotButton.remove(); mrtPlotButton = null; }
  if (mrtFitButton)  { mrtFitButton.remove();  mrtFitButton  = null; }
  if (mrtDoneButton) { mrtDoneButton.remove();  mrtDoneButton = null; }
}

function handleMrtPlotPoint() {
  if (!mrtLigandSlider || mrtDrugType === 'none') return;
  const x = int(mrtLigandSlider.value());
  const y = mrtEma;
  if (!mrtPlotData[mrtDrugType]) mrtPlotData[mrtDrugType] = [];
  // overwrite if same x
  const existing = mrtPlotData[mrtDrugType].findIndex(p => p.x === x);
  if (existing >= 0) mrtPlotData[mrtDrugType][existing].y = y;
  else {
    mrtPlotData[mrtDrugType].push({ x, y });
    mrtPlotData[mrtDrugType].sort((a, b) => a.x - b.x);
  }
}

function handleMrtFitSigmoid() {
  const pts = mrtPlotData[mrtDrugType];
  if (!pts || pts.length < 5) { alert('Plot at least 5 points first.'); return; }
  const fit = fitMrtHill(pts, mrtDrugType);
  if (fit) mrtFittedCurves[mrtDrugType] = fit;
}

function handleMrtDoneDrug() {
  mrtCompletedDrugs.add(mrtDrugType);
  hideMrtGraphButtons();
  hideMrtActionButtons();
  particles = [];
  attachedLigands = new Array(rectangles.length).fill(null);
  for (let i = 0; i < MRT_N; i++) mrtBasalStates[i] = (i % 2 === 0);
  mrtActivityHistory = [];
  mrtEma = 0.5;
  mrtDrugType = 'none';
  mrtPhase = 'selecting';
  detachmentTime = 20;

  if (mrtCompletedDrugs.size >= 5) {
    if (slideButton) {
      slideButton.html('Analyse →');
      slideButton.position(240, 590);
      slideButton.style('width', '160px');
      slideButton.style('height', '40px');
      slideButton.style('line-height', '20px');
      slideButton.style('font-size', '14px');
      slideButton.show();
    }
  } else {
    showMrtDrugButtons();
  }
}

// Hill fit for MRT — x is slider count (1-80), y is EMA (0-1)
// Supports curves that go below basal (inverse agonists) via a floor parameter
function fitMrtHill(points, drugType) {
  const isInverse = drugType === 'fullInverse' || drugType === 'partialInverse';
  const ecSteps = 40, nSteps = 20;
  let best = null;

  for (let i = 0; i < ecSteps; i++) {
    const t = i / (ecSteps - 1);
    const EC50 = Math.pow(10, Math.log10(MRT_SLIDER_MIN) + t * (Math.log10(MRT_SLIDER_MAX) - Math.log10(MRT_SLIDER_MIN)));
    for (let j = 0; j < nSteps; j++) {
      const n = 0.5 + (j / (nSteps - 1)) * 2.5;
      // Hill fraction 0→1 as x increases
      let Emax, Floor;
      if (isInverse) {
        // response = basal - (basal - floor) * hill
        // find optimal floor for this EC50, n
        let num = 0, den = 0;
        for (const p of points) {
          const h = hillG(p.x, EC50, n);
          num += (0.5 - p.y) * h;
          den += h * h;
        }
        const delta = den > 0 ? num / den : 0;
        Floor = Math.max(0, 0.5 - delta);
        Emax = 0.5;
      } else {
        // response = basal + (Emax - basal) * hill
        let num = 0, den = 0;
        for (const p of points) {
          const h = hillG(p.x, EC50, n);
          num += (p.y - 0.5) * h;
          den += h * h;
        }
        const delta = den > 0 ? num / den : 0;
        Emax = Math.min(1, 0.5 + delta);
        Floor = 0.5;
      }
      let sse = 0;
      for (const p of points) {
        const h = hillG(p.x, EC50, n);
        const yhat = isInverse ? Emax - (Emax - Floor) * h : Floor + (Emax - Floor) * h;
        sse += (p.y - yhat) ** 2;
      }
      const mse = sse / points.length;
      if (!best || mse < best.mse) best = { Emax, Floor, EC50, n, mse, isInverse };
    }
  }
  return best;
}

function sampleMrtCurve(params, gx1, gx2, gy1, gy2) {
  const gw = gx2 - gx1, gh = gy2 - gy1;
  const logMin = Math.log10(MRT_SLIDER_MIN), logMax = Math.log10(MRT_SLIDER_MAX);
  const out = [];
  for (let x = MRT_SLIDER_MIN; x <= MRT_SLIDER_MAX; x += 0.5) {
    const h = hillG(x, params.EC50, params.n);
    const y = params.isInverse
      ? params.Emax - (params.Emax - params.Floor) * h
      : params.Floor + (params.Emax - params.Floor) * h;
    out.push({
      x: gx1 + (Math.log10(x) - logMin) / (logMax - logMin) * gw,
      y: gy2 - y * gh,
    });
  }
  return out;
}

function washMrtLigand() {
  // Clears this drug's data so student can redo it
  if (mrtDrugType !== 'none') {
    delete mrtPlotData[mrtDrugType];
    delete mrtFittedCurves[mrtDrugType];
    mrtCompletedDrugs.delete(mrtDrugType);
  }
  hideMrtActionButtons();
  hideMrtGraphButtons();
  particles = [];
  attachedLigands = new Array(rectangles.length).fill(null);
  for (let i = 0; i < MRT_N; i++) mrtBasalStates[i] = (i % 2 === 0);
  mrtDrugType = 'none';
  mrtPhase = 'selecting';
  detachmentTime = 20;
  mrtActivityHistory = [];
  mrtEma = 0.5;
  showMrtDrugButtons();
}

function chooseMrtDrug() {
  hideMrtActionButtons();
  hideMrtGraphButtons();
  particles = [];
  attachedLigands = new Array(rectangles.length).fill(null);
  for (let i = 0; i < MRT_N; i++) mrtBasalStates[i] = (i % 2 === 0);
  mrtDrugType = 'none';
  mrtPhase = 'selecting';
  detachmentTime = 20;
  mrtActivityHistory = [];
  mrtEma = 0.5;
  showMrtDrugButtons();
}


// ─────────────────────────────────────────────────────────────────────────────
// MRT — Slide 1: Basal Activity
// Left: dual Gaussian frequency distribution (R vs R*)
// Right: animated 6-receptor membrane, live toggling, % counter
// ─────────────────────────────────────────────────────────────────────────────
function drawMrtBridgeSlide() {
  const BG   = [10, 42, 38];
  const TEAL = [46, 196, 160];
  const LT   = [180, 240, 225];
  const MID  = [120, 200, 180];
  const DIM  = [80, 150, 135];

  background(BG[0], BG[1], BG[2]);

  // Top accent bar
  fill(TEAL[0], TEAL[1], TEAL[2]); noStroke();
  rect(0, 0, 1280, 6);

  // Chapter label
  noStroke(); fill(TEAL[0], TEAL[1], TEAL[2]);
  textAlign(CENTER); textStyle(BOLD); textSize(11);
  text('CHAPTER III  ·  MODERN RECEPTOR THEORY', 640, 54);
  textStyle(NORMAL);

  // Big recap heading
  fill(LT[0], LT[1], LT[2]);
  textSize(40); textStyle(BOLD); textAlign(CENTER);
  text("You've seen the two-state model.", 640, 148);
  textStyle(NORMAL);

  // R ⇄ R* recap equation — large, centered
  const eqX = 640, eqY = 222;
  fill(210, 55, 55); textSize(32); textStyle(BOLD); textAlign(RIGHT);
  text('R', eqX - 58, eqY);
  fill(60, 195, 75); textAlign(LEFT);
  text('R*', eqX + 58, eqY);
  textStyle(NORMAL);
  // arrows
  stroke(60, 195, 75); strokeWeight(2.5);
  line(eqX - 36, eqY - 12, eqX + 36, eqY - 12);
  line(eqX + 28, eqY - 18, eqX + 36, eqY - 12);
  line(eqX + 28, eqY - 6,  eqX + 36, eqY - 12);
  stroke(210, 55, 55); strokeWeight(2.5);
  line(eqX + 36, eqY - 2, eqX - 36, eqY - 2);
  line(eqX - 28, eqY - 8,  eqX - 36, eqY - 2);
  line(eqX - 28, eqY + 4,  eqX - 36, eqY - 2);
  noStroke(); fill(MID[0], MID[1], MID[2]); textStyle(ITALIC); textSize(13); textAlign(CENTER);
  text('receptors toggle spontaneously — no drug needed', eqX, eqY + 26);
  textStyle(NORMAL);

  // Divider
  stroke(TEAL[0], TEAL[1], TEAL[2], 60); strokeWeight(1);
  line(240, 278, 1040, 278);

  // What's next section
  noStroke(); fill(LT[0], LT[1], LT[2]); textSize(22); textStyle(BOLD); textAlign(CENTER);
  text("Now let's see what happens when we add a drug.", 640, 326);
  textStyle(NORMAL);

  // Three-step preview cards
  const cards = [
    { icon: '⚗', title: 'Pick a drug type', body: 'Five classes of drugs,\nfive different outcomes.' },
    { icon: '📈', title: 'Build a curve',     body: 'Adjust ligand count, record\nR* activity at each dose.' },
    { icon: '🔬', title: 'See the pattern',   body: 'The curve shape tells you\nhow the drug shifts R⇄R*.' },
  ];
  const cardW = 280, cardH = 150, cardY = 390, gap = 32;
  const totalW = cards.length * cardW + (cards.length - 1) * gap;
  const startX = (1280 - totalW) / 2;
  for (let i = 0; i < cards.length; i++) {
    const cx = startX + i * (cardW + gap);
    fill(18, 68, 58); noStroke();
    rect(cx, cardY, cardW, cardH, 10);
    stroke(TEAL[0], TEAL[1], TEAL[2], 80); strokeWeight(1);
    rect(cx, cardY, cardW, cardH, 10);
    noStroke();
    fill(LT[0], LT[1], LT[2]); textSize(13); textStyle(BOLD); textAlign(LEFT);
    text(cards[i].icon + '  ' + cards[i].title, cx + 20, cardY + 34);
    textStyle(NORMAL); fill(MID[0], MID[1], MID[2]); textSize(11);
    text(cards[i].body, cx + 20, cardY + 60);
  }

  // Bottom accent bar
  fill(TEAL[0], TEAL[1], TEAL[2]); noStroke();
  rect(0, 714, 1280, 6);

  // Ensure button is visible and correctly configured every frame
  if (slideButton) {
    slideButton.html('Start investigating →');
    slideButton.style('width', '200px');
    slideButton.style('background-color', '#2EC4A0');
    slideButton.style('color', '#0f2e28');
    slideButton.position(540, 648);
    slideButton.show();
  }
  if (gotItButton) gotItButton.hide();
}

function drawMrtDrugOverlay() {
  // Full-canvas dim
  fill(0, 0, 0, 165); noStroke();
  rect(0, 0, width, height);

  // Panel — same geometry as Clark overlay
  const px = 190, py = 90, pw = 900, ph = 540;
  fill(20, 80, 68); noStroke(); rect(px, py, pw, 58, 10, 10, 0, 0);
  fill(12, 48, 42); stroke(46, 196, 160, 60); strokeWeight(1);
  rect(px, py + 58, pw, ph - 58, 0, 0, 10, 10);

  // Header
  noStroke(); fill(180, 240, 225);
  textAlign(CENTER); textStyle(BOLD); textSize(14);
  text("MS. FRIZZLE'S FIELD NOTES  \xB7  CHAPTER III  \xB7  LAB GOAL", px + pw / 2, py + 26);
  textStyle(NORMAL); fill(120, 200, 180); textSize(11);
  text('"Now you\'re going to build the evidence yourself."', px + pw / 2, py + 46);

  const tx = px + 52;
  const lh = 22;
  let ty = py + 88;

  noStroke(); fill(180, 240, 225);
  textAlign(CENTER); textStyle(BOLD); textSize(20);
  text('Your Goal: 5 Drugs. 5 Curves.', px + pw / 2, ty);
  textStyle(NORMAL); ty += 16;

  stroke(46, 196, 160, 55); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 24;

  fill(160, 225, 210); textAlign(LEFT); textSize(14);
  text('There are five drug types below. For each one:', tx, ty); ty += lh * 1.3;

  const steps = [
    ['1.', 'Pick the drug type from the buttons at the bottom.'],
    ['2.', 'Drag the slider to change how many ligands are in the environment.'],
    ['3.', 'Watch the R* activity stabilise on the graph, then hit Record It!'],
    ['4.', 'Repeat at different concentrations until you have 5 data points.'],
    ['5.', 'Hit Draw the Curve — then move on to the next drug type.'],
  ];
  for (const [num, desc] of steps) {
    fill(46, 196, 160); textStyle(BOLD); textSize(13); textAlign(LEFT);
    text(num, tx + 10, ty);
    fill(180, 240, 225); textStyle(NORMAL); textSize(13);
    text(desc, tx + 34, ty);
    ty += lh * 1.2;
  }

  ty += 8;
  stroke(46, 196, 160, 55); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 22;

  fill(120, 200, 180); textStyle(ITALIC); textSize(13); textAlign(LEFT);
  text('Once all five curves are done, we\'ll analyse them together and see', tx, ty); ty += lh;
  text('what each drug type reveals about the R⇄R* equilibrium.', tx, ty);
  textStyle(NORMAL);
}

function drawMrtBasalOverlay() {
  fill(0, 0, 0, 165); noStroke();
  rect(0, 0, width, height);

  const px = 190, py = 90, pw = 900, ph = 540;
  fill(20, 80, 68); noStroke(); rect(px, py, pw, 58, 10, 10, 0, 0);
  fill(250, 248, 240); stroke(175, 210, 200); strokeWeight(1);
  rect(px, py + 58, pw, ph - 58, 0, 0, 10, 10);

  noStroke(); fill(180, 240, 225);
  textAlign(CENTER); textStyle(BOLD); textSize(14);
  text("MS. FRIZZLE'S FIELD NOTES  \xB7  CHAPTER III  \xB7  FIELD STOP", px + pw / 2, py + 26);
  textStyle(NORMAL); fill(120, 210, 190); textSize(11);
  text('"Clark was right about binding. He just didn\'t know what happened next."', px + pw / 2, py + 46);

  const tx = px + 52, lh = 22;
  let ty = py + 88;

  noStroke(); fill(30, 38, 80);
  textAlign(CENTER); textStyle(BOLD); textSize(20);
  text('Modern Receptor Theory', px + pw / 2, ty);
  textStyle(NORMAL); ty += 16;

  stroke(175, 210, 200); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 22;

  fill(55, 60, 80); textAlign(LEFT); textSize(14);
  text("Clark modelled receptors as passive locks — inert until a drug came along to turn them on.", tx, ty); ty += lh;
  text("Modern Receptor Theory (MRT) shows that's not the full picture.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text('The Two-State Model', tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("Receptors spontaneously toggle between an inactive state (R) and an active state (R*).", tx, ty); ty += lh;
  text("No drug required. This basal activity is always happening.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text('Your Goal', tx, ty); textStyle(NORMAL); ty += lh;
  fill(55, 60, 80); textSize(14);
  text("Test 5 different drug types on these receptors and build a dose-response curve for each.", tx, ty); ty += lh;
  text("The shift in R* activity will tell you exactly what kind of drug you're dealing with.", tx, ty); ty += lh * 1.8;

  stroke(175, 210, 200); strokeWeight(1);
  line(px + 40, ty - 8, px + pw - 40, ty - 8); noStroke();

  fill(20, 100, 80); textStyle(BOLD); textSize(12);
  text('UP NEXT', tx, ty + 4);
  fill(55, 60, 80); textStyle(NORMAL); textSize(13);
  text('A quick tour of the lab.', tx + 76, ty + 4);
}

function drawMrtBasalScene() {
  // Advance flip animation — drug phase is 2× faster
  mrtFlipTimer++;
  if (mrtFlipTimer >= (mrtPhase === 'drug' ? 15 : 30)) {
    mrtFlipTimer = 0;
    // Count only FREE (unbound) receptors — bound ones are locked, invisible to equilibrium
    const freeRStar = mrtBasalStates.filter((s, i) =>  s && attachedLigands[i] === null).length;
    const freeR     = mrtBasalStates.filter((s, i) => !s && attachedLigands[i] === null).length;
    const freeCount = freeRStar + freeR;

    let goUp;

    // Free receptors always equilibrate 50:50 regardless of drug type
    // Drug effect comes purely from binding affinity, not from biasing free receptor flips
    const freeTarget = freeCount / 2;
    if      (freeRStar >= freeTarget + 1) goUp = false;
    else if (freeRStar <= freeTarget - 1) goUp = true;
    else                                   goUp = Math.random() < 0.5;

    // Only flip free (unbound) receptors
    const cands = mrtBasalStates
      .map((s, i) => {
        if (attachedLigands[i] !== null) return -1; // locked
        return goUp ? (!s ? i : -1) : (s ? i : -1);
      })
      .filter(i => i >= 0);

    if (cands.length > 0) {
      const idx = cands[floor(random(cands.length))];
      mrtBasalStates[idx] = !mrtBasalStates[idx];
    }

    // EMA: slower in basal (tight smoothing), faster in drug phase (show shift clearly)
    const alpha = mrtPhase === 'drug' ? 0.4 : 0.25;
    mrtEma = mrtEma * (1 - alpha) + (mrtBasalStates.filter(s => s).length / MRT_N) * alpha;
    mrtActivityHistory.push(mrtEma);
    if (mrtActivityHistory.length > 40) mrtActivityHistory.shift();
  }

  // ── Bridge slide (full-screen transition between basal and drug phase) ──
  if (mrtPhase === 'bridge') {
    drawMrtBridgeSlide();
    return;
  }

  // ── Background (matches heartGraph) ──
  background(173, 216, 230);

  // ── Red divider (matches heartGraph) ──
  stroke(255, 0, 0); strokeWeight(3);
  line(640, 0, 640, 720);

  // ── LEFT PANEL ──
  fill(255); noStroke();
  rect(0, 0, 640, 720);

  const rStarNow = mrtBasalStates.filter(s => s).length;
  const pctNow = round(rStarNow / MRT_N * 100);

  if (mrtPhase === 'basal') {
    // ── BASAL: time-series view ──
    noStroke(); textAlign(CENTER);
    fill(40, 45, 70); textStyle(BOLD); textSize(16);
    text('Modern Receptor Theory  —  Intestinal Muscle', 320, 36);
    textStyle(NORMAL);
    fill(90, 95, 120); textSize(13);
    text('The Receptors at Rest — No Drug Present', 320, 58);

    const eqCX = 320, eqCY = 100;
    fill(210, 55, 55); textSize(22); textStyle(BOLD); textAlign(RIGHT);
    text('R', eqCX - 52, eqCY + 8);
    fill(60, 195, 75); textAlign(LEFT);
    text('R*', eqCX + 52, eqCY + 8);
    textStyle(NORMAL);
    stroke(60, 195, 75); strokeWeight(2.5);
    line(eqCX - 32, eqCY - 8, eqCX + 32, eqCY - 8);
    line(eqCX + 24, eqCY - 14, eqCX + 32, eqCY - 8);
    line(eqCX + 24, eqCY - 2,  eqCX + 32, eqCY - 8);
    stroke(210, 55, 55); strokeWeight(2.5);
    line(eqCX + 32, eqCY + 3, eqCX - 32, eqCY + 3);
    line(eqCX - 24, eqCY - 3,  eqCX - 32, eqCY + 3);
    line(eqCX - 24, eqCY + 9,  eqCX - 32, eqCY + 3);
    noStroke(); fill(130, 120, 170); textAlign(CENTER); textStyle(ITALIC); textSize(11);
    text('spontaneous toggling', eqCX, eqCY + 24);
    textStyle(NORMAL);
    fill(60, 65, 90); textAlign(CENTER); textSize(12);
    text('Receptors flicker between inactive (R) and active (R*)', 320, eqCY + 46);
    text('constantly — even with no drug present.', 320, eqCY + 62);

    const px1 = 100, px2 = 580, py1 = 190, py2 = 450;
    const pw = px2 - px1, ph = py2 - py1;
    fill(248); noStroke(); rect(px1, py1, pw, ph);
    stroke(215); strokeWeight(1);
    for (let p = 10; p <= 100; p += 10) line(px1, py2 - (p/100)*ph, px2, py2 - (p/100)*ph);
    stroke(160); strokeWeight(1.5);
    line(px1, py1, px1, py2); line(px1, py2, px2, py2);
    noStroke(); fill(110); textSize(9); textAlign(RIGHT);
    for (let p = 10; p <= 100; p += 10) text(p+'%', px1-5, py2-(p/100)*ph+4);
    if (mrtActivityHistory.length > 1) {
      stroke(60, 195, 75); strokeWeight(2); noFill(); beginShape();
      for (let i = 0; i < mrtActivityHistory.length; i++)
        vertex(px1 + (i/(40-1))*pw, py2 - mrtActivityHistory[i]*ph);
      endShape();
    }
    noStroke(); fill(120, 125, 150); textSize(12); textAlign(LEFT);
    text('No drug present', 100, 472);
    fill(60, 195, 75); textStyle(BOLD); textAlign(RIGHT); textSize(12);
    text(`${rStarNow} / ${MRT_N} in R*  (${pctNow}% active)`, 580, 472);
    textStyle(NORMAL);
    // Instruction text is rendered as a DOM element in showMrtDrugButtons()

  } else {
    // ── SELECTING / DRUG PHASE: dose-response graph ──
    const drugNames = { fullAgonist: 'Full Agonist', partialAgonist: 'Partial Agonist',
      antagonist: 'Antagonist', partialInverse: 'Partial Inv. Agonist', fullInverse: 'Full Inv. Agonist' };
    const [dr, dg, db] = mrtPhase === 'selecting' ? [46, 196, 160] : mrtDrugColor;

    noStroke(); textAlign(CENTER);
    fill(40, 45, 70); textStyle(BOLD); textSize(16);
    text('Modern Receptor Theory  —  Dose-Response', 320, 24);
    textStyle(NORMAL); fill(90, 95, 120); textSize(12);
    text(mrtPhase === 'selecting'
      ? 'Pick a drug type below to start building curves.'
      : 'Adjust ligand count, let EMA settle, then Record It!', 320, 44);

    // Graph bounds — tall, with room above AND below basal
    const gx1 = 80, gx2 = 560, gy1 = 58, gy2 = 460;
    const gw = gx2 - gx1, gh = gy2 - gy1;
    const logMin = Math.log10(MRT_SLIDER_MIN), logMax = Math.log10(MRT_SLIDER_MAX);
    const basalY = gy2 - 0.5 * gh; // 50% mark

    fill(248); noStroke(); rect(gx1, gy1, gw, gh);

    // Gridlines every 10%
    stroke(215); strokeWeight(1);
    for (let p = 0; p <= 100; p += 10) {
      if (p === 50) continue;
      line(gx1, gy2 - (p/100)*gh, gx2, gy2 - (p/100)*gh);
    }
    // Basal dashed line
    stroke(160, 160, 210); strokeWeight(1.5);
    for (let dx = 0; dx < gw; dx += 10) line(gx1+dx, basalY, gx1+dx+5, basalY);

    // Axes
    stroke(160); strokeWeight(1.5);
    line(gx1, gy1, gx1, gy2); line(gx1, gy2, gx2, gy2);

    // Y-axis labels
    noStroke(); fill(110); textSize(9); textAlign(RIGHT);
    for (let p = 0; p <= 100; p += 20) text(p+'%', gx1-5, gy2-(p/100)*gh+3);
    noStroke(); fill(155, 155, 205); textSize(8); textAlign(LEFT);
    text('basal (50%)', gx1+4, basalY-4);

    // X-axis log ticks
    const xTicks = [1, 2, 5, 10, 20, 50, 80];
    noStroke(); fill(110); textSize(9); textAlign(CENTER);
    for (const v of xTicks) {
      const lx = gx1 + (Math.log10(v)-logMin)/(logMax-logMin)*gw;
      stroke(215); strokeWeight(1); line(lx, gy2, lx, gy2+4); noStroke();
      text(v, lx, gy2+14);
    }

    // Axis labels
    noStroke(); fill(60, 65, 90); textSize(11); textAlign(CENTER);
    text('Ligand count (log scale)', gx1+gw/2, gy2+18);
    push(); translate(gx1-38, gy1+gh/2); rotate(-HALF_PI);
    noStroke(); fill(60, 65, 90); textSize(11); textAlign(CENTER);
    text('% R* active', 0, 0); pop();

    // Empty-state prompt when no drug selected yet
    if (mrtPhase === 'selecting' && Object.values(mrtPlotData).every(p => !p.length)) {
      noStroke(); fill(180, 185, 210); textSize(13); textAlign(CENTER); textStyle(ITALIC);
      text('Your curves will appear here as you test each drug type.', gx1 + gw / 2, gy1 + gh / 2 - 10);
      textStyle(NORMAL);
    }

    // Draw all completed + current drug curves and points
    for (const [dtype, pts] of Object.entries(mrtPlotData)) {
      if (!pts.length) continue;
      const drug = MRT_DRUGS.find(d => d.type === dtype);
      if (!drug) continue;
      const dc = color(drug.r, drug.g, drug.b);
      // Points
      fill(dc); noStroke();
      for (const pt of pts) {
        const px = gx1 + (Math.log10(pt.x)-logMin)/(logMax-logMin)*gw;
        const py = gy2 - pt.y * gh;
        circle(px, py, 9);
      }
      // Fitted curve
      if (mrtFittedCurves[dtype]) {
        const curve = sampleMrtCurve(mrtFittedCurves[dtype], gx1, gx2, gy1, gy2);
        stroke(dc); strokeWeight(2.5); noFill();
        beginShape(); for (const p of curve) vertex(p.x, p.y); endShape();
        // Label at end of curve
        const last = curve[curve.length-1];
        noStroke(); fill(dc); textStyle(BOLD); textSize(9); textAlign(LEFT);
        text(drug.label, min(last.x+4, gx2-65), last.y+3);
        textStyle(NORMAL);
      }
    }

    // Ghost point — live preview of current (slider, EMA)
    if (mrtLigandSlider) {
      const sliderVal = int(mrtLigandSlider.value());
      const ghostX = gx1 + (Math.log10(max(1,sliderVal))-logMin)/(logMax-logMin)*gw;
      const ghostY = gy2 - mrtEma * gh;
      fill(dr, dg, db, 120); noStroke();
      circle(ghostX, ghostY, 12);
      // dashed crosshairs
      stroke(dr, dg, db, 80); strokeWeight(1);
      for (let dx2 = 0; dx2 < ghostX-gx1; dx2+=8) line(gx1+dx2, ghostY, gx1+dx2+4, ghostY);
      for (let dy2 = 0; dy2 < gy2-ghostY; dy2+=8) line(ghostX, ghostY+dy2, ghostX, ghostY+dy2+4);
    }

    // Status bar (only when a drug is active)
    if (mrtPhase === 'drug') {
      const pts = mrtPlotData[mrtDrugType] || [];
      const sliderVal2 = mrtLigandSlider ? int(mrtLigandSlider.value()) : 1;
      noStroke(); fill(dr, dg, db); textStyle(BOLD); textSize(11); textAlign(LEFT);
      text(drugNames[mrtDrugType]||'', gx1, gy2+32);
      textStyle(NORMAL); fill(60, 195, 75); textStyle(BOLD); textAlign(RIGHT); textSize(11);
      text(`${pctNow}% R* active  ·  ${pts.length}/5 pts`, gx2, gy2+32);
      textStyle(NORMAL);
      if (pts.length < 5) {
        noStroke(); fill(130,135,160); textSize(9); textAlign(CENTER);
        text(`Record ${5-pts.length} more point${5-pts.length!==1?'s':''} to unlock Draw the Curve!`, gx1+gw/2, gy2+44);
      }
      noStroke(); fill(80,85,110); textSize(10); textAlign(LEFT);
      text('Ligand count:', gx1, 510);
      fill(dr, dg, db); textStyle(BOLD); text(sliderVal2, gx1+78, 510); textStyle(NORMAL);
    }

    // Sync particles to slider
    if (mrtLigandSlider) {
      const target = int(mrtLigandSlider.value());
      const bounds = { x: 645, y: 305, w: 630, h: 260 };
      if (particles.length < target) {
        for (let i = particles.length; i < target; i++) {
          const pos = createVector(random(bounds.x+20,bounds.x+bounds.w-20), random(bounds.y+20,bounds.y+bounds.h-20));
          particles.push(new Ball(pos, p5.Vector.random2D().mult(3), 3, i, particles, color(dr,dg,db), false, bounds));
        }
      } else if (particles.length > target) {
        let rem = particles.length - target;
        particles = particles.filter(p => { if (rem>0&&p.attachedRectIndex===-1&&!p.animating){rem--;return false;} return true; });
      }
    }

  }

  // ── RIGHT PANEL (matches heartGraph exactly) ──
  image(Smallintestine, 740, -20, 400, 300);

  noFill(); stroke(255, 255, 102);
  rect(945, 80, 30, 30);
  line(945, 110, 640, 300);
  line(975, 110, 1280, 300);
  line(640, 300, 1280, 300);

  image(membrane, 640, 572, 640, 40);

  // 10 GPCRs — R or R* state
  const { gpcrPos, rects: hitRects, size } = getMrtLayout();
  for (let i = 0; i < MRT_N; i++) {
    const p = gpcrPos[i];
    const isActive = mrtBasalStates[i];

    if (isActive) {
      noTint();
      image(gpcr, p.x, p.y, size, size);
    } else {
      tint(210, 80, 80);
      image(gpcr, p.x, p.y, size, size);
      noTint();
    }

    // State label above each GPCR
    noStroke(); textAlign(CENTER); textStyle(BOLD); textSize(11);
    fill(isActive ? color(60, 195, 75) : color(210, 55, 55));
    text(isActive ? 'R*' : 'R', p.x + size / 2, 553);
    textStyle(NORMAL);
  }

  // Hitboxes — teal for R*, purple for R
  noStroke();
  for (let i = 0; i < hitRects.length; i++) {
    fill(mrtBasalStates[i] ? color(60, 195, 75) : color(210, 55, 55));
    rect(hitRects[i].x, hitRects[i].y, hitRects[i].w, hitRects[i].h);
  }

  // Balls
  for (let a of particles) {
    a.bounceOthers();
    a.update();
    a.display();
  }

  // Overlay + tour — drawn last so they sit on top of all sprites
  if (showMrtBasalOverlay) {
    drawMrtBasalOverlay();
    if (gotItButton) gotItButton.show();
    if (slideButton) slideButton.hide();
  } else if (showMrtDrugOverlay) {
    drawMrtDrugOverlay();
    if (gotItButton) {
      gotItButton.html('Got it — let\'s build curves!');
      gotItButton.style('width', '230px');
      gotItButton.position(852, 578);
      gotItButton.show();
    }
    if (slideButton) slideButton.hide();
  } else if (activeSpotlight) {
    if (gotItButton) gotItButton.hide();
    if (slideButton) slideButton.hide();
  } else {
    if (gotItButton && (mrtPhase === 'selecting' || mrtPhase === 'drug')) gotItButton.hide();
  }
  drawSpotlight();
}


function handleSlideButtonClick() {
  if (scene === 'clarkIntro') {
    if (clarkIntroPage < 2) {
      clarkIntroPage++;
    } else {
      initializeScene('achGraph');
    }
  } else if (scene === 'tissueTransition') {
    initializeScene('heartGraph');
  } else if (scene === 'dataCollected') {
    initializeScene('compareGraphs');
  } else if (scene === 'compareGraphs') {
    initializeScene('limitationTitle');
  } else if (scene === 'limitationTitle') {
    initializeScene('partialGraph');
  } else if (scene === 'clarkProblems') {
    if (problemPage === 1) {
      problemPage = 2;
      slideButton.html('Next Chapter →');
      slideButton.style('width', '194px');
    } else {
      initializeScene('mrtTitle');
    }
  } else if (scene === 'partialGraph') {
    initializeScene('mrtTitle');
  } else if (scene === 'ch2Analysis') {
    initializeScene('mrtTitle');
  } else if (scene === 'mrtTitle') {
    initializeScene('mrtBasal');
  } else if (scene === 'mrtBasal') {
    if (mrtPhase === 'basal') {
      mrtPhase = 'bridge';
      if (slideButton) {
        slideButton.html('Start investigating →');
        slideButton.style('width', '190px');
        slideButton.position(545, 645);
        slideButton.show();
      }
    } else if (mrtPhase === 'bridge') {
      mrtPhase = 'selecting';
      showMrtDrugOverlay = true;
      if (slideButton) slideButton.hide();
      showMrtDrugButtons();
    } else if (mrtCompletedDrugs.size >= 5) {
      initializeScene('mrtAnalysis');
    }
  } else if (scene === 'mrtAnalysis') {
    initializeScene('mrtPrinciples');
  } else if (scene === 'mrtPrinciples') {
    initializeScene('intro');
  }
}

function drawClarkIntroScene() {
  background(255, 210, 0);

  // Dark top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 72);

  fill(255, 210, 0); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  ·  CHAPTER I', width / 2, 30);
  textStyle(NORMAL);
  fill(255, 210, 100); textSize(12);
  text('Field Trip: Drug-Receptor Binding', width / 2, 54);

  noStroke(); textAlign(CENTER);

  if (clarkIntroPage === 1) {

    // ── Table of Contents ───────────────────────────────────────────────
    const TOC = [
      { x: 200,  label: 'The Clark Model',       sub: 'Dose-response experiments\n& occupancy theory',     col: [21,  38,  120] },
      { x: 480,  label: 'Where Clark Fell Short', sub: 'Spare receptors\n& partial agonism',               col: [204, 60,  60]  },
      { x: 760,  label: 'Modern Receptor Theory', sub: 'Efficacy, the two-state\nmodel & drug types',      col: [32,  168, 140] },
      { x: 1040, label: 'The Comparison',         sub: 'Clark vs. MRT — what\nchanged and why it matters', col: [120, 80,  220] },
    ];

    // ── Page heading ───────────────────────────────────────────────────
    noStroke(); fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(26);
    text('Your Field Trip, at a Glance', width / 2, 112);
    textStyle(NORMAL);
    fill(40, 28, 0); textSize(15);
    text('Four stops. Each one builds on the last.', width / 2, 146);

    // ── Card layout constants ──────────────────────────────────────────
    const cardW = 228, cardH = 242;
    const cardTopY = 218;   // pushed down to clear subtitle
    const badgeR  = 28;     // number badge radius

    // ── Connecting arrows between cards ───────────────────────────────
    const arrowY = cardTopY + cardH / 2;
    for (let i = 0; i < TOC.length - 1; i++) {
      const ax = (TOC[i].x + TOC[i + 1].x) / 2;
      fill(40, 28, 0, 160); noStroke();
      triangle(ax - 7, arrowY - 8, ax - 7, arrowY + 8, ax + 11, arrowY);
    }

    // ── Cards ─────────────────────────────────────────────────────────
    for (let i = 0; i < TOC.length; i++) {
      const s   = TOC[i];
      const [r, g, b] = s.col;
      const cLeft = s.x - cardW / 2;

      // Card background
      fill(r, g, b, 32); stroke(r, g, b); strokeWeight(2.5);
      rect(cLeft, cardTopY, cardW, cardH, 14);

      // Badge circle — centred on the top edge of the card
      fill(r, g, b); stroke(255, 255, 255, 220); strokeWeight(3);
      circle(s.x, cardTopY, badgeR * 2);

      // Number inside badge
      fill(255); noStroke(); textStyle(BOLD); textSize(22); textAlign(CENTER);
      text(i + 1, s.x, cardTopY + 8);

      // Chapter title
      fill(15, 22, 55); textStyle(BOLD); textSize(13); textAlign(CENTER);
      text(s.label, s.x, cardTopY + 62);
      textStyle(NORMAL);

      // Divider
      stroke(r, g, b, 110); strokeWeight(1.5);
      line(cLeft + 20, cardTopY + 74, cLeft + cardW - 20, cardTopY + 74);
      noStroke();

      // Sub-text — x is LEFT edge of box so wrapping stays inside the card
      fill(40, 30, 0); textSize(12); textAlign(CENTER);
      text(s.sub, cLeft + 16, cardTopY + 96, cardW - 32, 100);
    }

    // ── Animated bus driving left → right below cards ─────────────────
    const busW = 140, busH = 44;
    stroke(40, 28, 0, 50); strokeWeight(2);
    line(0, 548, width, 548); noStroke();
    const totalTravel = width + busW + 80;
    const busX = -busW + ((millis() * 0.13) % totalTravel);
    drawMSBus(busX, 494, busW, busH);

  } else if (clarkIntroPage === 2) {
    fill(15, 22, 55);
    textStyle(BOLD); textSize(36);
    text("Here's what you're doing on this trip.", width / 2, 155);
    textStyle(NORMAL);

    stroke(15, 22, 55); strokeWeight(1.5);
    line(200, 180, 1080, 180); noStroke();

    fill(40, 28, 0); textSize(18);
    text("You'll run two experiments with two different tissues using the same ligand (acetylcholine, Chari's Favourite!).", width / 2, 236);
    text('Your goal is to build your OWN dose-response curve.', width / 2, 264);
    text('Set a concentration, let the receptors respond, and record data points. Repeat.', width / 2, 292);
    text('Fit the curve, and watch two different dose-response curves form.', width / 2, 320);

    fill(15, 22, 55); textSize(20); textStyle(ITALIC);
    text("Throughout the experiment, you will stumble upon Clark's postulates, one at a time.", width / 2, 382);
    textStyle(NORMAL);

    fill(40, 28, 0); textSize(18);
    text("We'll then meet back for some analysis!", width / 2, 432);

    drawMSBus(1020, 540, 220, 68);
  }
}

function drawTissueTransitionScene() {
  background(255, 210, 0);

  // Dark top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 72);

  fill(255, 210, 0); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  ·  NEXT STOP', width / 2, 30);
  textStyle(NORMAL);
  fill(255, 210, 100); textSize(12);
  text('Intestinal Smooth Muscle', width / 2, 54);

  noStroke(); textAlign(CENTER);

  // Main statement
  fill(15, 22, 55); textStyle(BOLD); textSize(19);
  text("Let's repeat the same experiment, same ligand, same methodology.", width / 2, 176);
  textStyle(NORMAL);

  stroke(15, 22, 55); strokeWeight(1.5);
  line(200, 200, 1080, 200); noStroke();

  fill(40, 28, 0); textSize(19);
  text('But THIS time — intestinal smooth muscle, with more receptor density than the diaphragm muscle.', width / 2, 268);

  fill(15, 22, 55); textStyle(BOLD); textSize(22);
  text('Watch how the curve changes!', width / 2, 336);
  textStyle(NORMAL);

  fill(55, 40, 0); textSize(17);
  text("We'll come back for some analysis.", width / 2, 400);

  // Animated bus — loops across the bottom of the screen
  const busW = 420, busH = 128;
  const totalTravel = width + busW + 120;
  const busX = -busW + ((millis() * 0.38) % totalTravel);
  drawMSBus(busX, 520, busW, busH);
}

function drawDataCollectedScene() {
  background(255, 210, 0);

  // Dark top bar
  fill(15, 22, 55); noStroke();
  rect(0, 0, width, 72);
  fill(255, 210, 0); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  ·  FIELD STOP COMPLETE', width / 2, 30);
  textStyle(NORMAL); fill(255, 210, 100); textSize(12);
  text('Two tissues. Two curves. Time to think.', width / 2, 54);

  noStroke(); textAlign(CENTER);

  // Big celebratory heading
  fill(15, 22, 55); textStyle(BOLD); textSize(58);
  text('WOW!', width / 2, 192);
  textStyle(NORMAL);

  stroke(15, 22, 55); strokeWeight(2);
  line(320, 218, 960, 218); noStroke();

  fill(15, 22, 55); textStyle(BOLD); textSize(22);
  text("We've collected the data.", width / 2, 272);
  textStyle(NORMAL);

  fill(40, 28, 0); textSize(18);
  text('Two experiments. Two dose-response curves. Same ligand, two very different tissues.', width / 2, 330);

  fill(15, 22, 55); textSize(20); textStyle(BOLD);
  text("Now let's compare and contrast — and think about what Clark's model actually predicts.", width / 2, 400);
  textStyle(NORMAL);

  // Animated bus
  const busW = 380, busH = 116;
  const totalTravel = width + busW + 80;
  const busX = -busW + ((millis() * 0.28) % totalTravel);
  drawMSBus(busX, 490, busW, busH);
}

function drawCompareGraphsScene() {
  background(15, 25, 55);

  // Top bar
  fill(255, 210, 0); noStroke(); rect(0, 0, width, 58);
  fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text("THE MAGIC SCHOOL BUS  \xB7  CLARK'S OCCUPANCY THEORY", width / 2, 24);
  textStyle(NORMAL); fill(40, 28, 0); textSize(11);
  text("Two experiments, one framework — did Clark's model hold up?", width / 2, 44);
  fill(15, 22, 55); noStroke(); rect(0, 58, width, 10);

  // ── Left: graph panel ────────────────────────────────────────
  const gx = 50, gy = 88, gw = 560, gh = 430;

  fill(22, 35, 78); noStroke(); rect(30, 74, 600, 558, 10);

  fill(200, 215, 255); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('Dose–Response Comparison', gx + gw / 2, gy - 10); textStyle(NORMAL);

  // Grid
  stroke(35, 52, 100); strokeWeight(0.8);
  for (let i = 1; i < 5; i++) line(map(i, 0, 5, gx, gx + gw), gy, map(i, 0, 5, gx, gx + gw), gy + gh);
  for (let j = 1; j < 6; j++) line(gx, map(j, 0, 6, gy + gh, gy), gx + gw, map(j, 0, 6, gy + gh, gy));

  // Axes
  stroke(130, 155, 200); strokeWeight(2);
  line(gx, gy, gx, gy + gh);
  line(gx, gy + gh, gx + gw, gy + gh);

  // Labels
  noStroke(); fill(160, 180, 220); textAlign(CENTER); textSize(11);
  text('Concentration (nM)', gx + gw / 2, gy + gh + 20);
  push(); translate(gx - 28, gy + gh / 2); rotate(-HALF_PI);
  text('Response', 0, 0); pop();

  // Curves + points
  for (const snap of [achSnapshot, heartSnapshot]) {
    if (!snap || !snap.curve || !snap.curve.params) continue;
    const p = snap.curve.params;
    stroke(snap.color); strokeWeight(3); noFill();
    beginShape();
    for (let x = sliderMin; x <= sliderMax; x += 2) {
      const y = p.Emax * hillG(x, p.EC50, p.n);
      vertex(map(x, sliderMin, sliderMax, gx, gx + gw), map(y, yMin, yMax, gy + gh, gy));
    }
    endShape();
    if (snap.points && snap.points.length) {
      noStroke(); fill(snap.color);
      for (const pt of snap.points) {
        ellipse(map(pt.x, sliderMin, sliderMax, gx, gx + gw), map(pt.y, yMin, yMax, gy + gh, gy), 6, 6);
      }
    }
  }

  // Position "Click me!" at right edge of graph, between the two plateaued curves
  if (compareP5Button && !activeSpotlight &&
      achSnapshot?.curve?.params && heartSnapshot?.curve?.params) {
    const rightConc = sliderMin + (sliderMax - sliderMin) * 0.88;
    const pa = achSnapshot.curve.params, ph = heartSnapshot.curve.params;
    const ya = pa.Emax * hillG(rightConc, pa.EC50, pa.n);
    const yh = ph.Emax * hillG(rightConc, ph.EC50, ph.n);
    const btnSX = map(rightConc, sliderMin, sliderMax, gx, gx + gw) - 60;
    const btnSY = map((ya + yh) / 2, yMin, yMax, gy + gh, gy) - 16;
    compareP5Button.position(btnSX, btnSY);
  }

  // Legend
  const legX = gx + 10, legY = gy + 16;
  fill(15, 25, 55, 185); noStroke(); rect(legX - 6, legY - 12, 204, 50, 6);
  fill(220, 20, 60);   noStroke(); rect(legX, legY,      16, 8);
  fill(30, 144, 255);  noStroke(); rect(legX, legY + 22, 16, 8);
  fill(200, 215, 255); textAlign(LEFT); textSize(11);
  text('Diaphragm (4 receptors)', legX + 22, legY + 8);
  text('Intestine (6 receptors)',  legX + 22, legY + 30);

  // ── Right: summary text ───────────────────────────────────────
  const lx = 680;
  let ly = 110;

  fill(255, 240, 200); textStyle(BOLD); textSize(26); textAlign(LEFT);
  text("Summary of the", lx, ly); ly += 32;
  text("Experiment", lx, ly); ly += 48;

  fill(160, 185, 230); textStyle(NORMAL); textSize(13.5);
  text("From 1926 to 1950, Clark's model was", lx, ly); ly += 24;
  text("the widely accepted theory, and it was", lx, ly); ly += 24;
  text("because of the experiments we just did.", lx, ly); ly += 36;

  text("The graphs show us that the more", lx, ly); ly += 24;
  text("receptors a tissue has, the greater the", lx, ly); ly += 24;
  text("response. (See Graph)", lx, ly); ly += 44;

  stroke(80, 100, 160); strokeWeight(1); line(lx, ly, lx + 520, ly); noStroke(); ly += 28;

  fill(200, 215, 255); textStyle(BOLD); textSize(12.5);
  text("What your data confirmed:", lx, ly); ly += 24; textStyle(NORMAL);

  const bullets = [
    ["A1 & A3", "Drugs occupy receptors; one molecule per receptor."],
    ["A4",      "All receptors occupied = maximum response (Emax)."],
    ["A2",      "Response is proportional to receptor occupation."],
    ["A5",      "Click the graph to find out!"],
  ];
  for (const [label, body] of bullets) {
    fill(255, 213, 0); textStyle(BOLD); textSize(12);
    text(label, lx + 8, ly);
    fill(160, 185, 230);
    textStyle(NORMAL); textSize(12);
    text(body, lx + 62, ly);
    ly += 26;
  }
}

function drawClarkSummaryScene() {
  background(15, 25, 55);

  // MSB-style top bar on summary
  fill(255, 210, 0); noStroke(); rect(0, 0, width, 60);
  fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('FIELD TRIP DEBRIEF  ·  MS. FRIZZLE\'S NOTES', width / 2, 26);
  textStyle(NORMAL); textSize(11);
  text('Clark\'s Occupancy Theory  ·  What held up, and what didn\'t', width / 2, 46);
  fill(15, 22, 55); noStroke(); rect(0, 60, width, 18);
  noStroke();

  // ── Left: pivot text ─────────────────────────────────────────
  const lx = 78;
  let ly = 155;

  fill(255, 240, 200); textStyle(BOLD); textSize(34); textAlign(LEFT);
  text("Clark's model held up remarkably well.", lx, ly);
  textStyle(NORMAL); ly += 52;

  fill(160, 185, 230); textSize(16);
  text('His five postulates predicted the shape of dose-response curves,', lx, ly); ly += 28;
  text('explained why different tissues respond differently,', lx, ly); ly += 28;
  text('and gave pharmacology its first mathematical framework.', lx, ly); ly += 52;

  stroke(60, 80, 140); strokeWeight(1); line(lx, ly, 580, ly); noStroke(); ly += 36;

  fill(255, 220, 100); textStyle(BOLD); textSize(20);
  text('But two observations kept breaking it.', lx, ly);
  textStyle(NORMAL); ly += 44;

  fill(210, 100, 100); textStyle(BOLD); textSize(15);
  text('1.  Spare Receptors', lx + 8, ly); textStyle(NORMAL);
  fill(160, 185, 230); textSize(13);
  text('Maximum response could be reached without occupying all receptors.', lx + 28, ly + 20);
  ly += 56;

  fill(200, 140, 255); textStyle(BOLD); textSize(15);
  text('2.  Partial Agonism', lx + 8, ly); textStyle(NORMAL);
  fill(160, 185, 230); textSize(13);
  text('Some drugs saturated below Emax — no matter how much you added.', lx + 28, ly + 20);
  ly += 60;

  fill(130, 160, 215); textSize(14); textStyle(ITALIC);
  text('Clark had no mechanism for either. A new framework was needed.', lx, ly);
  textStyle(NORMAL);

  // ── Right: mini graph with both curves ───────────────────────
  const gx = 710, gy = 148, gw = 480, gh = 300;

  fill(22, 35, 78); noStroke(); rect(690, 128, 520, 420, 10);

  fill(200, 215, 255); textAlign(CENTER); textStyle(BOLD); textSize(12);
  text('Your data — both tissues', gx + gw / 2, gy - 12); textStyle(NORMAL);

  stroke(130, 150, 200); strokeWeight(1.5);
  line(gx, gy, gx, gy + gh); line(gx, gy + gh, gx + gw, gy + gh);
  stroke(40, 58, 100); strokeWeight(0.5);
  for (let i = 1; i <= 4; i++) {
    line(gx, map(i, 0, 4, gy + gh, gy), gx + gw, map(i, 0, 4, gy + gh, gy));
  }
  noStroke(); fill(130, 155, 205); textSize(10); textAlign(CENTER);
  text('Concentration →', gx + gw / 2, gy + gh + 16);
  push(); translate(gx - 20, gy + gh / 2); rotate(-HALF_PI);
  text('Response →', 0, 0); pop();

  drawMiniSnapshotCurve(achSnapshot, gx, gy, gw, gh);
  drawMiniSnapshotCurve(heartSnapshot, gx, gy, gw, gh);

  // Emax lines
  for (const [snap, col] of [[achSnapshot, color(220, 20, 60)], [heartSnapshot, color(30, 144, 255)]]) {
    if (!snap?.curve?.params) continue;
    let yMax = 0;
    for (let x = sliderMin; x <= sliderMax; x += 2) {
      const y = snap.curve.params.Emax * hillG(x, snap.curve.params.EC50, snap.curve.params.n);
      if (y > yMax) yMax = y;
    }
    const sy = map(yMax, yMin, yMax, gy + gh, gy);
    drawingContext.setLineDash([5, 4]);
    stroke(col); strokeWeight(1); line(gx, sy, gx + gw, sy);
    drawingContext.setLineDash([]);
    noStroke(); fill(col); textSize(10); textAlign(RIGHT);
    text(snap === achSnapshot ? 'Emax₁' : 'Emax₂', gx + gw - 4, sy - 4);
  }

  // Legend
  const lx2 = gx + 8, ly2 = gy + gh + 34;
  fill(220, 20, 60); noStroke(); rect(lx2, ly2 - 8, 14, 8);
  fill(140, 165, 215); textSize(10); textAlign(LEFT);
  text('Diaphragm (4)', lx2 + 20, ly2);
  fill(30, 144, 255); noStroke(); rect(lx2 + 130, ly2 - 8, 14, 8);
  fill(140, 165, 215); text('Intestine (6)', lx2 + 150, ly2);
}

function drawMiniSnapshotCurve(snapshot, gx, gy, gw, gh) {
  if (!snapshot?.curve?.params) return;
  const p = snapshot.curve.params;

  stroke(snapshot.color); strokeWeight(2.5); noFill();
  beginShape();
  for (let x = sliderMin; x <= sliderMax; x += 2) {
    const y = p.Emax * hillG(x, p.EC50, p.n);
    const sx = map(x, sliderMin, sliderMax, gx, gx + gw);
    const sy = map(y, yMin, yMax, gy + gh, gy);
    vertex(sx, sy);
  }
  endShape();

  if (snapshot.points?.length) {
    fill(snapshot.color); noStroke();
    for (const pt of snapshot.points) {
      const sx = map(pt.x, sliderMin, sliderMax, gx, gx + gw);
      const sy = map(pt.y, yMin, yMax, gy + gh, gy);
      ellipse(sx, sy, 5, 5);
    }
  }
}

function drawClarkProblemsScene() {
  if (problemPage === 1) {
    drawProblem1SpareReceptors();
  } else {
    drawProblem2PartialAgonism();
  }
}

function drawProblem1SpareReceptors() {
  background(15, 25, 55);

  // MSB ribbon — yellow with red underbar
  fill(255, 210, 0); noStroke(); rect(0, 0, width, 58);
  fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text("THE MAGIC SCHOOL BUS  \xB7  CHAPTER II", width / 2, 24);
  textStyle(NORMAL); fill(40, 28, 0); textSize(11);
  text("Where Clark's Model Fell Short  \xB7  Limitation 1 of 2", width / 2, 44);
  fill(204, 60, 60); noStroke(); rect(0, 58, width, 10);

  // Chapter heading
  fill(204, 60, 60); textAlign(LEFT); textStyle(BOLD); textSize(13);
  text('LIMITATION I', 78, 102);
  textStyle(NORMAL);
  fill(255, 220, 220); textAlign(LEFT); textStyle(BOLD); textSize(40);
  text('Spare Receptors', 78, 148);
  textStyle(NORMAL);

  fill(200, 155, 155); textSize(15); textStyle(ITALIC);
  text('"You don\'t need all receptors occupied to get maximum response."', 78, 178);
  textStyle(NORMAL);

  // Red divider
  stroke(204, 60, 60, 80); strokeWeight(1);
  line(78, 196, 620, 196); noStroke();

  // ── Left panel text ───────────────────────────────────────────
  const lx = 78; let ly = 222;

  fill(255, 210, 210); textAlign(LEFT); textStyle(BOLD); textSize(13);
  text('Clark predicted (Postulate 3):', lx, ly); textStyle(NORMAL); ly += 22;
  fill(190, 165, 165); textSize(13);
  text('\xB7  Emax only occurs when every receptor is occupied', lx + 10, ly); ly += 20;
  text('\xB7  Response = (occupied / total) \xD7 Emax', lx + 10, ly); ly += 32;

  fill(255, 210, 210); textStyle(BOLD); textSize(13);
  text('What was actually observed:', lx, ly); textStyle(NORMAL); ly += 22;
  fill(190, 165, 165); textSize(13);
  text('\xB7  Maximum response reached at only 30–50% occupancy', lx + 10, ly); ly += 20;
  text('\xB7  More drug beyond that threshold changes nothing', lx + 10, ly); ly += 20;
  text('\xB7  The leftover receptors are "spare" — a reserve', lx + 10, ly); ly += 32;

  fill(255, 210, 0); textStyle(BOLD); textSize(13);
  text('Why this breaks Clark:', lx, ly); textStyle(NORMAL); ly += 22;
  fill(215, 195, 155); textSize(13);
  text('If full occupancy is not required for Emax,', lx + 10, ly); ly += 20;
  text('response cannot simply equal occupancy \xD7 Emax.', lx + 10, ly); ly += 20;
  text('Postulate 3 is directly violated.', lx + 10, ly);

  // ── Right panel: receptor comparison diagrams ─────────────────
  const panelX = 650, panelY = 86;
  fill(22, 35, 78); noStroke(); rect(panelX, panelY, 590, 608, 10);

  fill(255, 210, 210); textAlign(CENTER); textStyle(BOLD); textSize(14);
  text('The Contradiction', panelX + 295, panelY + 30);
  textStyle(NORMAL);

  drawReceptorComparison(panelX + 140, panelY + 52, 6, 2,
    "Clark's Prediction",
    "33% occupancy  →  33% response",
    0.33, color(180, 140, 255));

  stroke(50, 65, 110); strokeWeight(1);
  line(panelX + 18, panelY + 280, panelX + 572, panelY + 280); noStroke();

  drawReceptorComparison(panelX + 140, panelY + 298, 6, 2,
    'Observed Reality',
    "33% occupancy  →  100% response  (!)",
    1.0, color(100, 220, 150));

  // Red accent box at bottom of right panel
  fill(204, 60, 60, 28); noStroke(); rect(panelX + 18, panelY + 526, 554, 62, 6);
  stroke(204, 60, 60, 100); strokeWeight(1);
  rect(panelX + 18, panelY + 526, 554, 62, 6); noStroke();
  fill(204, 60, 60); textAlign(LEFT); textStyle(BOLD); textSize(12);
  text('The key insight:', panelX + 34, panelY + 548);
  textStyle(NORMAL); fill(255, 200, 200); textSize(12);
  text('Some tissues have so many receptors that maximal response can be', panelX + 34, panelY + 566);
  text('achieved before all of them are occupied.', panelX + 34, panelY + 580);
}

function drawReceptorComparison(x, y, total, occupied, title, subtitle, responseFraction, col) {
  const rW = 40, rH = 54, gap = 14;

  noStroke(); fill(200, 215, 255);
  textAlign(LEFT); textStyle(BOLD); textSize(13);
  text(title, x, y + 14); textStyle(NORMAL);

  const recY = y + 50;
  for (let i = 0; i < total; i++) {
    const rx = x + i * (rW + gap);
    fill(35, 55, 115); stroke(75, 108, 185); strokeWeight(1.2);
    rect(rx, recY, rW, rH, 6);
    if (i < occupied) {
      noStroke(); fill(col);
      ellipse(rx + rW / 2, recY - 12, 22, 22);
      stroke(col); strokeWeight(1.5);
      line(rx + rW / 2, recY - 1, rx + rW / 2, recY + 2);
    }
  }

  const barY = recY + rH + 20;
  const barMaxW = total * (rW + gap) - gap;
  noStroke(); fill(38, 52, 100);
  rect(x, barY, barMaxW, 24, 5);
  fill(col);
  rect(x, barY, barMaxW * responseFraction, 24, 5);

  noStroke(); fill(col);
  textSize(12); textStyle(ITALIC); textAlign(LEFT);
  text(subtitle, x, barY + 42); textStyle(NORMAL);
}

function drawProblem2PartialAgonism() {
  background(15, 25, 55);

  // MSB ribbon — yellow with red underbar
  fill(255, 210, 0); noStroke(); rect(0, 0, width, 58);
  fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text("THE MAGIC SCHOOL BUS  \xB7  CHAPTER II", width / 2, 24);
  textStyle(NORMAL); fill(40, 28, 0); textSize(11);
  text("Where Clark's Model Fell Short  \xB7  Limitation 2 of 2", width / 2, 44);
  fill(204, 60, 60); noStroke(); rect(0, 58, width, 10);

  // Chapter heading
  fill(204, 60, 60); textAlign(LEFT); textStyle(BOLD); textSize(13);
  text('LIMITATION II', 78, 102);
  textStyle(NORMAL);
  fill(255, 220, 220); textAlign(LEFT); textStyle(BOLD); textSize(40);
  text('Partial Agonism', 78, 148);
  textStyle(NORMAL);

  fill(200, 155, 155); textSize(15); textStyle(ITALIC);
  text('"Some drugs can never produce a full response, no matter the dose."', 78, 178);
  textStyle(NORMAL);

  // Red divider
  stroke(204, 60, 60, 80); strokeWeight(1);
  line(78, 196, 620, 196); noStroke();

  // ── Left panel text ───────────────────────────────────────────
  const lx = 78; let ly = 222;

  fill(255, 210, 210); textAlign(LEFT); textStyle(BOLD); textSize(13);
  text('Clark predicted:', lx, ly); textStyle(NORMAL); ly += 22;
  fill(190, 165, 165); textSize(13);
  text('\xB7  Any drug that fills all receptors → full Emax', lx + 10, ly); ly += 20;
  text('\xB7  Occupancy alone determines response', lx + 10, ly); ly += 32;

  fill(255, 210, 210); textStyle(BOLD); textSize(13);
  text('What was actually observed:', lx, ly); textStyle(NORMAL); ly += 22;
  fill(190, 165, 165); textSize(13);
  text('\xB7  Some drugs fill every receptor yet only reach 50% Emax', lx + 10, ly); ly += 20;
  text('\xB7  No extra dose can push them to full response', lx + 10, ly); ly += 20;
  text('\xB7  These are called partial agonists', lx + 10, ly); ly += 32;

  fill(255, 210, 0); textStyle(BOLD); textSize(13);
  text('Why this breaks Clark:', lx, ly); textStyle(NORMAL); ly += 22;
  fill(215, 195, 155); textSize(13);
  text('Binding a receptor is not the same as activating it fully.', lx + 10, ly); ly += 20;
  text('A second property determines how strongly a drug activates', lx + 10, ly); ly += 20;
  text('the receptor it occupies: intrinsic efficacy.', lx + 10, ly);

  // ── Right panel: two-row comparison (identical structure to spare receptors) ──
  const panelX = 650, panelY = 86;
  fill(22, 35, 78); noStroke(); rect(panelX, panelY, 590, 608, 10);

  fill(255, 210, 210); textAlign(CENTER); textStyle(BOLD); textSize(14);
  text('The Contradiction', panelX + 295, panelY + 30);
  textStyle(NORMAL);

  drawReceptorComparison(panelX + 140, panelY + 52, 6, 6,
    'Full Agonist',
    '100% occupancy  →  100% response',
    1.0, color(100, 185, 255));

  stroke(50, 65, 110); strokeWeight(1);
  line(panelX + 18, panelY + 280, panelX + 572, panelY + 280); noStroke();

  drawReceptorComparison(panelX + 140, panelY + 298, 6, 6,
    'Partial Agonist',
    '100% occupancy  →  50% response  (!)',
    0.5, color(255, 155, 55));

  // Red accent box at bottom of right panel
  fill(204, 60, 60, 28); noStroke(); rect(panelX + 18, panelY + 526, 554, 62, 6);
  stroke(204, 60, 60, 100); strokeWeight(1);
  rect(panelX + 18, panelY + 526, 554, 62, 6); noStroke();
  fill(204, 60, 60); textAlign(LEFT); textStyle(BOLD); textSize(12);
  text('The new concept this introduced:', panelX + 34, panelY + 548);
  textStyle(NORMAL); fill(255, 200, 200); textSize(12);
  text('Intrinsic efficacy — a drug\'s ability to activate a receptor once bound.', panelX + 34, panelY + 566);
  text('Clark\'s model had no way to account for this.', panelX + 34, panelY + 580);
}

function drawGraphOverlay() {
  // Full-canvas dim
  fill(0, 0, 0, 165); noStroke();
  rect(0, 0, width, height);

  // Panel
  const px = 190, py = 90, pw = 900, ph = 540;
  fill(255, 210, 0); noStroke(); rect(px, py, pw, 58, 10, 10, 0, 0);
  fill(250, 248, 240); stroke(200, 192, 175); strokeWeight(1);
  rect(px, py + 58, pw, ph - 58, 0, 0, 10, 10);

  // MSB ribbon text
  noStroke(); fill(15, 22, 55);
  textAlign(CENTER); textStyle(BOLD); textSize(14);
  text("MS. FRIZZLE'S FIELD NOTES  ·  YOUR MISSION FOR THIS STOP", px + pw / 2, py + 26);
  textStyle(NORMAL); fill(40, 28, 0); textSize(11);
  text('"Take chances, make mistakes, get messy — but record everything!"', px + pw / 2, py + 46);

  const tx = px + 52;
  const lh = 22;
  let ty = py + 88;

  noStroke(); fill(30, 38, 80);
  textAlign(CENTER); textStyle(BOLD); textSize(20);
  text('Welcome to the lab', px + pw / 2, ty);
  textStyle(NORMAL); ty += 16;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty, px + pw - 40, ty); noStroke();
  ty += 22;

  fill(55, 60, 80); textAlign(LEFT); textSize(14);
  text("The right panel is an isolated strip of muscle (diaphragm) — our investigation site. With the Magic", tx, ty); ty += lh;
  text("School Bus magnification, we are able to watch the molecules live in action. White ligands are", tx, ty); ty += lh;
  text("acetylcholine molecules in solution. Four GPCRs sit in the magnified environment, giving us an", tx, ty); ty += lh;
  text("arbitrary receptor density of the tissue. Each one has a binding site for the ligand.", tx, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(BOLD); textSize(14);
  text('Your Goal', tx, ty); textStyle(NORMAL); ty += lh;

  fill(55, 60, 80); textSize(14);
  text('We want to create a dose-response curve based on this lab experiment. The graph on your left is your', tx, ty); ty += lh;
  text('starting point, and will populate as you record points. The graph plots:', tx, ty); ty += lh;

  fill(30, 38, 80); textSize(13);
  text('X axis', tx + 20, ty); fill(90, 95, 120); text("→  Acetylcholine Concentration  (set by the slider)", tx + 80, ty); ty += lh;
  fill(30, 38, 80); text('Y axis', tx + 20, ty); fill(90, 95, 120); text("→  Response  (based on Clark's understanding of binding rate, measured automatically,", tx + 80, ty); ty += lh;
  fill(90, 95, 120); text('shown by the ghost point)', tx + 80, ty); ty += lh * 1.3;

  fill(30, 38, 80); textStyle(ITALIC); textSize(13);
  text('Adjust the slider to a concentration, then wait 5–7 seconds for the binding rate to stabilise before', tx, ty); ty += lh;
  text('hitting "Record It!". Repeat for ~5 concentrations, then plot the curve.', tx, ty);
  textStyle(NORMAL); ty += lh * 1.8;

  stroke(210, 200, 182); strokeWeight(1);
  line(px + 40, ty - 8, px + pw - 40, ty - 8); noStroke();

  fill(180, 140, 0); textStyle(BOLD); textSize(12);
  text('UP NEXT', tx, ty + 4);
  fill(55, 60, 80); textStyle(NORMAL); textSize(13);
  text('A quick lab tour — we will walk through the field site before the experiment begins.', tx + 76, ty + 4);
}

function drawIntroScene() {
  background(255, 210, 0);

  // Dark navy top bar
  fill(15, 22, 55); noStroke(); rect(0, 0, width, 68);
  fill(255, 210, 0); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('THE MAGIC SCHOOL BUS  ·  RECEPTOR THEORY', width / 2, 26);
  textStyle(ITALIC); fill(255, 210, 100); textSize(11);
  text('"Every trip is different — and every trip is extraordinary!"', width / 2, 50);
  textStyle(NORMAL);

  // Thank you headline
  fill(15, 22, 55); textAlign(CENTER); textStyle(BOLD); textSize(46);
  text('Thank you for riding.', width / 2, 136);
  textStyle(NORMAL);

  // Navy accent line under headline
  stroke(15, 22, 55); strokeWeight(1.5);
  line(width / 2 - 220, 152, width / 2 + 220, 152); noStroke();

  // Subtitle
  fill(30, 40, 80); textSize(13); textAlign(CENTER);
  text("You've traced 100 years of pharmacology — from Clark's elegant first equation", width / 2, 178);
  text('to the two-state model that explains every drug type.', width / 2, 197);

  // Revisit prompt
  fill(15, 22, 55, 160); textSize(11); textStyle(ITALIC);
  text('Want to revisit a chapter? Jump back in anytime.', width / 2, 224);
  textStyle(NORMAL);

  // — Chapter cards —
  const CY = 252, CH = 280, GAP = 40;
  const CW = (width - GAP * 4) / 3; // ~373px each

  // Card 1 — Chapter I: Clark
  const c1x = GAP;
  fill(15, 22, 55); noStroke(); rect(c1x, CY, CW, CH, 10);
  fill(140, 80, 200); noStroke(); rect(c1x, CY, CW, 5, 10, 10, 0, 0);
  fill(200, 160, 255); textSize(10); textStyle(BOLD); textAlign(LEFT);
  text('CHAPTER I', c1x + 16, CY + 24);
  fill(255, 255, 255); textSize(18); textStyle(BOLD);
  text("Clark's Occupancy", c1x + 16, CY + 46);
  text('Theory', c1x + 16, CY + 67);
  textStyle(NORMAL);
  fill(190, 210, 240); textSize(11.5);
  text('Binding, saturation, and the dose-response curve. Five postulates, one elegant model, and the receptors that make it real.', c1x + 16, CY + 92, CW - 32, 100);
  fill(100, 120, 170); textSize(10); textStyle(ITALIC);
  text('Hill equation · dose-response · binding kinetics', c1x + 16, CY + 210, CW - 32, 30);
  textStyle(NORMAL);

  // Card 2 — Chapter II: The Problem with Clark
  const c2x = GAP * 2 + CW;
  fill(15, 22, 55); noStroke(); rect(c2x, CY, CW, CH, 10);
  fill(210, 80, 80); noStroke(); rect(c2x, CY, CW, 5, 10, 10, 0, 0);
  fill(255, 160, 140); textSize(10); textStyle(BOLD); textAlign(LEFT);
  text('CHAPTER II', c2x + 16, CY + 24);
  fill(255, 255, 255); textSize(18); textStyle(BOLD);
  text('The Problem', c2x + 16, CY + 46);
  text('with Clark', c2x + 16, CY + 67);
  textStyle(NORMAL);
  fill(190, 210, 240); textSize(11.5);
  text('Partial agonists. Spare receptors. The experiments that showed binding alone couldn\'t explain drug behavior.', c2x + 16, CY + 92, CW - 32, 100);
  fill(100, 120, 170); textSize(10); textStyle(ITALIC);
  text('partial agonism · spare receptors · limitations', c2x + 16, CY + 210, CW - 32, 30);
  textStyle(NORMAL);

  // Card 3 — Chapter III: Modern Receptor Theory
  const c3x = GAP * 3 + CW * 2;
  fill(15, 22, 55); noStroke(); rect(c3x, CY, CW, CH, 10);
  fill(46, 196, 160); noStroke(); rect(c3x, CY, CW, 5, 10, 10, 0, 0);
  fill(140, 210, 195); textSize(10); textStyle(BOLD); textAlign(LEFT);
  text('CHAPTER III', c3x + 16, CY + 24);
  fill(255, 255, 255); textSize(18); textStyle(BOLD);
  text('Modern Receptor', c3x + 16, CY + 46);
  text('Theory', c3x + 16, CY + 67);
  textStyle(NORMAL);
  fill(190, 210, 240); textSize(11.5);
  text('Five drug types. One equilibrium. R ⇌ R* explains full agonists, partial agonists, antagonists, and inverses.', c3x + 16, CY + 92, CW - 32, 100);
  fill(100, 120, 170); textSize(10); textStyle(ITALIC);
  text('two-state model · efficacy · R ⇌ R*', c3x + 16, CY + 210, CW - 32, 30);
  textStyle(NORMAL);

  // Small bus bottom right
  drawMSBus(1068, 580, 170, 52);
}


// ─────────────────────────────────────────────
// Button and Slider Event Handlers
// ─────────────────────────────────────────────
function handlePointButtonClick() {
  if (scene === 'partialGraph') { handlePartialPointClick(); return; }
  if (scene === 'spareGraph')   { handleSparePointClick();   return; }
  // Capture spotlight position: prefer a free-floating ligand (particles freeze
  // once activeSpotlight is set, so it stays put for the whole card display).
  // Fall back to an attached ligand, then to the first receptor slot.
  let spX = rectangles[0].x + rectangles[0].w / 2;
  let spY = rectangles[0].y + rectangles[0].h / 2;
  let spRadius = 62;
  let foundSpot = false;
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    if (!p.isInhibitor && p.attachedRectIndex === -1 && !p.animating) {
      spX = p.position.x;
      spY = p.position.y;
      spRadius = p.r + 36;
      foundSpot = true;
      break;
    }
  }
  if (!foundSpot) {
    for (let i = 0; i < attachedLigands.length; i++) {
      if (attachedLigands[i] !== null) {
        spX = attachedLigands[i].position.x;
        spY = attachedLigands[i].position.y;
        spRadius = attachedLigands[i].r + 36;
        break;
      }
    }
  }

  detachAllLigands();
  resetLigandProperties();

  const currentTime = millis();
  const elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;
  const averageAttachments = elapsedSeconds > 0 ? attachmentTimes.length / elapsedSeconds : 0;

  if (scene === 'achGraph' || scene === 'heartGraph') {
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);
    ghostPoint.y = constrain(averageAttachments, yMin, yMax);

    pointList.push({ x: ghostPoint.x, y: ghostPoint.y, alpha: 255 });

    // Keep counter consistent with array size; no hard cap
    pointCounter = pointList.length;

    // P1+P2 spotlight — fires on the first "Record It!" click in achGraph
    if (scene === 'achGraph' && pointCounter === 1) {
      // Nearest receptor to the spotted ligand
      let recX = rectangles[0].x + rectangles[0].w / 2;
      let recY = rectangles[0].y + rectangles[0].h / 2;
      let nearestDist = Infinity;
      for (const rect of rectangles) {
        const rcx = rect.x + rect.w / 2;
        const rcy = rect.y + rect.h / 2;
        const d = Math.sqrt((spX - rcx) ** 2 + (spY - rcy) ** 2);
        if (d < nearestDist) { nearestDist = d; recX = rcx; recY = rcy; }
      }
      // First plotted data point on the graph canvas
      const gpCx = map(ghostPoint.x, sliderMin, sliderMax, 80, w - 80);
      const gpCy = map(ghostPoint.y, yMin, yMax, h - 80, 80);

      // If ligand and receptor circles overlap, merge them into one
      const recR = 32;
      const holeDist = Math.sqrt((spX - recX) ** 2 + (spY - recY) ** 2);
      let primaryX = spX, primaryY = spY, primaryR = spRadius;
      let recHole = { x: recX, y: recY, shape: 'circle', radius: recR };
      if (holeDist < spRadius + recR) {
        // Merged: centre between the two, radius spans both
        primaryX = (spX + recX) / 2;
        primaryY = (spY + recY) / 2;
        primaryR = holeDist / 2 + Math.max(spRadius, recR) + 10;
        recHole = null; // absorbed into primary
      }

      showSpotlight(1, primaryX, primaryY, primaryR,
        "Woah, you found Clark's first assumptions!",
        [
          { label: "Assumption 1 — Drugs Occupy Receptors",
            text: "Drugs exert their effects by physically occupying receptors. Without that binding event, no pharmacological response is generated." },
          { label: "Assumption 3 — One Drug, One Receptor",
            text: "Each drug molecule binds to exactly one receptor. A single ligand occupies a single binding site — no sharing, no stacking." }
        ]
      );
      // Extra holes: receptor (if distinct) + graph data point
      const extraHoles = [];
      if (recHole) extraHoles.push(recHole);
      extraHoles.push({ x: gpCx, y: gpCy, shape: 'circle', radius: 26 });
      activeSpotlight.extraHoles = extraHoles;
      _buildSpotlightOverlay(activeSpotlight);
      clarkPostulateShown.add(2);
    }

    // If you use these, reset plotted state so user can re-plot after adding points
    graphPlotted = false;
    if (typeof fittedCurve !== 'undefined') fittedCurve = null;
  }
}



function handleContinueButtonClick() {
  lastConc = particles.length;

  if (scene === 'achGraph') {
    achSnapshot = snapshotCurrent('ACH', color(220, 20, 60));
    initializeScene('tissueTransition');
  } else if (scene === 'heartGraph') {
    heartSnapshot = snapshotCurrent('HEART', color(30, 144, 255));
    initializeScene('dataCollected');
  } else if (scene === 'partialGraph') {
    initializeScene('spareGraph');
  } else if (scene === 'spareGraph') {
    initializeScene('ch2Analysis');
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
//Postulate Button Stuff  
function handlePostulateButtonClick() {
  postulatePage++;

  if (postulatePage === 1) {
    postulateButton.html("Next Postulates");
    showEvidence = true;

    evidenceParticles = [];

    const miniX = 700;
    const miniY = 150;
    const miniWidth = 300;
    const miniHeight = 120; // increased height for mini simulation

    const receptorWidth = 80;
    const receptorHeight = 80;

    // Receptor positioned at bottom of mini simulation, on top of membrane
    const receptorX = miniX + miniWidth / 2 - receptorWidth / 2;
    const receptorY = miniY + miniHeight - receptorHeight / 2; // receptor overlaps membrane

    evidenceReceptor = { x: receptorX, y: receptorY };

    // Bounds for mini balls: full mini simulation area above receptor
    const bounds = {
      x: miniX,
      y: miniY,
      w: miniWidth,
      h: receptorY - miniY
    };

    for (let i = 0; i < evidenceConcentration; i++) {
      let pos = createVector(
        random(bounds.x + 10, bounds.x + bounds.w - 10),
        random(bounds.y + 10, bounds.y + bounds.h - 10)
      );
      let vel = p5.Vector.random2D().mult(1.5);
      evidenceParticles.push(new Ball(pos, vel, radius, i, [], "cyan", false, bounds));
    }

  } else if (postulatePage === 2) {
    postulateButton.html("Final Postulate");
    showEvidence = false;
    evidenceParticles = [];
  } else if (postulatePage >= 3) {
    postulateButton.hide();
    showEvidence = false;
    evidenceParticles = [];
  }
}

// ─────────────────────────────────────────────
// Spotlight System
// ─────────────────────────────────────────────

// Build (or rebuild) the off-screen overlay buffer.
// spotlight.shape === 'rect'  → punches a rounded-rect hole (rw, rh, rr)
// spotlight.shape === 'circle' (default) → punches a circle hole (radius)
function _buildSpotlightOverlay(spotlight) {
  if (spotlightOverlayG) { spotlightOverlayG.remove(); spotlightOverlayG = null; }
  if (!spotlight) return;
  spotlightOverlayG = createGraphics(1280, 720);
  spotlightOverlayG.pixelDensity(1);
  spotlightOverlayG.clear();
  spotlightOverlayG.fill(0, 0, 0, 190);
  spotlightOverlayG.noStroke();
  spotlightOverlayG.rect(0, 0, 1280, 720);
  spotlightOverlayG.erase();
  spotlightOverlayG.noStroke();

  // Helper — punch one hole based on shape
  function punchHole(h) {
    if (h.shape === 'rect') {
      spotlightOverlayG.rect(
        h.x - h.rw / 2, h.y - h.rh / 2,
        h.rw, h.rh, h.rr || 10
      );
    } else {
      spotlightOverlayG.circle(h.x, h.y, (h.radius || 50) * 2);
    }
  }

  punchHole(spotlight);
  if (spotlight.extraHoles) {
    for (const hole of spotlight.extraHoles) punchHole(hole);
  }

  spotlightOverlayG.noErase();
}

// geom: a number → circle with that radius
//       an object {w, h, r} → rounded rect
function showSpotlight(num, x, y, geom, heading, postulates, pending) {
  if (clarkPostulateShown.has(num)) return;
  if (activeSpotlight) return;
  clarkPostulateShown.add(num);
  const isRect = (typeof geom === 'object' && geom !== null);
  activeSpotlight = {
    num, x, y, heading, postulates,
    shape:  isRect ? 'rect'   : 'circle',
    radius: isRect ? null     : geom,
    rw:     isRect ? geom.w   : null,
    rh:     isRect ? geom.h   : null,
    rr:     isRect ? (geom.r || 10) : null,
  };
  pendingSpotlight = pending || null;
  spotlightShownAtMillis = millis();
  _buildSpotlightOverlay(activeSpotlight);
}

function dismissSpotlight() {
  activeSpotlight = null;
  const next = pendingSpotlight;
  pendingSpotlight = null;
  if (next) {
    clarkPostulateShown.add(next.num);
    activeSpotlight = next;
    pendingSpotlight = next.pending || null; // propagate chained spotlights
    spotlightShownAtMillis = millis();
    _buildSpotlightOverlay(activeSpotlight);
  } else {
    _buildSpotlightOverlay(null);
    // MRT tour just finished — reveal the Continue button
    if (scene === 'mrtBasal' && !showMrtBasalOverlay && slideButton) {
      slideButton.show();
    }
  }
}

function drawSpotlight() {
  if (!activeSpotlight || !spotlightOverlayG) return;

  image(spotlightOverlayG, 0, 0);

  const { x, y, heading, postulates, shape, radius, rw, rh, rr } = activeSpotlight;

  // Highlighted curve segment — drawn inside the spotlight hole for P3/P4
  if (activeSpotlight.highlightCurve && typeof fittedCurve !== 'undefined' && fittedCurve?.points?.length) {
    const hx1 = (shape === 'rect') ? x - rw / 2 : x - radius;
    const hx2 = (shape === 'rect') ? x + rw / 2 : x + radius;
    const pts = fittedCurve.points.filter(p => p.x >= hx1 && p.x <= hx2);
    if (pts.length >= 2) {
      stroke(255, 180, 0); strokeWeight(6); noFill();
      beginShape();
      for (const p of pts) vertex(p.x, p.y);
      endShape();
    }
  }

  // Ring around hole — teal for MRT, amber otherwise
  const _mrtRingNums = new Set([101, 102, 103, 104, 105, 106, 107]);
  const _ringColor = _mrtRingNums.has(activeSpotlight.num) ? color(46, 196, 160, 210) : color(255, 215, 60, 210);
  noFill(); stroke(_ringColor); strokeWeight(2.5);
  if (shape === 'rect') {
    rect(x - rw / 2 - 5, y - rh / 2 - 5, rw + 10, rh + 10, (rr || 10) + 3);
  } else {
    circle(x, y, radius * 2 + 10);
  }

  // Extra holes — amber rings for each
  if (activeSpotlight.extraHoles) {
    for (const h of activeSpotlight.extraHoles) {
      noFill(); stroke(255, 215, 60, 210); strokeWeight(2.5);
      if (h.shape === 'rect') {
        rect(h.x - h.rw / 2 - 5, h.y - h.rh / 2 - 5, h.rw + 10, h.rh + 10, (h.rr || 10) + 3);
      } else {
        circle(h.x, h.y, (h.radius || 50) * 2 + 10);
      }
    }
  }

  const posts = postulates || [];
  const cardW = 460;
  const cardH = 36 + 1 + posts.length * 72 + 26;

  // Compute outer bounds across ALL holes (primary + extra) for card placement
  function _holeEdges(h) {
    if (h.shape === 'rect') return { top: h.y - h.rh / 2, bottom: h.y + h.rh / 2 };
    const r = h.radius || 50;
    return { top: h.y - r, bottom: h.y + r };
  }
  const allHoles = [activeSpotlight].concat(activeSpotlight.extraHoles || []);
  let minTop    = Infinity;
  let maxBottom = -Infinity;
  for (const h of allHoles) {
    const e = _holeEdges(h);
    if (e.top    < minTop)    minTop    = e.top;
    if (e.bottom > maxBottom) maxBottom = e.bottom;
  }

  let cardX = constrain(x - cardW / 2, 10, width - cardW - 10);
  let cardY = (maxBottom + 18 + cardH > height - 16)
    ? minTop - 18 - cardH
    : maxBottom + 18;
  cardY = constrain(cardY, 10, height - cardH - 10);

  // Card style: yellow (Ch1 postulates <90), red (Ch2 nums 21/22/94-96), navy (lab tour)
  const ch2Nums = new Set([20, 21, 22, 23, 24, 25, 94, 95, 96, 97, 98, 99]);
  const mrtNums  = new Set([101, 102, 103, 104, 105, 106, 107]);
  const isPostulate = activeSpotlight.num < 90 && !ch2Nums.has(activeSpotlight.num);
  const isChapter2  = ch2Nums.has(activeSpotlight.num);
  const isMrt       = mrtNums.has(activeSpotlight.num);

  if (isMrt) {
    fill(18, 68, 58, 252);
    stroke(46, 196, 160, 210);
  } else if (isChapter2) {
    fill(204, 55, 55, 252);
    stroke(100, 18, 18, 210);
  } else if (isPostulate) {
    fill(255, 213, 0, 250);
    stroke(15, 22, 55, 210);
  } else {
    fill(12, 18, 48, 240);
    stroke(255, 215, 60, 180);
  }
  strokeWeight(1.5);
  rect(cardX, cardY, cardW, cardH, 10);

  // Heading
  noStroke();
  fill(isMrt ? color(180, 240, 225) : isChapter2 ? color(255, 235, 235) : isPostulate ? color(15, 22, 55) : color(255, 215, 60));
  textStyle(BOLD); textSize(13); textAlign(LEFT);
  text(heading, cardX + 16, cardY + 24);

  // Divider under heading
  stroke(isMrt ? color(46, 196, 160, 60) : isChapter2 ? color(255, 200, 200, 60) : isPostulate ? color(15, 22, 55, 55) : color(255, 215, 60, 55));
  strokeWeight(0.75);
  line(cardX + 16, cardY + 33, cardX + cardW - 16, cardY + 33);
  noStroke();

  // Each postulate block
  let curY = cardY + 50;
  for (let i = 0; i < posts.length; i++) {
    // Label
    fill(isMrt ? color(140, 220, 200, 220) : isChapter2 ? color(255, 210, 210, 220) : isPostulate ? color(15, 22, 55, 210) : color(255, 215, 60, 190));
    textStyle(BOLD); textSize(10.5); textAlign(LEFT);
    text(posts[i].label, cardX + 16, curY);
    curY += 15;

    // Body text
    fill(isMrt ? color(210, 245, 238) : isChapter2 ? color(255, 245, 245) : isPostulate ? color(30, 18, 0) : color(195, 215, 252));
    textStyle(NORMAL); textSize(11);
    text(posts[i].text, cardX + 16, curY, cardW - 32, 52);
    curY += 57;

    // Separator between postulates (not after last)
    if (i < posts.length - 1) {
      stroke(isMrt ? color(46, 196, 160, 35) : isChapter2 ? color(255, 200, 200, 35) : isPostulate ? color(15, 22, 55, 35) : color(255, 255, 255, 18));
      strokeWeight(0.5);
      line(cardX + 16, curY - 2, cardX + cardW - 16, curY - 2);
      noStroke();
    }
  }

  // Footer
  fill(isMrt ? color(100, 200, 180) : isChapter2 ? color(255, 190, 190) : isPostulate ? color(60, 40, 0) : color(105, 128, 172));
  textSize(10);
  textAlign(CENTER);
  text('Click anywhere to continue →', cardX + cardW / 2, cardY + cardH - 10);

  noStroke();
  textAlign(LEFT);
  textStyle(NORMAL);
}

// Mini Simulation as Evidence

function drawMiniEvidence() {
  if (!showEvidence) return;

  const miniX = 700;
  const miniY = 150;
  const miniWidth = 300;
  const miniHeight = 120; // updated to match handlePostulateButtonClick

  // Draw membrane image at the bottom of the mini simulation area
  image(membrane, miniX, miniY + miniHeight - 20, miniWidth, 40);

  // Draw receptor on top of membrane
  if (evidenceReceptor) {
    image(gpcr, evidenceReceptor.x, evidenceReceptor.y, 80, 80);
  }

  // Draw mini balls
  for (let p of evidenceParticles) {
    p.bounceOthers();
    p.update();
    p.display();
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
  if (scene === 'partialGraph') { handlePartialFitClick(); return; }
  if (scene === 'spareGraph')   { handleSpareFitClick();   return; }
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
  graphPlotted = true;

  // Trigger postulate spotlights after fitting
  if (scene === 'achGraph') {
    const curvePts = fittedCurve.points; // already in screen coords

    // A4: plateau — right 22% of the curve x range
    const p3x = constrain(map(sliderMax * 0.78, sliderMin, sliderMax, 80, w - 80), 80, w - 80);
    const plateauPts = curvePts.filter(p => p.x >= p3x - 110 && p.x <= p3x + 110);
    const p3y = plateauPts.length > 0
      ? plateauPts.reduce((s, p) => s + p.y, 0) / plateauPts.length
      : constrain(map(fit.Emax * 0.97, yMin, yMax, h - 80, 80), 80, h - 80);

    // A2: steep rise centred on EC50
    const p4x = constrain(map(fit.EC50, sliderMin, sliderMax, 80, w - 80), 80, w - 80);
    const risePts = curvePts.filter(p => p.x >= p4x - 95 && p.x <= p4x + 95);
    const p4y = risePts.length > 0
      ? risePts.reduce((s, p) => s + p.y, 0) / risePts.length
      : constrain(map(fit.Emax * 0.5, yMin, yMax, h - 80, 80), 140, h - 80);
    showSpotlight(3, p3x, p3y, { w: 220, h: 90, r: 8 },
      "Clark's Assumption 4 — Maximum Response",
      [
        { label: "For maximal response, all receptors must be occupied",
          text: "When every available receptor is occupied, the tissue reaches its maximum response (Emax). Beyond this point, adding more drug produces no further effect — the system is saturated." }
      ],
      { num: 4, x: p4x, y: p4y, shape: 'rect', rw: 190, rh: 260, rr: 8,
        highlightCurve: true,
        heading: "Clark's Assumption 2 — Proportional Response",
        postulates: [
          { label: "Response proportional to receptor occupation",
            text: "The pharmacological response is directly proportional to the fraction of receptors occupied. The steeper this part of the curve, the more sensitive the tissue is to changes in concentration." }
        ]
      }
    );
    activeSpotlight.highlightCurve = true; // mark P3 too
  }
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


// ── Chapter 2: Partial Agonism handlers ──────────────────────────────────────

function handlePartialPointClick() {
  const conc = slider.value();
  const y    = constrain(partialGhostY, yMin, yMax);
  partialPointList.push({ x: conc, y, alpha: 255 });
  partialPointCounter = partialPointList.length;

  // Spotlight 21 on first recorded point
  if (partialPointCounter === 1) {
    const gpCx = map(conc, sliderMin, sliderMax, 80, w - 80);
    const gpCy = map(y, yMin, yMax, h - 80, 80);
    showSpotlight(21, gpCx, gpCy, 44,
      "A New Ligand — Same Prediction",
      [
        { label: "Clark Says You Should Hit the Original Graph",
          text: "Same tissue. Same 4 receptors. Clark's assumptions predict that any drug filling those receptors should drive the response to the same maximum you found in Chapter 1. The dashed curve is that benchmark — collect enough points and let's see if Ligand B gets there." }
      ]
    );
  }
}

function handlePartialFitClick() {
  if (partialPointList.length < 3) {
    alert('Plot at least 3 points before fitting.');
    return;
  }
  const fit = fitHill(partialPointList);
  if (!fit) { alert('Could not fit a curve.'); return; }

  partialFittedCurve  = { points: samplePartialCurve(fit, 1), params: fit };
  partialGraphPlotted = true;

  // Spotlight 22 — spotlight the plateau of the actual curve vs the dashed prediction
  const curvePts = partialFittedCurve.points;
  const plateauX  = map(sliderMax * 0.80, sliderMin, sliderMax, 80, w - 80);
  const plateauPts = curvePts.filter(p => p.x >= plateauX - 100 && p.x <= plateauX + 100);
  const spX = plateauX;
  const spY = plateauPts.length > 0
    ? plateauPts.reduce((s, p) => s + p.y, 0) / plateauPts.length
    : map(fit.Emax, yMin, yMax, h - 80, 80);

  showSpotlight(22, spX, spY, { w: 240, h: 100, r: 8 },
    "It Didn't Match the Original Graph",
    [
      { label: "Binding is not the same as activation",
        text: "Ligand B filled the receptors — but the response plateaued well below your Chapter 1 curve. Full occupancy didn't mean full response. Clark's model predicts they should be identical. They aren't. His model has no answer for why." }
    ]
  );
  activeSpotlight.highlightCurve = true;
}

function samplePartialCurve(params, stepX = 1) {
  const out = [];
  for (let x = sliderMin; x <= sliderMax; x += stepX) {
    const y  = params.Emax * hillG(x, params.EC50, params.n);
    const sx = map(x, sliderMin, sliderMax, 80, w - 80);
    const sy = map(y, yMin, yMax, h - 80, 80);
    out.push({ x: sx, y: sy });
  }
  return out;
}

function handleSparePointClick() {
  const conc = slider.value();
  const y = constrain(spareGhostY, yMin, yMax);
  sparePointList.push({ x: conc, y, alpha: 255 });
  sparePointCounter = sparePointList.length;
  if (sparePointCounter === 1) {
    const gpCx = map(conc, sliderMin, sliderMax, 80, w - 80);
    const gpCy = map(y, yMin, yMax, h - 80, 80);
    showSpotlight(23, gpCx, gpCy, 44,
      "Ligand C — Watch Where It Lands",
      [{ label: "Compare It To The Original Graph",
         text: "Clark predicts Ligand C — with 8 receptors to fill — should need a higher concentration to reach the same maximum. But keep an eye on where this curve is going. Does it follow the Original Graph or does it peel away?" }]
    );
  }
}

function handleSpareFitClick() {
  if (sparePointList.length < 3) {
    alert('Plot at least 3 points before fitting.');
    return;
  }
  const spareFit = fitHill(sparePointList);
  if (!spareFit) { alert('Could not fit a curve.'); return; }

  spareFittedCurve  = { points: sampleSpareCurve(spareFit, 1), params: spareFit };
  spareGraphPlotted = true;

  // Spotlight 24 — plateau of Ligand C vs Original Graph
  const curvePts  = spareFittedCurve.points;
  const plateauX  = map(sliderMax * 0.45, sliderMin, sliderMax, 80, w - 80);
  const plateauPts = curvePts.filter(p => p.x >= plateauX - 80 && p.x <= plateauX + 80);
  const spX = plateauX;
  const spY = plateauPts.length > 0
    ? plateauPts.reduce((s, p) => s + p.y, 0) / plateauPts.length
    : map(spareFit.Emax * 0.95, yMin, yMax, h - 80, 80);

  showSpotlight(24, spX, spY, { w: 260, h: 110, r: 8 },
    "Same Emax — But It Got There Faster",
    [{ label: "The Receptors Were Not All Occupied",
       text: "Ligand C reached the same maximum response as your Chapter 1 curve — but at a fraction of the concentration. That means some receptors were never needed. Clark's model says you need to fill all of them. The data says otherwise." }]
  );
  activeSpotlight.highlightCurve = true;
}

function sampleSpareCurve(params, stepX = 1) {
  const out = [];
  // Only draw curve over the restricted slider range — don't extrapolate past what's measurable
  for (let x = sliderMin; x <= SPARE_SLIDER_MAX; x += stepX) {
    const y  = params.Emax * hillG(x, params.EC50, params.n);
    const sx = map(x, sliderMin, sliderMax, 80, w - 80);
    const sy = map(y, yMin, yMax, h - 80, 80);
    out.push({ x: sx, y: sy });
  }
  return out;
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
  stroke(180, 140, 0);
  strokeWeight(2);
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
  if (scene === 'title') {
    drawTitleScene();
    return;
  }

  if (scene === 'clarkIntro') {
    drawClarkIntroScene();
    return;
  }

  if (scene === 'tissueTransition') {
    drawTissueTransitionScene();
    return;
  }

  if (scene === 'dataCollected') {
    drawDataCollectedScene();
    return;
  }

  // ACH GRAPH — keeps VESSEL and 4 receptors

  if (scene === 'intro') {
    drawIntroScene();
    return;
  }

  if (scene === 'clarkSummary') {
    drawClarkSummaryScene();
    return;
  }

  if (scene === 'clarkProblems') {
    drawClarkProblemsScene();
    return;
  }

  if (scene === 'limitationTitle') {
    drawLimitationTitleScene();
    return;
  }

  if (scene === 'partialGraph') {
    drawPartialGraphScene();
    return;
  }

  if (scene === 'spareGraph') {
    drawSpareGraphScene();
    return;
  }

  if (scene === 'ch2Analysis') {
    drawCh2AnalysisScene();
    return;
  }

  if (scene === 'mrtTitle') {
    drawMrtTitleScene();
    return;
  }

  if (scene === 'mrtBasal') {
    drawMrtBasalScene();
    return;
  }

  if (scene === 'mrtAnalysis') {
    drawMrtAnalysisScene();
    return;
  }

  if (scene === 'mrtPrinciples') {
    drawMrtPrinciplesScene();
    return;
  }

  if (scene === 'achGraph') {
    // Gate UI behind overlay or spotlight
    if (showGraphOverlay) {
      hideUIElements();
      gotItButton.show();
    } else if (activeSpotlight) {
      hideUIElements();
      if (activeSpotlight.num === 93) slider.show(); // slider spotlight: keep it visible
    } else {
      showUIElements();
      gotItButton.hide();
      if (graphPlotted) continueButton.show(); else continueButton.hide();
    }

    background(173, 216, 230);
    stroke(180, 140, 0); strokeWeight(2);
    line(640, 0, 640, 720);

    // Left graph panel (white)
    fill(255); noStroke();
    rect(0, 0, 640, 720);

    // Grid
    stroke(180); strokeWeight(1);
    for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
    for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

    // Axes
    strokeWeight(2); stroke(0);
    line(80, h - 80, w - 80, h - 80);
    line(80, 80, 80, h - 80);

    // ── Graph labels ──
    noStroke(); textAlign(CENTER);

    // Scene title
    fill(40, 45, 70);
    textStyle(BOLD); textSize(16);
    text("Field Stop 1  —  Diaphragm Muscle", 320, 38);
    textStyle(NORMAL);

    // Graph area title
    fill(90, 95, 120); textSize(13);
    text('Concentration — Response Curve', 320, 68);

    // X axis label
    fill(60, 65, 90); textSize(13);
    text('Acetylcholine Concentration (units)', 320, 452);

    // Y axis label (rotated)
    push();
    translate(18, 250);
    rotate(-HALF_PI);
    textAlign(CENTER);
    fill(60, 65, 90); textSize(13);
    text('Binding Rate (binds / sec)', 0, 0);
    pop();

    // Live stats (compact, below x-axis)
    let currentTime = millis();
    let elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;
    let averageAttachments = elapsedSeconds > 0 ? attachmentTimes.length / elapsedSeconds : 0;

    fill(120, 125, 150); textSize(12); textAlign(LEFT);
    text(`Concentration: ${Math.round(concentration)}`, 85, 478);
    textAlign(RIGHT);
    text(`Rate: ${averageAttachments.toFixed(2)} /sec`, 555, 478);

    // Right side: simulation
    image(Diaphragm, 735, 10, 450, 250);

    noFill(); stroke(255, 255, 102);
    rect(945, 80, 30, 30);
    line(945, 110, 640, 300);
    line(975, 110, 1280, 300);
    line(640, 300, 1280, 300);

    image(membrane, 640, 580, 750, 105);

    // 4 GPCRs + binding rectangles
    image(gpcr, 630, 530, 200, 200);
    image(gpcr, 790, 530, 200, 200);
    image(gpcr, 950, 530, 200, 200);
    image(gpcr, 1100, 530, 200, 200);

    fill(255, 0, 0); noStroke();
    rect(925, 580, 10, 20);
    rect(755, 585, 10, 20);
    rect(1095, 555, 10, 20);
    rect(1225, 555, 10, 20);

    // Balls
    stroke(0); noFill();
    for (let a of particles) {
      if (!activeSpotlight) { a.bounceOthers(); a.update(); }
      a.display();
    }

    // Ghost point
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);
    currentTime = millis();
    elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;
    averageAttachments = elapsedSeconds > 0 ? attachmentTimes.length / elapsedSeconds : 0;
    ghostPoint.y = constrain(averageAttachments, yMin, yMax);

    const ghostXCoord = map(ghostPoint.x, sliderMin, sliderMax, 80, w - 80);
    const ghostYCoord = map(ghostPoint.y, yMin, yMax, h - 80, 80);
    fill(255, 0, 0, ghostPoint.alpha); noStroke();
    ellipse(ghostXCoord, ghostYCoord, 10, 10);

    // Plotted points
    for (let point of pointList) {
      const xCoord = map(point.x, sliderMin, sliderMax, 80, w - 80);
      const yCoord = map(point.y, yMin, yMax, h - 80, 80);
      fill(255, 0, 0, point.alpha); noStroke();
      ellipse(xCoord, yCoord, 10, 10);
    }

    // Fitted curve
    if (fittedCurve && fittedCurve.points?.length) {
      stroke(0, 0, 255); strokeWeight(3); noFill();
      beginShape();
      for (const p of fittedCurve.points) vertex(p.x, p.y);
      endShape();
    }

    // Overlay on top of everything
    if (showGraphOverlay) drawGraphOverlay();

    drawSpotlight();
    frameCounter++;
    return;
  }

  // HEART GRAPH — replaces VESSEL with HEART and draws 6 small GPCRs in a straight line
  if (scene === 'heartGraph') {
    if (activeSpotlight) { hideUIElements(); } else { showUIElements(); }

    background(173, 216, 230);
    stroke(180, 140, 0); strokeWeight(2);
    line(640, 0, 640, 720);

    fill(255); noStroke();
    rect(0, 0, 640, 720);

    stroke(180); strokeWeight(1);
    for (let i = 4; i <= h / (unit + 3); i++) line(80, 20 * i, w - 80, 20 * i);
    for (let i = 4; i <= w / (unit + 2.5); i++) line(20 * i, 80, 20 * i, h - 80);

    strokeWeight(2); stroke(0);
    line(80, h - 80, w - 80, h - 80);
    line(80, 80, 80, h - 80);

    // ── Graph labels (matching achGraph) ──
    noStroke(); textAlign(CENTER);

    fill(40, 45, 70);
    textStyle(BOLD); textSize(16);
    text("Field Stop 2  —  Intestinal Muscle", 320, 38);
    textStyle(NORMAL);

    fill(90, 95, 120); textSize(13);
    text('Concentration — Response Curve', 320, 68);

    fill(60, 65, 90); textSize(13);
    text('Acetylcholine Concentration (units)', 320, 452);

    push();
    translate(18, 250); rotate(-HALF_PI);
    textAlign(CENTER); fill(60, 65, 90); textSize(13);
    text('Binding Rate (binds / sec)', 0, 0);
    pop();

    let currentTime = millis();
    let elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;
    let averageAttachments = elapsedSeconds > 0 ? attachmentTimes.length / elapsedSeconds : 0;

    fill(120, 125, 150); textSize(12); textAlign(LEFT);
    text(`Concentration: ${Math.round(concentration)}`, 85, 478);
    textAlign(RIGHT);
    text(`Rate: ${averageAttachments.toFixed(2)} /sec`, 555, 478);

    // Replace vessel with heart
    image(Smallintestine, 740, -20, 400, 300);

    noFill(); stroke(255, 255, 102);
    rect(945, 80, 30, 30);
    line(945, 110, 640, 300);
    line(975, 110, 1280, 300);
    line(640, 300, 1280, 300);

    image(membrane, 635, 575, 665, 60);

    // 6 GPCRs (smaller) in one line + binding rectangles (10×20)
    {
      const { gpcrPos, rects, size } = getHeartLayout();
      for (const p of gpcrPos) image(gpcr, p.x, p.y, size, size);
      fill(255, 0, 0); noStroke();
      for (const r of rects) rect(r.x, r.y, r.w, r.h);
    }

    stroke(0); noFill();
    for (let a of particles) {
      if (!activeSpotlight) { a.bounceOthers(); a.update(); }
      a.display();
    }

    // Always show the live ghost preview (optional: show only when not plotted)
    ghostPoint.x = constrain(slider.value(), sliderMin, sliderMax);
    currentTime = millis();
    elapsedSeconds = (currentTime - lastBallCountChangeTime) / 1000;
    averageAttachments = elapsedSeconds > 0 ? attachmentTimes.length / elapsedSeconds : 0;
    ghostPoint.y = constrain(averageAttachments, yMin, yMax);

    const ghostXCoord = map(ghostPoint.x, sliderMin, sliderMax, 80, w - 80);
    const ghostYCoord = map(ghostPoint.y, yMin, yMax, h - 80, 80);
    fill(255, 0, 0, ghostPoint.alpha);
    noStroke();
    ellipse(ghostXCoord, ghostYCoord, 10, 10);

    // Draw all saved points (no cap)
    if (pointList.length > 0) {
      for (let point of pointList) {
        const xCoord = map(point.x, sliderMin, sliderMax, 80, w - 80);
        const yCoord = map(point.y, yMin, yMax, h - 80, 80);
        fill(255, 0, 0, point.alpha);
        noStroke();
        ellipse(xCoord, yCoord, 10, 10);
      }
    }
    
    // Draw fitted curve if available
    if (fittedCurve && fittedCurve.points?.length) {
      stroke(0, 0, 255);
      strokeWeight(3);
      noFill();
      beginShape();
      for (const p of fittedCurve.points) vertex(p.x, p.y);
      endShape();
    }
    

    frameCounter++;

    // New: allow unlimited points, but require minimum before showing Graph button

    if (graphPlotted === true) {
      continueButton.show();
    } else {
      continueButton.hide();
    }

    drawSpotlight();
    return;
  }

  if (scene === 'compareGraphs') {
    drawCompareGraphsScene();
    if (activeSpotlight) {
      if (slideButton) slideButton.hide();
      if (compareP5Button) compareP5Button.hide();
    } else {
      if (slideButton) slideButton.show();
      if (compareP5Button) compareP5Button.show();
    }
    drawSpotlight();
    return;
  }

  // Other scenes (if you add them later)
}

// ─────────────────────────────────────────────
// Additional Scene Draw Functions
// ─────────────────────────────────────────────
// Seeds approximate plot data + fitted curves for all 5 drug types (dev only)
function seedMrtDevData() {
  mrtPlotData = {};
  mrtFittedCurves = {};
  mrtCompletedDrugs = new Set(MRT_DRUGS.map(d => d.type));

  const configs = [
    { type: 'fullAgonist',    Emax: 0.92, Floor: 0.5,  EC50: 8,  n: 1.6, isInverse: false },
    { type: 'partialAgonist', Emax: 0.68, Floor: 0.5,  EC50: 12, n: 1.4, isInverse: false },
    { type: 'antagonist',     Emax: 0.52, Floor: 0.5,  EC50: 20, n: 0.8, isInverse: false },
    { type: 'partialInverse', Emax: 0.5,  Floor: 0.28, EC50: 10, n: 1.3, isInverse: true  },
    { type: 'fullInverse',    Emax: 0.5,  Floor: 0.05, EC50: 6,  n: 1.7, isInverse: true  },
  ];

  for (const cfg of configs) {
    mrtFittedCurves[cfg.type] = { Emax: cfg.Emax, Floor: cfg.Floor, EC50: cfg.EC50, n: cfg.n, isInverse: cfg.isInverse };
    // Generate 6 sample points along the curve with a little noise
    const pts = [];
    for (const x of [1, 3, 8, 20, 50, 80]) {
      const h = hillG(x, cfg.EC50, cfg.n);
      const y = cfg.isInverse
        ? cfg.Emax - (cfg.Emax - cfg.Floor) * h
        : cfg.Floor + (cfg.Emax - cfg.Floor) * h;
      pts.push({ x, y: constrain(y + random(-0.04, 0.04), 0, 1) });
    }
    mrtPlotData[cfg.type] = pts;
  }
}

// ─────────────────────────────────────────────
// MRT Analysis Scene
// ─────────────────────────────────────────────
function drawMrtAnalysisScene() {
  background(8, 32, 28);

  // ── HEADER ──
  fill(10, 42, 38); noStroke(); rect(0, 0, width, 4);
  fill(10, 42, 38); noStroke(); rect(0, 4, width, 46);
  fill(46, 196, 160); textAlign(CENTER); textStyle(BOLD); textSize(11);
  text('CHAPTER III  ·  MODERN RECEPTOR THEORY  ·  ANALYSIS', width / 2, 26);
  textStyle(NORMAL); fill(120, 200, 180); textSize(10);
  text('Five drugs. Five curves. What does it mean?', width / 2, 42);
  fill(6, 25, 22); noStroke(); rect(0, 50, width, 3);

  // ── 5 MINI-GRAPHS ──
  const COL_W = 256;
  const GY1 = 62, GY2 = 248;
  const logMin = Math.log10(MRT_SLIDER_MIN), logMax = Math.log10(MRT_SLIDER_MAX);

  const drugExplanations = [
    {
      drug: MRT_DRUGS[0],
      clark: ['Clark predicted this.',
              'More drug → more response → 100% Emax.',
              'He got the shape right.'],
      mrt:   ['Full agonists bind only R*,',
              'driving the equilibrium all the way right.',
              'Also explains spare receptors (Ch. II).']
    },
    {
      drug: MRT_DRUGS[1],
      clark: ['Clark couldn\'t explain this.',
              'His model predicts 100% at saturation —',
              'this curve never gets there.'],
      mrt:   ['Binds both R and R* — equilibrium stays split.',
              'The ceiling is the drug\'s intrinsic efficacy,',
              'not how many receptors are occupied.']
    },
    {
      drug: MRT_DRUGS[2],
      clark: ['Clark called it competitive antagonism.',
              'Blocks response without causing one.',
              'Descriptively right, mechanistically empty.'],
      mrt:   ['Binds R and R* with equal affinity.',
              'Zero net shift in the equilibrium.',
              'Occupies without changing state.']
    },
    {
      drug: MRT_DRUGS[3],
      clark: ['Impossible in Clark\'s model.',
              'A drug can\'t suppress baseline activity',
              'if baseline activity doesn\'t exist.'],
      mrt:   ['Preferentially binds R (inactive state).',
              'Suppresses spontaneous R* activity.',
              'Partial negative efficacy is real.']
    },
    {
      drug: MRT_DRUGS[4],
      clark: ['Doesn\'t exist in Clark\'s framework.',
              'Negative efficacy has no place',
              'in a pure occupancy model.'],
      mrt:   ['Locks all receptors into R.',
              'Drains the R* pool to near zero —',
              'the mirror image of a full agonist.']
    }
  ];

  for (let i = 0; i < 5; i++) {
    const entry = drugExplanations[i];
    const drug  = entry.drug;
    const [dr, dg, db] = [drug.r, drug.g, drug.b];
    const cx = i * COL_W;
    const gx1 = cx + 26, gx2 = cx + COL_W - 16;
    const gw = gx2 - gx1, gh = GY2 - GY1;

    // Card background
    fill(12, 46, 40); noStroke(); rect(cx, 53, COL_W - 1, GY2 - 53 + 4);

    // Drug name tab
    fill(dr, dg, db, 220); noStroke(); rect(cx, 53, COL_W - 1, 16);
    fill(255); textAlign(CENTER); textStyle(BOLD); textSize(9.5);
    text(drug.label, cx + COL_W / 2, 63);
    textStyle(NORMAL);

    // Graph area
    fill(6, 28, 24); noStroke(); rect(gx1, GY1, gw, gh);

    // Grid lines
    stroke(20, 65, 55); strokeWeight(1);
    for (let p = 25; p <= 100; p += 25) {
      const ly = GY2 - (p / 100) * gh;
      line(gx1, ly, gx2, ly);
    }

    // Basal 50% dashed
    stroke(46, 120, 100); strokeWeight(1);
    drawingContext.setLineDash([4, 3]);
    line(gx1, GY2 - 0.5 * gh, gx2, GY2 - 0.5 * gh);
    drawingContext.setLineDash([]);

    // Axes
    stroke(46, 110, 90); strokeWeight(1.2);
    line(gx1, GY1, gx1, GY2); line(gx1, GY2, gx2, GY2);

    // Y labels
    noStroke(); fill(80, 160, 140); textSize(8); textAlign(RIGHT);
    for (let p = 0; p <= 100; p += 50) text(p + '%', gx1 - 3, GY2 - (p / 100) * gh + 3);

    // Draw curve
    const curve = mrtFittedCurves[drug.type];
    if (curve) {
      const pts = sampleMrtCurve(curve, gx1, gx2, GY1, GY2);
      stroke(dr, dg, db); strokeWeight(2); noFill();
      beginShape();
      for (const p of pts) vertex(p.x, p.y);
      endShape();
    } else {
      // Fallback placeholder
      noStroke(); fill(40, 90, 75); textAlign(CENTER); textSize(9); textStyle(ITALIC);
      text('(no data)', cx + COL_W / 2, GY1 + gh / 2);
      textStyle(NORMAL);
    }

    // ── Explanation text below graph ──
    let ty = GY2 + 18;
    const tx = cx + 12;
    const tw = COL_W - 20;

    // Clark row
    fill(200, 80, 80); textStyle(BOLD); textSize(9.5); textAlign(LEFT);
    text('CLARK', tx, ty); ty += 14;
    textStyle(NORMAL); fill(230, 200, 200); textSize(9.5);
    for (const ln of entry.clark) { text(ln, tx, ty, tw, 14); ty += 14; }

    ty += 8;
    stroke(30, 90, 75); strokeWeight(1); line(cx + 10, ty, cx + COL_W - 10, ty); ty += 10; noStroke();

    // MRT row
    fill(46, 196, 160); textStyle(BOLD); textSize(9.5); textAlign(LEFT);
    text('MRT', tx, ty); ty += 14;
    textStyle(NORMAL); fill(140, 210, 195); textSize(9.5);
    for (const ln of entry.mrt) { text(ln, tx, ty, tw, 14); ty += 14; }
  }

  // Vertical dividers between columns
  stroke(20, 65, 55); strokeWeight(1);
  for (let i = 1; i < 5; i++) line(i * COL_W, 53, i * COL_W, 500);

  // ── CONCLUSION STRIP ──
  const cy = 502;
  fill(10, 42, 38); noStroke(); rect(0, cy, width, 720 - cy);
  stroke(46, 196, 160, 80); strokeWeight(1); line(0, cy, width, cy); noStroke();

  fill(180, 240, 225); textAlign(CENTER); textStyle(BOLD); textSize(28);
  text('This is Modern Receptor Theory.', width / 2, cy + 54);
  textStyle(NORMAL);

  fill(120, 200, 180); textSize(14); textAlign(CENTER);
  text('Not a patch on Clark\'s work — a complete replacement. It explains every curve above,', width / 2, cy + 90);
  text('including the ones Clark said couldn\'t exist.', width / 2, cy + 108);

  stroke(46, 196, 160, 40); strokeWeight(1); line(200, cy + 125, 1080, cy + 125); noStroke();

  fill(46, 196, 160); textSize(12); textStyle(ITALIC); textAlign(CENTER);
  text('Ariëns introduced intrinsic activity (1954)  ·  Stephenson formalized efficacy (1956)  ·  Black & Leff built the two-state model (1983)', width / 2, cy + 147);
  textStyle(NORMAL);

  fill(80, 150, 130); textSize(11); textAlign(CENTER);
  text('Together, they gave pharmacology a framework that still holds today.', width / 2, cy + 168);

  // "MRT Principles →" slideButton drawn over this strip — shows via initializeScene
}

// ─────────────────────────────────────────────
// MRT Principles Scene
// ─────────────────────────────────────────────
function drawMrtPrinciplesScene() {
  const BG   = [10, 42, 38];
  const TEAL = [46, 196, 160];
  const LT   = [180, 240, 225];
  const MID  = [100, 185, 165];
  const DIM  = [60, 120, 105];

  background(BG[0], BG[1], BG[2]);

  // ── Header ──
  fill(6, 28, 24); noStroke(); rect(0, 0, width, 58);
  fill(TEAL[0], TEAL[1], TEAL[2]); noStroke(); rect(0, 0, width, 4);
  fill(LT[0], LT[1], LT[2]); textAlign(CENTER); textStyle(BOLD); textSize(13);
  text('SUMMARY  ·  A CENTURY OF PHARMACOLOGY', width / 2, 28);
  textStyle(NORMAL); fill(MID[0], MID[1], MID[2]); textSize(11);
  text('From Clark\'s first experiments to the theory that explains everything.', width / 2, 46);

  // ── Timeline ──
  const events = [
    {
      year: '1926',
      who:  'A.J. Clark',
      what: 'Occupancy Theory',
      body: 'Effect is proportional to how many receptors a drug occupies. Clark assumed binding = activation — one drug property, one curve. Simple, powerful, and the foundation of quantitative pharmacology.',
      note: 'The sigmoid dose-response curve you built in Chapter I is Clark\'s model.',
      col:  [200, 180, 100],
    },
    {
      year: 'The cracks',
      who:  'Partial agonists · Spare receptors · Inverse agonists',
      what: 'Clark\'s Limitations',
      body: 'Partial agonists plateau below Emax even at full occupancy. Tissues with spare receptors reach full response at low occupancy. Some drugs suppress activity below baseline — impossible in Clark\'s world.',
      note: 'Binding and activation are not the same thing. Clark had no way to say that.',
      col:  [210, 80, 80],
    },
    {
      year: '1954 – 1983',
      who:  'Ariëns · Stephenson · Del Castillo & Katz · Black & Leff',
      what: 'Modern Receptor Theory',
      body: 'Receptors spontaneously toggle between R (inactive) and R* (active). Drugs don\'t activate receptors — they shift a pre-existing equilibrium. This single idea explains every drug type Clark couldn\'t account for.',
      note: 'Full agonists, partial agonists, antagonists, inverse agonists — all explained by R ⇌ R*.',
      col:  [46, 196, 160],
    },
  ];

  // Timeline bar — endpoints inset so cards stay on screen for any n
  const TY = 218;
  const n = events.length;
  const halfCard = n <= 2 ? 240 : n === 3 ? 165 : 97;
  const TX1 = halfCard + 10, TX2 = width - halfCard - 10;
  stroke(TEAL[0], TEAL[1], TEAL[2], 80); strokeWeight(2);
  line(TX1, TY, TX2, TY);

  const spacing = (TX2 - TX1) / (n - 1);

  for (let i = 0; i < n; i++) {
    const ev = events[i];
    const ex = TX1 + i * spacing;
    const [er, eg, eb] = ev.col;

    // Dot on timeline
    fill(er, eg, eb); noStroke();
    circle(ex, TY, 14);
    stroke(BG[0], BG[1], BG[2]); strokeWeight(2);
    noFill(); circle(ex, TY, 14); noStroke();

    // Year above
    fill(er, eg, eb); textAlign(CENTER); textStyle(BOLD); textSize(13);
    text(ev.year, ex, TY - 22);
    textStyle(NORMAL);

    // Tick
    stroke(er, eg, eb, 120); strokeWeight(1);
    line(ex, TY - 14, ex, TY + 14);

    // Card below
    const cardW = n <= 2 ? 480 : n === 3 ? 330 : 193;
    const cardX = ex - cardW / 2, cardY = TY + 22, cardH = 205;
    fill(8, 35, 30); noStroke(); rect(cardX, cardY, cardW, cardH, 6);
    stroke(er, eg, eb, 60); strokeWeight(1); rect(cardX, cardY, cardW, cardH, 6); noStroke();

    // Who + what
    const tsz = n <= 2 ? 12 : 10;
    fill(er, eg, eb); textStyle(BOLD); textSize(tsz); textAlign(LEFT);
    text(ev.who, cardX + 14, cardY + 20);
    fill(LT[0], LT[1], LT[2]); textStyle(BOLD); textSize(tsz + 2);
    text(ev.what, cardX + 14, cardY + 36);
    textStyle(NORMAL);

    // Body
    fill(MID[0], MID[1], MID[2]); textSize(n <= 2 ? 11 : 9.5); textAlign(LEFT);
    text(ev.body, cardX + 14, cardY + 50, cardW - 28, 100);

    // Note (italic, dimmer)
    fill(DIM[0], DIM[1], DIM[2]); textSize(n <= 2 ? 10.5 : 9); textStyle(ITALIC);
    text(ev.note, cardX + 14, cardY + 158, cardW - 28, 50);
    textStyle(NORMAL);
  }

  // ── Bottom strip ──
  fill(6, 24, 20); noStroke(); rect(0, 638, width, 82);
  stroke(TEAL[0], TEAL[1], TEAL[2], 50); strokeWeight(1); line(0, 638, width, 638); noStroke();

  fill(LT[0], LT[1], LT[2]); textStyle(BOLD); textSize(16); textAlign(CENTER);
  text('Modern Receptor Theory gives us one equation for all of it: R ⇌ R*', width / 2, 666);
  textStyle(NORMAL); fill(MID[0], MID[1], MID[2]); textSize(11);
  text('Full agonists, partial agonists, antagonists, inverse agonists — all explained by a single molecular equilibrium.', width / 2, 686);
  // "Finish →" slideButton sits here via initializeScene
}

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
  // Spotlight takes priority — click dismisses it, but guard against same-event dismiss
  if (activeSpotlight) {
    if (millis() - spotlightShownAtMillis > 300) dismissSpotlight();
    return;
  }

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
