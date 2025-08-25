// ===============================
// ball.js
// ===============================

// Attached ligand slots (one per receptor hitbox)
let attachedLigands = [null, null, null, null];

// Detach/bounce tuning
let detachmentTime = 20;
let graceDuration = 70;     // frames after detachment where re-attach is blocked
let animationDuration = 10; // frames for attach animation

// Default receptor hitboxes (ACH graph: 4)
let rectangles = [
  { x: 925,  y: 580, w: 10, h: 20 },
  { x: 755,  y: 585, w: 10, h: 20 },
  { x: 1095, y: 555, w: 10, h: 20 },
  { x: 1225, y: 555, w: 10, h: 20 }
];

// Constrain ball bouncing inside rect(640, 300, 1280, 305)
let boundingBox = { x: 640, y: 300, w: 640, h: 320 };

/**
 * Allow sketch.js to change receptor layout per scene.
 * Pass an array of {x,y,w,h} hitboxes.
 */
function setReceptorLayout(rects) {
  rectangles = Array.isArray(rects) ? rects.slice() : rectangles;
  attachedLigands = new Array(rectangles.length).fill(null);

  // NEW: inform sketch.js so occupancy slots match receptor count
  if (typeof initOccupancySlots === 'function') initOccupancySlots(rectangles.length);
}


// -------------------------------
// Ball class
// -------------------------------
function Ball(pos, vel, radius, identity, others, color, follow, isInhibitor = false, isLigand = false, isWhiteBall = false) {
  this.coeficient = 1;
  this.position = pos.copy ? pos.copy() : pos;
  this.velocity = vel.copy ? vel.copy() : vel;
  this.acceleration = createVector(0, 0); // keep defined to avoid NaNs
  this.id = identity;
  this.others = others;
  this.r = radius;
  this.color = isInhibitor ? "red" : color;
  this.mass = this.r * 5;
  this.follow = follow;
  this.history = [];
  this.attachedTime = 0;
  this.gracePeriod = 0;
  this.animating = false;
  this.animationProgress = 0;
  this.startPosition = null;
  this.targetPosition = null;
  this.storedVelocity = null;
  this.attachedRectIndex = -1;

  this.velocity.mult(5);

  this.isInhibitor = isInhibitor;
  this.isLigand = isLigand;
  this.isWhiteBall = isWhiteBall;
}

Ball.prototype.update = function () {
  if (this.animating) {
    this.updateAnimation();
  } else {
    this.updateNormal();
  }

  // optional trail
  let v = createVector(this.position.x, this.position.y);
  this.history.push(v);
  if (frameRate() < 30) this.history.splice(0, 10);
};

Ball.prototype.updateAnimation = function () {
  this.animationProgress += 1 / animationDuration;
  if (this.animationProgress >= 1) {
    this.position = this.targetPosition.copy();
    this.animating = false;
    this.animationProgress = 0;
    this.attachedTime = 0;
  } else {
    let t = easeInOutQuad(this.animationProgress);
    this.position = p5.Vector.lerp(this.startPosition, this.targetPosition, t);
  }
};

Ball.prototype.updateNormal = function () {
  this.velocity.add(this.acceleration);
  this.position.add(this.velocity);

  if (this.gracePeriod > 0) this.gracePeriod--;

  this.checkRectangleCollisions();
  this.handleBoundingBoxCollisions();
};

Ball.prototype.checkRectangleCollisions = function () {
  for (let i = 0; i < rectangles.length; i++) {
    let rect = rectangles[i];

    if (this.attachedRectIndex === i) {
      this.handleAttachment(i);
    } else if (this.isCollidingWithRectangle(rect)) {
      if (attachedLigands[i] === null && this.attachedRectIndex === -1 && this.gracePeriod === 0) {
        this.attachToRectangle(i);
      } else {
        this.bounceOffRectangle(rect);
      }
    }
  }
};

Ball.prototype.isCollidingWithRectangle = function (rect) {
  return (
    this.position.x + this.r > rect.x &&
    this.position.x - this.r < rect.x + rect.w &&
    this.position.y + this.r > rect.y &&
    this.position.y - this.r < rect.y + rect.h
  );
};

Ball.prototype.bounceOffRectangle = function (rect) {
  let closestX = Math.max(rect.x, Math.min(this.position.x, rect.x + rect.w));
  let closestY = Math.max(rect.y, Math.min(this.position.y, rect.y + rect.h));
  let distanceX = this.position.x - closestX;
  let distanceY = this.position.y - closestY;
  let distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

  if (distance < this.r) {
    let overlap = this.r - distance;
    let angle = Math.atan2(distanceY, distanceX);
    this.position.x += overlap * Math.cos(angle);
    this.position.y += overlap * Math.sin(angle);
    let normal = createVector(Math.cos(angle), Math.sin(angle));
    let dot = this.velocity.dot(normal);
    if (dot < 0) this.velocity.sub(normal.mult(2 * dot));
  }
};

// ===============================
// ball.js — FINAL versions
// ===============================

// ATTACH: unchanged logic + NEW occupancy hook
// ATTACH: unchanged logic + NEW occupancy hook (agonists only)
Ball.prototype.attachToRectangle = function (rectIndex) {
  // (existing) store current velocity so we can relaunch with same speed later
  this.storedVelocity = this.velocity.copy();

  // (existing) animate into the receptor center
  let rect = rectangles[rectIndex];
  this.startAnimation(createVector(rect.x + rect.w / 2, rect.y + rect.h / 2));

  // (existing) mark slot occupied
  attachedLigands[rectIndex] = this;
  this.attachedRectIndex = rectIndex;

  // NEW: notify occupancy tracker (only count agonists, not inhibitors)
  if (
    !this.isInhibitor &&
    typeof markReceptorBound === 'function' &&
    Number.isInteger(rectIndex) && rectIndex >= 0
  ) {
    markReceptorBound(rectIndex, millis());
  }

  // (existing) record an attachment timestamp for your old metric (safe to keep)
  if (!this.isInhibitor) {
    if (typeof attachmentTimes !== 'undefined' && attachmentTimes.push) {
      attachmentTimes.push(millis());
    }
  }
};



// DETACH: unchanged logic + NEW occupancy hook (agonists only, robust index)
Ball.prototype.detachFromRectangle = function (rectIndex) {
  // Some callers (e.g., detachAllLigands in sketch.js) don’t pass rectIndex
  if (rectIndex == null || rectIndex === undefined) {
    rectIndex = this.attachedRectIndex;
  }

  // (existing) relaunch with same speed as before attachment
  const speed = this.storedVelocity ? this.storedVelocity.mag() : 0;
  let randomDir = p5.Vector.random2D();
  randomDir.setMag(speed);
  this.velocity = randomDir;

  // NEW: notify occupancy tracker BEFORE clearing the slot (agonists only)
  if (
    !this.isInhibitor &&
    typeof markReceptorUnbound === 'function' &&
    Number.isInteger(rectIndex) && rectIndex >= 0
  ) {
    markReceptorUnbound(rectIndex, millis());
  }

  // (existing) clear receptor slot and set grace period
  if (Number.isInteger(rectIndex) && rectIndex >= 0) {
    attachedLigands[rectIndex] = null;
  }
  this.attachedRectIndex = -1;
  this.gracePeriod = graceDuration;
};



Ball.prototype.handleAttachment = function (rectIndex) {
  this.attachedTime++;
  if (this.attachedTime > detachmentTime) this.detachFromRectangle(rectIndex);
};



Ball.prototype.handleBoundingBoxCollisions = function () {
  if (this.position.x > boundingBox.x + boundingBox.w - this.r) {
    this.position.x = boundingBox.x + boundingBox.w - this.r;
    this.velocity.x *= -1 * this.coeficient;
  } else if (this.position.x < boundingBox.x + this.r) {
    this.position.x = boundingBox.x + this.r;
    this.velocity.x *= -1 * this.coeficient;
  }
  if (this.position.y > boundingBox.y + boundingBox.h - this.r) {
    this.position.y = boundingBox.y + boundingBox.h - this.r;
    this.velocity.y *= -1 * this.coeficient;
  } else if (this.position.y < boundingBox.y + this.r) {
    this.position.y = boundingBox.y + this.r;
    this.velocity.y *= -1 * this.coeficient;
  }
};

Ball.prototype.startAnimation = function (targetPos) {
  this.animating = true;
  this.animationProgress = 0;
  this.startPosition = this.position.copy();
  this.targetPosition = targetPos.copy();
  this.velocity.set(0, 0);
};

Ball.prototype.display = function () {
  stroke(0);
  fill(this.color);
  circle(this.position.x, this.position.y, this.r * 2);

  if (this.follow) {
    for (let i = 0; i < this.history.length; i++) {
      var pos = this.history[i];
      push();
      fill("magenta");
      noStroke();
      ellipse(pos.x, pos.y, 2, 2);
      pop();
    }
  }
};

Ball.prototype.bounceOthers = function () {
  if (this.animating || this.attachedRectIndex !== -1) return;

  for (let i = 0; i < this.others.length; i++) {
    if (this.others[i] === this || this.others[i].animating || this.others[i].attachedRectIndex !== -1) continue;
    let relative = p5.Vector.sub(this.others[i].position, this.position);
    let distance = relative.mag() - (this.r + this.others[i].r);

    if (distance < 0) {
      let movement = relative.copy().setMag(Math.abs(distance / 2));
      this.position.sub(movement);
      this.others[i].position.add(movement);

      let thisToOtherNormal = relative.copy().normalize();
      let approachSpeed =
        this.velocity.dot(thisToOtherNormal) +
        -this.others[i].velocity.dot(thisToOtherNormal);
      let approachVector = thisToOtherNormal.copy().setMag(approachSpeed);
      this.velocity.sub(approachVector);
      this.others[i].velocity.add(approachVector);
    }
  }
};

function easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}
