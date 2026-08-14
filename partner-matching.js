// ============================================================================
// 225 — TRAINING PARTNER MATCHING ENGINE
// ============================================================================
// Pure computation, no Firebase or DOM, for the same reason as analytics.js:
// everything here can be tested in isolation against known inputs.
//
// The problem: given a set of users, decide which of them would make a good
// training partner for a given lifter. "Good" is not a single quantity, so
// this is a multi-criteria similarity problem rather than a lookup.
//
// Four independent signals are computed and combined:
//
//   1. Strength profile similarity  — cosine similarity over movement-pattern
//                                     volume vectors (Salton, 1975)
//   2. Absolute strength proximity  — Gaussian decay on relative difference
//   3. Geographic proximity         — Haversine great-circle distance
//   4. Activity overlap             — Jaccard index over training modalities
//
// Cosine similarity is used for the profile comparison specifically because it
// is scale-invariant: it compares the SHAPE of a training split rather than
// its magnitude. A beginner and an advanced lifter who both train push, pull
// and legs in equal proportion score as highly similar in profile, and their
// difference in absolute load is then handled separately by signal 2. Using a
// raw distance metric would have conflated those two very different questions.
// ============================================================================

import { classifyExercise, volumeLoad, estimate1RM } from "./analytics.js";


// ---------------------------------------------------------------------------
// SIGNAL 1 — STRENGTH PROFILE SIMILARITY (cosine)
// ---------------------------------------------------------------------------

/**
 * Builds a movement-pattern volume vector for one lifter.
 * Returns proportions rather than totals, so the vector describes how someone
 * trains rather than how much.
 */
export function buildProfileVector(entries) {
  const totals = { push: 0, pull: 0, legs: 0, core: 0 };

  for (const e of entries) {
    if (e.type === "cardio") continue;
    const pattern = classifyExercise(e.exerciseId || e.exerciseName);
    if (pattern in totals) totals[pattern] += volumeLoad(e);
  }

  const sum = Object.values(totals).reduce((a, b) => a + b, 0);
  if (sum === 0) return null; // no usable strength history

  return {
    push: totals.push / sum,
    pull: totals.pull / sum,
    legs: totals.legs / sum,
    core: totals.core / sum,
  };
}

/**
 * Cosine similarity between two vectors: the cosine of the angle between them.
 *
 * Returns 1.0 for identical direction, 0.0 for orthogonal. Because all volume
 * components are non-negative, the result is bounded to [0, 1] here rather
 * than the full [-1, 1] the measure allows in general.
 */
export function cosineSimilarity(a, b) {
  if (!a || !b) return null;

  const keys = Object.keys(a);
  let dot = 0, magA = 0, magB = 0;

  for (const k of keys) {
    const va = a[k] || 0;
    const vb = b[k] || 0;
    dot += va * vb;
    magA += va * va;
    magB += vb * vb;
  }

  if (magA === 0 || magB === 0) return null;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}


// ---------------------------------------------------------------------------
// SIGNAL 2 — ABSOLUTE STRENGTH PROXIMITY
// ---------------------------------------------------------------------------

/**
 * A lifter's overall strength level, taken as the mean of their best estimated
 * 1RM in each movement pattern they train. Averaging across patterns rather
 * than using a single lift avoids ranking someone highly just because they
 * happen to train one exercise heavily.
 */
export function strengthLevel(entries) {
  const bests = { push: 0, pull: 0, legs: 0 };

  for (const e of entries) {
    if (e.type === "cardio") continue;
    const pattern = classifyExercise(e.exerciseId || e.exerciseName);
    if (!(pattern in bests)) continue;

    const est = estimate1RM(e.weight, e.reps);
    if (est && est.estimate > bests[pattern]) bests[pattern] = est.estimate;
  }

  const trained = Object.values(bests).filter((v) => v > 0);
  if (trained.length === 0) return null;

  return trained.reduce((a, b) => a + b, 0) / trained.length;
}

/**
 * Scores how closely two lifters match in absolute strength.
 *
 * Uses a Gaussian decay on the RELATIVE difference rather than the absolute
 * one, because a 50 lb gap means something very different at 135 than at 400.
 * Tolerance is the relative difference at which the score falls to about 0.61.
 *
 * Perfect equality is not the goal — a partner who is somewhat stronger is
 * often more useful than an identical one — but the further apart two lifters
 * are, the less they can share equipment, loading, or spotting duty.
 */
export function strengthProximity(levelA, levelB, tolerance = 0.35) {
  if (!levelA || !levelB || levelA <= 0 || levelB <= 0) return null;

  const relativeDiff = Math.abs(levelA - levelB) / ((levelA + levelB) / 2);
  return Math.exp(-Math.pow(relativeDiff / tolerance, 2));
}


// ---------------------------------------------------------------------------
// SIGNAL 3 — GEOGRAPHIC PROXIMITY (Haversine)
// ---------------------------------------------------------------------------

const EARTH_RADIUS_KM = 6371;

/**
 * Great-circle distance between two coordinates, in kilometres.
 *
 * The Haversine formula accounts for the curvature of the Earth. Treating
 * latitude and longitude as a flat plane produces errors that grow with
 * distance and with latitude — at 40 degrees north, a degree of longitude is
 * about 23% shorter than a degree of latitude, so flat-plane distance would
 * systematically distort east-west separation.
 */
export function haversineDistance(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);

  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Nearest pairing between two lifters' training locations.
 * Returns the shortest distance and which two places produced it, so the
 * interface can explain the match rather than just scoring it.
 */
export function nearestLocationPair(locationsA, locationsB) {
  if (!locationsA?.length || !locationsB?.length) return null;

  let best = null;

  for (const a of locationsA) {
    if (typeof a.lat !== "number" || typeof a.lng !== "number") continue;
    for (const b of locationsB) {
      if (typeof b.lat !== "number" || typeof b.lng !== "number") continue;

      const km = haversineDistance(a.lat, a.lng, b.lat, b.lng);
      if (!best || km < best.km) best = { km, from: a.name, to: b.name, activity: b.activity };
    }
  }

  return best;
}

/**
 * Converts a distance into a 0–1 score using exponential decay.
 * Half-life is the distance at which the score falls to 0.5, so the penalty
 * is gentle for nearby gyms and steep beyond commuting range.
 */
export function proximityScore(km, halfLifeKm = 8) {
  if (km === null || km === undefined || !isFinite(km)) return null;
  return Math.pow(0.5, km / halfLifeKm);
}


// ---------------------------------------------------------------------------
// SIGNAL 4 — ACTIVITY OVERLAP (Jaccard)
// ---------------------------------------------------------------------------

/**
 * Jaccard index: the size of the intersection over the size of the union.
 *
 * Chosen over a simple intersection count because it penalises mismatch in
 * both directions. Two lifters who both do only strength training score 1.0;
 * one who does five activities and another who does one of them scores 0.2,
 * which correctly reflects that most of their training would not overlap.
 */
export function jaccardIndex(setA, setB) {
  const a = new Set(setA);
  const b = new Set(setB);

  // An empty set means "we don't know what they train", not "they train
  // nothing in common with you". Returning 0 here would treat absence of
  // information as evidence of incompatibility, which then drags down the
  // composite score of anyone who simply hasn't filled in their profile yet.
  // Returning null lets the caller exclude the signal and renormalise instead.
  if (a.size === 0 || b.size === 0) return null;

  let intersection = 0;
  for (const item of a) if (b.has(item)) intersection++;

  const union = a.size + b.size - intersection;
  return union === 0 ? null : intersection / union;
}

/** Extracts the distinct activities a lifter trains, from locations and logged entries. */
export function activitySet(locations, entries) {
  const set = new Set();

  for (const loc of locations || []) {
    if (loc.activity) set.add(loc.activity.toLowerCase());
  }
  for (const e of entries || []) {
    if (e.type === "cardio" && e.activity) set.add("cardio");
    else if (e.type !== "cardio") set.add("strength");
  }

  return [...set];
}


// ---------------------------------------------------------------------------
// COMPOSITE SCORING
// ---------------------------------------------------------------------------

// Weights are a modelling decision, not a derived result. Geographic
// proximity carries the most weight because a partner who cannot physically
// be there is not a partner regardless of how well their training matches.
// These are stated openly so they can be challenged, and are configurable
// so the effect of changing them can be measured.
export const DEFAULT_WEIGHTS = {
  proximity: 0.35,
  profile: 0.25,
  strength: 0.25,
  activity: 0.15,
};

/**
 * Scores one candidate against the current user.
 *
 * Signals that cannot be computed — a candidate with no logged workouts, or
 * no saved locations — are excluded and the remaining weights are
 * renormalised, rather than being scored as zero. Scoring a missing signal as
 * zero would systematically punish new users for having no history yet, which
 * would make the feature useless precisely when someone most needs it.
 */
export function scoreCandidate(me, candidate, weights = DEFAULT_WEIGHTS) {
  const signals = {};
  const reasons = [];

  // --- profile similarity ---
  const cos = cosineSimilarity(me.profileVector, candidate.profileVector);
  if (cos !== null) {
    signals.profile = cos;
    if (cos > 0.9) reasons.push("Very similar training split");
    else if (cos > 0.75) reasons.push("Similar training split");
  }

  // --- absolute strength ---
  const prox = strengthProximity(me.strengthLevel, candidate.strengthLevel);
  if (prox !== null) {
    signals.strength = prox;
    if (prox > 0.8) reasons.push("Lifting comparable loads");
    else if (candidate.strengthLevel > me.strengthLevel * 1.15) reasons.push("Lifts heavier than you");
  }

  // --- geography ---
  const near = nearestLocationPair(me.locations, candidate.locations);
  if (near) {
    const geo = proximityScore(near.km);
    signals.proximity = geo;
    if (near.km < 1) reasons.push(`Trains at ${near.to}, right by you`);
    else if (near.km < 8) reasons.push(`Trains ${near.km.toFixed(1)} km away`);
  }

  // --- activity overlap ---
  const jac = jaccardIndex(me.activities, candidate.activities);
  if (jac !== null) {
    signals.activity = jac;
    if (jac === 1) reasons.push("Trains exactly the same activities");
    else if (jac >= 0.5) reasons.push("Overlapping activities");
  }

  const available = Object.keys(signals);
  if (available.length === 0) {
    return { score: null, signals: {}, reasons: [], coverage: 0, nearest: near, distanceKm: near ? round(near.km, 2) : null };
  }

  // Renormalise over the signals actually available for this pair.
  const weightSum = available.reduce((s, k) => s + (weights[k] || 0), 0);
  if (weightSum === 0) return { score: null, signals, reasons, coverage: 0, nearest: near, distanceKm: near ? round(near.km, 2) : null };

  let score = 0;
  for (const k of available) score += signals[k] * ((weights[k] || 0) / weightSum);

  return {
    score: round(score, 4),
    signals: Object.fromEntries(Object.entries(signals).map(([k, v]) => [k, round(v, 3)])),
    // How much of the model could actually be evaluated. A high score from one
    // signal is weaker evidence than a moderate score from all four, so this
    // is surfaced rather than hidden inside the number.
    coverage: round(available.reduce((s, k) => s + (weights[k] || 0), 0), 3),
    reasons,
    nearest: near,
    distanceKm: near ? round(near.km, 2) : null,
  };
}

/**
 * Ranks every candidate against the current user.
 *
 * Deliberately two-stage, mirroring how production recommender systems are
 * built: a cheap FEASIBILITY filter first, then a scoring pass over what
 * survives.
 *
 * The reason is a real weakness of weighted-sum scoring: it treats every
 * signal as compensatory, so a strong score on one can offset a near-zero on
 * another. Testing exposed this directly — a lifter 3,700 km away with a
 * similar training split still scored 0.43, because good profile and strength
 * matches outweighed a proximity score of essentially zero. But distance is
 * not a preference that trades against training style; a partner you cannot
 * physically meet is not a partner at all.
 *
 * Distance is therefore treated as a constraint rather than a term in the sum.
 * Candidates beyond maxDistanceKm are removed before scoring, and the weighted
 * model then ranks only those who could actually train together.
 */
export function rankPartners(me, candidates, options = {}) {
  const weights = options.weights || DEFAULT_WEIGHTS;
  const minCoverage = options.minCoverage ?? 0.2;
  const maxDistanceKm = options.maxDistanceKm ?? 40;

  // --- Stage 1: feasibility ---
  const feasible = candidates.filter((c) => {
    if (c.uid === me.uid) return false; // never match someone with themselves

    const near = nearestLocationPair(me.locations, c.locations);

    // Unknown distance is not disqualifying. Either party may simply not have
    // saved a location yet, and excluding them would make the feature useless
    // for new users — the same reasoning as renormalising missing signals.
    if (!near) return true;

    return near.km <= maxDistanceKm;
  });

  // --- Stage 2: ranking ---
  return feasible
    .map((c) => ({ candidate: c, ...scoreCandidate(me, c, weights) }))
    .filter((r) => r.score !== null && r.coverage >= minCoverage)
    .sort((a, b) => (b.score - a.score) || (b.coverage - a.coverage))
    .slice(0, options.limit ?? 10);
}

/** Assembles the feature bundle one user contributes to matching. */
export function buildMatchProfile(uid, username, entries, locations) {
  return {
    uid,
    username,
    profileVector: buildProfileVector(entries),
    strengthLevel: strengthLevel(entries),
    locations: locations || [],
    activities: activitySet(locations, entries),
    sessionCount: entries.length,
  };
}

function round(value, dp) {
  if (value === null || value === undefined || !isFinite(value)) return null;
  const f = Math.pow(10, dp);
  return Math.round(value * f) / f;
}
