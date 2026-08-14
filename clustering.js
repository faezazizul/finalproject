// ============================================================================
// 225 — TRAINING ARCHETYPE CLUSTERING
// ============================================================================
// Unsupervised learning: k-means with k-means++ seeding, plus silhouette
// analysis for selecting k. Pure computation, no libraries.
//
// The question this answers is different from the recommender's. The
// recommender asks "what should this person do next". Clustering asks "what
// KINDS of lifter exist in this population, and which kind is this person" —
// and crucially, nobody defines the kinds in advance. The categories are
// discovered from the data.
//
// WHY K-MEANS++ RATHER THAN RANDOM SEEDING
//
// Plain k-means picks starting centroids at random, which frequently places
// two of them inside the same natural cluster. Lloyd's algorithm then
// converges to a local optimum that splits one real group in half while
// merging two others. k-means++ instead chooses each new centroid with
// probability proportional to its squared distance from the nearest existing
// centroid, so seeds spread out. It is a small change with a large effect on
// solution quality, and it is why the difference is tested explicitly.
//
// WHY SILHOUETTE RATHER THAN INERTIA FOR CHOOSING K
//
// Inertia (within-cluster sum of squares) falls monotonically as k rises, so
// it can never identify a best k on its own — it always favours more clusters.
// The silhouette coefficient compares each point's cohesion with its own
// cluster against its separation from the nearest other cluster, so it peaks
// at a genuinely appropriate k rather than running away.
// ============================================================================

import { makeRandom } from "./ml-recommender.js";


// ---------------------------------------------------------------------------
// FEATURE EXTRACTION
// ---------------------------------------------------------------------------

// Dimensions were chosen to be interpretable rather than merely separable. A
// centroid in this space can be read directly as a description of a training
// style, which is what lets each discovered cluster be given a meaningful name.
export const FEATURE_NAMES = ["push", "pull", "legs", "core", "cardioShare", "compoundRatio"];

/**
 * Builds a feature vector describing how one person trains.
 *
 * Every dimension is a proportion in 0–1, so no scaling step is required.
 * That matters: k-means uses Euclidean distance, so a single unnormalised
 * dimension with a larger numeric range would dominate the distance
 * calculation and effectively become the only feature that mattered.
 */
export function buildFeatureVector(entries, classifyFn, lookupFn) {
  if (!entries || entries.length === 0) return null;

  const totals = { push: 0, pull: 0, legs: 0, core: 0 };
  let strengthVolume = 0;
  let compoundVolume = 0;
  let cardioSessions = 0;

  for (const e of entries) {
    if (e.type === "cardio") { cardioSessions++; continue; }

    const volume = (e.sets || 0) * (e.reps || 0) * (e.weight || 0);
    if (volume <= 0) continue;

    const pattern = classifyFn(e.exerciseId || e.exerciseName);
    if (pattern in totals) totals[pattern] += volume;

    strengthVolume += volume;

    const meta = lookupFn(e.exerciseId || e.exerciseName);
    if (meta && meta.compound) compoundVolume += volume;
  }

  if (strengthVolume === 0 && cardioSessions === 0) return null;

  const patternTotal = totals.push + totals.pull + totals.legs + totals.core;

  return [
    patternTotal === 0 ? 0 : totals.push / patternTotal,
    patternTotal === 0 ? 0 : totals.pull / patternTotal,
    patternTotal === 0 ? 0 : totals.legs / patternTotal,
    patternTotal === 0 ? 0 : totals.core / patternTotal,
    entries.length === 0 ? 0 : cardioSessions / entries.length,
    strengthVolume === 0 ? 0 : compoundVolume / strengthVolume,
  ];
}


// ---------------------------------------------------------------------------
// DISTANCE AND SEEDING
// ---------------------------------------------------------------------------

export function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * k-means++ seeding.
 *
 * The first centroid is uniformly random. Every subsequent one is drawn with
 * probability proportional to D(x)², where D(x) is the distance from x to its
 * nearest already-chosen centroid — so points far from existing seeds are
 * strongly favoured.
 */
export function kMeansPlusPlusInit(vectors, k, rand) {
  const centroids = [];
  const first = Math.floor(rand() * vectors.length);
  centroids.push([...vectors[first]]);

  while (centroids.length < k) {
    const distances = vectors.map((v) => {
      let best = Infinity;
      for (const c of centroids) {
        const d = euclideanDistance(v, c);
        if (d < best) best = d;
      }
      return best * best;
    });

    const total = distances.reduce((a, b) => a + b, 0);

    // Every point already coincides with a centroid — no meaningful spread
    // remains, so fall back to an arbitrary pick rather than dividing by zero.
    if (total === 0) {
      centroids.push([...vectors[Math.floor(rand() * vectors.length)]]);
      continue;
    }

    let threshold = rand() * total;
    let chosen = 0;
    for (let i = 0; i < distances.length; i++) {
      threshold -= distances[i];
      if (threshold <= 0) { chosen = i; break; }
    }
    centroids.push([...vectors[chosen]]);
  }

  return centroids;
}


// ---------------------------------------------------------------------------
// K-MEANS
// ---------------------------------------------------------------------------

/**
 * Lloyd's algorithm with k-means++ seeding.
 *
 * Runs to convergence — defined as no point changing cluster — or to
 * maxIterations, whichever comes first. Convergence is reported so a run that
 * hit the iteration cap can be distinguished from one that genuinely settled.
 *
 * Empty clusters are reseeded to the point furthest from its own centroid
 * rather than being dropped, so the caller always receives exactly k clusters.
 */
export function kMeans(vectors, k, options = {}) {
  const maxIterations = options.maxIterations ?? 100;
  const rand = makeRandom(options.seed ?? 42);

  if (!vectors || vectors.length === 0) return null;
  if (k < 1) return null;
  if (k >= vectors.length) {
    // More clusters requested than points: each point becomes its own cluster.
    return {
      centroids: vectors.map((v) => [...v]),
      assignments: vectors.map((_, i) => i),
      iterations: 0,
      converged: true,
      inertia: 0,
      k: vectors.length,
    };
  }

  let centroids = kMeansPlusPlusInit(vectors, k, rand);
  let assignments = new Array(vectors.length).fill(-1);
  let iterations = 0;
  let converged = false;

  while (iterations < maxIterations) {
    iterations++;
    let changed = false;

    // Assignment step
    for (let i = 0; i < vectors.length; i++) {
      let best = 0, bestDistance = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const d = euclideanDistance(vectors[i], centroids[c]);
        if (d < bestDistance) { bestDistance = d; best = c; }
      }
      if (assignments[i] !== best) { assignments[i] = best; changed = true; }
    }

    if (!changed) { converged = true; break; }

    // Update step
    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const counts = new Array(k).fill(0);

    for (let i = 0; i < vectors.length; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < vectors[i].length; d++) sums[c][d] += vectors[i][d];
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // Reseed an empty cluster onto the worst-served point.
        let worst = 0, worstDistance = -1;
        for (let i = 0; i < vectors.length; i++) {
          const d = euclideanDistance(vectors[i], centroids[assignments[i]]);
          if (d > worstDistance) { worstDistance = d; worst = i; }
        }
        centroids[c] = [...vectors[worst]];
        continue;
      }
      centroids[c] = sums[c].map((s) => s / counts[c]);
    }
  }

  // Within-cluster sum of squares
  let inertia = 0;
  for (let i = 0; i < vectors.length; i++) {
    const d = euclideanDistance(vectors[i], centroids[assignments[i]]);
    inertia += d * d;
  }

  return { centroids, assignments, iterations, converged, inertia, k };
}


// ---------------------------------------------------------------------------
// CLUSTER QUALITY
// ---------------------------------------------------------------------------

/**
 * Mean silhouette coefficient across all points.
 *
 * For each point: a = mean distance to others in its own cluster,
 * b = lowest mean distance to any other cluster, s = (b − a) / max(a, b).
 *
 * Ranges from −1 to 1. Above roughly 0.5 indicates well-separated clusters;
 * near 0 means clusters overlap; negative means points are on average closer
 * to a different cluster than their own.
 *
 * Points alone in their cluster score 0 by convention — there is no
 * within-cluster distance to measure, and scoring them 1 would reward
 * degenerate solutions that isolate outliers.
 */
export function silhouetteScore(vectors, assignments, k) {
  if (vectors.length <= k || k < 2) return null;

  const byCluster = Array.from({ length: k }, () => []);
  assignments.forEach((c, i) => byCluster[c].push(i));

  let total = 0;

  for (let i = 0; i < vectors.length; i++) {
    const own = assignments[i];
    const sameCluster = byCluster[own].filter((j) => j !== i);

    if (sameCluster.length === 0) continue; // contributes 0

    const a = sameCluster.reduce((s, j) => s + euclideanDistance(vectors[i], vectors[j]), 0) / sameCluster.length;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === own || byCluster[c].length === 0) continue;
      const mean = byCluster[c].reduce((s, j) => s + euclideanDistance(vectors[i], vectors[j]), 0) / byCluster[c].length;
      if (mean < b) b = mean;
    }

    if (!isFinite(b)) continue;
    total += (b - a) / Math.max(a, b);
  }

  return total / vectors.length;
}

/**
 * Selects k by maximising the silhouette coefficient over a candidate range.
 *
 * Returns the full sweep alongside the winner, because the shape of the curve
 * is itself informative: a flat curve means the data has no strong cluster
 * structure at any k, which is a real finding rather than a failure.
 */
export function chooseK(vectors, options = {}) {
  const minK = options.minK ?? 2;
  const maxK = Math.min(options.maxK ?? 6, vectors.length - 1);

  if (vectors.length < 3) return null;

  const sweep = [];
  for (let k = minK; k <= maxK; k++) {
    const result = kMeans(vectors, k, options);
    if (!result) continue;
    const silhouette = silhouetteScore(vectors, result.assignments, k);
    sweep.push({ k, silhouette, inertia: result.inertia, converged: result.converged });
  }

  if (sweep.length === 0) return null;

  const best = sweep.reduce((a, b) => ((b.silhouette ?? -Infinity) > (a.silhouette ?? -Infinity) ? b : a));
  return { bestK: best.k, bestSilhouette: best.silhouette, sweep };
}


// ---------------------------------------------------------------------------
// INTERPRETATION
// ---------------------------------------------------------------------------

/**
 * Turns a centroid into a human-readable archetype.
 *
 * This is where the interpretability of the feature space pays off: because
 * every dimension is a meaningful proportion, a centroid can be read directly.
 * The labels are assigned by rules over centroid values, not learned — the
 * CLUSTERS are discovered, the NAMES are how they get described.
 */
export function describeCluster(centroid) {
  const [push, pull, legs, core, cardioShare, compoundRatio] = centroid;

  const traits = [];
  let label;

  if (cardioShare > 0.5) {
    label = "Endurance Focused";
    traits.push("Majority of sessions are cardio");
  } else if (compoundRatio > 0.75 && legs > 0.3) {
    label = "Strength Athlete";
    traits.push("Heavy emphasis on compound lifts");
  } else if (compoundRatio < 0.5) {
    label = "Hypertrophy Focused";
    traits.push("Leans toward isolation work");
  } else if (legs < 0.2) {
    label = "Upper Body Focused";
    traits.push("Comparatively little lower-body volume");
  } else if (Math.abs(push - pull) < 0.08 && legs > 0.25) {
    label = "Balanced Generalist";
    traits.push("Push, pull and legs are evenly distributed");
  } else if (push > pull * 1.4) {
    label = "Press Dominant";
    traits.push("Pressing volume well above pulling");
  } else if (pull > push * 1.4) {
    label = "Pull Dominant";
    traits.push("Pulling volume well above pressing");
  } else {
    label = "Mixed Training";
    traits.push("No single dominant emphasis");
  }

  if (cardioShare > 0.2 && cardioShare <= 0.5) traits.push("Regular cardio alongside lifting");
  if (core > 0.15) traits.push("Consistent core work");
  if (legs > 0.4) traits.push("Lower-body heavy");

  return {
    label,
    traits,
    profile: {
      push: round(push * 100, 1),
      pull: round(pull * 100, 1),
      legs: round(legs * 100, 1),
      core: round(core * 100, 1),
      cardioShare: round(cardioShare * 100, 1),
      compoundRatio: round(compoundRatio * 100, 1),
    },
  };
}

/** Full pipeline: choose k, cluster, describe, and locate one target member. */
export function discoverArchetypes(members, options = {}) {
  const usable = members.filter((m) => m.vector && m.vector.length === FEATURE_NAMES.length);
  if (usable.length < 3) {
    return { status: "insufficient", message: "At least three profiles are needed to identify training archetypes.", members: usable.length };
  }

  const vectors = usable.map((m) => m.vector);
  const selection = chooseK(vectors, options);
  if (!selection) return { status: "insufficient", message: "Could not evaluate cluster structure.", members: usable.length };

  const result = kMeans(vectors, selection.bestK, options);
  const clusters = result.centroids.map((centroid, i) => {
    const memberIndices = result.assignments.map((c, idx) => (c === i ? idx : -1)).filter((x) => x >= 0);
    return {
      index: i,
      size: memberIndices.length,
      centroid,
      ...describeCluster(centroid),
      memberIds: memberIndices.map((idx) => usable[idx].id),
    };
  });

  let own = null;
  if (options.targetId) {
    const position = usable.findIndex((m) => m.id === options.targetId);
    if (position >= 0) own = clusters[result.assignments[position]];
  }

  return {
    status: "ok",
    k: selection.bestK,
    silhouette: round(selection.bestSilhouette, 3),
    sweep: selection.sweep.map((s) => ({ k: s.k, silhouette: round(s.silhouette, 3), inertia: round(s.inertia, 4) })),
    clusters,
    own,
    members: usable.length,
    converged: result.converged,
  };
}

function round(value, dp) {
  if (value === null || value === undefined || !isFinite(value)) return null;
  const f = Math.pow(10, dp);
  return Math.round(value * f) / f;
}
