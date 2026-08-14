// ============================================================================
// 225 — TRAINING ANALYTICS ENGINE
// ============================================================================
// Pure computation. No Firebase, no DOM, no side effects — every function here
// takes data in and returns a result, which is what makes the whole module
// testable in isolation.
//
// The models below are drawn from published strength and conditioning
// literature rather than invented for this project:
//
//   Epley (1985), Brzycki (1993), Lombardi (1989)
//       — submaximal load-to-1RM prediction equations.
//   Gabbett (2016), Hulin et al. (2014)
//       — acute:chronic workload ratio as an injury-risk indicator.
//   Ordinary least squares regression with a t-test on the slope
//       — used here to decide whether an apparent plateau is real or noise.
// ============================================================================


import { lookupExercise } from "./exercises.js";


// ---------------------------------------------------------------------------
// ONE-REPETITION MAXIMUM ESTIMATION
// ---------------------------------------------------------------------------
// A lifter's true 1RM is the heaviest load they can move once. Testing it
// directly is fatiguing and risky, so it is normally predicted from a
// submaximal set. Each published equation fits real lifting data slightly
// differently, and they disagree most at high repetition counts.

export const ONE_RM_FORMULAS = {
  // Epley: linear in reps. Tends to over-predict above ~10 reps.
  epley: (weight, reps) => (reps === 1 ? weight : weight * (1 + reps / 30)),

  // Brzycki: hyperbolic. Breaks down as reps approach 37, so it is capped.
  brzycki: (weight, reps) => (reps >= 37 ? null : weight * (36 / (37 - reps))),

  // Lombardi: power law. More conservative at high repetition counts.
  lombardi: (weight, reps) => weight * Math.pow(reps, 0.1),
};

/**
 * Estimates 1RM from a submaximal set.
 *
 * Accuracy degrades as repetitions rise — the equations are fitted on sets of
 * roughly 1–10 reps, so a set of 20 is an extrapolation rather than a
 * prediction. Rather than hide that, the returned confidence band widens with
 * rep count so the interface can present the number with appropriate caution.
 *
 * Using the median of the three formulas rather than the mean keeps a single
 * badly-behaved equation (Brzycki at high reps) from dragging the estimate.
 */
export function estimate1RM(weight, reps) {
  if (!weight || weight <= 0 || !reps || reps <= 0) return null;
  if (reps === 1) {
    return { estimate: weight, low: weight, high: weight, confidence: "measured", spread: 0 };
  }

  const values = Object.values(ONE_RM_FORMULAS)
    .map((fn) => fn(weight, reps))
    .filter((v) => v !== null && isFinite(v));

  if (values.length === 0) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 1
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;

  // Disagreement between formulas is itself a useful uncertainty signal.
  const spread = sorted[sorted.length - 1] - sorted[0];

  // Extrapolation error grows with reps; ±2% per rep beyond the first is a
  // deliberately conservative band on top of the observed formula spread.
  const extrapolationError = median * 0.02 * (reps - 1);
  const margin = spread / 2 + extrapolationError;

  let confidence;
  if (reps <= 5) confidence = "high";
  else if (reps <= 10) confidence = "moderate";
  else confidence = "low";

  return {
    estimate: round(median, 1),
    low: round(median - margin, 1),
    high: round(median + margin, 1),
    confidence,
    spread: round(spread, 1),
  };
}

/**
 * Best estimated 1RM from a single session, taken across all its sets.
 * A lifter's heaviest set is not always their most predictive one — a
 * moderate load for many reps can imply a higher maximum than a heavy single.
 */
export function sessionBest1RM(session) {
  if (!session) return null;

  // Current schema stores one exercise per document with uniform sets.
  const est = estimate1RM(session.weight, session.reps);
  return est ? est.estimate : null;
}


// ---------------------------------------------------------------------------
// ORDINARY LEAST SQUARES REGRESSION
// ---------------------------------------------------------------------------

/**
 * Fits y = slope·x + intercept by least squares, and reports how much the fit
 * should be believed.
 *
 * r2        — proportion of variance explained (0 to 1).
 * stdError  — standard error of the slope estimate.
 * tStat     — slope divided by its standard error.
 * pApprox   — approximate two-tailed significance of the slope.
 *
 * The t-statistic is the part that matters for plateau detection: a slope of
 * zero and a slope that is merely indistinguishable from zero are different
 * claims, and only the second is honest with three data points.
 */
export function linearRegression(points) {
  const n = points.length;
  if (n < 2) return null;

  const sumX = points.reduce((s, p) => s + p.x, 0);
  const sumY = points.reduce((s, p) => s + p.y, 0);
  const meanX = sumX / n;
  const meanY = sumY / n;

  let sxx = 0, sxy = 0, syy = 0;
  for (const p of points) {
    const dx = p.x - meanX;
    const dy = p.y - meanY;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }

  // All x values identical — slope is undefined, not zero.
  if (sxx === 0) return null;

  const slope = sxy / sxx;
  const intercept = meanY - slope * meanX;

  // Residual sum of squares
  let rss = 0;
  for (const p of points) {
    const predicted = slope * p.x + intercept;
    rss += Math.pow(p.y - predicted, 2);
  }

  const r2 = syy === 0 ? 1 : Math.max(0, 1 - rss / syy);

  // Standard error requires at least one degree of freedom beyond the fit.
  let stdError = null, tStat = null, pApprox = null;
  if (n > 2) {
    const residualVariance = rss / (n - 2);
    stdError = Math.sqrt(residualVariance / sxx);
    if (stdError > 0) {
      tStat = slope / stdError;
      pApprox = approximatePValue(Math.abs(tStat), n - 2);
    } else {
      // Perfect fit: the slope is exact, so treat it as fully significant.
      tStat = slope === 0 ? 0 : Infinity;
      pApprox = slope === 0 ? 1 : 0;
    }
  }

  return {
    slope,
    intercept,
    r2,
    stdError,
    tStat,
    pApprox,
    n,
  };
}

/**
 * Approximate two-tailed p-value for a t-statistic.
 *
 * Uses a normal approximation with a small-sample correction rather than a
 * full incomplete beta function. This is adequate for flagging plateaus, and
 * the approximation is stated openly rather than presented as exact.
 */
function approximatePValue(t, df) {
  if (!isFinite(t)) return 0;
  if (df <= 0) return 1;

  // Convert t to an approximately standard normal z (Wallace-style correction)
  const z = t * (1 - 1 / (4 * df)) / Math.sqrt(1 + (t * t) / (2 * df));
  return 2 * (1 - normalCDF(z));
}

/** Standard normal CDF via the Abramowitz & Stegun 7.1.26 error-function approximation. */
function normalCDF(z) {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);

  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;

  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1 + sign * y);
}


// ---------------------------------------------------------------------------
// PLATEAU DETECTION
// ---------------------------------------------------------------------------

/**
 * Decides whether a lift has stalled.
 *
 * The naive approach — "is the last value lower than the first?" — is wrong,
 * because week-to-week strength is noisy. Two sessions can differ by 10 lb for
 * reasons that have nothing to do with training.
 *
 * Instead a line is fitted through the estimated 1RM history and its slope is
 * tested against zero. Three outcomes are possible, and the third is the one
 * naive approaches miss entirely:
 *
 *   progressing  — slope significantly positive
 *   regressing   — slope significantly negative
 *   plateau      — slope not significantly different from zero
 *   insufficient — too little data to make any claim
 *
 * @param sessions  chronologically sorted entries with createdAt, weight, reps
 * @param options.minSessions   fewest sessions before any claim is made
 * @param options.alpha         significance threshold for the slope test
 * @param options.windowDays    only consider sessions within this recent window
 */
export function detectPlateau(sessions, options = {}) {
  const minSessions = options.minSessions ?? 4;
  const alpha = options.alpha ?? 0.10;
  const windowDays = options.windowDays ?? 56; // eight weeks

  const usable = sessions
    .filter((s) => s.timestamp && s.weight > 0 && s.reps > 0)
    .sort((a, b) => a.timestamp - b.timestamp);

  if (usable.length < minSessions) {
    return {
      status: "insufficient",
      reason: `Needs at least ${minSessions} sessions; found ${usable.length}.`,
      sessions: usable.length,
    };
  }

  const latest = usable[usable.length - 1].timestamp;
  const cutoff = latest - windowDays * 86400000;
  const recent = usable.filter((s) => s.timestamp >= cutoff);

  if (recent.length < minSessions) {
    return {
      status: "insufficient",
      reason: `Only ${recent.length} sessions in the last ${windowDays} days.`,
      sessions: recent.length,
    };
  }

  // x in days since the first session in the window, y in estimated 1RM.
  const origin = recent[0].timestamp;
  const points = recent
    .map((s) => {
      const est = estimate1RM(s.weight, s.reps);
      return est ? { x: (s.timestamp - origin) / 86400000, y: est.estimate } : null;
    })
    .filter(Boolean);

  const fit = linearRegression(points);
  if (!fit || fit.tStat === null) {
    return { status: "insufficient", reason: "Regression could not be computed.", sessions: points.length };
  }

  const perWeek = fit.slope * 7;
  const significant = fit.pApprox !== null && fit.pApprox < alpha;

  let status;
  if (!significant) status = "plateau";
  else if (fit.slope > 0) status = "progressing";
  else status = "regressing";

  const spanDays = points[points.length - 1].x;

  return {
    status,
    slopePerWeek: round(perWeek, 2),
    r2: round(fit.r2, 3),
    pValue: fit.pApprox === null ? null : round(fit.pApprox, 4),
    significant,
    sessions: points.length,
    spanDays: round(spanDays, 0),
    current1RM: round(points[points.length - 1].y, 1),
    // Projected gain over four weeks if the current trend continued.
    projected4Week: round(perWeek * 4, 1),
  };
}


// ---------------------------------------------------------------------------
// TRAINING LOAD AND INJURY RISK
// ---------------------------------------------------------------------------

/** Volume load for one entry: the standard tonnage measure, sets × reps × load. */
export function volumeLoad(entry) {
  if (entry.type === "cardio") {
    // Cardio has no tonnage. Distance is used as a stand-in so cardio still
    // contributes to total workload, scaled to sit on a comparable order of
    // magnitude to lifting volume. This is a modelling choice, not a
    // physiological equivalence, and is stated as such.
    return (entry.distance || 0) * 100;
  }
  return (entry.sets || 0) * (entry.reps || 0) * (entry.weight || 0);
}

/**
 * Acute:chronic workload ratio.
 *
 * Acute load is the total of the last 7 days. Chronic load is the average
 * 7-day load across the last 28. Their ratio describes how a lifter's recent
 * work compares to what they are conditioned for.
 *
 * Published thresholds (Gabbett 2016):
 *   below 0.8   — undertrained; detraining risk
 *   0.8 to 1.3  — the "sweet spot" associated with lowest injury incidence
 *   1.3 to 1.5  — elevated
 *   above 1.5   — high risk; sharp spike in reported injury rates
 *
 * The ratio is undefined without a chronic baseline, which is a real state
 * rather than an error: a lifter in their first month simply has no
 * established baseline to compare against.
 */
export function computeACWR(entries, referenceDate) {
  const now = referenceDate ?? Math.max(...entries.map((e) => e.timestamp || 0));
  const DAY = 86400000;

  const acuteCutoff = now - 7 * DAY;
  const chronicCutoff = now - 28 * DAY;

  const acute = entries
    .filter((e) => e.timestamp > acuteCutoff && e.timestamp <= now)
    .reduce((sum, e) => sum + volumeLoad(e), 0);

  const chronicTotal = entries
    .filter((e) => e.timestamp > chronicCutoff && e.timestamp <= now)
    .reduce((sum, e) => sum + volumeLoad(e), 0);

  const chronicWeekly = chronicTotal / 4;

  // Require a real baseline before reporting a ratio.
  const earliest = Math.min(...entries.map((e) => e.timestamp || Infinity));
  const historyDays = isFinite(earliest) ? (now - earliest) / DAY : 0;

  if (chronicWeekly === 0 || historyDays < 14) {
    return {
      status: "baseline",
      acute: round(acute, 0),
      chronicWeekly: round(chronicWeekly, 0),
      ratio: null,
      historyDays: round(historyDays, 0),
      message: "Building a baseline — at least two weeks of history is needed before a ratio is meaningful.",
    };
  }

  const ratio = acute / chronicWeekly;

  let status, message;
  if (ratio < 0.8) {
    status = "undertrained";
    message = "Recent work is below your established baseline.";
  } else if (ratio <= 1.3) {
    status = "optimal";
    message = "Recent work is in the range associated with the lowest injury incidence.";
  } else if (ratio <= 1.5) {
    status = "elevated";
    message = "Recent work is climbing faster than your baseline supports.";
  } else {
    status = "high";
    message = "A sharp spike relative to your baseline. Published data associates this range with markedly higher injury rates.";
  }

  return {
    status,
    ratio: round(ratio, 2),
    acute: round(acute, 0),
    chronicWeekly: round(chronicWeekly, 0),
    historyDays: round(historyDays, 0),
    message,
  };
}

/** Weekly volume series, oldest first, for charting workload over time. */
export function weeklyVolumeSeries(entries, weeks = 12) {
  if (entries.length === 0) return [];

  const DAY = 86400000;
  const now = Math.max(...entries.map((e) => e.timestamp || 0));
  const series = [];

  for (let i = weeks - 1; i >= 0; i--) {
    const end = now - i * 7 * DAY;
    const start = end - 7 * DAY;
    const total = entries
      .filter((e) => e.timestamp > start && e.timestamp <= end)
      .reduce((sum, e) => sum + volumeLoad(e), 0);
    series.push({ weekEnding: end, volume: round(total, 0) });
  }

  return series;
}


// ---------------------------------------------------------------------------
// MOVEMENT PATTERN BALANCE
// ---------------------------------------------------------------------------

// Before the exercise catalog existed, exercise names were free text and had
// to be classified by keyword matching. That heuristic is retained only as a
// fallback for entries logged under the old scheme — anything logged through
// the catalog is classified by authoritative lookup instead.
const LEGACY_KEYWORD_PATTERNS = [
  { pattern: "push", keywords: ["bench", "press", "push", "dip", "fly", "flye", "tricep", "overhead", "ohp"] },
  { pattern: "pull", keywords: ["row", "pull", "chin", "lat", "curl", "shrug", "face pull", "rear delt"] },
  { pattern: "legs", keywords: ["squat", "lunge", "leg", "calf", "deadlift", "rdl", "hip thrust", "glute", "hamstring", "quad"] },
  { pattern: "core", keywords: ["ab", "plank", "crunch", "core", "oblique", "sit-up", "situp", "hanging leg"] },
];

/**
 * Determines the movement pattern for an exercise.
 *
 * Resolution order matters:
 *   1. Catalog lookup by ID or exact name — authoritative, no guessing.
 *   2. Keyword matching — legacy fallback for free-text entries.
 *   3. "other" — unclassifiable.
 *
 * Step 2 exists purely for backward compatibility. It is genuinely unreliable
 * ("Leg Press" and "Chest Press" both contain "press"), which is precisely why
 * the catalog replaced it.
 */
export function classifyExercise(idOrName) {
  if (!idOrName) return "other";

  const known = lookupExercise(idOrName);
  if (known) return known.pattern;

  const lower = String(idOrName).toLowerCase();
  for (const group of LEGACY_KEYWORD_PATTERNS) {
    if (group.keywords.some((k) => lower.includes(k))) return group.pattern;
  }
  return "other";
}

/**
 * Muscle group for an exercise. Only available for catalog entries — legacy
 * free-text entries return null rather than a guess, since inferring muscle
 * group from a keyword is far less reliable than inferring movement pattern.
 */
export function muscleGroupOf(idOrName) {
  const known = lookupExercise(idOrName);
  return known ? known.group : null;
}

/**
 * Volume distribution across muscle groups.
 * Entries that predate the catalog are counted separately rather than being
 * silently dropped or misattributed.
 */
export function muscleGroupVolume(entries) {
  const totals = {};
  let uncatalogued = 0;

  for (const e of entries) {
    if (e.type === "cardio") continue;
    const group = muscleGroupOf(e.exerciseId || e.exerciseName);
    if (!group) {
      uncatalogued += volumeLoad(e);
      continue;
    }
    totals[group] = (totals[group] || 0) + volumeLoad(e);
  }

  const total = Object.values(totals).reduce((a, b) => a + b, 0);

  return {
    totals: Object.fromEntries(Object.entries(totals).map(([k, v]) => [k, round(v, 0)])),
    sharePercent: total === 0 ? {} : Object.fromEntries(
      Object.entries(totals).map(([k, v]) => [k, round((v / total) * 100, 1)])
    ),
    uncatalogued: round(uncatalogued, 0),
    total: round(total, 0),
  };
}

/**
 * Push-to-pull volume ratio.
 *
 * Chronic imbalance between pushing and pulling volume is a commonly cited
 * contributor to shoulder problems. A ratio near 1.0 indicates balance;
 * substantially above 1.0 means pressing volume exceeds pulling volume.
 */
export function movementBalance(entries) {
  const totals = { push: 0, pull: 0, legs: 0, core: 0, other: 0 };

  for (const e of entries) {
    if (e.type === "cardio") continue;
    // Prefer the catalog ID; fall back to the stored name for legacy entries.
    totals[classifyExercise(e.exerciseId || e.exerciseName)] += volumeLoad(e);
  }

  const strengthTotal = totals.push + totals.pull + totals.legs + totals.core + totals.other;
  if (strengthTotal === 0) {
    return { status: "insufficient", totals, pushPullRatio: null };
  }

  const pushPullRatio = totals.pull === 0 ? null : totals.push / totals.pull;

  let status, message;
  if (totals.pull === 0 && totals.push > 0) {
    status = "imbalanced";
    message = "Pressing volume recorded with no pulling volume.";
  } else if (pushPullRatio === null) {
    status = "insufficient";
    message = "Not enough upper-body volume to assess balance.";
  } else if (pushPullRatio > 1.5) {
    status = "push-dominant";
    message = "Pressing volume substantially exceeds pulling volume.";
  } else if (pushPullRatio < 0.67) {
    status = "pull-dominant";
    message = "Pulling volume substantially exceeds pressing volume.";
  } else {
    status = "balanced";
    message = "Pressing and pulling volume are reasonably balanced.";
  }

  const share = {};
  for (const key of Object.keys(totals)) {
    share[key] = round((totals[key] / strengthTotal) * 100, 1);
  }

  return {
    status,
    message,
    pushPullRatio: pushPullRatio === null ? null : round(pushPullRatio, 2),
    totals: Object.fromEntries(Object.entries(totals).map(([k, v]) => [k, round(v, 0)])),
    sharePercent: share,
  };
}


// ---------------------------------------------------------------------------
// UTILITY
// ---------------------------------------------------------------------------

function round(value, dp) {
  if (value === null || value === undefined || !isFinite(value)) return null;
  const factor = Math.pow(10, dp);
  return Math.round(value * factor) / factor;
}

/**
 * Normalises raw Firestore workout documents into the shape this module
 * expects. Keeping this conversion in one place means the analytics functions
 * never need to know about Firestore timestamps.
 */
export function normaliseEntries(rawWorkouts) {
  return rawWorkouts
    .filter((w) => w.createdAt)
    .map((w) => ({
      type: w.type || "strength",
      exerciseId: w.exerciseId || null,
      exerciseName: w.exerciseName,
      activity: w.activity,
      sets: w.sets,
      reps: w.reps,
      weight: w.weight,
      distance: w.distance,
      durationSeconds: w.durationSeconds,
      timestamp: typeof w.createdAt.toMillis === "function" ? w.createdAt.toMillis() : w.createdAt,
    }))
    .sort((a, b) => a.timestamp - b.timestamp);
}
