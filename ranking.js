// ============================================================================
// 225 — RANKING AND ACHIEVEMENT ENGINE
// ============================================================================
// Pure computation, no Firebase or DOM, for the same reason as the other
// engines: it can be unit-tested against known values in isolation.
//
// The problem this solves: raw lifted weight is not comparable between people.
// A 60 kg lifter benching 100 kg and a 110 kg lifter benching 140 kg are not
// separated by 40 kg of ability — relative to bodyweight the first is the more
// impressive lift. Any leaderboard built on raw load simply ranks people by
// how heavy they are.
//
// Strength sports solved this with allometric scaling coefficients. This
// module uses DOTS (Dynamic Objective Team Scoring), the coefficient adopted
// by the International Powerlifting Federation to replace the older Wilks
// formula. DOTS fits a fourth-order polynomial to bodyweight and divides total
// load by it, producing a score comparable across bodyweights.
//
// Separate coefficients exist for men and women because the fitted
// bodyweight-to-strength curves differ measurably in the underlying
// competition data. That separation is the published standard rather than an
// assumption made here, and an OPEN category is provided for anyone who
// prefers not to specify — scored on the men's curve, which is the more
// conservative of the two and therefore never inflates a score.
// ============================================================================


// ---------------------------------------------------------------------------
// DOTS COEFFICIENT
// ---------------------------------------------------------------------------

// Published DOTS polynomial coefficients. Bodyweight is in kilograms.
const DOTS_COEFFICIENTS = {
  male:   { a: -0.000001093, b: 0.0007391293, c: -0.1918759221, d: 24.0900756, e: -307.75076 },
  female: { a: -0.0000010706, b: 0.0005158568, c: -0.1126655495, d: 13.6175032, e: -57.96288 },
};

// Categories map onto the published curves. "open" deliberately uses the
// men's curve so that declining to specify can never produce a higher score
// than specifying would.
export const CATEGORIES = {
  male:   { label: "Men's",  curve: "male" },
  female: { label: "Women's", curve: "female" },
  open:   { label: "Open",   curve: "male" },
};

export const LB_PER_KG = 2.2046226218;

export const lbToKg = (lb) => lb / LB_PER_KG;
export const kgToLb = (kg) => kg * LB_PER_KG;

/**
 * DOTS multiplier for a given bodyweight and category.
 *
 * The polynomial is only fitted over the range of real competition
 * bodyweights, so inputs are clamped to 40–210 kg. Outside that range the
 * curve can bend back on itself and produce nonsense, which would be far worse
 * than a slightly conservative score at the extremes.
 */
export function dotsCoefficient(bodyweightKg, category = "open") {
  const cat = CATEGORIES[category] || CATEGORIES.open;
  const k = DOTS_COEFFICIENTS[cat.curve];

  const bw = Math.min(210, Math.max(40, bodyweightKg));
  const denominator = k.a * Math.pow(bw, 4) + k.b * Math.pow(bw, 3) + k.c * Math.pow(bw, 2) + k.d * bw + k.e;

  if (denominator <= 0) return null;
  return 500 / denominator;
}

/**
 * DOTS score for a total, in kilograms.
 * Higher is stronger relative to bodyweight, comparable across bodyweights
 * and — because each category uses its own fitted curve — across categories.
 */
export function dotsScore(totalKg, bodyweightKg, category = "open") {
  if (!totalKg || totalKg <= 0 || !bodyweightKg || bodyweightKg <= 0) return null;
  const coeff = dotsCoefficient(bodyweightKg, category);
  return coeff === null ? null : round(totalKg * coeff, 1);
}


// ---------------------------------------------------------------------------
// COMPETITIVE TOTAL
// ---------------------------------------------------------------------------

// The three competition lifts. Catalog variants map onto the same slot, so a
// front squat counts toward the squat slot if no back squat has been logged.
const TOTAL_SLOTS = {
  squat:    ["squat_back", "squat_front", "squat_hack", "squat_goblet"],
  bench:    ["bench_press_barbell", "bench_press_incline", "bench_press_dumbbell", "close_grip_bench"],
  deadlift: ["deadlift", "deadlift_sumo", "romanian_deadlift", "rack_pull"],
};

/**
 * Best estimated 1RM in each competition slot, and their total.
 *
 * Reports which slots are missing rather than substituting zero, because a
 * total assembled from two lifts is not comparable to one assembled from
 * three. A score is still produced from a partial total, but flagged as such
 * so the interface can present it honestly.
 */
export function competitiveTotal(entries, estimate1RMFn) {
  const bests = { squat: 0, bench: 0, deadlift: 0 };

  for (const e of entries) {
    if (e.type === "cardio" || !e.weight || !e.reps) continue;
    const id = e.exerciseId;
    if (!id) continue;

    for (const [slot, ids] of Object.entries(TOTAL_SLOTS)) {
      if (!ids.includes(id)) continue;
      const est = estimate1RMFn(e.weight, e.reps);
      if (est && est.estimate > bests[slot]) bests[slot] = est.estimate;
    }
  }

  const missing = Object.entries(bests).filter(([, v]) => v === 0).map(([k]) => k);
  const totalLb = bests.squat + bests.bench + bests.deadlift;

  return {
    bests: {
      squat: round(bests.squat, 1),
      bench: round(bests.bench, 1),
      deadlift: round(bests.deadlift, 1),
    },
    totalLb: round(totalLb, 1),
    missing,
    complete: missing.length === 0,
  };
}


// ---------------------------------------------------------------------------
// TIERS
// ---------------------------------------------------------------------------

// Thresholds are DOTS scores. Because DOTS already normalises for bodyweight
// and category, one threshold set serves everyone — which is the point of
// using a coefficient rather than separate raw-weight tables per group.
//
// Boundaries are aligned to recognised competitive bands: roughly novice,
// trained, competitive club level, regional, national and international.
export const TIERS = [
  { id: "bronze",   label: "Bronze",   min: 0,   accent: "#A5713C" },
  { id: "silver",   label: "Silver",   min: 200, accent: "#8C939A" },
  { id: "gold",     label: "Gold",     min: 275, accent: "#B8912F" },
  { id: "platinum", label: "Platinum", min: 350, accent: "#4E8B84" },
  { id: "diamond",  label: "Diamond",  min: 425, accent: "#3E7CA6" },
  { id: "master",   label: "Master",   min: 500, accent: "#7B4E9E" },
];

// Each tier is divided into three divisions, counting down, so progress is
// visible between tier promotions rather than only at them.
const DIVISIONS = ["III", "II", "I"];

/**
 * Places a DOTS score into a tier and division, and reports progress toward
 * the next promotion.
 */
export function tierForScore(score) {
  if (score === null || score === undefined || score < 0) return null;

  let index = 0;
  for (let i = TIERS.length - 1; i >= 0; i--) {
    if (score >= TIERS[i].min) { index = i; break; }
  }

  const tier = TIERS[index];
  const next = TIERS[index + 1] || null;

  // The top tier is open-ended, so it is given a nominal span for division
  // purposes rather than being left undivided.
  const span = next ? next.min - tier.min : 150;
  const within = Math.min(0.999, Math.max(0, (score - tier.min) / span));

  const divisionIndex = Math.min(DIVISIONS.length - 1, Math.floor(within * DIVISIONS.length));

  return {
    tier: tier.id,
    label: tier.label,
    division: DIVISIONS[divisionIndex],
    accent: tier.accent,
    score: round(score, 1),
    progressPercent: round(within * 100, 1),
    nextTier: next ? next.label : null,
    pointsToNext: next ? round(next.min - score, 1) : null,
  };
}

/** Full ranking for one lifter. */
export function computeRank(entries, profile, estimate1RMFn) {
  const category = profile?.category && CATEGORIES[profile.category] ? profile.category : "open";
  const bodyweightLb = profile?.bodyweightLb;

  const total = competitiveTotal(entries, estimate1RMFn);

  if (!bodyweightLb || bodyweightLb <= 0) {
    return {
      status: "needs-bodyweight",
      message: "Add your bodyweight in your profile to be ranked. Strength is scored relative to bodyweight, so it cannot be computed without it.",
      total,
      category,
    };
  }

  if (total.totalLb <= 0) {
    return {
      status: "needs-lifts",
      message: "Log a squat, bench press and deadlift to be ranked.",
      total,
      category,
    };
  }

  const score = dotsScore(lbToKg(total.totalLb), lbToKg(bodyweightLb), category);
  const rank = tierForScore(score);

  return {
    status: total.complete ? "ranked" : "partial",
    message: total.complete
      ? null
      : `Provisional — no ${total.missing.join(" or ")} logged yet, so your total is incomplete.`,
    ...rank,
    total,
    category,
    categoryLabel: CATEGORIES[category].label,
    bodyweightLb: round(bodyweightLb, 1),
  };
}


// ---------------------------------------------------------------------------
// ACHIEVEMENTS
// ---------------------------------------------------------------------------

/**
 * Longest run of consecutive calendar days containing at least one session,
 * and the current ongoing run.
 *
 * Days are compared as UTC calendar dates rather than raw millisecond gaps,
 * so two sessions 20 hours apart that fall either side of midnight count as
 * two days, which is what a user would expect.
 */
export function computeStreaks(entries, referenceDate) {
  if (entries.length === 0) return { longest: 0, current: 0, totalDays: 0 };

  const DAY = 86400000;
  const dayNumber = (ms) => Math.floor(ms / DAY);

  const days = [...new Set(entries.map((e) => dayNumber(e.timestamp)))].sort((a, b) => a - b);

  let longest = 1, run = 1;
  for (let i = 1; i < days.length; i++) {
    run = days[i] === days[i - 1] + 1 ? run + 1 : 1;
    if (run > longest) longest = run;
  }

  // The current streak only counts if training happened today or yesterday;
  // otherwise it has been broken.
  const today = dayNumber(referenceDate ?? Date.now());
  const last = days[days.length - 1];

  let current = 0;
  if (today - last <= 1) {
    current = 1;
    for (let i = days.length - 1; i > 0; i--) {
      if (days[i] === days[i - 1] + 1) current++;
      else break;
    }
  }

  return { longest, current, totalDays: days.length };
}

// Achievements are defined as data rather than code so the set can be extended
// without touching the evaluation logic. Each `test` receives a single stats
// object and returns either a boolean or a progress fraction.
export const ACHIEVEMENTS = [
  // --- consistency ---
  { id: "first_session",  name: "First Rep",        description: "Log your first session",              group: "Consistency", test: (s) => s.sessions >= 1 },
  { id: "sessions_10",    name: "Getting Going",    description: "Log 10 sessions",                     group: "Consistency", test: (s) => s.sessions / 10 },
  { id: "sessions_50",    name: "Committed",        description: "Log 50 sessions",                     group: "Consistency", test: (s) => s.sessions / 50 },
  { id: "sessions_200",   name: "Two Hundred Club", description: "Log 200 sessions",                    group: "Consistency", test: (s) => s.sessions / 200 },
  { id: "streak_7",       name: "Seven Straight",   description: "Train 7 days in a row",               group: "Consistency", test: (s) => s.longestStreak / 7 },
  { id: "streak_30",      name: "Iron Habit",       description: "Train 30 days in a row",              group: "Consistency", test: (s) => s.longestStreak / 30 },

  // --- volume ---
  { id: "volume_100k",    name: "Six Figures",      description: "Move 100,000 lb in total",            group: "Volume", test: (s) => s.totalVolume / 100000 },
  { id: "volume_1m",      name: "Millionaire",      description: "Move 1,000,000 lb in total",          group: "Volume", test: (s) => s.totalVolume / 1000000 },
  { id: "cardio_50mi",    name: "Fifty Miles",      description: "Cover 50 miles of cardio",            group: "Volume", test: (s) => s.totalDistance / 50 },
  { id: "cardio_250mi",   name: "Long Hauler",      description: "Cover 250 miles of cardio",           group: "Volume", test: (s) => s.totalDistance / 250 },

  // --- strength milestones ---
  { id: "bench_225",      name: "The 225 Club",     description: "Estimated bench press of 225 lb",     group: "Strength", test: (s) => s.bestBench / 225 },
  { id: "squat_315",      name: "Three Plates",     description: "Estimated squat of 315 lb",           group: "Strength", test: (s) => s.bestSquat / 315 },
  { id: "deadlift_405",   name: "Four Plates",      description: "Estimated deadlift of 405 lb",        group: "Strength", test: (s) => s.bestDeadlift / 405 },
  { id: "bw_bench",       name: "Bodyweight Bench", description: "Bench your own bodyweight",           group: "Strength", test: (s) => (s.bodyweight ? s.bestBench / s.bodyweight : 0) },
  { id: "double_bw_dl",   name: "Double Bodyweight", description: "Deadlift twice your bodyweight",     group: "Strength", test: (s) => (s.bodyweight ? s.bestDeadlift / (s.bodyweight * 2) : 0) },

  // --- training quality ---
  { id: "balanced",       name: "In Balance",       description: "Keep push and pull volume within 10% across 20+ sessions", group: "Quality",
    test: (s) => (s.sessions >= 20 && s.pushPullRatio !== null && s.pushPullRatio >= 0.9 && s.pushPullRatio <= 1.1) },
  { id: "all_groups",     name: "Full Coverage",    description: "Train all six muscle groups",         group: "Quality", test: (s) => s.groupsTrained / 6 },
  { id: "in_the_zone",    name: "In The Zone",      description: "Keep your workload ratio in the optimal range", group: "Quality", test: (s) => s.acwrOptimal === true },

  // --- social ---
  { id: "first_post",     name: "Going Public",     description: "Share a workout to the feed",         group: "Social", test: (s) => s.posts >= 1 },
  { id: "followers_10",   name: "Getting Noticed",  description: "Reach 10 followers",                  group: "Social", test: (s) => s.followers / 10 },
  { id: "published_plan", name: "Coach",            description: "Publish a training plan",             group: "Social", test: (s) => s.plansPublished >= 1 },
  { id: "subscribed_10",  name: "Programmer",       description: "Have 10 people subscribe to your plans", group: "Social", test: (s) => s.planSubscribers / 10 },
];

/**
 * Evaluates every achievement against a stats object.
 *
 * Tests returning a fraction give partial progress, so the interface can show
 * how close someone is rather than a bare locked/unlocked state — which is
 * what makes the system motivating rather than merely decorative.
 */
export function evaluateAchievements(stats) {
  return ACHIEVEMENTS.map((a) => {
    const raw = a.test(stats);
    const progress = typeof raw === "boolean" ? (raw ? 1 : 0) : Math.max(0, Math.min(1, raw || 0));

    return {
      id: a.id,
      name: a.name,
      description: a.description,
      group: a.group,
      unlocked: progress >= 1,
      progressPercent: round(progress * 100, 0),
    };
  });
}

/** Assembles the stats object the achievement tests expect. */
export function buildAchievementStats(entries, extras = {}) {
  const strength = entries.filter((e) => e.type !== "cardio");
  const cardio = entries.filter((e) => e.type === "cardio");

  const totalVolume = strength.reduce((s, e) => s + (e.sets || 0) * (e.reps || 0) * (e.weight || 0), 0);
  const totalDistance = cardio.reduce((s, e) => s + (e.distance || 0), 0);
  const streaks = computeStreaks(entries, extras.now);

  return {
    sessions: entries.length,
    totalVolume,
    totalDistance,
    longestStreak: streaks.longest,
    currentStreak: streaks.current,
    bestBench: extras.bestBench || 0,
    bestSquat: extras.bestSquat || 0,
    bestDeadlift: extras.bestDeadlift || 0,
    bodyweight: extras.bodyweight || null,
    pushPullRatio: extras.pushPullRatio ?? null,
    groupsTrained: extras.groupsTrained || 0,
    acwrOptimal: extras.acwrOptimal === true,
    posts: extras.posts || 0,
    followers: extras.followers || 0,
    plansPublished: extras.plansPublished || 0,
    planSubscribers: extras.planSubscribers || 0,
  };
}

function round(value, dp) {
  if (value === null || value === undefined || !isFinite(value)) return null;
  const f = Math.pow(10, dp);
  return Math.round(value * f) / f;
}
