// ============================================================================
// 225 — EXERCISE CATALOG
// ============================================================================
// A controlled vocabulary of exercises, replacing free-text entry.
//
// Why this exists:
//
//   Free text made every downstream comparison unreliable. "Bench Press",
//   "bench press", "Barbell Bench" and "BB Bench" are one exercise to a
//   lifter and four distinct strings to a database. That breaks per-exercise
//   progress tracking, makes cross-user comparison meaningless, and forced
//   movement-pattern classification to be done by keyword guessing.
//
//   With a fixed catalog, each entry carries a stable ID and authoritative
//   metadata, so classification becomes a lookup rather than a heuristic.
//
// The trade-off is real and worth stating: a closed vocabulary cannot record
// an exercise it does not know about. The catalog is therefore deliberately
// broad in common training movements and deliberately excludes highly
// specialised variations. Extending it means adding an entry here — a
// controlled, reviewable change rather than uncontrolled string entry.
// ============================================================================

export const MUSCLE_GROUPS = ["Chest", "Back", "Shoulders", "Legs", "Arms", "Core"];

// pattern    — movement pattern, used by the analytics engine for balance
// group      — primary muscle group, used for grouping in the interface
// compound   — multi-joint movements, which carry most of a session's load
export const EXERCISES = [
  // ----- CHEST -----
  { id: "bench_press_barbell",   name: "Barbell Bench Press",       group: "Chest", pattern: "push", compound: true },
  { id: "bench_press_incline",   name: "Incline Barbell Press",     group: "Chest", pattern: "push", compound: true },
  { id: "bench_press_decline",   name: "Decline Barbell Press",     group: "Chest", pattern: "push", compound: true },
  { id: "bench_press_dumbbell",  name: "Dumbbell Bench Press",      group: "Chest", pattern: "push", compound: true },
  { id: "incline_dumbbell",      name: "Incline Dumbbell Press",    group: "Chest", pattern: "push", compound: true },
  { id: "chest_press_machine",   name: "Chest Press Machine",       group: "Chest", pattern: "push", compound: true },
  { id: "dumbbell_fly",          name: "Dumbbell Fly",              group: "Chest", pattern: "push", compound: false },
  { id: "cable_fly",             name: "Cable Fly",                 group: "Chest", pattern: "push", compound: false },
  { id: "pec_deck",              name: "Pec Deck",                  group: "Chest", pattern: "push", compound: false },
  { id: "push_up",               name: "Push-Up",                   group: "Chest", pattern: "push", compound: true },
  { id: "dip_chest",             name: "Chest Dip",                 group: "Chest", pattern: "push", compound: true },

  // ----- BACK -----
  { id: "deadlift",              name: "Deadlift",                  group: "Back", pattern: "legs", compound: true },
  { id: "deadlift_sumo",         name: "Sumo Deadlift",             group: "Back", pattern: "legs", compound: true },
  { id: "rack_pull",             name: "Rack Pull",                 group: "Back", pattern: "pull", compound: true },
  { id: "row_barbell",           name: "Barbell Row",               group: "Back", pattern: "pull", compound: true },
  { id: "row_dumbbell",          name: "Dumbbell Row",              group: "Back", pattern: "pull", compound: true },
  { id: "row_tbar",              name: "T-Bar Row",                 group: "Back", pattern: "pull", compound: true },
  { id: "row_cable_seated",      name: "Seated Cable Row",          group: "Back", pattern: "pull", compound: true },
  { id: "row_machine",           name: "Machine Row",               group: "Back", pattern: "pull", compound: true },
  { id: "lat_pulldown",          name: "Lat Pulldown",              group: "Back", pattern: "pull", compound: true },
  { id: "pull_up",               name: "Pull-Up",                   group: "Back", pattern: "pull", compound: true },
  { id: "chin_up",               name: "Chin-Up",                   group: "Back", pattern: "pull", compound: true },
  { id: "face_pull",             name: "Face Pull",                 group: "Back", pattern: "pull", compound: false },
  { id: "shrug",                 name: "Shrug",                     group: "Back", pattern: "pull", compound: false },
  { id: "back_extension",        name: "Back Extension",            group: "Back", pattern: "pull", compound: false },

  // ----- SHOULDERS -----
  { id: "overhead_press",        name: "Overhead Press",            group: "Shoulders", pattern: "push", compound: true },
  { id: "shoulder_press_db",     name: "Dumbbell Shoulder Press",   group: "Shoulders", pattern: "push", compound: true },
  { id: "arnold_press",          name: "Arnold Press",              group: "Shoulders", pattern: "push", compound: true },
  { id: "shoulder_press_machine",name: "Shoulder Press Machine",    group: "Shoulders", pattern: "push", compound: true },
  { id: "lateral_raise",         name: "Lateral Raise",             group: "Shoulders", pattern: "push", compound: false },
  { id: "front_raise",           name: "Front Raise",               group: "Shoulders", pattern: "push", compound: false },
  { id: "rear_delt_fly",         name: "Rear Delt Fly",             group: "Shoulders", pattern: "pull", compound: false },
  { id: "upright_row",           name: "Upright Row",               group: "Shoulders", pattern: "pull", compound: false },

  // ----- LEGS -----
  { id: "squat_back",            name: "Back Squat",                group: "Legs", pattern: "legs", compound: true },
  { id: "squat_front",           name: "Front Squat",               group: "Legs", pattern: "legs", compound: true },
  { id: "squat_goblet",          name: "Goblet Squat",              group: "Legs", pattern: "legs", compound: true },
  { id: "squat_hack",            name: "Hack Squat",                group: "Legs", pattern: "legs", compound: true },
  { id: "leg_press",             name: "Leg Press",                 group: "Legs", pattern: "legs", compound: true },
  { id: "romanian_deadlift",     name: "Romanian Deadlift",         group: "Legs", pattern: "legs", compound: true },
  { id: "lunge",                 name: "Lunge",                     group: "Legs", pattern: "legs", compound: true },
  { id: "split_squat_bulgarian", name: "Bulgarian Split Squat",     group: "Legs", pattern: "legs", compound: true },
  { id: "step_up",               name: "Step-Up",                   group: "Legs", pattern: "legs", compound: true },
  { id: "hip_thrust",            name: "Hip Thrust",                group: "Legs", pattern: "legs", compound: true },
  { id: "leg_extension",         name: "Leg Extension",             group: "Legs", pattern: "legs", compound: false },
  { id: "leg_curl",              name: "Leg Curl",                  group: "Legs", pattern: "legs", compound: false },
  { id: "calf_raise",            name: "Calf Raise",                group: "Legs", pattern: "legs", compound: false },
  { id: "glute_bridge",          name: "Glute Bridge",              group: "Legs", pattern: "legs", compound: false },

  // ----- ARMS -----
  { id: "curl_barbell",          name: "Barbell Curl",              group: "Arms", pattern: "pull", compound: false },
  { id: "curl_dumbbell",         name: "Dumbbell Curl",             group: "Arms", pattern: "pull", compound: false },
  { id: "curl_hammer",           name: "Hammer Curl",               group: "Arms", pattern: "pull", compound: false },
  { id: "curl_preacher",         name: "Preacher Curl",             group: "Arms", pattern: "pull", compound: false },
  { id: "curl_cable",            name: "Cable Curl",                group: "Arms", pattern: "pull", compound: false },
  { id: "tricep_pushdown",       name: "Tricep Pushdown",           group: "Arms", pattern: "push", compound: false },
  { id: "tricep_overhead",       name: "Overhead Tricep Extension", group: "Arms", pattern: "push", compound: false },
  { id: "skull_crusher",         name: "Skull Crusher",             group: "Arms", pattern: "push", compound: false },
  { id: "close_grip_bench",      name: "Close-Grip Bench Press",    group: "Arms", pattern: "push", compound: true },
  { id: "dip_tricep",            name: "Tricep Dip",                group: "Arms", pattern: "push", compound: true },

  // ----- CORE -----
  { id: "plank",                 name: "Plank",                     group: "Core", pattern: "core", compound: false },
  { id: "hanging_leg_raise",     name: "Hanging Leg Raise",         group: "Core", pattern: "core", compound: false },
  { id: "cable_crunch",          name: "Cable Crunch",              group: "Core", pattern: "core", compound: false },
  { id: "crunch",                name: "Crunch",                    group: "Core", pattern: "core", compound: false },
  { id: "sit_up",                name: "Sit-Up",                    group: "Core", pattern: "core", compound: false },
  { id: "russian_twist",         name: "Russian Twist",             group: "Core", pattern: "core", compound: false },
  { id: "ab_wheel",              name: "Ab Wheel Rollout",          group: "Core", pattern: "core", compound: false },
  { id: "dead_bug",              name: "Dead Bug",                  group: "Core", pattern: "core", compound: false },
];

// Indexes built once at load, so lookups are constant-time rather than
// scanning the array on every classification call.
const BY_ID = new Map(EXERCISES.map((e) => [e.id, e]));
const BY_NAME = new Map(EXERCISES.map((e) => [e.name.toLowerCase(), e]));

/**
 * Resolves an exercise from either its catalog ID or its display name.
 *
 * Entries logged before the catalog existed stored only free text, so name
 * lookup is attempted as a fallback. Anything still unresolved returns null
 * and is handled by the legacy keyword path in the analytics engine.
 */
export function lookupExercise(idOrName) {
  if (!idOrName) return null;
  return BY_ID.get(idOrName) || BY_NAME.get(String(idOrName).toLowerCase()) || null;
}

/** Exercises grouped by muscle group, in catalog order, for building menus. */
export function exercisesByGroup() {
  const grouped = {};
  for (const g of MUSCLE_GROUPS) grouped[g] = [];
  for (const e of EXERCISES) {
    if (grouped[e.group]) grouped[e.group].push(e);
  }
  return grouped;
}

/** Display name for an exercise, falling back to whatever was stored. */
export function exerciseName(idOrName) {
  const found = lookupExercise(idOrName);
  return found ? found.name : idOrName;
}

export const EXERCISE_COUNT = EXERCISES.length;
