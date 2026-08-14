// Import the pieces we need directly from Firebase's servers (no install needed)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import {
  getFirestore,
  collection,
  addDoc,
  setDoc,
  getDoc,
  deleteDoc,
  doc,
  query,
  where,
  orderBy,
  limit,
  onSnapshot,
  getDocs,
  writeBatch,
  updateDoc,
  serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// PHOTO UPLOAD — disabled (requires Firebase Blaze plan for Cloud Storage).
// To re-enable, uncomment this import block.
// import {
//   getStorage,
//   ref,
//   uploadBytes,
//   getDownloadURL,
// } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-storage.js";

import { firebaseConfig } from "./firebase-config.js";

// Analytics and matching are kept in separate modules deliberately: they are
// pure computation with no Firebase or DOM access, which is what allows them
// to be unit-tested in isolation (see analytics.test.mjs, partner-matching.test.mjs).
import {
  normaliseEntries,
  estimate1RM,
  detectPlateau,
  computeACWR,
  weeklyVolumeSeries,
  movementBalance,
  muscleGroupVolume,
  classifyExercise,
} from "./analytics.js";

import { exercisesByGroup, exerciseName, lookupExercise } from "./exercises.js";

import {
  buildInteractionMatrix,
  trainMatrixFactorization,
  recommend,
  evaluate,
} from "./ml-recommender.js";

import {
  buildFeatureVector,
  discoverArchetypes,
} from "./clustering.js";

import {
  computeRank,
  computeStreaks,
  buildAchievementStats,
  evaluateAchievements,
} from "./ranking.js";

import {
  buildProfileVector,
  strengthLevel,
  activitySet,
  rankPartners,
} from "./partner-matching.js";

console.log("225 script.js loaded — version: ml-clustering-v1");

// Connect to your specific Firebase project
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
// const storage = getStorage(app); // PHOTO UPLOAD — disabled

// Escapes special HTML characters in user-provided text before it's inserted
// via innerHTML, so a workout name or caption like "<script>...</script>"
// is displayed as harmless text instead of being executed as real HTML.
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Builds the grouped option list for an exercise selector. Options are nested
// under <optgroup> by muscle group so a 65-entry catalog stays navigable.
function buildExerciseOptions(selectedId) {
  const grouped = exercisesByGroup();
  let html = '<option value="">Select an exercise...</option>';

  for (const [group, list] of Object.entries(grouped)) {
    html += `<optgroup label="${group}">`;
    for (const ex of list) {
      const sel = ex.id === selectedId ? " selected" : "";
      html += `<option value="${ex.id}"${sel}>${ex.name}</option>`;
    }
    html += "</optgroup>";
  }

  // A plan saved before the catalog existed may reference an exercise that is
  // no longer selectable. Rather than silently losing it, it is added as a
  // one-off option so editing the plan does not destroy data.
  if (selectedId && !lookupExercise(selectedId)) {
    html += `<option value="${escapeHtml(selectedId)}" selected>${escapeHtml(selectedId)} (legacy)</option>`;
  }

  return html;
}

// Shows a small inline status message under a form, instead of an
// interruptive alert() popup. type is "error" or "success".
function showMessage(element, text, type) {
  element.textContent = text;
  element.className = "form-message " + type;

  if (type === "success" && text) {
    setTimeout(() => {
      element.textContent = "";
      element.className = "form-message";
    }, 3000);
  }
}

// Finds and deletes any feed posts that were shared from a given workout,
// so removing a workout also cleans up its copy in the feed. Uses a
// one-time getDocs() (not onSnapshot) since this only needs to run once,
// right after the workout itself is deleted.
async function deleteLinkedPosts(workoutId) {
  const linkedPostsQuery = query(collection(db, "posts"), where("workoutId", "==", workoutId));
  const snapshot = await getDocs(linkedPostsQuery);
  const deletions = snapshot.docs.map((postDoc) => deleteDoc(doc(db, "posts", postDoc.id)));
  await Promise.all(deletions);
}

// Formats a number of seconds as m:ss, or h:mm:ss once it passes an hour.
// Used for run durations and paces, which are stored as plain seconds so
// they can be compared and charted numerically.
function formatDuration(totalSeconds) {
  const seconds = Math.round(totalSeconds);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

// Pace is derived, not stored — storing it would mean it could drift out of
// sync with the distance and duration it comes from.
function paceSecondsPerMile(distance, durationSeconds) {
  if (!distance || distance <= 0) return 0;
  return durationSeconds / distance;
}

// Holds the logged-in user's profile document, so posts and comments can
// stamp a username without re-reading it from the database every time.
let currentProfile = null;

// Usernames must be 3-20 characters, letters/numbers/underscores only.
// Returns an error string, or null if the username is valid.
function validateUsername(username) {
  if (username.length < 3) return "Username must be at least 3 characters.";
  if (username.length > 20) return "Username must be 20 characters or fewer.";
  if (!/^[a-zA-Z0-9_]+$/.test(username)) return "Username can only contain letters, numbers, and underscores.";
  return null;
}

// Creates the user's profile document and claims their username, in one
// atomic batch — either both writes succeed or neither does, so we never
// end up with a profile that has no reserved username (or vice versa).
// Usernames are stored lowercase as the document ID, which is what makes
// them unique: two documents can't share one path.
async function createProfile(uid, email, username) {
  const batch = writeBatch(db);

  batch.set(doc(db, "usernames", username.toLowerCase()), {
    uid: uid,
    claimedAt: serverTimestamp(),
  });

  batch.set(doc(db, "users", uid), {
    username: username,
    usernameLower: username.toLowerCase(),
    email: email,
    displayName: "",
    bio: "",
    goal: "",
    createdAt: serverTimestamp(),
  });

  await batch.commit();
}

// PHOTO UPLOAD — disabled (requires Firebase Blaze plan for Cloud Storage).
// Uploads a profile photo to Cloud Storage and returns its public URL.
// The file is stored at a path keyed by the user's ID, which is what the
// Storage security rules check — so a user can only ever overwrite their
// own photo. Using the same path each time also means an old photo is
// replaced rather than accumulating orphaned files.
// async function uploadProfilePhoto(uid, file) {
//   const storageRef = ref(storage, `profile-photos/${uid}`);
//   await uploadBytes(storageRef, file);
//   return await getDownloadURL(storageRef);
// }

// Reads a user's profile document. Returns null if they don't have one yet
// (which is the case for accounts created before profiles existed).
async function fetchProfile(uid) {
  const snapshot = await getDoc(doc(db, "users", uid));
  return snapshot.exists() ? snapshot.data() : null;
}

// Keeps track of which workout/feed list listeners are currently active,
// so we can stop them when a different user logs in.
let unsubscribeWorkoutList = null;
let unsubscribeFeedList = null;

// Feed filter state: "all" shows every post, "following" shows only posts
// from people the current user follows.
let feedMode = "all";
let followingIds = [];

// The most recent workout data from the live listener, kept so the progress
// chart can re-render without issuing its own database query.
let cachedWorkouts = [];
let myLocationsCache = []; // own training places, reused by partner matching
let myPlansCount = 0;      // feeds the "Coach" achievement
let chartMetric = "weight";   // strength: "weight" | "volume"
let cardioMetric = "distance"; // cardio: "distance" | "pace" | "duration"
let progressType = "strength"; // which progress tab is active

document.addEventListener("DOMContentLoaded", () => {
  // --- Login / Sign up elements ---
  const loginView = document.getElementById("login-view");
  const loginForm = document.getElementById("login-form");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const submitButton = document.getElementById("submit-button");
  const switchLink = document.getElementById("switch-mode");
  const switchText = document.getElementById("switch-text");
  const loginMessage = document.getElementById("login-message");
  const usernameField = document.getElementById("username-field");
  const usernameInput = document.getElementById("username");

  // --- Profile setup (legacy accounts) ---
  const profileSetupView = document.getElementById("profile-setup-view");
  const profileSetupForm = document.getElementById("profile-setup-form");
  const setupUsernameInput = document.getElementById("setup-username");
  const setupSubmitButton = document.getElementById("setup-submit-button");
  const setupMessage = document.getElementById("setup-message");
  const setupLogout = document.getElementById("setup-logout");

  // --- Profile tab ---
  const tabProfileBtn = document.getElementById("tab-profile-btn");
  const profileView = document.getElementById("profile-view");
  const profileForm = document.getElementById("profile-form");
  const profileAvatar = document.getElementById("profile-avatar");
  const profileUsernameEl = document.getElementById("profile-username");
  const profileMetaEl = document.getElementById("profile-meta");
  const profileDisplayNameInput = document.getElementById("profile-display-name");
  // const profilePhotoInput = document.getElementById("profile-photo"); // PHOTO UPLOAD — disabled
  const profileBioInput = document.getElementById("profile-bio");
  const profileGoalInput = document.getElementById("profile-goal");
  const profileBodyweightInput = document.getElementById("profile-bodyweight");
  const profileCategoryInput = document.getElementById("profile-category");
  const rankPanel = document.getElementById("rank-panel");
  const achievementsPanel = document.getElementById("achievements-panel");
  const achievementCount = document.getElementById("achievement-count");
  const profileSubmitButton = document.getElementById("profile-submit-button");
  const profileMessage = document.getElementById("profile-message");
  const statWorkouts = document.getElementById("stat-workouts");
  const statPosts = document.getElementById("stat-posts");
  const statFollowing = document.getElementById("stat-following");
  const statFollowers = document.getElementById("stat-followers");

  // --- People tab ---
  const tabPeopleBtn = document.getElementById("tab-people-btn");
  const peopleView = document.getElementById("people-view");
  const peopleSearchInput = document.getElementById("people-search");
  const peopleSearchButton = document.getElementById("people-search-button");
  const peopleResults = document.getElementById("people-results");
  const followingListEl = document.getElementById("following-list");

  // --- Feed filter ---
  const filterAllBtn = document.getElementById("filter-all");
  const filterFollowingBtn = document.getElementById("filter-following");

  // --- Plans tab ---
  const tabPlansBtn = document.getElementById("tab-plans-btn");
  const plansView = document.getElementById("plans-view");
  const togglePlanFormBtn = document.getElementById("toggle-plan-form");
  const planForm = document.getElementById("plan-form");
  const planTitleInput = document.getElementById("plan-title");
  const planDescriptionInput = document.getElementById("plan-description");
  const planLevelInput = document.getElementById("plan-level");
  const planDaysInput = document.getElementById("plan-days");
  const planExerciseRows = document.getElementById("plan-exercise-rows");
  const addExerciseRowBtn = document.getElementById("add-exercise-row");
  const planSubmitButton = document.getElementById("plan-submit-button");
  const planMessage = document.getElementById("plan-message");
  const myPlansList = document.getElementById("my-plans-list");
  const subscribedPlansList = document.getElementById("subscribed-plans-list");
  const browsePlansList = document.getElementById("browse-plans-list");

  // --- Detail views ---
  const mainContent = document.getElementById("main-content");
  const planDetailView = document.getElementById("plan-detail-view");
  const planBackBtn = document.getElementById("plan-back-btn");
  const planDetailTitle = document.getElementById("plan-detail-title");
  const planDetailAuthor = document.getElementById("plan-detail-author");
  const planDetailInfo = document.getElementById("plan-detail-info");
  const planDetailDescription = document.getElementById("plan-detail-description");
  const planDetailExercises = document.getElementById("plan-detail-exercises");
  const planSubscribeBtn = document.getElementById("plan-subscribe-btn");

  const userDetailView = document.getElementById("user-detail-view");
  const userBackBtn = document.getElementById("user-back-btn");
  const userDetailAvatar = document.getElementById("user-detail-avatar");
  const userDetailUsername = document.getElementById("user-detail-username");
  const userDetailMeta = document.getElementById("user-detail-meta");
  const userDetailBio = document.getElementById("user-detail-bio");
  const userDetailFollowBtn = document.getElementById("user-detail-follow-btn");
  const userDetailPlans = document.getElementById("user-detail-plans");
  const userDetailPosts = document.getElementById("user-detail-posts");

  // --- Insights ---
  const tabInsightsBtn = document.getElementById("tab-insights-btn");
  const insightsView = document.getElementById("insights-view");
  const acwrPanel = document.getElementById("acwr-panel");
  const trendsPanel = document.getElementById("trends-panel");
  const volumeChart = document.getElementById("volume-chart");
  const balancePanel = document.getElementById("balance-panel");
  const mlPanel = document.getElementById("ml-panel");
  const archetypePanel = document.getElementById("archetype-panel");
  const trainModelBtn = document.getElementById("train-model-btn");

  // --- Partner matching ---
  const matchesList = document.getElementById("matches-list");
  const refreshMatchesBtn = document.getElementById("refresh-matches");

  // --- Training locations ---
  const toggleLocationFormBtn = document.getElementById("toggle-location-form");
  const locationForm = document.getElementById("location-form");
  const locationNameInput = document.getElementById("location-name");
  const locationActivityInput = document.getElementById("location-activity");
  const locationWhenInput = document.getElementById("location-when");
  const locationSearchInput = document.getElementById("location-search");
  const locationSearchBtn = document.getElementById("location-search-btn");
  const locationPicked = document.getElementById("location-picked");
  const locationSubmitButton = document.getElementById("location-submit-button");
  const locationMessage = document.getElementById("location-message");
  const myLocationsList = document.getElementById("my-locations-list");
  const userDetailLocations = document.getElementById("user-detail-locations");

  // --- Dashboard elements ---
  const dashboardView = document.getElementById("dashboard-view");
  const logoutButton = document.getElementById("logout-button");
  const workoutForm = document.getElementById("workout-form");
  const exerciseNameInput = document.getElementById("exercise-name");
  const setsInput = document.getElementById("sets");
  const repsInput = document.getElementById("reps");
  const weightInput = document.getElementById("weight");
  const workoutList = document.getElementById("workout-list");
  const workoutSubmitButton = document.getElementById("workout-submit-button");
  const workoutMessage = document.getElementById("workout-message");

  // --- Cardio logging ---
  const logStrengthBtn = document.getElementById("log-strength-btn");
  const logCardioBtn = document.getElementById("log-cardio-btn");
  const cardioForm = document.getElementById("cardio-form");
  const cardioActivityInput = document.getElementById("cardio-activity");
  const cardioDistanceInput = document.getElementById("cardio-distance");
  const cardioMinutesInput = document.getElementById("cardio-minutes");
  const cardioSecondsInput = document.getElementById("cardio-seconds");
  const cardioSubmitButton = document.getElementById("cardio-submit-button");
  const cardioMessage = document.getElementById("cardio-message");

  // --- Progress chart ---
  const progressStrengthBtn = document.getElementById("progress-strength-btn");
  const progressCardioBtn = document.getElementById("progress-cardio-btn");
  const progressExerciseSelect = document.getElementById("progress-exercise");
  const progressActivitySelect = document.getElementById("progress-activity");
  const strengthMetrics = document.getElementById("strength-metrics");
  const cardioMetrics = document.getElementById("cardio-metrics");
  const progressChart = document.getElementById("progress-chart");
  const metricWeightBtn = document.getElementById("metric-weight");
  const metricVolumeBtn = document.getElementById("metric-volume");
  const metricDistanceBtn = document.getElementById("metric-distance");
  const metricPaceBtn = document.getElementById("metric-pace");
  const metricDurationBtn = document.getElementById("metric-duration");
  const statBest = document.getElementById("stat-best");
  const statBestLabel = document.getElementById("stat-best-label");
  const statVolume = document.getElementById("stat-volume");
  const statVolumeLabel = document.getElementById("stat-volume-label");
  const statSessions = document.getElementById("stat-sessions");

  // --- Tabs ---
  const tabWorkoutsBtn = document.getElementById("tab-workouts-btn");
  const tabFeedBtn = document.getElementById("tab-feed-btn");
  const workoutsView = document.getElementById("workouts-view");
  const feedView = document.getElementById("feed-view");
  const feedList = document.getElementById("feed-list");

  // Populate the exercise selector from the catalog
  exerciseNameInput.innerHTML = buildExerciseOptions();

  let isSignUpMode = false;

  // ---------- LOGIN / SIGN UP MODE TOGGLE ----------
  switchLink.addEventListener("click", (event) => {
    event.preventDefault();
    isSignUpMode = !isSignUpMode;

    submitButton.textContent = isSignUpMode ? "Sign Up" : "Log In";
    switchText.textContent = isSignUpMode ? "Already have an account?" : "Don't have an account?";
    switchLink.textContent = isSignUpMode ? "Log in" : "Sign up";
    usernameField.style.display = isSignUpMode ? "block" : "none"; // username is only needed when creating an account
  });

  // ---------- LOGIN / SIGN UP SUBMIT ----------
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage(loginMessage, "", "");

    const email = emailInput.value;
    const password = passwordInput.value;
    const username = usernameInput.value.trim();

    if (password.length < 6) {
      showMessage(loginMessage, "Password must be at least 6 characters.", "error");
      return;
    }

    // Username is only required (and only validated) when signing up.
    if (isSignUpMode) {
      const usernameError = validateUsername(username);
      if (usernameError) {
        showMessage(loginMessage, usernameError, "error");
        return;
      }
    }

    // Disable the button and show a loading label so the click feels
    // acknowledged immediately, and to prevent double-submits.
    submitButton.disabled = true;
    submitButton.textContent = isSignUpMode ? "Signing up..." : "Logging in...";

    try {
      if (isSignUpMode) {
        // Check the username is free before creating the account, so we don't
        // leave a user with an account but no profile if the name is taken.
        const existing = await getDoc(doc(db, "usernames", username.toLowerCase()));
        if (existing.exists()) {
          showMessage(loginMessage, "That username is already taken.", "error");
          return;
        }

        const credential = await createUserWithEmailAndPassword(auth, email, password);
        await createProfile(credential.user.uid, email, username);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }

      emailInput.value = "";
      passwordInput.value = "";
      usernameInput.value = "";
    } catch (error) {
      console.error(error.code, error.message);
      showMessage(loginMessage, error.message, "error");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = isSignUpMode ? "Sign Up" : "Log In";
    }
  });

  // ---------- PROFILE SETUP (accounts created before profiles existed) ----------
  profileSetupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage(setupMessage, "", "");

    const user = auth.currentUser;
    if (!user) return;

    const username = setupUsernameInput.value.trim();
    const usernameError = validateUsername(username);
    if (usernameError) {
      showMessage(setupMessage, usernameError, "error");
      return;
    }

    setupSubmitButton.disabled = true;
    setupSubmitButton.textContent = "Saving...";

    try {
      const existing = await getDoc(doc(db, "usernames", username.toLowerCase()));
      if (existing.exists()) {
        showMessage(setupMessage, "That username is already taken.", "error");
        return;
      }

      await createProfile(user.uid, user.email, username);
      currentProfile = await fetchProfile(user.uid);
      setupUsernameInput.value = "";
      enterDashboard(user);
    } catch (error) {
      console.error(error.code, error.message);
      showMessage(setupMessage, error.message, "error");
    } finally {
      setupSubmitButton.disabled = false;
      setupSubmitButton.textContent = "Continue";
    }
  });

  setupLogout.addEventListener("click", (event) => {
    event.preventDefault();
    signOut(auth);
  });

  // ---------- LOGOUT ----------
  logoutButton.addEventListener("click", () => {
    signOut(auth);
  });

  // ---------- LOG A NEW WORKOUT ----------
  workoutForm.addEventListener("submit", (event) => {
    event.preventDefault();
    showMessage(workoutMessage, "", "");

    const user = auth.currentUser;
    if (!user) return; // safety check, shouldn't happen since form is only visible when logged in

    workoutSubmitButton.disabled = true;
    workoutSubmitButton.textContent = "Saving...";

    const selectedId = exerciseNameInput.value;
    if (!selectedId) {
      showMessage(workoutMessage, "Choose an exercise from the list.", "error");
      workoutSubmitButton.disabled = false;
      workoutSubmitButton.textContent = "Save Workout";
      return;
    }

    addDoc(collection(db, "workouts"), {
      userId: user.uid,
      exerciseId: selectedId,
      exerciseName: exerciseName(selectedId),
      sets: Number(setsInput.value),
      reps: Number(repsInput.value),
      weight: Number(weightInput.value),
      createdAt: serverTimestamp(),
    })
      .then(() => {
        workoutForm.reset();
        showMessage(workoutMessage, "Workout saved.", "success");
      })
      .catch((error) => {
        console.error(error.code, error.message);
        showMessage(workoutMessage, "Couldn't save that workout: " + error.message, "error");
      })
      .finally(() => {
        workoutSubmitButton.disabled = false;
        workoutSubmitButton.textContent = "Save Workout";
      });
  });

  // ---------- LOG TYPE TOGGLE (strength vs cardio) ----------
  logStrengthBtn.addEventListener("click", () => {
    logStrengthBtn.classList.add("active");
    logCardioBtn.classList.remove("active");
    workoutForm.style.display = "block";
    cardioForm.style.display = "none";
  });

  logCardioBtn.addEventListener("click", () => {
    logCardioBtn.classList.add("active");
    logStrengthBtn.classList.remove("active");
    cardioForm.style.display = "block";
    workoutForm.style.display = "none";
  });

  // ---------- LOG A CARDIO SESSION ----------
  cardioForm.addEventListener("submit", (event) => {
    event.preventDefault();
    showMessage(cardioMessage, "", "");

    const user = auth.currentUser;
    if (!user) return;

    const distance = Number(cardioDistanceInput.value);
    const minutes = Number(cardioMinutesInput.value) || 0;
    const seconds = Number(cardioSecondsInput.value) || 0;
    const durationSeconds = minutes * 60 + seconds;

    if (distance <= 0) {
      showMessage(cardioMessage, "Distance must be greater than zero.", "error");
      return;
    }
    if (durationSeconds <= 0) {
      showMessage(cardioMessage, "Enter how long the session took.", "error");
      return;
    }

    cardioSubmitButton.disabled = true;
    cardioSubmitButton.textContent = "Saving...";

    // Same collection as strength workouts — the "type" field is what
    // distinguishes them, so no new security rules or indexes are needed.
    addDoc(collection(db, "workouts"), {
      userId: user.uid,
      type: "cardio",
      activity: cardioActivityInput.value,
      distance: distance,
      durationSeconds: durationSeconds,
      createdAt: serverTimestamp(),
    })
      .then(() => {
        cardioForm.reset();
        showMessage(cardioMessage, "Session saved.", "success");
      })
      .catch((error) => {
        console.error(error.code, error.message);
        showMessage(cardioMessage, "Couldn't save that session: " + error.message, "error");
      })
      .finally(() => {
        cardioSubmitButton.disabled = false;
        cardioSubmitButton.textContent = "Save Run";
      });
  });

  // ---------- TAB SWITCHING ----------
  // Small helper so adding a tab doesn't mean touching three separate handlers.
  function selectTab(activeBtn, activeView) {
    [tabWorkoutsBtn, tabFeedBtn, tabInsightsBtn, tabPeopleBtn, tabPlansBtn, tabProfileBtn].forEach((btn) => btn.classList.remove("active"));
    [workoutsView, feedView, insightsView, peopleView, plansView, profileView].forEach((view) => (view.style.display = "none"));
    activeBtn.classList.add("active");
    activeView.style.display = "block";
  }

  tabWorkoutsBtn.addEventListener("click", () => selectTab(tabWorkoutsBtn, workoutsView));
  tabFeedBtn.addEventListener("click", () => selectTab(tabFeedBtn, feedView));
  tabInsightsBtn.addEventListener("click", () => {
    selectTab(tabInsightsBtn, insightsView);
    renderInsights();
  });
  tabPeopleBtn.addEventListener("click", () => {
    selectTab(tabPeopleBtn, peopleView);
    loadFollowingList();
    loadPartnerMatches();
  });
  tabPlansBtn.addEventListener("click", () => {
    selectTab(tabPlansBtn, plansView);
    loadPlansTab();
  });
  tabProfileBtn.addEventListener("click", () => {
    selectTab(tabProfileBtn, profileView);
    loadProfileStats(); // refresh counts each time the tab is opened
    loadMyLocations();  // map must be sized after its container is visible
    renderRankAndAchievements();
  });

  // ---------- DELETE A WORKOUT / SHARE TO FEED ----------
  // One listener on the whole list, instead of one per button, since
  // workout entries get created and destroyed dynamically by onSnapshot.
  workoutList.addEventListener("click", (event) => {
    // --- Delete button ---
    if (event.target.classList.contains("delete-btn")) {
      const workoutId = event.target.dataset.id;
      const confirmed = confirm("Delete this workout? Any feed post sharing it will be removed too. This can't be undone.");
      if (!confirmed) return;

      deleteDoc(doc(db, "workouts", workoutId))
        .then(() => deleteLinkedPosts(workoutId))
        .catch((error) => {
          console.error(error.code, error.message);
          alert("Couldn't delete that workout: " + error.message);
        });
      return;
    }

    // --- Share button ---
    if (event.target.classList.contains("share-btn")) {
      const user = auth.currentUser;
      if (!user) return;

      const button = event.target;
      const caption = prompt("Add a caption for this post (optional):", "");
      if (caption === null) return; // user clicked Cancel

      const originalLabel = button.textContent;
      button.disabled = true;
      button.textContent = "Sharing...";

      // Build the post from whichever set of data attributes this entry carries.
      const isCardio = button.dataset.type === "cardio";
      const activityData = isCardio
        ? {
            type: "cardio",
            activity: button.dataset.activity,
            distance: Number(button.dataset.distance),
            durationSeconds: Number(button.dataset.duration),
          }
        : {
            type: "strength",
            exerciseName: button.dataset.exercise,
            sets: Number(button.dataset.sets),
            reps: Number(button.dataset.reps),
            weight: Number(button.dataset.weight),
          };

      addDoc(collection(db, "posts"), {
        authorId: user.uid,
        authorUsername: currentProfile ? currentProfile.username : "unknown",
        workoutId: button.dataset.workoutId,
        caption: caption,
        createdAt: serverTimestamp(),
        ...activityData, // spread merges the type-specific fields in
      })
        .then(() => {
          button.textContent = "Shared!";
          setTimeout(() => {
            button.textContent = originalLabel;
          }, 2000);
        })
        .catch((error) => {
          console.error(error.code, error.message);
          alert("Couldn't share that workout: " + error.message);
          button.textContent = originalLabel;
        })
        .finally(() => {
          button.disabled = false;
        });
    }
  });

  // ---------- LIKES ----------
  // Fetches the current like count and whether the logged-in user has
  // already liked this post, then sets the button's visual state to match.
  // Runs once per post each time the feed re-renders (a one-time read,
  // not a live listener — see the note on this tradeoff further down).
  async function initLikeButton(button) {
    const user = auth.currentUser;
    if (!user) return;

    const postId = button.dataset.postId;
    const likesSnapshot = await getDocs(collection(db, "posts", postId, "likes"));
    const likeCountEl = button.querySelector(".like-count");
    const heartEl = button.querySelector(".heart");

    likeCountEl.textContent = likesSnapshot.size;

    const likedByMe = likesSnapshot.docs.some((likeDoc) => likeDoc.id === user.uid);
    button.dataset.liked = likedByMe ? "true" : "false";
    button.classList.toggle("liked", likedByMe);
    heartEl.textContent = likedByMe ? "♥" : "♡";
  }

  // Handles clicks on any like button in the feed (event delegation,
  // same pattern used for delete/share in the Workouts tab).
  feedList.addEventListener("click", (event) => {
    // --- Comment toggle (expand/collapse a post's comment section) ---
    const commentToggle = event.target.closest(".comment-btn");
    if (commentToggle) {
      const postEl = commentToggle.closest(".post");
      const section = postEl.querySelector(".comments-section");
      const listEl = section.querySelector(".comment-list");
      const isOpen = section.style.display !== "none";

      if (isOpen) {
        section.style.display = "none";
      } else {
        section.style.display = "block";
        loadComments(commentToggle.dataset.postId, listEl); // fetch only when actually opened
      }
      return;
    }

    // --- Submit a new comment ---
    const commentSubmit = event.target.closest(".comment-submit");
    if (commentSubmit) {
      const section = commentSubmit.closest(".comments-section");
      const inputEl = section.querySelector(".comment-input");
      const listEl = section.querySelector(".comment-list");
      submitComment(commentSubmit.dataset.postId, inputEl, listEl, commentSubmit);
      return;
    }

    // --- Delete one of your own comments ---
    const commentDelete = event.target.closest(".comment-delete");
    if (commentDelete) {
      const postId = commentDelete.dataset.postId;
      const commentId = commentDelete.dataset.commentId;
      const listEl = commentDelete.closest(".comment-list");

      if (!confirm("Delete this comment?")) return;

      deleteDoc(doc(db, "posts", postId, "comments", commentId))
        .then(() => loadComments(postId, listEl))
        .catch((error) => {
          console.error(error.code, error.message);
          alert("Couldn't delete that comment: " + error.message);
        });
      return;
    }

    // --- Like / unlike ---
    const button = event.target.closest(".like-btn");
    if (!button) return;

    const user = auth.currentUser;
    if (!user) return;

    const postId = button.dataset.postId;
    const isLiked = button.dataset.liked === "true";
    const likeRef = doc(db, "posts", postId, "likes", user.uid);
    const likeCountEl = button.querySelector(".like-count");
    const heartEl = button.querySelector(".heart");

    button.disabled = true;

    const action = isLiked
      ? deleteDoc(likeRef)
      : setDoc(likeRef, { likedAt: serverTimestamp() });

    action
      .then(() => {
        const currentCount = Number(likeCountEl.textContent);
        likeCountEl.textContent = isLiked ? currentCount - 1 : currentCount + 1;
        button.dataset.liked = isLiked ? "false" : "true";
        button.classList.toggle("liked", !isLiked);
        heartEl.textContent = isLiked ? "♡" : "♥";
      })
      .catch((error) => {
        console.error(error.code, error.message);
        alert("Couldn't update like: " + error.message);
      })
      .finally(() => {
        button.disabled = false;
      });
  });

  // Lets the user press Enter in a comment box instead of clicking Post.
  // Uses delegation on the feed container, same reasoning as the click handler.
  feedList.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    if (!event.target.classList.contains("comment-input")) return;

    event.preventDefault();
    const section = event.target.closest(".comments-section");
    const listEl = section.querySelector(".comment-list");
    const buttonEl = section.querySelector(".comment-submit");
    submitComment(buttonEl.dataset.postId, event.target, listEl, buttonEl);
  });

  // ---------- COMMENTS ----------
  // Fetches and renders the comments for one post, on demand.
  // Deliberately lazy: comments are only fetched when a user actually
  // expands that post's comment section, rather than for all 50 feed posts
  // on every render — which would mean 50 extra database reads per refresh
  // for comments most users never open.
  async function loadComments(postId, listEl) {
    const user = auth.currentUser;
    if (!user) return;

    listEl.innerHTML = '<p class="comment-empty">Loading...</p>';

    try {
      const commentsQuery = query(
        collection(db, "posts", postId, "comments"),
        orderBy("createdAt", "asc") // oldest first, so a conversation reads top to bottom
      );
      const snapshot = await getDocs(commentsQuery);

      if (snapshot.empty) {
        listEl.innerHTML = '<p class="comment-empty">No comments yet. Be the first.</p>';
        return;
      }

      listEl.innerHTML = "";

      snapshot.forEach((commentDoc) => {
        const comment = commentDoc.data();
        const isMine = comment.authorId === user.uid;

        const commentEl = document.createElement("div");
        commentEl.className = "comment";
        commentEl.innerHTML = `
          <div class="comment-body">
            <span class="comment-author">@${escapeHtml(comment.authorUsername || "unknown")}</span>
            <span class="comment-text">${escapeHtml(comment.text)}</span>
          </div>
          ${isMine ? `<button class="comment-delete" data-post-id="${postId}" data-comment-id="${commentDoc.id}" title="Delete comment">✕</button>` : ""}
        `;
        listEl.appendChild(commentEl);
      });
    } catch (error) {
      console.error(error.code, error.message);
      listEl.innerHTML = '<p class="comment-empty">Couldn\'t load comments.</p>';
    }
  }

  // Adds a new comment, then refreshes that post's comment list.
  async function submitComment(postId, inputEl, listEl, buttonEl) {
    const user = auth.currentUser;
    if (!user) return;

    const text = inputEl.value.trim(); // .trim() removes leading/trailing spaces
    if (!text) return;                 // ignore empty or whitespace-only comments

    buttonEl.disabled = true;
    buttonEl.textContent = "...";

    try {
      await addDoc(collection(db, "posts", postId, "comments"), {
        authorId: user.uid,
        authorUsername: currentProfile ? currentProfile.username : "unknown",
        text: text,
        createdAt: serverTimestamp(),
      });
      inputEl.value = "";
      await loadComments(postId, listEl);
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't post that comment: " + error.message);
    } finally {
      buttonEl.disabled = false;
      buttonEl.textContent = "Post";
    }
  }

  // ---------- PROGRESS CHART ----------
  // Picks "round" axis values (1, 2, 5, 10, 20, 50...) instead of whatever
  // the raw data happens to be, so the y-axis reads 0/50/100/150 rather
  // than 0/47/94/141. Standard technique for any chart axis.
  function niceScale(min, max, tickCount) {
    const range = (max - min) || 1;
    const rawStep = range / tickCount;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;

    let step;
    if (normalized <= 1) step = 1;
    else if (normalized <= 2) step = 2;
    else if (normalized <= 5) step = 5;
    else step = 10;
    step *= magnitude;

    const niceMin = Math.floor(min / step) * step;
    let niceMax = Math.ceil(max / step) * step;

    // If every value is identical (e.g. bodyweight exercises all logged at
    // 0 lb), min and max collapse to the same number — which would divide
    // by zero when mapping values to pixels. Force a visible range instead.
    if (niceMax === niceMin) niceMax = niceMin + step;

    return { min: niceMin, max: niceMax, step: step };
  }

  // Builds an SVG line chart by hand. Charts are just coordinate math:
  // map each data value onto an x/y pixel position, then draw lines
  // between those points. viewBox makes the whole thing scale to fit
  // whatever width the container happens to be.
  // options.formatValue — how axis labels and tooltips are displayed
  // options.baseline    — "zero" anchors the axis at 0 (right for weight and
  //                       volume, where magnitude matters); "auto" zooms to
  //                       the data's own range (right for pace, where all the
  //                       meaningful variation sits in a narrow band and
  //                       including 0 would flatten the line into nothing).
  function buildChartSVG(points, options) {
    const opts = options || {};
    const format = opts.formatValue || ((v) => Math.round(v * 100) / 100); // default: trim float noise
    const W = 640, H = 240;
    const padL = 52, padR = 18, padT = 18, padB = 36;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const values = points.map((p) => p.value);
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);

    // "auto" pads the range by 10% so points don't sit flush against the edges
    const padding = (dataMax - dataMin) * 0.1 || 1;
    const rangeMin = opts.baseline === "auto" ? dataMin - padding : Math.min(dataMin, 0);
    const rangeMax = opts.baseline === "auto" ? dataMax + padding : dataMax;

    const scale = niceScale(rangeMin, rangeMax, 4);

    // Convert a data value into a vertical pixel position. SVG's y-axis
    // grows downward, so this is inverted: high values get small y.
    const toY = (value) =>
      padT + plotH - ((value - scale.min) / (scale.max - scale.min)) * plotH;

    // Spread points evenly across the width. A single point sits centered.
    const toX = (index) =>
      points.length === 1
        ? padL + plotW / 2
        : padL + (index / (points.length - 1)) * plotW;

    // Horizontal gridlines + y-axis labels
    let gridlines = "";
    for (let v = scale.min; v <= scale.max; v += scale.step) {
      const y = toY(v);
      gridlines += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" class="chart-grid" />`;
      gridlines += `<text x="${padL - 10}" y="${y + 4}" class="chart-axis-label" text-anchor="end">${format(v)}</text>`;
    }

    // The line itself, plus a soft filled area beneath it
    const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(p.value)}`).join(" ");
    const areaPath = `${linePath} L ${toX(points.length - 1)} ${padT + plotH} L ${toX(0)} ${padT + plotH} Z`;

    const dots = points
      .map((p, i) => `<circle cx="${toX(i)}" cy="${toY(p.value)}" r="4" class="chart-dot"><title>${p.label}: ${format(p.value)}</title></circle>`)
      .join("");

    // Only label the first and last dates, so they never overlap
    let dateLabels = "";
    if (points.length > 0) {
      dateLabels += `<text x="${toX(0)}" y="${H - 12}" class="chart-axis-label" text-anchor="start">${points[0].label}</text>`;
      if (points.length > 1) {
        dateLabels += `<text x="${toX(points.length - 1)}" y="${H - 12}" class="chart-axis-label" text-anchor="end">${points[points.length - 1].label}</text>`;
      }
    }

    return `
      <svg viewBox="0 0 ${W} ${H}" class="chart-svg" preserveAspectRatio="xMidYMid meet">
        ${gridlines}
        ${points.length > 1 ? `<path d="${areaPath}" class="chart-area" />` : ""}
        ${points.length > 1 ? `<path d="${linePath}" class="chart-line" />` : ""}
        ${dots}
        ${dateLabels}
      </svg>
    `;
  }

  // Rebuilds both dropdowns from whatever the user has actually logged,
  // preserving their current selections where possible.
  function refreshExerciseOptions() {
    // Legacy workouts logged before cardio existed have no "type" field,
    // so anything without one is treated as strength.
    const strength = cachedWorkouts.filter((w) => (w.type || "strength") === "strength");
    const cardio = cachedWorkouts.filter((w) => w.type === "cardio");

    fillSelect(progressExerciseSelect, [...new Set(strength.map((w) => w.exerciseName))].sort(), "Log a workout to see progress");
    // Note: still keyed on display name so entries logged before the catalog
    // existed continue to appear alongside catalogued ones.
    fillSelect(progressActivitySelect, [...new Set(cardio.map((w) => w.activity))].sort(), "Log a run to see progress");
  }

  // A Set discards duplicates automatically, giving one entry per name.
  function fillSelect(selectEl, names, emptyText) {
    const previous = selectEl.value;

    if (names.length === 0) {
      selectEl.innerHTML = `<option value="">${emptyText}</option>`;
      return;
    }

    selectEl.innerHTML = names
      .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
      .join("");

    selectEl.value = names.includes(previous) ? previous : names[0];
  }

  // Draws the chart and summary stats for whichever tab and item is selected.
  function renderProgress() {
    if (progressType === "cardio") {
      renderCardioProgress();
    } else {
      renderStrengthProgress();
    }
  }

  function renderStrengthProgress() {
    const exercise = progressExerciseSelect.value;

    const sessions = cachedWorkouts
      .filter((w) => (w.type || "strength") === "strength" && w.exerciseName === exercise && w.createdAt)
      .sort((a, b) => a.createdAt.toMillis() - b.createdAt.toMillis()); // oldest first, left to right

    statBestLabel.textContent = "Best set";
    statVolumeLabel.textContent = "Total volume";

    if (sessions.length === 0) {
      showEmptyProgress("No data for this exercise yet.");
      return;
    }

    const points = sessions.map((w) => ({
      value: chartMetric === "weight" ? w.weight : w.sets * w.reps * w.weight,
      label: shortDate(w.createdAt),
    }));

    progressChart.innerHTML = buildChartSVG(points);

    const bestWeight = Math.max(...sessions.map((w) => w.weight));
    const totalVolume = sessions.reduce((sum, w) => sum + w.sets * w.reps * w.weight, 0);

    statBest.textContent = `${bestWeight} lb`;
    statVolume.textContent = totalVolume.toLocaleString() + " lb"; // toLocaleString adds thousands separators
    statSessions.textContent = sessions.length;
  }

  function renderCardioProgress() {
    const activity = progressActivitySelect.value;

    const sessions = cachedWorkouts
      .filter((w) => w.type === "cardio" && w.activity === activity && w.createdAt)
      .sort((a, b) => a.createdAt.toMillis() - b.createdAt.toMillis());

    statBestLabel.textContent = "Best pace";
    statVolumeLabel.textContent = "Total distance";

    if (sessions.length === 0) {
      showEmptyProgress("No data for this activity yet.");
      return;
    }

    // Each metric needs a value, a display format, and the right baseline.
    let points, chartOptions;
    if (cardioMetric === "distance") {
      points = sessions.map((w) => ({ value: w.distance, label: shortDate(w.createdAt) }));
      chartOptions = { baseline: "zero" };
    } else if (cardioMetric === "pace") {
      points = sessions.map((w) => ({ value: paceSecondsPerMile(w.distance, w.durationSeconds), label: shortDate(w.createdAt) }));
      chartOptions = { formatValue: formatDuration, baseline: "auto" }; // zoom in; a 0-anchored pace axis hides all variation
    } else {
      points = sessions.map((w) => ({ value: w.durationSeconds, label: shortDate(w.createdAt) }));
      chartOptions = { formatValue: formatDuration, baseline: "zero" };
    }

    progressChart.innerHTML = buildChartSVG(points, chartOptions);

    // Best pace is the LOWEST number — faster is better, unlike every other
    // metric in the app where higher is the improvement.
    const bestPace = Math.min(...sessions.map((w) => paceSecondsPerMile(w.distance, w.durationSeconds)));
    const totalDistance = sessions.reduce((sum, w) => sum + w.distance, 0);

    statBest.textContent = `${formatDuration(bestPace)}/mi`;
    statVolume.textContent = `${Math.round(totalDistance * 100) / 100} mi`;
    statSessions.textContent = sessions.length;
  }

  function showEmptyProgress(message) {
    progressChart.innerHTML = `<p class="empty-state">${message}</p>`;
    statBest.textContent = "—";
    statVolume.textContent = "—";
    statSessions.textContent = "—";
  }

  function shortDate(timestamp) {
    return timestamp.toDate().toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  // ---------- PROGRESS TYPE + METRIC TOGGLES ----------
  progressStrengthBtn.addEventListener("click", () => {
    progressType = "strength";
    progressStrengthBtn.classList.add("active");
    progressCardioBtn.classList.remove("active");
    progressExerciseSelect.style.display = "block";
    progressActivitySelect.style.display = "none";
    strengthMetrics.style.display = "flex";
    cardioMetrics.style.display = "none";
    renderProgress();
  });

  progressCardioBtn.addEventListener("click", () => {
    progressType = "cardio";
    progressCardioBtn.classList.add("active");
    progressStrengthBtn.classList.remove("active");
    progressActivitySelect.style.display = "block";
    progressExerciseSelect.style.display = "none";
    cardioMetrics.style.display = "flex";
    strengthMetrics.style.display = "none";
    renderProgress();
  });

  progressExerciseSelect.addEventListener("change", renderProgress);
  progressActivitySelect.addEventListener("change", renderProgress);

  // Small helper so each metric button isn't six near-identical lines.
  function wireMetricButton(button, group, metricName, setter) {
    button.addEventListener("click", () => {
      group.querySelectorAll(".metric-btn").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      setter(metricName);
      renderProgress();
    });
  }

  wireMetricButton(metricWeightBtn, strengthMetrics, "weight", (m) => (chartMetric = m));
  wireMetricButton(metricVolumeBtn, strengthMetrics, "volume", (m) => (chartMetric = m));
  wireMetricButton(metricDistanceBtn, cardioMetrics, "distance", (m) => (cardioMetric = m));
  wireMetricButton(metricPaceBtn, cardioMetrics, "pace", (m) => (cardioMetric = m));
  wireMetricButton(metricDurationBtn, cardioMetrics, "duration", (m) => (cardioMetric = m));

  // ---------- PLAN BUILDER ----------
  // Exercise rows are built in the DOM rather than stored in a variable —
  // the form itself holds the state until submit, which keeps this simple.
  function addExerciseRow(prefill) {
    const data = prefill || {};
    const row = document.createElement("div");
    row.className = "exercise-row";
    row.innerHTML = `
      <input type="number" class="ex-day" placeholder="Day" min="1" max="7" value="${data.day || ""}" />
      <select class="ex-name">${buildExerciseOptions(data.exerciseId || data.name)}</select>
      <input type="number" class="ex-sets" placeholder="Sets" min="1" value="${data.sets || ""}" />
      <input type="text" class="ex-reps" placeholder="Reps" maxlength="12" value="${escapeHtml(data.reps || "")}" />
      <button type="button" class="remove-row-btn" title="Remove">✕</button>
    `;
    planExerciseRows.appendChild(row);
  }

  // Reads every row out of the DOM and turns it into a clean array.
  // Rows missing a name are skipped rather than saved as blanks.
  function collectExercises() {
    return [...planExerciseRows.querySelectorAll(".exercise-row")]
      .map((row) => {
        const id = row.querySelector(".ex-name").value;
        return {
          day: Number(row.querySelector(".ex-day").value) || 1,
          exerciseId: id,
          name: exerciseName(id),
          sets: Number(row.querySelector(".ex-sets").value) || 0,
          reps: row.querySelector(".ex-reps").value.trim(),
        };
      })
      .filter((ex) => ex.exerciseId !== "");
  }

  addExerciseRowBtn.addEventListener("click", () => addExerciseRow());

  planExerciseRows.addEventListener("click", (event) => {
    if (event.target.classList.contains("remove-row-btn")) {
      event.target.closest(".exercise-row").remove();
    }
  });

  togglePlanFormBtn.addEventListener("click", () => {
    const isOpen = planForm.style.display !== "none";
    planForm.style.display = isOpen ? "none" : "block";
    togglePlanFormBtn.textContent = isOpen ? "New Plan" : "Cancel";
    if (!isOpen && planExerciseRows.children.length === 0) {
      addExerciseRow(); // start with one blank row so the form isn't empty
    }
  });

  planForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage(planMessage, "", "");

    const user = auth.currentUser;
    if (!user) return;

    const exercises = collectExercises();
    if (exercises.length === 0) {
      showMessage(planMessage, "Add at least one exercise with a name.", "error");
      return;
    }

    planSubmitButton.disabled = true;
    planSubmitButton.textContent = "Publishing...";

    try {
      await addDoc(collection(db, "plans"), {
        authorId: user.uid,
        authorUsername: currentProfile ? currentProfile.username : "unknown",
        title: planTitleInput.value.trim(),
        description: planDescriptionInput.value.trim(),
        level: planLevelInput.value,
        daysPerWeek: Number(planDaysInput.value) || 1,
        exercises: exercises, // Firestore stores arrays of objects natively
        createdAt: serverTimestamp(),
      });

      planForm.reset();
      planExerciseRows.innerHTML = "";
      planForm.style.display = "none";
      togglePlanFormBtn.textContent = "New Plan";
      showMessage(planMessage, "Plan published.", "success");
      loadPlansTab();
    } catch (error) {
      console.error(error.code, error.message);
      showMessage(planMessage, "Couldn't publish that plan: " + error.message, "error");
    } finally {
      planSubmitButton.disabled = false;
      planSubmitButton.textContent = "Publish Plan";
    }
  });

  // ---------- PLAN LISTS ----------
  // Renders a set of plan cards into a container.
  function renderPlanCards(container, plans, emptyText) {
    if (plans.length === 0) {
      container.innerHTML = `<p class="empty-state">${emptyText}</p>`;
      return;
    }

    container.innerHTML = "";

    plans.forEach((plan) => {
      const card = document.createElement("div");
      card.className = "plan-card";
      card.dataset.planId = plan.id;
      card.innerHTML = `
        <div class="plan-card-title">${escapeHtml(plan.title)}</div>
        <div class="plan-card-meta">${escapeHtml(plan.level)} · ${plan.daysPerWeek}x per week · ${plan.exercises.length} exercises</div>
        <div class="plan-card-author">by @${escapeHtml(plan.authorUsername || "unknown")}</div>
      `;
      container.appendChild(card);
    });
  }

  // Loads all three plan lists. Queries filter on a single field only and
  // sort in JavaScript afterwards — at this scale that avoids needing
  // composite indexes, at the cost of not scaling to thousands of plans.
  async function loadPlansTab() {
    const user = auth.currentUser;
    if (!user) return;

    try {
      const [mineSnap, allSnap, subsSnap] = await Promise.all([
        getDocs(query(collection(db, "plans"), where("authorId", "==", user.uid))),
        getDocs(query(collection(db, "plans"), limit(30))),
        getDocs(collection(db, "users", user.uid, "subscriptions")),
      ]);

      const byNewest = (a, b) => (b.createdAt?.toMillis() || 0) - (a.createdAt?.toMillis() || 0);
      const toPlan = (d) => ({ id: d.id, ...d.data() });

      const mine = mineSnap.docs.map(toPlan).sort(byNewest);
      myPlansCount = mine.length;
      renderPlanCards(myPlansList, mine, "You haven't published a plan yet.");

      const subscribedIds = subsSnap.docs.map((d) => d.id);
      const all = allSnap.docs.map(toPlan);

      const subscribed = all.filter((p) => subscribedIds.includes(p.id)).sort(byNewest);
      renderPlanCards(subscribedPlansList, subscribed, "You're not subscribed to any plans yet.");

      // Browse excludes your own plans and anything you already follow
      const browse = all
        .filter((p) => p.authorId !== user.uid && !subscribedIds.includes(p.id))
        .sort(byNewest);
      renderPlanCards(browsePlansList, browse, "No other plans published yet.");
    } catch (error) {
      console.error(error.code, error.message);
      browsePlansList.innerHTML = '<p class="empty-state">Couldn\'t load plans.</p>';
    }
  }

  // ---------- PLAN DETAIL ----------
  async function openPlanDetail(planId) {
    const user = auth.currentUser;
    if (!user) return;

    showDetail(planDetailView);
    planDetailTitle.textContent = "Loading...";
    planDetailExercises.innerHTML = "";

    try {
      const [planSnap, subSnap] = await Promise.all([
        getDoc(doc(db, "plans", planId)),
        getDoc(doc(db, "plans", planId, "subscribers", user.uid)),
      ]);

      if (!planSnap.exists()) {
        planDetailTitle.textContent = "Plan not found";
        return;
      }

      const plan = planSnap.data();
      planDetailTitle.textContent = plan.title;
      planDetailAuthor.textContent = "@" + (plan.authorUsername || "unknown");
      planDetailAuthor.dataset.uid = plan.authorId;
      planDetailInfo.textContent = ` · ${plan.level} · ${plan.daysPerWeek}x per week`;
      planDetailDescription.textContent = plan.description || "";

      // The author can't subscribe to their own plan
      const isOwn = plan.authorId === user.uid;
      planSubscribeBtn.style.display = isOwn ? "none" : "block";
      setSubscribeButton(subSnap.exists());
      planSubscribeBtn.dataset.planId = planId;

      renderPlanExercises(plan.exercises || []);
    } catch (error) {
      console.error(error.code, error.message);
      planDetailTitle.textContent = "Couldn't load this plan";
    }
  }

  // Groups exercises by day so the plan reads like an actual program
  // rather than one flat list.
  function renderPlanExercises(exercises) {
    if (exercises.length === 0) {
      planDetailExercises.innerHTML = '<p class="empty-state">No exercises listed.</p>';
      return;
    }

    const days = [...new Set(exercises.map((ex) => ex.day))].sort((a, b) => a - b);

    planDetailExercises.innerHTML = days
      .map((day) => {
        const rows = exercises
          .filter((ex) => ex.day === day)
          .map(
            (ex) => `
            <div class="plan-exercise">
              <span class="plan-exercise-name">${escapeHtml(ex.name)}</span>
              <span class="plan-exercise-detail">${ex.sets || "—"} × ${escapeHtml(ex.reps || "—")}</span>
            </div>`
          )
          .join("");
        return `<div class="plan-day"><div class="plan-day-label">Day ${day}</div>${rows}</div>`;
      })
      .join("");
  }

  function setSubscribeButton(isSubscribed) {
    planSubscribeBtn.dataset.subscribed = isSubscribed ? "true" : "false";
    planSubscribeBtn.textContent = isSubscribed ? "Subscribed" : "Subscribe";
    planSubscribeBtn.classList.toggle("following", isSubscribed);
  }

  // Subscribing writes both sides at once, same as following a user:
  // the plan's subscriber list and the user's own subscription list.
  planSubscribeBtn.addEventListener("click", async () => {
    const user = auth.currentUser;
    if (!user) return;

    const planId = planSubscribeBtn.dataset.planId;
    const isSubscribed = planSubscribeBtn.dataset.subscribed === "true";

    planSubscribeBtn.disabled = true;

    try {
      const batch = writeBatch(db);
      const subscriberRef = doc(db, "plans", planId, "subscribers", user.uid);
      const subscriptionRef = doc(db, "users", user.uid, "subscriptions", planId);

      if (isSubscribed) {
        batch.delete(subscriberRef);
        batch.delete(subscriptionRef);
      } else {
        batch.set(subscriberRef, { since: serverTimestamp() });
        batch.set(subscriptionRef, { since: serverTimestamp() });
      }

      await batch.commit();
      setSubscribeButton(!isSubscribed);
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't update subscription: " + error.message);
    } finally {
      planSubscribeBtn.disabled = false;
    }
  });

  // ---------- USER DETAIL (viewing someone else's profile) ----------
  async function openUserDetail(uid) {
    const user = auth.currentUser;
    if (!user) return;

    showDetail(userDetailView);
    userDetailUsername.textContent = "Loading...";
    userDetailPlans.innerHTML = "";
    userDetailPosts.innerHTML = "";

    try {
      const profileSnap = await getDoc(doc(db, "users", uid));
      if (!profileSnap.exists()) {
        userDetailUsername.textContent = "User not found";
        return;
      }

      const profile = profileSnap.data();
      userDetailUsername.textContent = "@" + profile.username;
      renderAvatar(userDetailAvatar, profile);

      const joined = profile.createdAt ? profile.createdAt.toDate().toLocaleDateString() : "recently";
      const goalText = profile.goal ? ` · ${profile.goal}` : "";
      userDetailMeta.textContent = `${profile.displayName || ""}${goalText} · joined ${joined}`.replace(/^ · /, "");
      userDetailBio.textContent = profile.bio || "";

      // You can't follow yourself
      const isMe = uid === user.uid;
      userDetailFollowBtn.style.display = isMe ? "none" : "block";
      userDetailFollowBtn.dataset.uid = uid;
      const isFollowing = followingIds.includes(uid);
      userDetailFollowBtn.dataset.following = isFollowing ? "true" : "false";
      userDetailFollowBtn.textContent = isFollowing ? "Following" : "Follow";
      userDetailFollowBtn.classList.toggle("following", isFollowing);

      // Their plans and posts, fetched in parallel
      const [plansSnap, postsSnap] = await Promise.all([
        getDocs(query(collection(db, "plans"), where("authorId", "==", uid))),
        getDocs(query(collection(db, "posts"), where("authorId", "==", uid))),
      ]);

      const byNewest = (a, b) => (b.createdAt?.toMillis() || 0) - (a.createdAt?.toMillis() || 0);

      const plans = plansSnap.docs.map((d) => ({ id: d.id, ...d.data() })).sort(byNewest);
      renderPlanCards(userDetailPlans, plans, "No plans published.");

      const posts = postsSnap.docs.map((d) => d.data()).sort(byNewest).slice(0, 20);
      renderUserPosts(posts);

      loadUserLocations(uid); // separate call so the map sizes after this view is visible
    } catch (error) {
      console.error(error.code, error.message);
      userDetailUsername.textContent = "Couldn't load this profile";
    }
  }

  // A read-only version of the feed post, without likes or comments.
  function renderUserPosts(posts) {
    if (posts.length === 0) {
      userDetailPosts.innerHTML = '<p class="empty-state">No posts yet.</p>';
      return;
    }

    userDetailPosts.innerHTML = posts
      .map((post) => {
        const summary =
          post.type === "cardio"
            ? `${escapeHtml(post.activity)} — ${post.distance} mi in ${formatDuration(post.durationSeconds)} (${formatDuration(paceSecondsPerMile(post.distance, post.durationSeconds))}/mi)`
            : `${escapeHtml(post.exerciseName || "Workout")} — ${post.sets} sets × ${post.reps} reps @ ${post.weight} lb`;
        const date = post.createdAt ? post.createdAt.toDate().toLocaleDateString() : "";
        return `
          <div class="post">
            ${post.caption ? `<div class="post-caption">${escapeHtml(post.caption)}</div>` : ""}
            <div class="post-workout">${summary}</div>
            <div class="post-date">${date}</div>
          </div>`;
      })
      .join("");
  }

  userDetailFollowBtn.addEventListener("click", async () => {
    const targetId = userDetailFollowBtn.dataset.uid;
    const isFollowing = userDetailFollowBtn.dataset.following === "true";

    userDetailFollowBtn.disabled = true;

    try {
      await toggleFollow(targetId, !isFollowing);
      userDetailFollowBtn.dataset.following = (!isFollowing).toString();
      userDetailFollowBtn.textContent = !isFollowing ? "Following" : "Follow";
      userDetailFollowBtn.classList.toggle("following", !isFollowing);
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't update follow: " + error.message);
    } finally {
      userDetailFollowBtn.disabled = false;
    }
  });

  // ---------- DETAIL VIEW NAVIGATION ----------
  // Detail views replace the tabbed content rather than opening a modal,
  // which keeps scrolling and the back button behaving predictably.
  function showDetail(view) {
    mainContent.style.display = "none";
    planDetailView.style.display = "none";
    userDetailView.style.display = "none";
    view.style.display = "block";
    window.scrollTo(0, 0);
  }

  function closeDetail() {
    planDetailView.style.display = "none";
    userDetailView.style.display = "none";
    mainContent.style.display = "block";
  }

  planBackBtn.addEventListener("click", closeDetail);
  userBackBtn.addEventListener("click", closeDetail);

  // Any plan card anywhere in the app opens that plan.
  document.addEventListener("click", (event) => {
    const planCard = event.target.closest(".plan-card");
    if (planCard) {
      openPlanDetail(planCard.dataset.planId);
      return;
    }

    // Any element marked as a user link opens that person's profile.
    const userLink = event.target.closest(".user-link");
    if (userLink && userLink.dataset.uid) {
      event.preventDefault();
      openUserDetail(userLink.dataset.uid);
    }
  });

  // ---------- TRAINING LOCATIONS ----------
  // Maps are created lazily and reused. Leaflet measures its container when
  // the map is created, so a map built inside a hidden element renders at
  // zero size — invalidateSize() forces it to re-measure once visible.
  let profileMap = null;
  let profileLayer = null;
  let userDetailMap = null;
  let userDetailLayer = null;
  let pendingLocation = null; // coordinates chosen but not yet saved

  // Each activity gets its own pin colour so a profile map reads at a glance.
  const ACTIVITY_COLORS = {
    Strength: "#12382A",
    Cardio: "#7D6127",
    Pilates: "#8A4B7D",
    Yoga: "#2E7D52",
    Swimming: "#2A6B8F",
    Climbing: "#B23A2E",
    Other: "#5A6B62",
  };

  function createMap(containerId, onMapClick) {
    // Leaflet loads from a CDN, so it may be unavailable offline or on a
    // restricted network. Failing loudly here would break the whole tab, so
    // the map degrades to a message and the location list still works.
    if (typeof L === "undefined") {
      const container = document.getElementById(containerId);
      if (container) {
        container.innerHTML = '<p class="empty-state map-fallback">Map unavailable — check your connection.</p>';
      }
      return null;
    }

    // Default view covers the DC / Maryland area; overridden by fitBounds
    // whenever the user actually has pins to show.
    const map = L.map(containerId).setView([38.92, -77.02], 11);

    // OpenStreetMap tiles are free to use but their tile servers require
    // attribution, which the control below provides.
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(map);

    if (onMapClick) map.on("click", onMapClick);
    return map;
  }

  // Circle markers are used rather than Leaflet's default pin images, which
  // depend on icon files that are easy to mis-path. Circles need no assets
  // and can be coloured directly.
  function drawLocationPins(map, layerRef, locations) {
    if (!map) return null; // map failed to load; the list below still renders
    if (layerRef) map.removeLayer(layerRef);

    const layer = L.layerGroup();
    const coords = [];

    locations.forEach((loc) => {
      if (typeof loc.lat !== "number" || typeof loc.lng !== "number") return;

      L.circleMarker([loc.lat, loc.lng], {
        radius: 9,
        color: "#FDFBF6",
        weight: 2,
        fillColor: ACTIVITY_COLORS[loc.activity] || ACTIVITY_COLORS.Other,
        fillOpacity: 0.95,
      })
        .bindPopup(`<strong>${escapeHtml(loc.name)}</strong><br>${escapeHtml(loc.activity)}${loc.when ? " · " + escapeHtml(loc.when) : ""}`)
        .addTo(layer);

      coords.push([loc.lat, loc.lng]);
    });

    layer.addTo(map);

    // Zoom to fit all pins, with padding so markers aren't flush to the edge
    if (coords.length === 1) {
      map.setView(coords[0], 14);
    } else if (coords.length > 1) {
      map.fitBounds(coords, { padding: [40, 40] });
    }

    return layer;
  }

  // Geocoding uses Nominatim, OpenStreetMap's free lookup service. Its usage
  // policy allows roughly one request per second and asks that it not be used
  // for heavy automated traffic — acceptable for manual searches at this
  // scale, but it is a shared community service, not a commercial API.
  async function geocode(queryText) {
    const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(queryText)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Lookup service unavailable");
    const results = await response.json();
    return results.length ? { lat: Number(results[0].lat), lng: Number(results[0].lon), label: results[0].display_name } : null;
  }

  function setPendingLocation(lat, lng, label) {
    pendingLocation = { lat, lng };
    locationPicked.textContent = `Selected: ${label}`;

    if (profileMap) {
      // A single temporary marker showing where the pin will land
      if (profileMap._pendingMarker) profileMap.removeLayer(profileMap._pendingMarker);
      profileMap._pendingMarker = L.circleMarker([lat, lng], {
        radius: 10,
        color: "#7D6127",
        weight: 3,
        fillColor: "#D4AF6A",
        fillOpacity: 0.6,
      }).addTo(profileMap);
      profileMap.setView([lat, lng], 15);
    }
  }

  function handleMapClick(event) {
    const { lat, lng } = event.latlng;
    setPendingLocation(lat, lng, `${lat.toFixed(4)}, ${lng.toFixed(4)}`);
  }

  locationSearchBtn.addEventListener("click", async () => {
    const term = locationSearchInput.value.trim();
    if (!term) return;

    locationSearchBtn.disabled = true;
    locationSearchBtn.textContent = "...";
    locationPicked.textContent = "Searching...";

    try {
      const result = await geocode(term);
      if (!result) {
        locationPicked.textContent = "Nothing found — try a fuller address, or click the map instead.";
        return;
      }
      setPendingLocation(result.lat, result.lng, result.label);
    } catch (error) {
      console.error(error);
      locationPicked.textContent = "Lookup failed — click the map to place a pin instead.";
    } finally {
      locationSearchBtn.disabled = false;
      locationSearchBtn.textContent = "Search";
    }
  });

  toggleLocationFormBtn.addEventListener("click", () => {
    const isOpen = locationForm.style.display !== "none";
    locationForm.style.display = isOpen ? "none" : "block";
    toggleLocationFormBtn.textContent = isOpen ? "Add Place" : "Cancel";
  });

  locationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage(locationMessage, "", "");

    const user = auth.currentUser;
    if (!user) return;

    if (!pendingLocation) {
      showMessage(locationMessage, "Pick a spot first — search for it or click the map.", "error");
      return;
    }

    locationSubmitButton.disabled = true;
    locationSubmitButton.textContent = "Saving...";

    try {
      await addDoc(collection(db, "users", user.uid, "locations"), {
        name: locationNameInput.value.trim(),
        activity: locationActivityInput.value,
        when: locationWhenInput.value.trim(),
        lat: pendingLocation.lat,
        lng: pendingLocation.lng,
        createdAt: serverTimestamp(),
      });

      locationForm.reset();
      pendingLocation = null;
      locationPicked.textContent = "No location selected yet.";
      if (profileMap && profileMap._pendingMarker) {
        profileMap.removeLayer(profileMap._pendingMarker);
        profileMap._pendingMarker = null;
      }
      locationForm.style.display = "none";
      toggleLocationFormBtn.textContent = "Add Place";
      showMessage(locationMessage, "Place saved.", "success");
      loadMyLocations();
    } catch (error) {
      console.error(error.code, error.message);
      showMessage(locationMessage, "Couldn't save that place: " + error.message, "error");
    } finally {
      locationSubmitButton.disabled = false;
      locationSubmitButton.textContent = "Save Place";
    }
  });

  // Loads the current user's own locations into the Profile tab.
  async function loadMyLocations() {
    const user = auth.currentUser;
    if (!user) return;

    // The map has to exist and be visible before Leaflet can size it
    if (!profileMap) profileMap = createMap("profile-map", handleMapClick);
    if (profileMap) profileMap.invalidateSize();

    try {
      const snapshot = await getDocs(collection(db, "users", user.uid, "locations"));
      const locations = snapshot.docs.map((d) => ({ id: d.id, ...d.data() }));
      myLocationsCache = locations; // reused by partner matching

      profileLayer = drawLocationPins(profileMap, profileLayer, locations);

      if (locations.length === 0) {
        myLocationsList.innerHTML = '<p class="empty-state">You haven\'t added any training places yet.</p>';
        return;
      }

      // Interest counts, fetched per location
      const counts = await Promise.all(
        locations.map((loc) => getDocs(collection(db, "users", user.uid, "locations", loc.id, "interested")))
      );

      myLocationsList.innerHTML = locations
        .map((loc, i) => `
          <div class="location-row">
            <div class="location-info">
              <span class="location-dot" style="background:${ACTIVITY_COLORS[loc.activity] || ACTIVITY_COLORS.Other}"></span>
              <div>
                <div class="location-name">${escapeHtml(loc.name)}</div>
                <div class="location-meta">${escapeHtml(loc.activity)}${loc.when ? " · " + escapeHtml(loc.when) : ""} · ${counts[i].size} interested</div>
              </div>
            </div>
            <button class="delete-btn location-delete" data-loc-id="${loc.id}" title="Remove this place">✕</button>
          </div>`)
        .join("");
    } catch (error) {
      console.error(error.code, error.message);
      myLocationsList.innerHTML = '<p class="empty-state">Couldn\'t load your places.</p>';
    }
  }

  myLocationsList.addEventListener("click", async (event) => {
    const button = event.target.closest(".location-delete");
    if (!button) return;

    const user = auth.currentUser;
    if (!user) return;
    if (!confirm("Remove this training place?")) return;

    try {
      await deleteDoc(doc(db, "users", user.uid, "locations", button.dataset.locId));
      loadMyLocations();
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't remove that place: " + error.message);
    }
  });

  // Loads another user's locations into their public profile view.
  async function loadUserLocations(uid) {
    const user = auth.currentUser;
    if (!user) return;

    if (!userDetailMap) userDetailMap = createMap("user-detail-map", null);
    if (userDetailMap) userDetailMap.invalidateSize();

    try {
      const snapshot = await getDocs(collection(db, "users", uid, "locations"));
      const locations = snapshot.docs.map((d) => ({ id: d.id, ...d.data() }));

      userDetailLayer = drawLocationPins(userDetailMap, userDetailLayer, locations);

      if (locations.length === 0) {
        userDetailLocations.innerHTML = '<p class="empty-state">No training places listed.</p>';
        return;
      }

      // For each place, count interest and check whether this viewer is in it
      const interestSnaps = await Promise.all(
        locations.map((loc) => getDocs(collection(db, "users", uid, "locations", loc.id, "interested")))
      );

      userDetailLocations.innerHTML = locations
        .map((loc, i) => {
          const interestedIds = interestSnaps[i].docs.map((d) => d.id);
          const isInterested = interestedIds.includes(user.uid);
          return `
            <div class="location-row">
              <div class="location-info">
                <span class="location-dot" style="background:${ACTIVITY_COLORS[loc.activity] || ACTIVITY_COLORS.Other}"></span>
                <div>
                  <div class="location-name">${escapeHtml(loc.name)}</div>
                  <div class="location-meta">${escapeHtml(loc.activity)}${loc.when ? " · " + escapeHtml(loc.when) : ""} · ${interestedIds.length} interested</div>
                </div>
              </div>
              <button class="follow-btn interest-btn ${isInterested ? "following" : ""}"
                      data-owner="${uid}" data-loc-id="${loc.id}" data-interested="${isInterested}">
                ${isInterested ? "Interested" : "I'm in"}
              </button>
            </div>`;
        })
        .join("");
    } catch (error) {
      console.error(error.code, error.message);
      userDetailLocations.innerHTML = '<p class="empty-state">Couldn\'t load their places.</p>';
    }
  }

  // Registering interest writes a single document keyed by the viewer's UID,
  // so the same person can never register interest twice.
  userDetailLocations.addEventListener("click", async (event) => {
    const button = event.target.closest(".interest-btn");
    if (!button) return;

    const user = auth.currentUser;
    if (!user) return;

    const ownerId = button.dataset.owner;
    const locId = button.dataset.locId;
    const isInterested = button.dataset.interested === "true";
    const ref = doc(db, "users", ownerId, "locations", locId, "interested", user.uid);

    button.disabled = true;

    try {
      if (isInterested) {
        await deleteDoc(ref);
      } else {
        await setDoc(ref, {
          username: currentProfile ? currentProfile.username : "unknown",
          since: serverTimestamp(),
        });
      }
      loadUserLocations(ownerId); // refresh so the count stays accurate
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't update interest: " + error.message);
    } finally {
      button.disabled = false;
    }
  });

  // ---------- INSIGHTS ----------
  // All four panels are computed from cachedWorkouts, which the existing
  // workout listener already holds — so opening this tab costs no extra reads.
  function renderInsights() {
    const entries = normaliseEntries(cachedWorkouts);

    if (entries.length === 0) {
      acwrPanel.innerHTML = '<p class="empty-state">Log some sessions to see your training load.</p>';
      trendsPanel.innerHTML = '<p class="empty-state">Log an exercise at least four times to see its trend.</p>';
      volumeChart.innerHTML = '<p class="empty-state">No volume data yet.</p>';
      balancePanel.innerHTML = '<p class="empty-state">Log strength work to see your split.</p>';
      return;
    }

    renderACWR(entries);
    renderTrends(entries);
    renderVolumeChart(entries);
    renderBalance(entries);
  }

  function renderACWR(entries) {
    const acwr = computeACWR(entries);

    if (acwr.status === "baseline") {
      acwrPanel.innerHTML = `
        <div class="acwr-headline">
          <span class="acwr-value muted">—</span>
          <span class="risk-badge risk-baseline">Building baseline</span>
        </div>
        <p class="insight-note">${escapeHtml(acwr.message)}</p>
        <p class="insight-note">${acwr.historyDays} days of history so far.</p>`;
      return;
    }

    // Position on a 0–2 scale, since the meaningful range tops out around 2.0
    const pct = Math.min(100, (acwr.ratio / 2) * 100);

    acwrPanel.innerHTML = `
      <div class="acwr-headline">
        <span class="acwr-value">${acwr.ratio}</span>
        <span class="risk-badge risk-${acwr.status}">${acwr.status}</span>
      </div>
      <div class="acwr-track">
        <div class="acwr-zone zone-low"></div>
        <div class="acwr-zone zone-optimal"></div>
        <div class="acwr-zone zone-elevated"></div>
        <div class="acwr-zone zone-high"></div>
        <div class="acwr-marker" style="left:${pct}%"></div>
      </div>
      <div class="acwr-scale">
        <span>0.8</span><span>1.3</span><span>1.5</span>
      </div>
      <p class="insight-note">${escapeHtml(acwr.message)}</p>
      <div class="insight-figures">
        <div><span class="fig-value">${acwr.acute.toLocaleString()}</span><span class="fig-label">Last 7 days</span></div>
        <div><span class="fig-value">${acwr.chronicWeekly.toLocaleString()}</span><span class="fig-label">Weekly baseline</span></div>
      </div>`;
  }

  function renderTrends(entries) {
    const strength = entries.filter((e) => e.type !== "cardio" && e.weight > 0);
    const names = [...new Set(strength.map((e) => e.exerciseName))];

    const results = names
      .map((name) => ({ name, plateau: detectPlateau(strength.filter((e) => e.exerciseName === name)) }))
      .filter((r) => r.plateau.status !== "insufficient")
      // Plateaus and regressions first — those are the ones needing attention.
      .sort((a, b) => {
        const rank = { regressing: 0, plateau: 1, progressing: 2 };
        return rank[a.plateau.status] - rank[b.plateau.status];
      });

    if (results.length === 0) {
      const best = strength
        .map((e) => ({ name: e.exerciseName, est: estimate1RM(e.weight, e.reps) }))
        .filter((r) => r.est);

      if (best.length === 0) {
        trendsPanel.innerHTML = '<p class="empty-state">Log an exercise at least four times to see its trend.</p>';
        return;
      }

      // Not enough history for a trend, but a current estimate is still useful.
      const byName = {};
      for (const b of best) {
        if (!byName[b.name] || b.est.estimate > byName[b.name].estimate) byName[b.name] = b.est;
      }

      trendsPanel.innerHTML =
        Object.entries(byName)
          .map(([name, est]) => `
            <div class="trend-row">
              <div>
                <div class="trend-name">${escapeHtml(name)}</div>
                <div class="trend-detail">Estimated 1RM ${est.estimate} lb · ${est.low}–${est.high} lb · ${est.confidence} confidence</div>
              </div>
              <span class="risk-badge risk-baseline">Need 4+ sessions</span>
            </div>`)
          .join("") +
        '<p class="insight-note">Log each lift at least four times for plateau detection.</p>';
      return;
    }

    trendsPanel.innerHTML = results
      .map((r) => {
        const p = r.plateau;
        const sign = p.slopePerWeek > 0 ? "+" : "";
        const label = { progressing: "Progressing", plateau: "Plateau", regressing: "Regressing" }[p.status];
        return `
          <div class="trend-row">
            <div>
              <div class="trend-name">${escapeHtml(r.name)}</div>
              <div class="trend-detail">
                Est. 1RM ${p.current1RM} lb · ${sign}${p.slopePerWeek} lb/week
                · ${p.sessions} sessions over ${p.spanDays} days
              </div>
              <div class="trend-stats">r² ${p.r2} · p ${p.pValue}${p.significant ? "" : " (not significant)"}</div>
            </div>
            <span class="risk-badge risk-${p.status}">${label}</span>
          </div>`;
      })
      .join("");
  }

  function renderVolumeChart(entries) {
    const series = weeklyVolumeSeries(entries, 12).filter((w) => w.volume > 0);

    if (series.length < 2) {
      volumeChart.innerHTML = '<p class="empty-state">Not enough weeks logged yet.</p>';
      return;
    }

    const points = series.map((w) => ({
      value: w.volume,
      label: new Date(w.weekEnding).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    }));

    volumeChart.innerHTML = buildChartSVG(points, {
      baseline: "zero",
      formatValue: (v) => (v >= 1000 ? Math.round(v / 1000) + "k" : Math.round(v)),
    });
  }

  function renderBalance(entries) {
    const bal = movementBalance(entries);

    if (bal.status === "insufficient") {
      balancePanel.innerHTML = '<p class="empty-state">Log strength work to see your split.</p>';
      return;
    }

    const order = ["push", "pull", "legs", "core", "other"];
    const bars = order
      .filter((k) => bal.sharePercent[k] > 0)
      .map((k) => `
        <div class="balance-row">
          <span class="balance-label">${k}</span>
          <div class="balance-track"><div class="balance-fill fill-${k}" style="width:${bal.sharePercent[k]}%"></div></div>
          <span class="balance-pct">${bal.sharePercent[k]}%</span>
        </div>`)
      .join("");

    const ratioText = bal.pushPullRatio === null ? "—" : bal.pushPullRatio.toFixed(2);

    balancePanel.innerHTML = `
      ${bars}
      <div class="insight-figures">
        <div><span class="fig-value">${ratioText}</span><span class="fig-label">Push : pull ratio</span></div>
        <div><span class="risk-badge risk-${bal.status === "balanced" ? "optimal" : "elevated"}">${bal.status}</span></div>
      </div>
      <p class="insight-note">${escapeHtml(bal.message)}</p>`;
  }

  // ---------- PARTNER MATCHING ----------
  // Raw workouts stay private under the security rules, so matching cannot read
  // other users' training data directly. Instead each user publishes a small
  // DERIVED summary — training split proportions and an overall strength level
  // — onto their own public profile document. Individual sessions are never
  // exposed; only the aggregate needed for comparison.
  async function publishMatchProfile() {
    const user = auth.currentUser;
    if (!user || cachedWorkouts.length === 0) return;

    const entries = normaliseEntries(cachedWorkouts);
    const vector = buildProfileVector(entries);
    const level = strengthLevel(entries);
    if (!vector && !level) return;

    try {
      await updateDoc(doc(db, "users", user.uid), {
        matchProfile: {
          vector: vector,
          strengthLevel: level,
          sessionCount: entries.length,
          updatedAt: Date.now(),
        },
      });
    } catch (error) {
      // Non-critical: matching degrades rather than breaking the app.
      console.warn("Couldn't publish match profile:", error.message);
    }
  }

  async function loadPartnerMatches() {
    const user = auth.currentUser;
    if (!user) return;

    const entries = normaliseEntries(cachedWorkouts);
    const myVector = buildProfileVector(entries);
    const myLevel = strengthLevel(entries);

    matchesList.innerHTML = '<p class="empty-state">Finding partners...</p>';

    try {
      // Candidate pool is capped. This is the naive approach: fetch users and
      // score client-side. It is appropriate at this scale and would need a
      // server-side pre-filter (for example by geographic cell) to grow.
      const usersSnap = await getDocs(query(collection(db, "users"), limit(40)));

      const candidates = await Promise.all(
        usersSnap.docs
          .filter((d) => d.id !== user.uid)
          .map(async (d) => {
            const data = d.data();
            const locsSnap = await getDocs(collection(db, "users", d.id, "locations"));
            const locations = locsSnap.docs.map((l) => l.data());
            const mp = data.matchProfile || {};
            return {
              uid: d.id,
              username: data.username || "unknown",
              displayName: data.displayName || "",
              photoURL: data.photoURL || null,
              profileVector: mp.vector || null,
              strengthLevel: mp.strengthLevel || null,
              locations,
              activities: activitySet(locations, []),
              sessionCount: mp.sessionCount || 0,
            };
          })
      );

      const me = {
        uid: user.uid,
        profileVector: myVector,
        strengthLevel: myLevel,
        locations: myLocationsCache,
        activities: activitySet(myLocationsCache, entries),
      };

      const ranked = rankPartners(me, candidates, { limit: 6 });

      if (ranked.length === 0) {
        matchesList.innerHTML = '<p class="empty-state">No suggestions yet. Suggestions improve as you log workouts and add training places — and as more people join.</p>';
        return;
      }

      matchesList.innerHTML = ranked
        .map((r) => {
          const c = r.candidate;
          const pct = Math.round(r.score * 100);
          const reasons = r.reasons.length
            ? r.reasons.map((t) => `<span class="reason-chip">${escapeHtml(t)}</span>`).join("")
            : '<span class="reason-chip">Limited data so far</span>';

          return `
            <div class="match-row">
              <div class="match-main user-link" data-uid="${c.uid}">
                <div class="person-avatar match-avatar"></div>
                <div>
                  <div class="person-username">@${escapeHtml(c.username)}</div>
                  <div class="match-reasons">${reasons}</div>
                </div>
              </div>
              <div class="match-score">
                <span class="match-pct">${pct}%</span>
                <span class="match-coverage">${Math.round(r.coverage * 100)}% signal</span>
              </div>
            </div>`;
        })
        .join("");

      // Avatars are rendered after insertion so the shared helper can be reused
      ranked.forEach((r, i) => {
        const el = matchesList.querySelectorAll(".match-avatar")[i];
        if (el) renderAvatar(el, r.candidate);
      });
    } catch (error) {
      console.error(error.code, error.message);
      matchesList.innerHTML = '<p class="empty-state">Couldn\'t load suggestions.</p>';
    }
  }

  refreshMatchesBtn.addEventListener("click", loadPartnerMatches);

  // ---------- MACHINE LEARNING ----------
  // Raw workouts are private under the security rules, so the model cannot be
  // trained on other users' sessions directly. Instead each user publishes an
  // anonymised aggregate — which exercises they log and at what relative
  // volume — to their own public profile. No dates, loads, or individual
  // sessions are exposed; only the interaction pattern the model needs.
  async function publishExerciseProfile() {
    const user = auth.currentUser;
    if (!user || cachedWorkouts.length === 0) return;

    const entries = normaliseEntries(cachedWorkouts);
    const matrix = buildInteractionMatrix([{ userId: user.uid, workouts: entries }]);
    if (matrix.interactions.length === 0) return;

    const exercises = {};
    for (const x of matrix.interactions) {
      exercises[matrix.itemIds[x.item]] = Math.round(x.rating * 1000) / 1000;
    }

    const featureVector = buildFeatureVector(entries, classifyExercise, lookupExercise);

    try {
      await updateDoc(doc(db, "users", user.uid), {
        exerciseProfile: { exercises, featureVector, updatedAt: Date.now() },
      });
    } catch (error) {
      console.warn("Couldn't publish exercise profile:", error.message);
    }
  }

  async function trainAndRecommend() {
    const user = auth.currentUser;
    if (!user) return;

    trainModelBtn.disabled = true;
    trainModelBtn.textContent = "Training...";
    mlPanel.innerHTML = '<p class="empty-state">Gathering profiles...</p>';

    try {
      const usersSnap = await getDocs(query(collection(db, "users"), limit(100)));

      // Reconstruct interactions from the published aggregates.
      const population = [];
      for (const d of usersSnap.docs) {
        const p = d.data().exerciseProfile;
        if (!p || !p.exercises) continue;
        population.push({ id: d.id, exercises: p.exercises, featureVector: p.featureVector || null });
      }

      if (population.length < 3) {
        mlPanel.innerHTML = `<p class="empty-state">Only ${population.length} profile(s) available. Collaborative filtering needs several users before it can find patterns — this is the cold-start problem, and it is expected on a new deployment.</p>`;
        archetypePanel.innerHTML = '<p class="empty-state">Not enough profiles to discover archetypes yet.</p>';
        return;
      }

      // Build index maps and the interaction list
      const userIds = population.map((p) => p.id);
      const itemIds = [...new Set(population.flatMap((p) => Object.keys(p.exercises)))];
      const itemIndex = new Map(itemIds.map((id, i) => [id, i]));

      const interactions = [];
      population.forEach((p, u) => {
        for (const [exId, rating] of Object.entries(p.exercises)) {
          interactions.push({ user: u, item: itemIndex.get(exId), rating });
        }
      });

      mlPanel.innerHTML = '<p class="empty-state">Training model...</p>';

      // Yield to the browser so the loading text paints before the training
      // loop blocks the main thread.
      await new Promise((r) => setTimeout(r, 30));

      const model = trainMatrixFactorization(interactions, {
        factors: 8, epochs: 120, nItems: itemIds.length, seed: 42,
      });

      const myIndex = userIds.indexOf(user.uid);
      if (myIndex < 0 || !model) {
        mlPanel.innerHTML = '<p class="empty-state">Your profile is not in the training set yet. Log a workout and try again.</p>';
        return;
      }

      const known = interactions.filter((x) => x.user === myIndex).map((x) => x.item);
      const recs = recommend(model, myIndex, known, 5);

      // Held-out evaluation, so the recommendations come with a measured
      // quality figure rather than being presented as authoritative.
      let evaluation = null;
      try {
        evaluation = evaluate(interactions, { factors: 8, epochs: 120, k: 5, nItems: itemIds.length });
      } catch (e) {
        console.warn("Evaluation skipped:", e.message);
      }

      if (recs.length === 0) {
        mlPanel.innerHTML = '<p class="empty-state">No new exercises to suggest — you already log everything in the training set.</p>';
      } else {
        const evalRow = evaluation ? `
          <div class="model-metrics">
            <div><span class="fig-value">${(evaluation.results.model.precision * 100).toFixed(1)}%</span><span class="fig-label">Precision@5</span></div>
            <div><span class="fig-value">${(evaluation.results.popularity.precision * 100).toFixed(1)}%</span><span class="fig-label">Popularity baseline</span></div>
            <div><span class="fig-value">${evaluation.results.liftOverPopularity !== null ? (evaluation.results.liftOverPopularity > 0 ? "+" : "") + (evaluation.results.liftOverPopularity * 100).toFixed(0) + "%" : "—"}</span><span class="fig-label">Lift over baseline</span></div>
          </div>
          <p class="insight-note">Measured on held-out interactions from ${evaluation.results.users} profiles.</p>` : "";

        mlPanel.innerHTML = `
          <div class="rec-list">
            ${recs.map((r, i) => `
              <div class="rec-row">
                <span class="rec-rank">${i + 1}</span>
                <span class="rec-name">${escapeHtml(exerciseName(itemIds[r.item]))}</span>
                <div class="rec-bar"><div class="rec-fill" style="width:${Math.round(r.score * 100)}%"></div></div>
                <span class="rec-score">${r.score.toFixed(2)}</span>
              </div>`).join("")}
          </div>
          ${evalRow}`;
      }

      // --- archetypes ---
      const members = population
        .filter((p) => p.featureVector)
        .map((p) => ({ id: p.id, vector: p.featureVector }));

      const archetypes = discoverArchetypes(members, { seed: 42, targetId: user.uid });

      if (archetypes.status !== "ok") {
        archetypePanel.innerHTML = `<p class="empty-state">${escapeHtml(archetypes.message)}</p>`;
      } else {
        const own = archetypes.own;
        archetypePanel.innerHTML = `
          ${own ? `
            <div class="archetype-head">
              <span class="archetype-label">${escapeHtml(own.label)}</span>
              <span class="archetype-size">${own.size} of ${archetypes.members} lifters</span>
            </div>
            <ul class="archetype-traits">
              ${own.traits.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}
            </ul>` : '<p class="insight-note">Log more sessions to be placed in an archetype.</p>'}

          <div class="cluster-summary">
            ${archetypes.clusters.map((c) => `
              <div class="cluster-chip${own && c.index === own.index ? " current" : ""}">
                <span class="cluster-name">${escapeHtml(c.label)}</span>
                <span class="cluster-count">${c.size}</span>
              </div>`).join("")}
          </div>
          <p class="insight-note">${archetypes.k} archetypes found automatically, silhouette ${archetypes.silhouette}. The number of groups was selected by cluster quality, not chosen in advance.</p>`;
      }
    } catch (error) {
      console.error(error.code, error.message);
      mlPanel.innerHTML = '<p class="empty-state">Couldn\'t train the model.</p>';
    } finally {
      trainModelBtn.disabled = false;
      trainModelBtn.textContent = "Train Model";
    }
  }

  trainModelBtn.addEventListener("click", trainAndRecommend);

  // ---------- RANK AND ACHIEVEMENTS ----------
  function renderRankAndAchievements() {
    const entries = normaliseEntries(cachedWorkouts);
    const rank = computeRank(entries, currentProfile || {}, estimate1RM);

    // --- rank card ---
    if (rank.status === "needs-bodyweight" || rank.status === "needs-lifts") {
      rankPanel.innerHTML = `<p class="empty-state">${escapeHtml(rank.message)}</p>`;
    } else {
      const t = rank.total;
      rankPanel.innerHTML = `
        <div class="rank-head">
          <div class="rank-badge" style="--tier-accent:${rank.accent}">
            <span class="rank-tier">${rank.label}</span>
            <span class="rank-div">${rank.division}</span>
          </div>
          <div class="rank-meta">
            <div class="rank-score">${rank.score} <span class="rank-unit">DOTS</span></div>
            <div class="rank-category">${escapeHtml(rank.categoryLabel)} · ${rank.bodyweightLb} lb bodyweight</div>
          </div>
        </div>

        ${rank.nextTier ? `
          <div class="rank-progress-track">
            <div class="rank-progress-fill" style="width:${rank.progressPercent}%;background:${rank.accent}"></div>
          </div>
          <p class="insight-note">${rank.pointsToNext} points to ${escapeHtml(rank.nextTier)}.</p>
        ` : '<p class="insight-note">Top tier reached.</p>'}

        ${rank.message ? `<p class="insight-note provisional">${escapeHtml(rank.message)}</p>` : ""}

        <div class="lift-breakdown">
          ${["squat", "bench", "deadlift"].map((k) => `
            <div class="lift-cell${t.bests[k] ? "" : " lift-missing"}">
              <span class="lift-value">${t.bests[k] || "—"}</span>
              <span class="lift-label">${k}</span>
            </div>`).join("")}
          <div class="lift-cell lift-total">
            <span class="lift-value">${t.totalLb}</span>
            <span class="lift-label">total</span>
          </div>
        </div>`;
    }

    // --- achievements ---
    const balance = movementBalance(entries);
    const acwr = computeACWR(entries);
    const groups = muscleGroupVolume(entries);

    const stats = buildAchievementStats(entries, {
      bestBench: rank.total?.bests?.bench || 0,
      bestSquat: rank.total?.bests?.squat || 0,
      bestDeadlift: rank.total?.bests?.deadlift || 0,
      bodyweight: currentProfile?.bodyweightLb || null,
      pushPullRatio: balance.pushPullRatio,
      groupsTrained: Object.keys(groups.totals).length,
      acwrOptimal: acwr.status === "optimal",
      posts: Number(statPosts.textContent) || 0,
      followers: Number(statFollowers.textContent) || 0,
      plansPublished: myPlansCount,
      planSubscribers: 0,
    });

    const achievements = evaluateAchievements(stats);
    const unlocked = achievements.filter((a) => a.unlocked).length;
    achievementCount.textContent = `${unlocked} of ${achievements.length} unlocked`;

    // Unlocked first, then whatever the user is closest to earning — which is
    // more useful than a fixed order, since it surfaces the next realistic goal.
    const ordered = [...achievements].sort((a, b) => {
      if (a.unlocked !== b.unlocked) return a.unlocked ? -1 : 1;
      return b.progressPercent - a.progressPercent;
    });

    achievementsPanel.innerHTML = `
      <div class="achievement-grid">
        ${ordered.map((a) => `
          <div class="achievement ${a.unlocked ? "unlocked" : "locked"}" title="${escapeHtml(a.description)}">
            <div class="achievement-name">${escapeHtml(a.name)}</div>
            <div class="achievement-desc">${escapeHtml(a.description)}</div>
            ${a.unlocked
              ? '<div class="achievement-status">Unlocked</div>'
              : `<div class="achievement-track"><div class="achievement-fill" style="width:${a.progressPercent}%"></div></div>
                 <div class="achievement-status">${a.progressPercent}%</div>`}
          </div>`).join("")}
      </div>`;

    // Streak summary, shown alongside the achievements it feeds
    const streaks = computeStreaks(entries);
    if (streaks.totalDays > 0) {
      achievementsPanel.insertAdjacentHTML("afterbegin", `
        <div class="insight-figures" style="border-top:none;padding-top:0;margin-bottom:18px">
          <div><span class="fig-value">${streaks.current}</span><span class="fig-label">Current streak</span></div>
          <div><span class="fig-value">${streaks.longest}</span><span class="fig-label">Longest streak</span></div>
          <div><span class="fig-value">${streaks.totalDays}</span><span class="fig-label">Days trained</span></div>
        </div>`);
    }
  }

  // ---------- PROFILE ----------
  // Turns a username into a consistent color, so each user's letter avatar
  // looks distinct but never changes between sessions or devices.
  // Works by summing the character codes into a rough hash, then mapping
  // that number onto the 0-359 hue circle. Same input always gives the same
  // output, which is the whole point — no color needs to be stored anywhere.
  function avatarColor(username) {
    let hash = 0;
    for (let i = 0; i < username.length; i++) {
      hash = username.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    // Deep, saturated jewel tones: dark enough for the cream lettering to
    // stay readable, and consistent in weight with the dark green palette.
    return `hsl(${hue}, 38%, 32%)`;
  }

  // Renders an avatar into a given element: the uploaded photo if there is
  // one, otherwise the first letter of the username on a generated color.
  // Photo support is kept so the feature works immediately if Cloud Storage
  // is enabled later. Reused for the profile header and the People list.
  function renderAvatar(element, profile) {
    if (profile.photoURL) {
      element.innerHTML = `<img src="${escapeHtml(profile.photoURL)}" alt="" />`;
      element.classList.add("has-photo");
      element.style.backgroundColor = "";
    } else {
      const name = profile.username || "?";
      element.textContent = name.charAt(0).toUpperCase();
      element.classList.remove("has-photo");
      element.style.backgroundColor = avatarColor(name);
    }
  }

  // Fills the Profile tab from the cached currentProfile object.
  function renderProfile() {
    if (!currentProfile) return;

    profileUsernameEl.textContent = "@" + currentProfile.username;
    renderAvatar(profileAvatar, currentProfile);

    const joined = currentProfile.createdAt
      ? currentProfile.createdAt.toDate().toLocaleDateString()
      : "recently";
    profileMetaEl.textContent = `${currentProfile.email} · joined ${joined}`;

    // Pre-fill the edit fields with existing values (|| "" guards against
    // undefined on profiles saved before a field was added).
    profileDisplayNameInput.value = currentProfile.displayName || "";
    profileBioInput.value = currentProfile.bio || "";
    profileGoalInput.value = currentProfile.goal || "";
    profileBodyweightInput.value = currentProfile.bodyweightLb || "";
    profileCategoryInput.value = currentProfile.category || "open";
  }

  // Saves edits to the profile. Only updates the editable fields —
  // username, email, and createdAt are deliberately left untouched.
  profileForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    showMessage(profileMessage, "", "");

    const user = auth.currentUser;
    if (!user) return;

    profileSubmitButton.disabled = true;
    profileSubmitButton.textContent = "Saving...";

    try {
      const updates = {
        displayName: profileDisplayNameInput.value.trim(),
        bio: profileBioInput.value.trim(),
        goal: profileGoalInput.value,
        bodyweightLb: Number(profileBodyweightInput.value) || null,
        category: profileCategoryInput.value || "open",
      };

      // PHOTO UPLOAD — disabled (requires Firebase Blaze plan).
      // If the user picked a file, upload it first and add the resulting
      // URL to the updates. Checked client-side for fast feedback; Storage
      // rules enforce the same limits server-side regardless.
      // const file = profilePhotoInput.files[0];
      // if (file) {
      //   if (!file.type.startsWith("image/")) {
      //     showMessage(profileMessage, "That file isn't an image.", "error");
      //     return;
      //   }
      //   if (file.size > 5 * 1024 * 1024) {
      //     showMessage(profileMessage, "That image is larger than 5MB.", "error");
      //     return;
      //   }
      //
      //   profileSubmitButton.textContent = "Uploading photo...";
      //   updates.photoURL = await uploadProfilePhoto(user.uid, file);
      // }

      // updateDoc changes only the fields you pass, leaving the rest alone.
      // (setDoc without merge would overwrite the whole document.)
      await updateDoc(doc(db, "users", user.uid), updates);

      currentProfile = { ...currentProfile, ...updates }; // update the local cache too
      renderProfile();
      renderRankAndAchievements(); // bodyweight or category may have changed
      showMessage(profileMessage, "Profile saved.", "success");
    } catch (error) {
      console.error(error.code, error.message);
      showMessage(profileMessage, "Couldn't save profile: " + error.message, "error");
    } finally {
      profileSubmitButton.disabled = false;
      profileSubmitButton.textContent = "Save Profile";
    }
  });

  // Counts the user's workouts and posts. Uses one-time reads, refreshed
  // whenever the Profile tab is opened, rather than live listeners.
  async function loadProfileStats() {
    const user = auth.currentUser;
    if (!user) return;

    statWorkouts.textContent = "…";
    statPosts.textContent = "…";
    statFollowing.textContent = "…";
    statFollowers.textContent = "…";

    try {
      const [workoutsSnap, postsSnap, followingSnap, followersSnap] = await Promise.all([
        getDocs(query(collection(db, "workouts"), where("userId", "==", user.uid))),
        getDocs(query(collection(db, "posts"), where("authorId", "==", user.uid))),
        getDocs(collection(db, "users", user.uid, "following")),
        getDocs(collection(db, "users", user.uid, "followers")),
      ]);
      statWorkouts.textContent = workoutsSnap.size;
      statPosts.textContent = postsSnap.size;
      statFollowing.textContent = followingSnap.size;
      statFollowers.textContent = followersSnap.size;
    } catch (error) {
      console.error(error.code, error.message);
      statWorkouts.textContent = "—";
      statPosts.textContent = "—";
      statFollowing.textContent = "—";
      statFollowers.textContent = "—";
    }
  }

  // ---------- FOLLOWING ----------
  // Loads the IDs of everyone the current user follows, and caches them.
  async function loadFollowingIds() {
    const user = auth.currentUser;
    if (!user) return [];

    const snapshot = await getDocs(collection(db, "users", user.uid, "following"));
    followingIds = snapshot.docs.map((d) => d.id);
    return followingIds;
  }

  // Follows or unfollows a user. Writes to BOTH sides of the relationship
  // in one atomic batch: my "following" list and their "followers" list.
  // If only one write succeeded, the two lists would silently disagree.
  async function toggleFollow(targetId, shouldFollow) {
    const user = auth.currentUser;
    if (!user || targetId === user.uid) return;

    const batch = writeBatch(db);
    const followingRef = doc(db, "users", user.uid, "following", targetId);
    const followerRef = doc(db, "users", targetId, "followers", user.uid);

    if (shouldFollow) {
      batch.set(followingRef, { since: serverTimestamp() });
      batch.set(followerRef, { since: serverTimestamp() });
    } else {
      batch.delete(followingRef);
      batch.delete(followerRef);
    }

    await batch.commit();
    await loadFollowingIds(); // refresh the cache so the feed filter stays accurate
  }

  // Renders a list of user profiles with follow/unfollow buttons.
  function renderPeople(container, profiles, emptyText) {
    const user = auth.currentUser;

    if (profiles.length === 0) {
      container.innerHTML = `<p class="empty-state">${emptyText}</p>`;
      return;
    }

    container.innerHTML = "";

    profiles.forEach((profile) => {
      const isMe = profile.uid === user.uid;
      const isFollowing = followingIds.includes(profile.uid);

      const row = document.createElement("div");
      row.className = "person-row";
      row.innerHTML = `
        <div class="person-info user-link" data-uid="${profile.uid}">
          <div class="person-avatar"></div>
          <div>
            <div class="person-username">@${escapeHtml(profile.username)}</div>
            <div class="person-bio">${escapeHtml(profile.displayName || profile.bio || "")}</div>
          </div>
        </div>
        ${isMe ? '<span class="person-you">You</span>' : `
          <button class="follow-btn ${isFollowing ? "following" : ""}" data-uid="${profile.uid}" data-following="${isFollowing}">
            ${isFollowing ? "Following" : "Follow"}
          </button>
        `}
      `;
      container.appendChild(row);
      renderAvatar(row.querySelector(".person-avatar"), profile);
    });
  }

  // Searches users by username prefix.
  // Firestore can only match from the START of a string, not anywhere inside
  // it — the \uf8ff character is a very high Unicode value, so the range
  // "faez" to "faez\uf8ff" captures every username beginning with "faez".
  // True "contains" search would need a dedicated search service.
  async function searchPeople(term) {
    const cleaned = term.trim().toLowerCase();
    if (!cleaned) {
      peopleResults.innerHTML = '<p class="empty-state">Search for a username to find people to follow.</p>';
      return;
    }

    peopleResults.innerHTML = '<p class="empty-state">Searching...</p>';

    try {
      const searchQuery = query(
        collection(db, "users"),
        where("usernameLower", ">=", cleaned),
        where("usernameLower", "<=", cleaned + "\uf8ff"),
        limit(10)
      );
      const snapshot = await getDocs(searchQuery);
      const profiles = snapshot.docs.map((d) => ({ uid: d.id, ...d.data() }));
      renderPeople(peopleResults, profiles, "No users found with that username.");
    } catch (error) {
      console.error(error.code, error.message);
      peopleResults.innerHTML = '<p class="empty-state">Search failed. Check the console for details.</p>';
    }
  }

  // Loads and displays the full list of people the user follows.
  async function loadFollowingList() {
    await loadFollowingIds();

    if (followingIds.length === 0) {
      followingListEl.innerHTML = '<p class="empty-state">You\'re not following anyone yet.</p>';
      return;
    }

    followingListEl.innerHTML = '<p class="empty-state">Loading...</p>';

    try {
      // One read per followed user. Fine at this scale; a larger app would
      // denormalize username/photo into the following document instead.
      const profiles = await Promise.all(
        followingIds.map(async (uid) => {
          const snap = await getDoc(doc(db, "users", uid));
          return snap.exists() ? { uid: uid, ...snap.data() } : null;
        })
      );
      renderPeople(followingListEl, profiles.filter(Boolean), "You're not following anyone yet.");
    } catch (error) {
      console.error(error.code, error.message);
      followingListEl.innerHTML = '<p class="empty-state">Couldn\'t load your following list.</p>';
    }
  }

  peopleSearchButton.addEventListener("click", () => searchPeople(peopleSearchInput.value));

  peopleSearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      searchPeople(peopleSearchInput.value);
    }
  });

  // Follow buttons appear in two different lists, so listen on the whole
  // People view and let delegation sort out which button was clicked.
  peopleView.addEventListener("click", async (event) => {
    const button = event.target.closest(".follow-btn");
    if (!button) return;

    const targetId = button.dataset.uid;
    const isFollowing = button.dataset.following === "true";

    button.disabled = true;
    button.textContent = isFollowing ? "..." : "...";

    try {
      await toggleFollow(targetId, !isFollowing);

      button.dataset.following = (!isFollowing).toString();
      button.textContent = !isFollowing ? "Following" : "Follow";
      button.classList.toggle("following", !isFollowing);

      loadFollowingList(); // keep the lower list in sync after any change
    } catch (error) {
      console.error(error.code, error.message);
      alert("Couldn't update follow: " + error.message);
    } finally {
      button.disabled = false;
    }
  });

  // ---------- FEED FILTER ----------
  filterAllBtn.addEventListener("click", () => {
    feedMode = "all";
    filterAllBtn.classList.add("active");
    filterFollowingBtn.classList.remove("active");
    listenForFeed();
  });

  filterFollowingBtn.addEventListener("click", async () => {
    feedMode = "following";
    filterFollowingBtn.classList.add("active");
    filterAllBtn.classList.remove("active");
    await loadFollowingIds();
    listenForFeed();
  });

  // ---------- LIVE COMMUNITY FEED ----------
  function listenForFeed() {
    // Stop any existing listener first — switching filters creates a new
    // query, and leaving the old one running would cause both to write
    // into the same list element.
    if (unsubscribeFeedList) {
      unsubscribeFeedList();
      unsubscribeFeedList = null;
    }

    let feedQuery;

    if (feedMode === "following") {
      if (followingIds.length === 0) {
        feedList.innerHTML = '<p class="empty-state">You\'re not following anyone yet. Find people in the People tab.</p>';
        return;
      }

      // Firestore's "in" operator accepts at most 30 values, so this caps
      // the personalized feed at your 30 most recently loaded follows.
      // Scaling past that is why real apps precompute each user's feed
      // when a post is created ("fan-out on write") instead of querying live.
      feedQuery = query(
        collection(db, "posts"),
        where("authorId", "in", followingIds.slice(0, 30)),
        orderBy("createdAt", "desc"),
        limit(50)
      );
    } else {
      feedQuery = query(
        collection(db, "posts"),
        orderBy("createdAt", "desc"),
        limit(50)
      );
    }

    unsubscribeFeedList = onSnapshot(feedQuery, (snapshot) => {
      if (snapshot.empty) {
        feedList.innerHTML = '<p class="empty-state">No posts yet — share a workout to get things started.</p>';
        return;
      }

      feedList.innerHTML = "";

      snapshot.forEach((docSnap) => {
        const post = docSnap.data();
        const date = post.createdAt ? post.createdAt.toDate().toLocaleDateString() : "Just now";

        const postEl = document.createElement("div");
        postEl.className = "post";

        // Posts shared before cardio existed have no type field, so anything
        // without one is rendered as a strength post.
        const summary = post.type === "cardio"
          ? `${escapeHtml(post.activity)} — ${post.distance} mi in ${formatDuration(post.durationSeconds)} (${formatDuration(paceSecondsPerMile(post.distance, post.durationSeconds))}/mi)`
          : `${escapeHtml(post.exerciseName || "Workout")} — ${post.sets} sets × ${post.reps} reps @ ${post.weight} lb`;

        postEl.innerHTML = `
          <div class="post-author"><a href="#" class="user-link" data-uid="${post.authorId}">@${escapeHtml(post.authorUsername || "unknown")}</a></div>
          ${post.caption ? `<div class="post-caption">${escapeHtml(post.caption)}</div>` : ""}
          <div class="post-workout">${summary}</div>
          <div class="post-footer">
            <div class="post-date">${date}</div>
            <div class="post-actions">
              <button class="comment-btn" data-post-id="${docSnap.id}">Comments</button>
              <button class="like-btn" data-post-id="${docSnap.id}" data-liked="false">
                <span class="heart">♡</span> <span class="like-count">0</span>
              </button>
            </div>
          </div>
          <div class="comments-section" style="display: none;">
            <div class="comment-list"></div>
            <div class="comment-form">
              <input type="text" class="comment-input" placeholder="Add a comment..." maxlength="300" />
              <button class="comment-submit" data-post-id="${docSnap.id}">Post</button>
            </div>
          </div>
        `;
        feedList.appendChild(postEl);

        const likeButton = postEl.querySelector(".like-btn");
        initLikeButton(likeButton);
      });
    });
  }

  // ---------- LIVE WORKOUT HISTORY LIST ----------
  function listenForWorkouts(userId) {
    const workoutsQuery = query(
      collection(db, "workouts"),
      where("userId", "==", userId),
      orderBy("createdAt", "desc")
    );

    // onSnapshot keeps this list updated in real time — no manual refresh needed
    unsubscribeWorkoutList = onSnapshot(workoutsQuery, (snapshot) => {
      // Cache the raw data so the progress chart can use it without
      // running its own query — the listener already has everything.
      cachedWorkouts = snapshot.docs.map((d) => d.data());
      refreshExerciseOptions();
      renderProgress();
      publishMatchProfile();    // derived summary for partner matching
      publishExerciseProfile(); // anonymised interaction profile for the ML model

      if (snapshot.empty) {
        workoutList.innerHTML = '<p class="empty-state">No workouts logged yet — add your first one above.</p>';
        return;
      }

      workoutList.innerHTML = ""; // clear the list before re-rendering

      snapshot.forEach((doc) => {
        const workout = doc.data();
        const date = workout.createdAt ? workout.createdAt.toDate().toLocaleDateString() : "Just now";

        const entry = document.createElement("div");
        entry.className = "workout-entry";

        // Cardio and strength entries carry different fields, so each gets
        // its own title, detail line, and share-button payload.
        const isCardio = workout.type === "cardio";
        const title = isCardio ? workout.activity : workout.exerciseName;
        const detail = isCardio
          ? `${workout.distance} mi · ${formatDuration(workout.durationSeconds)} · ${formatDuration(paceSecondsPerMile(workout.distance, workout.durationSeconds))}/mi`
          : `${workout.sets} sets × ${workout.reps} reps @ ${workout.weight} lb`;

        const shareData = isCardio
          ? `data-type="cardio"
             data-activity="${escapeHtml(workout.activity)}"
             data-distance="${workout.distance}"
             data-duration="${workout.durationSeconds}"`
          : `data-type="strength"
             data-exercise="${escapeHtml(workout.exerciseName)}"
             data-sets="${workout.sets}"
             data-reps="${workout.reps}"
             data-weight="${workout.weight}"`;

        entry.innerHTML = `
          <div>
            <div class="workout-entry-name">${escapeHtml(title)}${isCardio ? ' <span class="entry-tag">Cardio</span>' : ""}</div>
            <div class="workout-entry-detail">${detail}</div>
          </div>
          <div class="workout-entry-right">
            <div class="workout-entry-date">${date}</div>
            <button class="share-btn" data-workout-id="${doc.id}" ${shareData}>Share</button>
            <button class="delete-btn" data-id="${doc.id}" title="Delete this entry">✕</button>
          </div>
        `;
        workoutList.appendChild(entry);
      });
    });
  }

  // ---------- SWITCH BETWEEN LOGIN, PROFILE SETUP, AND DASHBOARD ----------
  function enterDashboard(user) {
    loginView.style.display = "none";
    profileSetupView.style.display = "none";
    dashboardView.style.display = "block";
    document.body.classList.add("dashboard-mode");
    closeDetail(); // always land on the tabbed view, not a stale detail page
    renderProfile();
    listenForWorkouts(user.uid);
    loadFollowingIds().then(() => listenForFeed());
    // Load own locations up front so partner matching works without first
    // visiting the Profile tab.
    getDocs(collection(db, "users", user.uid, "locations"))
      .then((snap) => { myLocationsCache = snap.docs.map((d) => d.data()); })
      .catch(() => {}); // follow list must load before the feed can filter by it
  }

  function showProfileSetup() {
    loginView.style.display = "none";
    dashboardView.style.display = "none";
    profileSetupView.style.display = "block";
    document.body.classList.remove("dashboard-mode");
  }

  function showLogin() {
    dashboardView.style.display = "none";
    profileSetupView.style.display = "none";
    loginView.style.display = "block";
    document.body.classList.remove("dashboard-mode");
    currentProfile = null; // clear the cached profile so it can't leak to the next user
    followingIds = [];     // same for the follow list
    cachedWorkouts = [];   // and the chart data
    myLocationsCache = [];
    myPlansCount = 0;
    feedMode = "all";      // reset the feed filter for the next session

    // Stop listening to the previous user's data, if any
    if (unsubscribeWorkoutList) {
      unsubscribeWorkoutList();
      unsubscribeWorkoutList = null;
    }
    if (unsubscribeFeedList) {
      unsubscribeFeedList();
      unsubscribeFeedList = null;
    }
  }

  // Fires on page load, and every time login state changes
  onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log("Logged in as:", user.email);
      currentProfile = await fetchProfile(user.uid);

      if (currentProfile) {
        enterDashboard(user);
      } else {
        // Account exists but has no profile — either created before profiles
        // were added, or signup failed partway through. Ask for a username.
        showProfileSetup();
      }
    } else {
      console.log("No user is logged in.");
      showLogin();
    }
  });
});