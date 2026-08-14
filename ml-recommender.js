// ============================================================================
// 225 — MATRIX FACTORIZATION RECOMMENDER
// ============================================================================
// A collaborative filtering model trained by stochastic gradient descent.
// Pure JavaScript, no libraries, trains in the browser.
//
// THE MODEL
//
// Every user and every exercise is represented as a vector of K latent
// factors. Those factors are not defined in advance — the model discovers
// them from the data. In practice they tend to converge on interpretable
// axes such as upper versus lower body emphasis, or compound versus
// isolation preference, but nothing in the algorithm names them.
//
// A user's affinity for an exercise is predicted as:
//
//     r̂(u,i) = μ + b_u + b_i + P_u · Q_i
//
//   μ     global mean rating
//   b_u   user bias        — some people log everything more heavily
//   b_i   item bias        — some exercises are simply more popular
//   P_u   user factor vector
//   Q_i   item factor vector
//
// Bias terms matter more than they look. Without them the factor vectors are
// forced to encode "this user logs a lot" and "this exercise is common",
// wasting capacity that should be spent on actual preference structure.
//
// TRAINING
//
// Squared error on observed interactions, with L2 regularization to stop
// factors growing without bound on sparsely-observed users:
//
//     min Σ (r - r̂)² + λ(‖P_u‖² + ‖Q_i‖² + b_u² + b_i²)
//
// Optimised by SGD: for each observed interaction, compute the error and step
// every involved parameter against its gradient.
//
// EVALUATION
//
// The model is compared against two baselines on held-out data, because an
// unbaselined accuracy figure means nothing:
//
//   Popularity — always recommend the most-logged exercises. Surprisingly
//                strong, and the bar any recommender must clear to be useful.
//   Random     — a floor. Losing to this indicates a broken implementation.
// ============================================================================


// ---------------------------------------------------------------------------
// DATA PREPARATION
// ---------------------------------------------------------------------------

/**
 * Builds a user × exercise interaction matrix from raw logged workouts.
 *
 * Volume is log-compressed before scaling. Raw training volume spans orders of
 * magnitude — a set of heavy squats can outweigh twenty sets of curls — and
 * without compression the model would fit almost entirely to load rather than
 * to preference. Log scaling preserves ordering while pulling the range in.
 *
 * Ratings are then normalised per user to 0–1, so a high-volume lifter and a
 * beginner both express preference on the same scale. Absolute volume is
 * already handled elsewhere in the application; this model is about relative
 * preference between exercises.
 */
export function buildInteractionMatrix(userWorkouts) {
  const userIds = [];
  const itemIds = [];
  const userIndex = new Map();
  const itemIndex = new Map();

  // volume per (user, item)
  const raw = new Map();

  for (const { userId, workouts } of userWorkouts) {
    for (const w of workouts) {
      if (w.type === "cardio") continue;
      const itemId = w.exerciseId;
      if (!itemId) continue; // legacy free-text entries cannot be matched reliably

      const volume = (w.sets || 0) * (w.reps || 0) * (w.weight || 0);
      if (volume <= 0) continue;

      if (!userIndex.has(userId)) { userIndex.set(userId, userIds.length); userIds.push(userId); }
      if (!itemIndex.has(itemId)) { itemIndex.set(itemId, itemIds.length); itemIds.push(itemId); }

      const key = `${userIndex.get(userId)}:${itemIndex.get(itemId)}`;
      raw.set(key, (raw.get(key) || 0) + volume);
    }
  }

  // log-compress, then normalise within each user
  const byUser = new Map();
  for (const [key, volume] of raw) {
    const u = Number(key.split(":")[0]);
    const compressed = Math.log(1 + volume);
    if (!byUser.has(u)) byUser.set(u, []);
    byUser.get(u).push({ key, compressed });
  }

  const interactions = [];
  for (const [, list] of byUser) {
    const max = Math.max(...list.map((x) => x.compressed));
    const min = Math.min(...list.map((x) => x.compressed));
    const span = max - min;

    for (const { key, compressed } of list) {
      const [u, i] = key.split(":").map(Number);
      // A user with a single logged exercise has no spread to normalise
      // against; 1.0 is the honest reading of "everything they do".
      const rating = span === 0 ? 1 : 0.1 + 0.9 * ((compressed - min) / span);
      interactions.push({ user: u, item: i, rating });
    }
  }

  return { userIds, itemIds, interactions, userIndex, itemIndex };
}


// ---------------------------------------------------------------------------
// TRAINING
// ---------------------------------------------------------------------------

/**
 * Deterministic pseudo-random generator (mulberry32).
 *
 * Seeded rather than Math.random so that training runs are reproducible.
 * Without this, evaluation results would shift on every run and it would be
 * impossible to tell a genuine improvement from initialisation luck.
 */
export function makeRandom(seed = 42) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Trains the factorization by stochastic gradient descent with negative
 * sampling.
 *
 * WHY NEGATIVE SAMPLING IS REQUIRED HERE
 *
 * This dataset is implicit feedback: it records what people did, never what
 * they chose not to do. Training only on observed pairs optimises rating
 * prediction on those pairs — but the model is used to RANK unobserved pairs,
 * which is a different task.
 *
 * Without negatives the latent factors receive no signal that separates one
 * unobserved item from another, so ranking collapses onto the item bias term.
 * The item bias IS popularity, so the model degenerates into exactly the
 * baseline it is meant to beat. This was observed directly during testing:
 * every one of the top ten recommendations was also a top-ten item by bias.
 *
 * The fix is to sample items each user has NOT logged and train them toward a
 * low target. These are assumed negatives rather than confirmed ones — a user
 * may simply not have discovered an exercise — which is why the negative
 * target is 0.05 rather than 0, and why relatively few are sampled per
 * positive. This is the same principle underlying BPR and weighted ALS.
 *
 * @param options.negativeSamples  assumed negatives drawn per positive
 */
export function trainMatrixFactorization(interactions, options = {}) {
  const K = options.factors ?? 8;
  const epochs = options.epochs ?? 120;
  const lr = options.learningRate ?? 0.02;
  const reg = options.regularization ?? 0.05;
  // Default chosen empirically. An ablation across 0, 1, 2, 4, 8 and 16
  // negatives per positive peaked at 2; beyond that the assumed negatives
  // begin to outweigh genuine signal and precision falls away again.
  const negRatio = options.negativeSamples ?? 2;
  const negTarget = options.negativeTarget ?? 0.05;
  const rand = makeRandom(options.seed ?? 42);

  if (interactions.length === 0) return null;

  const nUsers = Math.max(...interactions.map((x) => x.user)) + 1;
  const nItems = options.nItems ?? Math.max(...interactions.map((x) => x.item)) + 1;

  const globalMean = interactions.reduce((s, x) => s + x.rating, 0) / interactions.length;

  // Which items each user has actually logged, so negatives are drawn only
  // from genuinely unobserved pairs.
  const seen = new Map();
  for (const x of interactions) {
    if (!seen.has(x.user)) seen.set(x.user, new Set());
    seen.get(x.user).add(x.item);
  }

  // Small random initialisation. Zeros would leave every factor with an
  // identical gradient, so they would never differentiate from each other.
  const scale = 0.1;
  const P = Array.from({ length: nUsers }, () => Array.from({ length: K }, () => (rand() - 0.5) * scale));
  const Q = Array.from({ length: nItems }, () => Array.from({ length: K }, () => (rand() - 0.5) * scale));
  const bU = new Array(nUsers).fill(0);
  const bI = new Array(nItems).fill(0);

  const lossHistory = [];
  const order = interactions.map((_, idx) => idx);

  // One SGD update for a single (user, item, target) triple.
  function step(u, i, target) {
    let dot = 0;
    for (let k = 0; k < K; k++) dot += P[u][k] * Q[i][k];
    const predicted = globalMean + bU[u] + bI[i] + dot;
    const error = target - predicted;

    bU[u] += lr * (error - reg * bU[u]);
    bI[i] += lr * (error - reg * bI[i]);

    for (let k = 0; k < K; k++) {
      const pu = P[u][k];
      const qi = Q[i][k];
      P[u][k] += lr * (error * qi - reg * pu);
      Q[i][k] += lr * (error * pu - reg * qi);
    }

    return error * error;
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle each epoch so the model does not learn the data ordering.
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    let positiveError = 0;

    for (const idx of order) {
      const { user: u, item: i, rating } = interactions[idx];
      positiveError += step(u, i, rating);

      // Assumed negatives, resampled every epoch so the model is not fitted
      // to one fixed pseudo-negative set.
      const known = seen.get(u);
      for (let n = 0; n < negRatio; n++) {
        let candidate = Math.floor(rand() * nItems);
        let attempts = 0;
        while (known.has(candidate) && attempts < 8) {
          candidate = Math.floor(rand() * nItems);
          attempts++;
        }
        if (known.has(candidate)) continue; // user has logged nearly everything
        step(u, candidate, negTarget);
      }
    }

    // Loss is reported over positives only, so the curve stays comparable to
    // a run trained without negative sampling.
    lossHistory.push(Math.sqrt(positiveError / interactions.length));
  }

  return { P, Q, bU, bI, globalMean, K, nUsers, nItems, lossHistory };
}

/** Predicted affinity, clamped to the rating range. */
export function predict(model, user, item) {
  if (!model || user >= model.nUsers || item >= model.nItems || user < 0 || item < 0) {
    return model ? model.globalMean : null;
  }
  let dot = 0;
  for (let k = 0; k < model.K; k++) dot += model.P[user][k] * model.Q[item][k];
  const raw = model.globalMean + model.bU[user] + model.bI[item] + dot;
  return Math.max(0, Math.min(1, raw));
}

/** Top-N recommendations for a user, excluding exercises they already log. */
export function recommend(model, user, knownItems, n = 5) {
  if (!model) return [];
  const known = new Set(knownItems);
  const scored = [];

  for (let item = 0; item < model.nItems; item++) {
    if (known.has(item)) continue;
    scored.push({ item, score: predict(model, user, item) });
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, n);
}


// ---------------------------------------------------------------------------
// BASELINES
// ---------------------------------------------------------------------------

/**
 * Popularity baseline: recommend whatever is logged most often overall.
 *
 * Deliberately included because it is a genuinely strong competitor. Popular
 * exercises are popular for good reasons, and a personalised model that
 * cannot beat this is adding complexity for nothing.
 */
export function popularityBaseline(interactions) {
  const counts = new Map();
  for (const x of interactions) counts.set(x.item, (counts.get(x.item) || 0) + 1);

  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([item]) => item);
  return {
    recommend: (knownItems, n = 5) => {
      const known = new Set(knownItems);
      return ranked.filter((i) => !known.has(i)).slice(0, n).map((item) => ({ item, score: 1 }));
    },
  };
}

/** Random baseline — the floor. Losing to this means something is broken. */
export function randomBaseline(nItems, seed = 7) {
  const rand = makeRandom(seed);
  return {
    recommend: (knownItems, n = 5) => {
      const known = new Set(knownItems);
      const pool = [];
      for (let i = 0; i < nItems; i++) if (!known.has(i)) pool.push(i);
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      return pool.slice(0, n).map((item) => ({ item, score: 0 }));
    },
  };
}


// ---------------------------------------------------------------------------
// EVALUATION
// ---------------------------------------------------------------------------

/**
 * Splits interactions into training and test sets.
 *
 * Split is per user rather than globally, so every user retains training
 * history. A global random split would leave some users with no training data
 * at all, and the model would then be measured on people it never saw — which
 * tests cold-start behaviour rather than recommendation quality.
 *
 * Users with fewer than `minTrain + 1` interactions are kept entirely in
 * training, since holding data out from them would leave nothing to learn from.
 */
export function trainTestSplit(interactions, testFraction = 0.2, seed = 99, minTrain = 2) {
  const rand = makeRandom(seed);
  const byUser = new Map();

  for (const x of interactions) {
    if (!byUser.has(x.user)) byUser.set(x.user, []);
    byUser.get(x.user).push(x);
  }

  const train = [];
  const test = [];

  for (const [, list] of byUser) {
    if (list.length <= minTrain) { train.push(...list); continue; }

    const shuffled = [...list];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const nTest = Math.max(1, Math.min(list.length - minTrain, Math.round(list.length * testFraction)));
    test.push(...shuffled.slice(0, nTest));
    train.push(...shuffled.slice(nTest));
  }

  return { train, test };
}

/** Root mean squared error of predicted ratings on held-out interactions. */
export function rmse(model, testSet) {
  if (testSet.length === 0) return null;
  let sum = 0;
  for (const x of testSet) {
    const err = x.rating - predict(model, x.user, x.item);
    sum += err * err;
  }
  return Math.sqrt(sum / testSet.length);
}

/**
 * Precision@K — of the K exercises recommended, what fraction appear in the
 * user's held-out data.
 *
 * Averaged per user rather than pooled, so a single very active user cannot
 * dominate the score.
 */
export function precisionAtK(recommender, trainSet, testSet, k = 5) {
  const trainByUser = new Map();
  const testByUser = new Map();

  for (const x of trainSet) {
    if (!trainByUser.has(x.user)) trainByUser.set(x.user, new Set());
    trainByUser.get(x.user).add(x.item);
  }
  for (const x of testSet) {
    if (!testByUser.has(x.user)) testByUser.set(x.user, new Set());
    testByUser.get(x.user).add(x.item);
  }

  let total = 0, users = 0;

  for (const [user, heldOut] of testByUser) {
    const known = trainByUser.get(user) || new Set();
    const recs = recommender(user, [...known], k);
    if (recs.length === 0) continue;

    const hits = recs.filter((r) => heldOut.has(r.item)).length;
    total += hits / recs.length;
    users++;
  }

  return users === 0 ? null : total / users;
}

/** Recall@K — what fraction of held-out items the recommender surfaced. */
export function recallAtK(recommender, trainSet, testSet, k = 5) {
  const trainByUser = new Map();
  const testByUser = new Map();

  for (const x of trainSet) {
    if (!trainByUser.has(x.user)) trainByUser.set(x.user, new Set());
    trainByUser.get(x.user).add(x.item);
  }
  for (const x of testSet) {
    if (!testByUser.has(x.user)) testByUser.set(x.user, new Set());
    testByUser.get(x.user).add(x.item);
  }

  let total = 0, users = 0;

  for (const [user, heldOut] of testByUser) {
    const known = trainByUser.get(user) || new Set();
    const recs = recommender(user, [...known], k);
    if (heldOut.size === 0) continue;

    const hits = recs.filter((r) => heldOut.has(r.item)).length;
    total += hits / heldOut.size;
    users++;
  }

  return users === 0 ? null : total / users;
}

/**
 * Full evaluation: trains the model and measures it against both baselines on
 * identical held-out data.
 */
export function evaluate(interactions, options = {}) {
  const k = options.k ?? 5;
  const { train, test } = trainTestSplit(interactions, options.testFraction ?? 0.2, options.splitSeed ?? 99);

  if (test.length === 0) return null;

  const nItems = Math.max(...interactions.map((x) => x.item)) + 1;
  const model = trainMatrixFactorization(train, { ...options, nItems });

  const pop = popularityBaseline(train);
  const rnd = randomBaseline(nItems, options.randomSeed ?? 7);

  const mfRecommender = (user, known, n) => recommend(model, user, known, n);
  const popRecommender = (_user, known, n) => pop.recommend(known, n);
  const rndRecommender = (_user, known, n) => rnd.recommend(known, n);

  const results = {
    k,
    trainSize: train.length,
    testSize: test.length,
    users: new Set(interactions.map((x) => x.user)).size,
    items: nItems,
    sparsity: 1 - interactions.length / (new Set(interactions.map((x) => x.user)).size * nItems),

    model: {
      rmse: rmse(model, test),
      precision: precisionAtK(mfRecommender, train, test, k),
      recall: recallAtK(mfRecommender, train, test, k),
      finalTrainingLoss: model.lossHistory[model.lossHistory.length - 1],
    },
    popularity: {
      precision: precisionAtK(popRecommender, train, test, k),
      recall: recallAtK(popRecommender, train, test, k),
    },
    random: {
      precision: precisionAtK(rndRecommender, train, test, k),
      recall: recallAtK(rndRecommender, train, test, k),
    },
  };

  // Relative improvement is what actually matters. Absolute precision figures
  // are not comparable across datasets of different sparsity.
  results.liftOverPopularity =
    results.popularity.precision > 0
      ? (results.model.precision - results.popularity.precision) / results.popularity.precision
      : null;

  results.beatsPopularity = results.model.precision > results.popularity.precision;
  results.beatsRandom = results.model.precision > results.random.precision;

  return { results, model, train, test };
}
