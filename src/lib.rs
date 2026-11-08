//! Thin harness crate: shared helpers for mtree benches and examples.

use mtree::distance::Distance;
use mtree::stats::DescendantCounter;
use mtree::{
    node_params_for_capacity, reset_knn_visit_stats, take_knn_visit_stats, EntryId, KnnConfig,
    MTree, DEFAULT_MAX_NODE_SIZE, DEFAULT_MIN_NODE_SIZE, DEFAULT_SPLIT_SAMPLING,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

pub type Key2d = (i64, i64);
/// Payload: `true` = active for filtered queries (~50% of inserts).
pub type Active = bool;
/// Monomorphe Experiment-Tree (Phase-4/5 Hotpath: keine DynDistance auf dem Query-Pfad).
pub type ExperimentTree = MTree<Key2d, Active, f64, DescendantCounter, Euclid2d>;

pub const QUERY_COUNT: usize = 256;
pub const TIMED_ROUNDS: usize = 5;

pub const LOG_SIZES: &[usize] = &[
    50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
];
pub const KS: &[usize] = &[1, 10, 50];

#[derive(Clone)]
pub struct Euclid2d;

impl Distance<Key2d> for Euclid2d {
    type Output = f64;
    fn distance(&self, a: &Key2d, b: &Key2d) -> f64 {
        let dx = (a.0 - b.0) as f64;
        let dy = (a.1 - b.1) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    pub min_node_size: usize,
    pub max_node_size: usize,
    pub split_sampling: usize,
}

/// Matches mtree constructor defaults (`DEFAULT_*`).
pub const FIXED_PARAMS: TreeParams = TreeParams {
    min_node_size: DEFAULT_MIN_NODE_SIZE,
    max_node_size: DEFAULT_MAX_NODE_SIZE,
    split_sampling: DEFAULT_SPLIT_SAMPLING,
};

#[derive(Clone, Copy, Debug)]
pub enum Policy {
    Fixed,
    Scaled,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Fixed => "fixed",
            Policy::Scaled => "scaled",
        }
    }

    pub fn params(self, n: usize) -> TreeParams {
        match self {
            Policy::Fixed => FIXED_PARAMS,
            Policy::Scaled => {
                let (min, max, split) = node_params_for_capacity(n);
                TreeParams {
                    min_node_size: min,
                    max_node_size: max,
                    split_sampling: split,
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryMode {
    NeedleNone,
    NeedleHint,
    FromEntry,
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QueryMode::NeedleNone => "needle_none",
            QueryMode::NeedleHint => "needle_hint",
            QueryMode::FromEntry => "from_entry",
        }
    }

    pub fn all() -> [QueryMode; 3] {
        [
            QueryMode::NeedleNone,
            QueryMode::NeedleHint,
            QueryMode::FromEntry,
        ]
    }
}

/// Unique 2D points with alternating active flag (~50%).
pub fn gen_active_data(n: usize, seed: u64) -> Vec<(Key2d, Active)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let k = (
            rng.gen_range(-10_000..10_000),
            rng.gen_range(-10_000..10_000),
        );
        if seen.insert(k) {
            let active = out.len() % 2 == 0;
            out.push((k, active));
        }
    }
    out
}

/// Derive `min` / `split` from `max` the same way as `node_params_for_capacity`
/// (without the √n choice of `max`).
pub fn params_from_max(max_node_size: usize) -> TreeParams {
    let max = max_node_size.max(2);
    let min = (max / 4).max(2);
    let split = (max / 4).clamp(5, 20);
    TreeParams {
        min_node_size: min,
        max_node_size: max,
        split_sampling: split,
    }
}

pub fn new_tree_with_params(params: TreeParams) -> ExperimentTree {
    MTree::with_params(
        params.min_node_size,
        params.max_node_size,
        params.split_sampling,
        Euclid2d,
    )
}

pub fn new_tree(policy: Policy, n: usize) -> ExperimentTree {
    match policy {
        Policy::Fixed => new_tree_with_params(FIXED_PARAMS),
        Policy::Scaled => MTree::with_capacity_hint(n, Euclid2d),
    }
}

pub fn build_tree_with_params(
    params: TreeParams,
    data: &[(Key2d, Active)],
) -> (ExperimentTree, Vec<EntryId>) {
    let mut tree = new_tree_with_params(params);
    let mut ids = Vec::with_capacity(data.len());
    for &(key, active) in data {
        ids.push(tree.insert(key, active).unwrap());
    }
    (tree, ids)
}

/// Build timing (ms) plus tree + entry ids — used by param-sweep side metrics.
pub fn build_tree_with_params_timed(
    params: TreeParams,
    data: &[(Key2d, Active)],
) -> (ExperimentTree, Vec<EntryId>, f64) {
    let start = Instant::now();
    let (tree, ids) = build_tree_with_params(params, data);
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    (tree, ids, build_ms)
}

pub fn build_tree(
    policy: Policy,
    n: usize,
    data: &[(Key2d, Active)],
) -> (ExperimentTree, Vec<EntryId>) {
    build_tree_with_params(policy.params(n), data)
}

/// Side-metric batch size for insert / erase / update_key micro-timings.
pub const MUTATE_BATCH: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct MutateSideMetrics {
    /// Median µs / insert over `MUTATE_BATCH` fresh keys (then erased).
    pub insert_us: f64,
    /// Median µs / erase_by_id over `MUTATE_BATCH` entries (then re-inserted).
    pub erase_us: f64,
    /// Median µs / update_key over `MUTATE_BATCH` key moves (restored after).
    pub update_us: f64,
}

fn fresh_key_2d(rng: &mut StdRng, occupied: &std::collections::HashSet<Key2d>) -> Key2d {
    loop {
        let k = (
            rng.gen_range(-10_000..10_000),
            rng.gen_range(-10_000..10_000),
        );
        if !occupied.contains(&k) {
            return k;
        }
    }
}

/// Short mutate side metrics; restores tree membership afterward.
pub fn measure_mutate_side(
    tree: &mut ExperimentTree,
    ids: &[EntryId],
    data: &[(Key2d, Active)],
    rounds: usize,
) -> MutateSideMetrics {
    let mut rng = StdRng::seed_from_u64(77);
    let mut occupied: std::collections::HashSet<Key2d> =
        data.iter().map(|&(k, _)| k).collect();

    let batch = MUTATE_BATCH.min(ids.len().max(1));
    let mut insert_samples = Vec::with_capacity(rounds);
    let mut erase_samples = Vec::with_capacity(rounds);
    let mut update_samples = Vec::with_capacity(rounds);

    // Warmup once (discarded).
    {
        let mut tmp_ids = Vec::with_capacity(batch);
        for _ in 0..batch {
            let key = fresh_key_2d(&mut rng, &occupied);
            occupied.insert(key);
            tmp_ids.push(tree.insert(key, true).expect("warmup insert"));
        }
        for id in tmp_ids {
            let key = tree.key(id).expect("warmup key");
            occupied.remove(&key);
            assert!(tree.erase_by_id(id));
        }
    }

    for _ in 0..rounds {
        // Insert batch.
        let mut tmp_ids = Vec::with_capacity(batch);
        let t0 = Instant::now();
        for _ in 0..batch {
            let key = fresh_key_2d(&mut rng, &occupied);
            occupied.insert(key);
            tmp_ids.push(tree.insert(key, true).expect("insert"));
        }
        insert_samples.push(t0.elapsed().as_secs_f64() * 1e6 / batch as f64);

        // Erase those inserts.
        let t1 = Instant::now();
        for &id in &tmp_ids {
            let key = tree.key(id).expect("erase key");
            occupied.remove(&key);
            assert!(tree.erase_by_id(id));
        }
        erase_samples.push(t1.elapsed().as_secs_f64() * 1e6 / batch as f64);

        // update_key on existing entries, then restore.
        let targets: Vec<EntryId> = (0..batch).map(|i| ids[i % ids.len()]).collect();
        let originals: Vec<Key2d> = targets
            .iter()
            .map(|&id| tree.key(id).expect("update key"))
            .collect();
        let news: Vec<Key2d> = (0..batch)
            .map(|_| {
                let key = fresh_key_2d(&mut rng, &occupied);
                occupied.insert(key);
                key
            })
            .collect();

        let t2 = Instant::now();
        for i in 0..batch {
            occupied.remove(&originals[i]);
            tree.update_key(targets[i], news[i]).expect("update_key");
        }
        update_samples.push(t2.elapsed().as_secs_f64() * 1e6 / batch as f64);

        for i in 0..batch {
            occupied.remove(&news[i]);
            tree.update_key(targets[i], originals[i])
                .expect("restore update_key");
            occupied.insert(originals[i]);
        }
    }

    MutateSideMetrics {
        insert_us: median_f64(&mut insert_samples),
        erase_us: median_f64(&mut erase_samples),
        update_us: median_f64(&mut update_samples),
    }
}

pub fn sample_query_ids(ids: &[EntryId], count: usize, seed: u64) -> Vec<EntryId> {
    let mut rng = StdRng::seed_from_u64(seed);
    let take = count.min(ids.len());
    (0..take)
        .map(|_| ids[rng.gen_range(0..ids.len())])
        .collect()
}

/// Prefetch keys so timed loops do not pay lookup cost unevenly across modes.
pub fn prefetch_queries(
    tree: &ExperimentTree,
    ids: &[EntryId],
) -> Vec<(EntryId, Key2d)> {
    ids.iter()
        .map(|&id| {
            let key = tree.key(id).expect("query id must resolve");
            (id, key)
        })
        .collect()
}

fn is_active(_id: EntryId, active: &Active) -> bool {
    *active
}

pub fn run_query_batch(
    tree: &ExperimentTree,
    queries: &[(EntryId, Key2d)],
    k: usize,
    mode: QueryMode,
    filter: bool,
) {
    for &(id, key) in queries {
        let hits = match (mode, filter) {
            (QueryMode::NeedleNone, false) => tree.knn_search(&key, k, ()),
            (QueryMode::NeedleNone, true) => {
                tree.knn_search(&key, k, KnnConfig::new().with_active(is_active))
            }
            (QueryMode::NeedleHint, false) => {
                tree.knn_search(&key, k, KnnConfig::new().seed(id))
            }
            (QueryMode::NeedleHint, true) => tree.knn_search(
                &key,
                k,
                KnnConfig::new().seed(id).with_active(is_active),
            ),
            (QueryMode::FromEntry, false) => tree
                .knn_from_entry(id, k, false, ())
                .expect("from_entry id"),
            (QueryMode::FromEntry, true) => tree
                .knn_from_entry(id, k, false, KnnConfig::new().with_active(is_active))
                .expect("from_entry id"),
        };
        black_box(hits);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeasureResult {
    pub avg_ms: f64,
    pub avg_routing: f64,
    pub avg_leaves: f64,
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Convert `(i64, i64)` keys to `[f64; 2]` for spatial baseline crates.
pub fn key_to_f64(key: Key2d) -> [f64; 2] {
    [key.0 as f64, key.1 as f64]
}

pub fn points_f64(data: &[(Key2d, Active)]) -> Vec<[f64; 2]> {
    data.iter().map(|&(key, _)| key_to_f64(key)).collect()
}

pub fn query_points_f64(queries: &[(EntryId, Key2d)]) -> Vec<[f64; 2]> {
    queries.iter().map(|&(_, key)| key_to_f64(key)).collect()
}

/// Warmup once, then `rounds` timed passes of `run`; returns median avg ms per query.
pub fn measure_query_median<F>(query_count: usize, rounds: usize, mut run: F) -> f64
where
    F: FnMut(),
{
    run();
    let q = query_count as f64;
    let mut ms_samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        run();
        ms_samples.push(start.elapsed().as_secs_f64() * 1000.0 / q);
    }
    median_f64(&mut ms_samples)
}

/// Warmup once, then `rounds` timed passes; returns medians of per-query averages.
pub fn measure_median(
    tree: &ExperimentTree,
    queries: &[(EntryId, Key2d)],
    k: usize,
    mode: QueryMode,
    filter: bool,
    rounds: usize,
) -> MeasureResult {
    run_query_batch(tree, queries, k, mode, filter);

    let q = queries.len() as f64;
    let mut ms_samples = Vec::with_capacity(rounds);
    let mut routing_samples = Vec::with_capacity(rounds);
    let mut leaf_samples = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        reset_knn_visit_stats();
        let start = Instant::now();
        run_query_batch(tree, queries, k, mode, filter);
        let elapsed = start.elapsed();
        let s = take_knn_visit_stats();
        ms_samples.push(elapsed.as_secs_f64() * 1000.0 / q);
        routing_samples.push(s.routing_nodes as f64 / q);
        leaf_samples.push(s.leaves as f64 / q);
    }

    MeasureResult {
        avg_ms: median_f64(&mut ms_samples),
        avg_routing: median_f64(&mut routing_samples),
        avg_leaves: median_f64(&mut leaf_samples),
    }
}
