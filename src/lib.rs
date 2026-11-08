//! Thin harness crate: shared helpers for mtree benches and examples.

use mtree::distance::Distance;
use mtree::{
    node_params_for_capacity, reset_knn_visit_stats, take_knn_visit_stats, EntryId, KnnConfig,
    MTree,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

pub type Key2d = (i64, i64);
/// Payload: `true` = active for filtered queries (~50% of inserts).
pub type Active = bool;

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
    fn clone_box(&self) -> Box<dyn Distance<Key2d, Output = f64> + Send + Sync> {
        Box::new(Euclid2d)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    pub min_node_size: usize,
    pub max_node_size: usize,
    pub split_sampling: usize,
}

pub const FIXED_PARAMS: TreeParams = TreeParams {
    min_node_size: 5,
    max_node_size: 100,
    split_sampling: 20,
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

pub fn new_tree(policy: Policy, n: usize) -> MTree<Key2d, Active> {
    match policy {
        Policy::Fixed => MTree::with_params(5, 100, 20, Euclid2d),
        Policy::Scaled => MTree::with_capacity_hint(n, Euclid2d),
    }
}

pub fn build_tree(
    policy: Policy,
    n: usize,
    data: &[(Key2d, Active)],
) -> (MTree<Key2d, Active>, Vec<EntryId>) {
    let mut tree = new_tree(policy, n);
    let mut ids = Vec::with_capacity(data.len());
    for &(key, active) in data {
        ids.push(tree.insert(key, active).unwrap());
    }
    (tree, ids)
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
    tree: &MTree<Key2d, Active>,
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
    tree: &MTree<Key2d, Active>,
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
    tree: &MTree<Key2d, Active>,
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
