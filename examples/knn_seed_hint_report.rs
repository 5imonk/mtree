// Copyright (c) 2020 CN Group, TU Wien
//! Timed + visit-count report for k-NN seed hints (experiment branch).
//!
//! ```bash
//! cargo run --release --example knn_seed_hint_report
//! ```
//!
//! Uses `mtree` with feature `knn_stats` (enabled by this crate).
//!
//! Policies: `fixed` (5,100,20) vs `scaled` (`node_params_for_capacity(n)`).
//! Modes: `none`, `nearest` (1-NN), `kth` (k-th NN), `random` (mean over 8 draws).

use mtree::distance::Distance;
use mtree::{node_params_for_capacity, EntryId, KnnConfig, MTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use mtree::{reset_knn_visit_stats, take_knn_visit_stats};

const NEEDLE_COUNT: usize = 256;
const RANDOM_DRAWS: usize = 8;

#[derive(Clone)]
struct Euclid2d;
impl Distance<(i64, i64)> for Euclid2d {
    type Output = f64;
    fn distance(&self, a: &(i64, i64), b: &(i64, i64)) -> f64 {
        let dx = (a.0 - b.0) as f64;
        let dy = (a.1 - b.1) as f64;
        (dx * dx + dy * dy).sqrt()
    }
    fn clone_box(&self) -> Box<dyn Distance<(i64, i64), Output = f64> + Send + Sync> {
        Box::new(Euclid2d)
    }
}

#[derive(Clone, Copy, Debug)]
struct TreeParams {
    min_node_size: usize,
    max_node_size: usize,
    split_sampling: usize,
}

const FIXED_PARAMS: TreeParams = TreeParams {
    min_node_size: 5,
    max_node_size: 100,
    split_sampling: 20,
};

fn scaled_params(n: usize) -> TreeParams {
    let (min, max, split) = node_params_for_capacity(n);
    TreeParams {
        min_node_size: min,
        max_node_size: max,
        split_sampling: split,
    }
}

#[derive(Clone, Copy)]
enum Policy {
    Fixed,
    Scaled,
}

impl Policy {
    fn as_str(self) -> &'static str {
        match self {
            Policy::Fixed => "fixed",
            Policy::Scaled => "scaled",
        }
    }

    fn params(self, n: usize) -> TreeParams {
        match self {
            Policy::Fixed => FIXED_PARAMS,
            Policy::Scaled => scaled_params(n),
        }
    }
}

fn new_tree(policy: Policy, n: usize) -> MTree<(i64, i64), String> {
    match policy {
        Policy::Fixed => MTree::with_params(5, 100, 20, Euclid2d),
        Policy::Scaled => MTree::with_capacity_hint(n, Euclid2d),
    }
}

fn gen_data(n: usize, seed: u64) -> Vec<((i64, i64), String)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while out.len() < n {
        let k = (
            rng.gen_range(-10_000..10_000),
            rng.gen_range(-10_000..10_000),
        );
        if seen.insert(k) {
            out.push((k, format!("v{}", i)));
            i += 1;
        }
    }
    out
}

fn gen_needles(count: usize, seed: u64) -> Vec<(i64, i64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (
                rng.gen_range(-10_000..10_000),
                rng.gen_range(-10_000..10_000),
            )
        })
        .collect()
}

fn random_ids_for_draw(ids: &[EntryId], n_needles: usize, draw: usize) -> Vec<EntryId> {
    let mut rng = StdRng::seed_from_u64(100 + draw as u64);
    (0..n_needles)
        .map(|_| ids[rng.gen_range(0..ids.len())])
        .collect()
}

#[derive(Clone, Copy)]
enum SeedMode {
    None,
    Nearest,
    Kth,
    Random,
}

impl SeedMode {
    fn as_str(self) -> &'static str {
        match self {
            SeedMode::None => "none",
            SeedMode::Nearest => "nearest",
            SeedMode::Kth => "kth",
            SeedMode::Random => "random",
        }
    }
}

struct Row {
    n: usize,
    policy: &'static str,
    min: usize,
    max: usize,
    split: usize,
    k: usize,
    mode: &'static str,
    avg_ms: f64,
    avg_routing: f64,
    avg_leaves: f64,
}

fn run_queries(
    tree: &MTree<(i64, i64), String>,
    needles: &[(i64, i64)],
    seed_ids: Option<&[EntryId]>,
    k: usize,
) {
    match seed_ids {
        None => {
            for needle in needles {
                let _ = tree.knn_search(needle, k, ());
            }
        }
        Some(seeds) => {
            for (needle, seed) in needles.iter().zip(seeds.iter()) {
                let _ = tree.knn_search(needle, k, KnnConfig::new().seed(*seed));
            }
        }
    }
}

fn measure_with_seeds(
    tree: &MTree<(i64, i64), String>,
    needles: &[(i64, i64)],
    seed_ids: Option<&[EntryId]>,
    k: usize,
    _mode: SeedMode,
) -> (f64, f64, f64) {
    run_queries(tree, needles, seed_ids, k);

    reset_knn_visit_stats();

    let start = Instant::now();
    run_queries(tree, needles, seed_ids, k);
    let elapsed = start.elapsed();

    let q = needles.len() as f64;
    let s = take_knn_visit_stats();
    let (avg_routing, avg_leaves) = (s.routing_nodes as f64 / q, s.leaves as f64 / q);

    (
        elapsed.as_secs_f64() * 1000.0 / q,
        avg_routing,
        avg_leaves,
    )
}

fn measure_random_mean(
    tree: &MTree<(i64, i64), String>,
    needles: &[(i64, i64)],
    ids: &[EntryId],
    k: usize,
) -> (f64, f64, f64) {
    let mut sum_ms = 0.0;
    let mut sum_routing = 0.0;
    let mut sum_leaves = 0.0;
    for draw in 0..RANDOM_DRAWS {
        let seeds = random_ids_for_draw(ids, needles.len(), draw);
        let (ms, routing, leaves) =
            measure_with_seeds(tree, needles, Some(&seeds), k, SeedMode::Random);
        sum_ms += ms;
        sum_routing += routing;
        sum_leaves += leaves;
    }
    let r = RANDOM_DRAWS as f64;
    (sum_ms / r, sum_routing / r, sum_leaves / r)
}

fn print_row(row: &Row, none_ms: f64, none_routing: f64) {
    let speedup = if row.avg_ms > 0.0 {
        none_ms / row.avg_ms
    } else {
        f64::NAN
    };
    let routing_ratio = if row.avg_routing > 0.0 {
        none_routing / row.avg_routing
    } else if none_routing == 0.0 {
        1.0
    } else {
        f64::NAN
    };
    println!(
        "{},{},{},{},{},{},{},{:.6},{:.2},{:.2},{:.4},{:.4}",
        row.n,
        row.policy,
        row.min,
        row.max,
        row.split,
        row.k,
        row.mode,
        row.avg_ms,
        row.avg_routing,
        row.avg_leaves,
        speedup,
        routing_ratio
    );
}

fn main() {
    let sizes = [100usize, 200, 500, 1_000, 10_000, 50_000, 100_000];
    let ks = [1usize, 10, 50];
    let needles = gen_needles(NEEDLE_COUNT, 7);

    println!(
        "n,policy,min,max,split,k,mode,avg_ms,avg_routing,avg_leaves,speedup_vs_none,routing_ratio_vs_none"
    );

    for &n in &sizes {
        let data = gen_data(n, 42);
        for policy in [Policy::Fixed, Policy::Scaled] {
            let params = policy.params(n);
            eprint!(
                "building n={n} policy={} min={} max={} split={}... ",
                policy.as_str(),
                params.min_node_size,
                params.max_node_size,
                params.split_sampling
            );
            let mut tree = new_tree(policy, n);
            let mut ids = Vec::with_capacity(n);
            for (key, val) in &data {
                ids.push(tree.insert(*key, val.clone()).unwrap());
            }
            let nearest_ids: Vec<_> = needles
                .iter()
                .map(|needle| tree.knn_search(needle, 1, ())[0].0)
                .collect();
            eprintln!("done");

            let mut none_ms = [0.0f64; 3];
            let mut none_routing = [0.0f64; 3];

            for (ki, &k) in ks.iter().enumerate() {
                // knn may return fewer than k (e.g. aggressive pruning); use farthest hit as kth seed.
                let kth_ids: Vec<_> = needles
                    .iter()
                    .map(|needle| {
                        let hits = tree.knn_search(needle, k, ());
                        hits.last().expect("knn_search returned empty").0
                    })
                    .collect();

                for mode in [
                    SeedMode::None,
                    SeedMode::Nearest,
                    SeedMode::Kth,
                    SeedMode::Random,
                ] {
                    let (avg_ms, avg_routing, avg_leaves) = match mode {
                        SeedMode::None => measure_with_seeds(&tree, &needles, None, k, mode),
                        SeedMode::Nearest => {
                            measure_with_seeds(&tree, &needles, Some(&nearest_ids), k, mode)
                        }
                        SeedMode::Kth => {
                            measure_with_seeds(&tree, &needles, Some(&kth_ids), k, mode)
                        }
                        SeedMode::Random => measure_random_mean(&tree, &needles, &ids, k),
                    };
                    let row = Row {
                        n,
                        policy: policy.as_str(),
                        min: params.min_node_size,
                        max: params.max_node_size,
                        split: params.split_sampling,
                        k,
                        mode: mode.as_str(),
                        avg_ms,
                        avg_routing,
                        avg_leaves,
                    };
                    if matches!(mode, SeedMode::None) {
                        none_ms[ki] = row.avg_ms;
                        none_routing[ki] = row.avg_routing;
                    }
                    print_row(&row, none_ms[ki], none_routing[ki]);
                }
            }
        }
    }
}
