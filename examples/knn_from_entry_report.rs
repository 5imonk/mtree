// Copyright (c) 2020 CN Group, TU Wien
//! Compare k-NN from a stored tree entry vs needle search with/without seed hint.
//!
//! ```bash
//! cargo run --release --example knn_from_entry_report
//! ```
//!
//! Queries are stored `EntryId`s (not random space needles).
//!
//! Modes:
//! - `from_entry`: `knn_from_entry(id, k, false, ())` — Self excluded
//! - `needle_none`: `knn_search(&key, k, ())` — Self included at dist 0
//! - `needle_hint`: `knn_search(&key, k, KnnConfig::new().seed(id))` — Self included
//!
//! Policies: `fixed` (library default 16,64,16) vs `scaled` (`node_params_for_capacity(n)`).

use mtree::distance::Distance;
use mtree::{node_params_for_capacity, EntryId, KnnConfig, MTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use mtree::{reset_knn_visit_stats, take_knn_visit_stats};

const QUERY_COUNT: usize = 256;

#[derive(Clone)]
struct Euclid2d;
impl Distance<(i64, i64)> for Euclid2d {
    type Output = f64;
    fn distance(&self, a: &(i64, i64), b: &(i64, i64)) -> f64 {
        let dx = (a.0 - b.0) as f64;
        let dy = (a.1 - b.1) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

type Tree = mtree::MTree<(i64, i64), String, f64, mtree::stats::DescendantCounter, Euclid2d>;

#[derive(Clone, Copy, Debug)]
struct TreeParams {
    min_node_size: usize,
    max_node_size: usize,
    split_sampling: usize,
}

const FIXED_PARAMS: TreeParams = TreeParams {
    min_node_size: 16,
    max_node_size: 64,
    split_sampling: 16,
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

fn new_tree(policy: Policy, n: usize) -> Tree {
    match policy {
        Policy::Fixed => MTree::with_distance(Euclid2d),
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

fn sample_query_ids(ids: &[EntryId], count: usize, seed: u64) -> Vec<EntryId> {
    let mut rng = StdRng::seed_from_u64(seed);
    let take = count.min(ids.len());
    (0..take)
        .map(|_| ids[rng.gen_range(0..ids.len())])
        .collect()
}

#[derive(Clone, Copy)]
enum Mode {
    FromEntry,
    NeedleNone,
    NeedleHint,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Mode::FromEntry => "from_entry",
            Mode::NeedleNone => "needle_none",
            Mode::NeedleHint => "needle_hint",
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

fn run_mode(
    tree: &Tree,
    queries: &[EntryId],
    k: usize,
    mode: Mode,
) {
    for &id in queries {
        match mode {
            Mode::FromEntry => {
                let _ = tree.knn_from_entry(id, k, false, ());
            }
            Mode::NeedleNone => {
                let key = tree.key(id).expect("query id");
                let _ = tree.knn_search(&key, k, ());
            }
            Mode::NeedleHint => {
                let key = tree.key(id).expect("query id");
                let _ = tree.knn_search(&key, k, KnnConfig::new().seed(id));
            }
        }
    }
}

fn measure(
    tree: &Tree,
    queries: &[EntryId],
    k: usize,
    mode: Mode,
) -> (f64, f64, f64) {
    run_mode(tree, queries, k, mode);

    reset_knn_visit_stats();

    let start = Instant::now();
    run_mode(tree, queries, k, mode);
    let elapsed = start.elapsed();

    let q = queries.len() as f64;
    let s = take_knn_visit_stats();
    let (avg_routing, avg_leaves) = (s.routing_nodes as f64 / q, s.leaves as f64 / q);

    (
        elapsed.as_secs_f64() * 1000.0 / q,
        avg_routing,
        avg_leaves,
    )
}

fn print_row(row: &Row, baseline_ms: f64, baseline_routing: f64) {
    let speedup = if row.avg_ms > 0.0 {
        baseline_ms / row.avg_ms
    } else {
        f64::NAN
    };
    let routing_ratio = if row.avg_routing > 0.0 {
        baseline_routing / row.avg_routing
    } else if baseline_routing == 0.0 {
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

    println!(
        "n,policy,min,max,split,k,mode,avg_ms,avg_routing,avg_leaves,speedup_vs_needle_none,routing_ratio_vs_needle_none"
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
            let queries = sample_query_ids(&ids, QUERY_COUNT, 11);
            eprintln!("done");

            let mut baseline_ms = [0.0f64; 3];
            let mut baseline_routing = [0.0f64; 3];

            for (ki, &k) in ks.iter().enumerate() {
                // Baseline first so speedup columns are defined.
                for mode in [Mode::NeedleNone, Mode::NeedleHint, Mode::FromEntry] {
                    let (avg_ms, avg_routing, avg_leaves) = measure(&tree, &queries, k, mode);
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
                    if matches!(mode, Mode::NeedleNone) {
                        baseline_ms[ki] = row.avg_ms;
                        baseline_routing[ki] = row.avg_routing;
                    }
                    print_row(&row, baseline_ms[ki], baseline_routing[ki]);
                }
            }
        }
    }
}
