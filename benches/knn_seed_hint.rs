// Copyright (c) 2020 CN Group, TU Wien
// Benchmark: plain k-NN with vs without seed hint (experiment branch).
//
// Run: cargo bench --bench knn_seed_hint
// Groups:
// - knn_seed_hint: space needles; modes none/nearest/kth/random
// - knn_from_entry_vs_needle: stored entries; from_entry / needle_none / needle_hint
// Policies: fixed (5,100,20) vs scaled (node_params_for_capacity(n)).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mtree::distance::Distance;
use mtree::{EntryId, KnnConfig, MTree};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Duration;

const NEEDLE_COUNT: usize = 256;

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
}

fn new_tree(policy: Policy, n: usize) -> Tree {
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

struct Prepared {
    tree: Tree,
    ids: Vec<EntryId>,
    needles: Vec<(i64, i64)>,
    nearest_ids: Vec<EntryId>,
    kth_ids: HashMap<usize, Vec<EntryId>>,
}

fn prepare(n: usize, policy: Policy, ks: &[usize]) -> Prepared {
    let data = gen_data(n, 42);
    let needles = gen_needles(NEEDLE_COUNT, 7);
    let mut tree = new_tree(policy, n);
    let mut ids = Vec::with_capacity(n);
    for (k, v) in &data {
        ids.push(tree.insert(*k, v.clone()).unwrap());
    }
    let nearest_ids: Vec<_> = needles
        .iter()
        .map(|needle| tree.knn_search(needle, 1, ())[0].0)
        .collect();
    let mut kth_ids = HashMap::new();
    for &k in ks {
        let ids_for_k: Vec<_> = needles
            .iter()
            .map(|needle| {
                let hits = tree.knn_search(needle, k, ());
                hits.last().expect("knn_search returned empty").0
            })
            .collect();
        kth_ids.insert(k, ids_for_k);
    }
    Prepared {
        tree,
        ids,
        needles,
        nearest_ids,
        kth_ids,
    }
}

fn bench_knn_seed_hint(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_seed_hint");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let ks = [1usize, 10, 50];
    let sizes = [100usize, 200, 500, 1_000, 10_000, 50_000, 100_000];

    for &n in &sizes {
        for policy in [Policy::Fixed, Policy::Scaled] {
            let prep = prepare(n, policy, &ks);
            let mut rng = StdRng::seed_from_u64(99);
            let prefix = format!("n={n}/policy={}/k=", policy.as_str());

            for &k in &ks {
                let id_none = BenchmarkId::new(format!("{prefix}{k}"), "none");
                group.bench_with_input(id_none, &prep, |b, prep| {
                    b.iter(|| {
                        for needle in &prep.needles {
                            black_box(prep.tree.knn_search(needle, k, ()));
                        }
                    })
                });

                let id_nearest = BenchmarkId::new(format!("{prefix}{k}"), "nearest");
                group.bench_with_input(id_nearest, &prep, |b, prep| {
                    b.iter(|| {
                        for (needle, seed) in prep.needles.iter().zip(prep.nearest_ids.iter()) {
                            black_box(
                                prep.tree
                                    .knn_search(needle, k, KnnConfig::new().seed(*seed)),
                            );
                        }
                    })
                });

                let kth_seeds = prep.kth_ids.get(&k).expect("kth ids").clone();
                let id_kth = BenchmarkId::new(format!("{prefix}{k}"), "kth");
                group.bench_function(id_kth, |b| {
                    b.iter(|| {
                        for (needle, seed) in prep.needles.iter().zip(kth_seeds.iter()) {
                            black_box(
                                prep.tree
                                    .knn_search(needle, k, KnnConfig::new().seed(*seed)),
                            );
                        }
                    })
                });

                let random_seeds: Vec<EntryId> = (0..prep.needles.len())
                    .map(|_| prep.ids[rng.gen_range(0..prep.ids.len())])
                    .collect();
                let id_random = BenchmarkId::new(format!("{prefix}{k}"), "random");
                group.bench_function(id_random, |b| {
                    b.iter(|| {
                        for (needle, seed) in prep.needles.iter().zip(random_seeds.iter()) {
                            black_box(
                                prep.tree
                                    .knn_search(needle, k, KnnConfig::new().seed(*seed)),
                            );
                        }
                    })
                });
            }
        }
    }

    group.finish();
}

struct PreparedFromEntry {
    tree: Tree,
    /// Sampled query entries with prefetched keys.
    queries: Vec<(EntryId, (i64, i64))>,
}

fn prepare_from_entry(n: usize, policy: Policy) -> PreparedFromEntry {
    let data = gen_data(n, 42);
    let mut tree = new_tree(policy, n);
    let mut ids = Vec::with_capacity(n);
    for (k, v) in &data {
        ids.push(tree.insert(*k, v.clone()).unwrap());
    }
    let mut rng = StdRng::seed_from_u64(11);
    let take = NEEDLE_COUNT.min(ids.len());
    let queries: Vec<_> = (0..take)
        .map(|_| {
            let id = ids[rng.gen_range(0..ids.len())];
            let key = tree.key(id).expect("id");
            (id, key)
        })
        .collect();
    PreparedFromEntry { tree, queries }
}

fn bench_knn_from_entry_vs_needle(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_from_entry_vs_needle");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let ks = [1usize, 10, 50];
    let sizes = [100usize, 200, 500, 1_000, 10_000, 50_000, 100_000];

    for &n in &sizes {
        for policy in [Policy::Fixed, Policy::Scaled] {
            let prep = prepare_from_entry(n, policy);
            let prefix = format!("n={n}/policy={}/k=", policy.as_str());

            for &k in &ks {
                let id_none = BenchmarkId::new(format!("{prefix}{k}"), "needle_none");
                group.bench_with_input(id_none, &prep, |b, prep| {
                    b.iter(|| {
                        for (_, key) in &prep.queries {
                            black_box(prep.tree.knn_search(key, k, ()));
                        }
                    })
                });

                let id_hint = BenchmarkId::new(format!("{prefix}{k}"), "needle_hint");
                group.bench_with_input(id_hint, &prep, |b, prep| {
                    b.iter(|| {
                        for (id, key) in &prep.queries {
                            black_box(
                                prep.tree
                                    .knn_search(key, k, KnnConfig::new().seed(*id)),
                            );
                        }
                    })
                });

                let id_from = BenchmarkId::new(format!("{prefix}{k}"), "from_entry");
                group.bench_with_input(id_from, &prep, |b, prep| {
                    b.iter(|| {
                        for (id, _) in &prep.queries {
                            black_box(prep.tree.knn_from_entry(*id, k, false, ()));
                        }
                    })
                });
            }
        }
    }

    group.finish();
}

/// Subset matrix matching the objective report (scaled, modes × filter).
fn bench_knn_query_variants(c: &mut Criterion) {
    use mtree_experiment::{
        build_tree, gen_active_data, prefetch_queries, run_query_batch, sample_query_ids, Policy,
        QueryMode, QUERY_COUNT,
    };

    let mut group = c.benchmark_group("knn_query_variants");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let sizes = [500usize, 5_000, 50_000];
    let ks = [1usize, 50];

    for &n in &sizes {
        let data = gen_active_data(n, 42);
        let (tree, ids) = build_tree(Policy::Scaled, n, &data);
        let query_ids = sample_query_ids(&ids, QUERY_COUNT, 11);
        let queries = prefetch_queries(&tree, &query_ids);

        for &k in &ks {
            for filter in [false, true] {
                for mode in QueryMode::all() {
                    let filter_label = if filter { "on" } else { "off" };
                    let id = BenchmarkId::new(
                        format!("n={n}/k={k}/filter={filter_label}"),
                        mode.as_str(),
                    );
                    group.bench_with_input(id, &(&tree, &queries), |b, (tree, queries)| {
                        b.iter(|| {
                            run_query_batch(tree, queries, k, mode, filter);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_knn_seed_hint,
    bench_knn_from_entry_vs_needle,
    bench_knn_query_variants
);
criterion_main!(benches);
