//! Benchmark: RkNN mutate / discover / refresh phases (mtree 2.1 opt-in graph).
//!
//! Run: cargo bench --bench rknn_graph
//!
//! Mirrors `examples/rknn_mutate_cost_report.rs` with a smaller (n, k) grid.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mtree_experiment::{
    build_graph_tree, build_plain_tree, gen_active_data, refresh_affected_knn, reverse_owners,
    Key2d, RKNN_MUTATE_BATCH,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Duration;

fn fresh_key(rng: &mut StdRng, occupied: &HashSet<Key2d>) -> Key2d {
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

fn bench_mutate_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("rknn_mutate_insert");
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in &[100usize, 500, 1_000] {
        for &k in &[5usize, 20] {
            let data = gen_active_data(n, 42 + n as u64);
            let (graph, _) = build_graph_tree(k, n + 50, &data);
            let (plain, _) = build_plain_tree(n + 50, &data);
            let occupied: HashSet<Key2d> = data.iter().map(|&(key, _)| key).collect();

            group.bench_with_input(
                BenchmarkId::new(format!("graph_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut tree = graph.clone();
                    let mut occ = occupied.clone();
                    let mut rng = StdRng::seed_from_u64(7);
                    b.iter(|| {
                        let key = fresh_key(&mut rng, &occ);
                        occ.insert(key);
                        let id = tree.insert(key, true).unwrap();
                        black_box(id);
                        assert!(tree.erase_by_id(id));
                        occ.remove(&key);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("plain_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut tree = plain.clone();
                    let mut occ = occupied.clone();
                    let mut rng = StdRng::seed_from_u64(7);
                    b.iter(|| {
                        let key = fresh_key(&mut rng, &occ);
                        occ.insert(key);
                        let id = tree.insert(key, true).unwrap();
                        black_box(id);
                        assert!(tree.erase_by_id(id));
                        occ.remove(&key);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_mutate_erase(c: &mut Criterion) {
    let mut group = c.benchmark_group("rknn_mutate_erase");
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in &[100usize, 500, 1_000] {
        for &k in &[5usize, 20] {
            let data = gen_active_data(n, 42 + n as u64);
            let (graph, graph_ids) = build_graph_tree(k, n + 50, &data);
            let (plain, plain_ids) = build_plain_tree(n + 50, &data);

            group.bench_with_input(
                BenchmarkId::new(format!("graph_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut tree = graph.clone();
                    let mut ids = graph_ids.clone();
                    let mut rng = StdRng::seed_from_u64(11);
                    b.iter(|| {
                        let idx = rng.gen_range(0..ids.len());
                        let id = ids[idx];
                        let key = tree.key(id).unwrap();
                        let active = tree.payload(id).unwrap();
                        assert!(tree.erase_by_id(id));
                        ids[idx] = tree.insert(key, active).unwrap();
                        black_box(ids[idx]);
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("plain_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut tree = plain.clone();
                    let mut ids = plain_ids.clone();
                    let mut rng = StdRng::seed_from_u64(11);
                    b.iter(|| {
                        let idx = rng.gen_range(0..ids.len());
                        let id = ids[idx];
                        let key = tree.key(id).unwrap();
                        let active = tree.payload(id).unwrap();
                        assert!(tree.erase_by_id(id));
                        ids[idx] = tree.insert(key, active).unwrap();
                        black_box(ids[idx]);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_discover(c: &mut Criterion) {
    let mut group = c.benchmark_group("rknn_discover");
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in &[100usize, 500, 1_000] {
        for &k in &[5usize, 20] {
            let data = gen_active_data(n, 42 + n as u64);
            let (tree, ids) = build_graph_tree(k, n + 50, &data);
            let occupied: HashSet<Key2d> = data.iter().map(|&(key, _)| key).collect();

            group.bench_with_input(
                BenchmarkId::new(format!("rknn_search_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut rng = StdRng::seed_from_u64(13);
                    b.iter(|| {
                        let key = fresh_key(&mut rng, &occupied);
                        black_box(tree.rknn_search(&key, k).unwrap());
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("reverse_scan_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    let mut rng = StdRng::seed_from_u64(17);
                    b.iter(|| {
                        let idx = rng.gen_range(0..ids.len());
                        black_box(reverse_owners(&tree, ids[idx], &ids));
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_refresh(c: &mut Criterion) {
    let mut group = c.benchmark_group("rknn_refresh");
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in &[100usize, 500, 1_000] {
        for &k in &[5usize, 20] {
            let data = gen_active_data(n, 42 + n as u64);
            let (tree, ids) = build_graph_tree(k, n + RKNN_MUTATE_BATCH, &data);
            let occupied: HashSet<Key2d> = data.iter().map(|&(key, _)| key).collect();
            let mut rng = StdRng::seed_from_u64(19);
            // Prefetch a representative affected set via rknn_search.
            let needle = fresh_key(&mut rng, &occupied);
            let affected: Vec<_> = tree
                .rknn_search(&needle, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            // Fallback: refresh a small fixed sample if RkNN is empty.
            let targets = if affected.is_empty() {
                ids.iter().copied().take(k.min(ids.len())).collect()
            } else {
                affected
            };

            group.bench_with_input(
                BenchmarkId::new(format!("knn_from_entry_n{n}_k{k}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        refresh_affected_knn(&tree, &targets, k);
                        black_box(());
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mutate_insert,
    bench_mutate_erase,
    bench_discover,
    bench_refresh
);
criterion_main!(benches);
