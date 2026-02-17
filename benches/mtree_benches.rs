// Copyright (c) 2020 CN Group, TU Wien
// Benchmarks: MTree vs naive Vec-based operations

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mtree::distance::{Distance, EuclideanDistance, Point};
use mtree::MTree;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::time::Duration;

/// Euclidean distance for (i64, i64) – same as in tests
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

fn new_tree() -> MTree<(i64, i64), String> {
    MTree::with_params(5, 100, 20, Euclid2d)
}

/// Deterministic data: n points in [-1000, 1000]^2
fn gen_data(n: usize, seed: u64) -> Vec<((i64, i64), String)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let k = (rng.gen_range(-1000..1000), rng.gen_range(-1000..1000));
            (k, format!("v{}", i))
        })
        .collect()
}

// ---- Naive implementations ----

fn naive_range_search(
    data: &[((i64, i64), String)],
    needle: &(i64, i64),
    radius: f64,
) -> Vec<((i64, i64), String)> {
    let dist = Euclid2d;
    data.iter()
        .filter(|(k, _)| dist.distance(k, needle) <= radius)
        .cloned()
        .collect()
}

fn naive_search_min_max(
    data: &[((i64, i64), String)],
    needle: &(i64, i64),
    min_radius: f64,
    max_radius: f64,
) -> Vec<((i64, i64), String)> {
    let dist = Euclid2d;
    data.iter()
        .filter(|(k, _)| {
            let d = dist.distance(k, needle);
            d >= min_radius && d <= max_radius
        })
        .cloned()
        .collect()
}

fn naive_knn_search(
    data: &[((i64, i64), String)],
    needle: &(i64, i64),
    k: usize,
) -> Vec<((i64, i64), String)> {
    let dist = Euclid2d;
    let mut with_dist: Vec<_> = data
        .iter()
        .map(|(k, v)| (dist.distance(k, needle), *k, v.clone()))
        .collect();
    with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    with_dist
        .into_iter()
        .take(k)
        .map(|(_, k, v)| (k, v))
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    for n in [100, 1_000, 10_000, 50_000] {
        let data = gen_data(n, 42);
        group.bench_with_input(BenchmarkId::new("mtree", n), &data, |b, data| {
            b.iter_batched(
                || new_tree(),
                |mut tree| {
                    for (k, v) in data {
                        tree.insert(*k, v.clone());
                    }
                    black_box(tree.size())
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("naive_vec", n), &data, |b, data| {
            b.iter_batched(
                || Vec::with_capacity(data.len()),
                |mut vec| {
                    for (k, v) in data {
                        vec.push((*k, v.clone()));
                    }
                    black_box(vec.len())
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_range_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_search");
    let needle = (0, 0);
    let radius = 200.0;
    for n in [100, 1_000, 10_000] {
        let data = gen_data(n, 123);
        let mut tree = new_tree();
        for (k, v) in &data {
            tree.insert(*k, v.clone());
        }
        let tree = std::sync::Arc::new(std::sync::Mutex::new(tree));
        group.bench_with_input(
            BenchmarkId::new("mtree", n),
            &(tree.clone(), &needle, radius),
            |b, (tree, needle, radius)| {
                b.iter(|| {
                    let t = tree.lock().unwrap();
                    let r: Vec<_> = t.range_search(needle, *radius).collect();
                    black_box(r)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("naive", n),
            &(data.as_slice(), &needle, radius),
            |b, (data, needle, radius)| {
                b.iter(|| {
                    let r = naive_range_search(data, needle, *radius);
                    black_box(r)
                })
            },
        );
    }
    group.finish();
}

fn bench_search_min_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_min_max");
    let needles = [(0, 0), (100, 100)];
    let min_r = 10.0;
    let max_r = 300.0;
    for n in [100, 1_000, 10_000] {
        let data = gen_data(n, 456);
        let mut tree = new_tree();
        for (k, v) in &data {
            tree.insert(*k, v.clone());
        }
        let tree = std::sync::Arc::new(std::sync::Mutex::new(tree));
        group.bench_with_input(
            BenchmarkId::new("mtree", n),
            &(tree.clone(), &needles[0], min_r, max_r),
            |b, (tree, needle, min_r, max_r)| {
                b.iter(|| {
                    let t = tree.lock().unwrap();
                    let r: Vec<_> = t.search(needle, *min_r, *max_r).collect();
                    black_box(r)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("naive", n),
            &(data.as_slice(), &needles[0], min_r, max_r),
            |b, (data, needle, min_r, max_r)| {
                b.iter(|| {
                    let r = naive_search_min_max(data, needle, *min_r, *max_r);
                    black_box(r)
                })
            },
        );
    }
    group.finish();
}

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");
    let needle = (0, 0);
    let k = 50;
    for n in [100, 1_000, 10_000] {
        let data = gen_data(n, 789);
        let mut tree = new_tree();
        for (k, v) in &data {
            tree.insert(*k, v.clone());
        }
        let tree = std::sync::Arc::new(std::sync::Mutex::new(tree));
        let tree_clone = tree.clone();
        group.bench_with_input(BenchmarkId::new("mtree", n), &(), |b, _| {
            b.iter(|| {
                let t = tree_clone.lock().unwrap();
                let r = t.knn_search(&needle, k);
                black_box(r)
            })
        });
        group.bench_with_input(
            BenchmarkId::new("naive", n),
            &(data.as_slice(), k),
            |b, (data, k)| {
                b.iter(|| {
                    let r = naive_knn_search(data, &needle, *k);
                    black_box(r)
                })
            },
        );
    }
    group.finish();
}

fn bench_erase(c: &mut Criterion) {
    let mut group = c.benchmark_group("erase");
    for n in [100, 1_000] {
        let data = gen_data(n, 999);
        let to_erase: Vec<_> = data.iter().take(50.min(n)).map(|(k, _)| *k).collect();
        group.bench_with_input(
            BenchmarkId::new("mtree", n),
            &(data.clone(), to_erase.clone()),
            |b, (data, to_erase)| {
                b.iter_batched(
                    || {
                        let mut tree = new_tree();
                        for (k, v) in data {
                            tree.insert(*k, v.clone());
                        }
                        tree
                    },
                    |mut tree| {
                        for key in to_erase {
                            tree.erase(&key);
                        }
                        black_box(tree.size())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("naive_vec", n),
            &(data.clone(), to_erase.clone()),
            |b, (data, to_erase)| {
                b.iter_batched(
                    || data.clone(),
                    |mut vec| {
                        for key in to_erase {
                            vec.retain(|(k, _)| k != key);
                        }
                        black_box(vec.len())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

// ---- Benchmarks für Point mit SIMD-Optimierung ----

/// Generiert Point-Daten für Benchmarks
fn gen_point_data(n: usize, dims: usize, seed: u64) -> Vec<(Point, String)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let coords: Vec<f64> = (0..dims)
                .map(|_| rng.gen_range(-1000.0..1000.0))
                .collect();
            (Point::new(coords), format!("v{}", i))
        })
        .collect()
}

/// Naive range search für Point (zum Vergleich)
fn naive_point_range_search(
    data: &[(Point, String)],
    needle: &Point,
    radius: f64,
) -> Vec<(Point, String)> {
    let dist = EuclideanDistance;
    data.iter()
        .filter(|(k, _)| dist.distance(k, needle) <= radius)
        .cloned()
        .collect()
}

/// Naive k-NN search für Point (zum Vergleich)
fn naive_point_knn_search(
    data: &[(Point, String)],
    needle: &Point,
    k: usize,
) -> Vec<(Point, String)> {
    let dist = EuclideanDistance;
    let mut with_dist: Vec<_> = data
        .iter()
        .map(|(k, v)| (dist.distance(k, needle), k.clone(), v.clone()))
        .collect();
    with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    with_dist
        .into_iter()
        .take(k)
        .map(|(_, k, v)| (k, v))
        .collect()
}

fn bench_point_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_insert");
    let dims = 3; // 3D points
    for n in [100, 1_000, 10_000] {
        let point_data = gen_point_data(n, dims, 42);
        
        // Benchmark Point mit SIMD
        group.bench_with_input(BenchmarkId::new("point_simd", n), &point_data, |b, data| {
            b.iter_batched(
                || MTree::<Point, String, f64>::with_distance(EuclideanDistance),
                |mut tree| {
                    for (k, v) in data {
                        tree.insert(k.clone(), v.clone());
                    }
                    black_box(tree.size())
                },
                BatchSize::SmallInput,
            )
        });
        
        // Benchmark naive Vec (zum Vergleich)
        group.bench_with_input(BenchmarkId::new("naive_vec", n), &point_data, |b, data| {
            b.iter_batched(
                || Vec::with_capacity(data.len()),
                |mut vec| {
                    for (k, v) in data {
                        vec.push((k.clone(), v.clone()));
                    }
                    black_box(vec.len())
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_point_range_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_range_search");
    let dims = 3;
    let radius = 200.0;
    for n in [100, 1_000, 10_000] {
        let point_data = gen_point_data(n, dims, 123);
        
        // Setup Point tree
        let mut point_tree = MTree::<Point, String, f64>::with_distance(EuclideanDistance);
        let point_needle = Point::new(vec![0.0, 0.0, 0.0]);
        for (k, v) in &point_data {
            point_tree.insert(k.clone(), v.clone());
        }
        let point_tree = std::sync::Arc::new(std::sync::Mutex::new(point_tree));
        
        // Benchmark Point mit SIMD
        group.bench_with_input(
            BenchmarkId::new("point_simd", n),
            &(point_tree.clone(), &point_needle, radius),
            |b, (tree, needle, radius)| {
                b.iter(|| {
                    let t = tree.lock().unwrap();
                    let r: Vec<_> = t.range_search(needle, *radius).collect();
                    black_box(r)
                })
            },
        );
        
        // Benchmark naive (zum Vergleich)
        group.bench_with_input(
            BenchmarkId::new("naive", n),
            &(point_data.as_slice(), &point_needle, radius),
            |b, (data, needle, radius)| {
                b.iter(|| {
                    let r = naive_point_range_search(data, needle, *radius);
                    black_box(r)
                })
            },
        );
    }
    group.finish();
}

fn bench_point_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_knn_search");
    let dims = 3;
    let k = 50;
    for n in [100, 1_000, 10_000] {
        let point_data = gen_point_data(n, dims, 789);
        
        // Setup Point tree
        let mut point_tree = MTree::<Point, String, f64>::with_distance(EuclideanDistance);
        let point_needle = Point::new(vec![0.0, 0.0, 0.0]);
        for (k, v) in &point_data {
            point_tree.insert(k.clone(), v.clone());
        }
        let point_tree = std::sync::Arc::new(std::sync::Mutex::new(point_tree));
        
        // Benchmark Point mit SIMD
        let point_tree_clone = point_tree.clone();
        group.bench_with_input(BenchmarkId::new("point_simd", n), &(), |b, _| {
            b.iter(|| {
                let t = point_tree_clone.lock().unwrap();
                let r = t.knn_search(&point_needle, k);
                black_box(r)
            })
        });
        
        // Benchmark naive (zum Vergleich)
        group.bench_with_input(
            BenchmarkId::new("naive", n),
            &(point_data.as_slice(), k),
            |b, (data, k)| {
                b.iter(|| {
                    let r = naive_point_knn_search(data, &point_needle, *k);
                    black_box(r)
                })
            },
        );
    }
    group.finish();
}

fn bench_point_erase(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_erase");
    let dims = 3;
    for n in [100, 1_000] {
        let point_data = gen_point_data(n, dims, 999);
        let to_erase: Vec<_> = point_data.iter().take(50.min(n)).map(|(k, _)| k.clone()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("point", n),
            &(point_data.clone(), to_erase.clone()),
            |b, (data, to_erase)| {
                b.iter_batched(
                    || {
                        let mut tree = MTree::<Point, String, f64>::with_distance(EuclideanDistance);
                        for (k, v) in data {
                            tree.insert(k.clone(), v.clone());
                        }
                        tree
                    },
                    |mut tree| {
                        for key in to_erase {
                            tree.erase(&key);
                        }
                        black_box(tree.size())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}


criterion_group!(
    name = benches;
    config = Criterion::default().
        sample_size(100).
        warm_up_time(Duration::from_secs(3)).
        measurement_time(Duration::from_secs(10));
    targets = bench_insert,
    bench_range_search,
    bench_search_min_max,
    bench_knn_search,
    bench_erase,
    bench_point_insert,
    bench_point_range_search,
    bench_point_knn_search,
    bench_point_erase,
);
criterion_main!(benches);
