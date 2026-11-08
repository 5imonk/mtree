//! Latency comparison: mtree k-NN vs Euclidean + metric baselines (2D).
//!
//! ```bash
//! cargo run --release --example knn_baseline_latency_report
//! ```
//!
//! Writes CSV to stdout and `reports/knn_baseline_latency.csv`, plus markdown
//! pivot tables to stderr.
//!
//! Methodology: same points/queries as the query-matrix report; no active filter;
//! 256 queries; 1 warmup + 5 timed rounds (median avg ms/query). Build time is a
//! one-shot side metric per (impl, n). `mtree_needle_hint` is unfair (seed knowledge).
//!
//! Fairer metric peers: vpsearch, vp_tree, ball-tree. Euclidean ceiling: kiddo/rstar.

use ball_tree::BallTree;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use mtree_experiment::{
    build_tree, gen_active_data, measure_query_median, points_f64, prefetch_queries,
    query_points_f64, run_query_batch, sample_query_ids, Policy, QueryMode, KS, LOG_SIZES,
    QUERY_COUNT, TIMED_ROUNDS,
};
use rstar::RTree;
use std::collections::HashSet;
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::num::NonZero;
use std::path::Path;
use std::time::Instant;
use vp_tree::{Distance as VpTreeDistance, Querry, VpTree};
use vpsearch::{BestCandidate, MetricSpace};

#[derive(Clone, Copy, Debug)]
struct MetricPoint {
    xy: [f64; 2],
}

impl MetricSpace for MetricPoint {
    type UserData = ();
    type Distance = f64;

    fn distance(&self, other: &Self, _: &Self::UserData) -> Self::Distance {
        let dx = self.xy[0] - other.xy[0];
        let dy = self.xy[1] - other.xy[1];
        (dx * dx + dy * dy).sqrt()
    }
}

impl VpTreeDistance<MetricPoint> for MetricPoint {
    fn distance(&self, other: &MetricPoint) -> f64 {
        self.distance_heuristic(other).sqrt()
    }

    fn distance_heuristic(&self, other: &MetricPoint) -> f64 {
        let dx = self.xy[0] - other.xy[0];
        let dy = self.xy[1] - other.xy[1];
        dx * dx + dy * dy
    }
}

/// k-NN accumulator for vpsearch (adapted from vpsearch's knn example; not exported).
struct CountBasedNeighborhood {
    max_item_count: usize,
    max_observed_distance: f64,
    distance_x_index: Vec<(f64, usize)>,
}

impl CountBasedNeighborhood {
    fn new(item_count: usize) -> Self {
        Self {
            max_item_count: item_count,
            max_observed_distance: f64::INFINITY,
            distance_x_index: Vec::new(),
        }
    }

    fn insert_index(&mut self, index: usize, distance: f64) {
        self.distance_x_index.push((distance, index));
        if self.distance_x_index.len() > 1 {
            let mut n = self.distance_x_index.len() - 1;
            while n > 0 && self.distance_x_index[n].0 < self.distance_x_index[n - 1].0 {
                self.distance_x_index.swap(n, n - 1);
                n -= 1;
            }
            self.distance_x_index.truncate(self.max_item_count);
        }
        self.max_observed_distance = self.distance_x_index.last().unwrap().0;
    }
}

impl BestCandidate<MetricPoint, ()> for CountBasedNeighborhood {
    type Output = HashSet<usize>;

    fn consider(&mut self, _: &MetricPoint, distance: f64, candidate_index: usize, _: &()) {
        if self.max_item_count == 0 {
            return;
        }
        if distance < self.max_observed_distance || self.distance_x_index.len() < self.max_item_count
        {
            self.insert_index(candidate_index, distance);
        }
    }

    fn distance(&self) -> f64 {
        self.max_observed_distance
    }

    fn result(self, _: &()) -> Self::Output {
        self.distance_x_index
            .into_iter()
            .map(|(_, index)| index)
            .collect()
    }
}

#[derive(Clone, Copy)]
enum ImplKind {
    MtreeNeedleNone,
    MtreeNeedleHint,
    Kiddo,
    Rstar,
    Vpsearch,
    VpTree,
    BallTree,
}

impl ImplKind {
    fn as_str(self) -> &'static str {
        match self {
            ImplKind::MtreeNeedleNone => "mtree_needle_none",
            ImplKind::MtreeNeedleHint => "mtree_needle_hint",
            ImplKind::Kiddo => "kiddo",
            ImplKind::Rstar => "rstar",
            ImplKind::Vpsearch => "vpsearch",
            ImplKind::VpTree => "vp_tree",
            ImplKind::BallTree => "ball_tree",
        }
    }

    fn all() -> [ImplKind; 7] {
        [
            ImplKind::MtreeNeedleNone,
            ImplKind::MtreeNeedleHint,
            ImplKind::Kiddo,
            ImplKind::Rstar,
            ImplKind::Vpsearch,
            ImplKind::VpTree,
            ImplKind::BallTree,
        ]
    }
}

#[derive(Clone)]
struct Row {
    n: usize,
    k: usize,
    impl_name: &'static str,
    avg_ms: f64,
    build_ms: f64,
    ratio_vs_kiddo: f64,
}

fn write_csv_header(out: &mut dyn Write) {
    writeln!(out, "n,k,impl,avg_ms,build_ms,ratio_vs_kiddo").unwrap();
}

fn write_csv_row(out: &mut dyn Write, row: &Row) {
    writeln!(
        out,
        "{},{},{},{:.6},{:.3},{:.4}",
        row.n, row.k, row.impl_name, row.avg_ms, row.build_ms, row.ratio_vs_kiddo
    )
    .unwrap();
}

fn print_markdown_matrices(rows: &[Row]) {
    eprintln!();
    eprintln!("# Markdown matrices (median avg ms / query)");
    eprintln!();
    eprintln!("Metric peers (fairer vs mtree): vpsearch, vp_tree, ball_tree.");
    eprintln!("Euclidean ceiling: kiddo / rstar.");
    eprintln!("mtree_needle_hint is unfair (seed EntryId). Build_ms is side metric only.");
    eprintln!();

    for &k in KS {
        eprintln!("## k={k}");
        eprintln!();
        eprint!("| n |");
        for kind in ImplKind::all() {
            eprint!(" {} |", kind.as_str());
        }
        eprintln!();
        eprint!("| --- |");
        for _ in ImplKind::all() {
            eprint!(" ---: |");
        }
        eprintln!();

        for &n in LOG_SIZES {
            eprint!("| {n} |");
            for kind in ImplKind::all() {
                let ms = rows
                    .iter()
                    .find(|r| r.n == n && r.k == k && r.impl_name == kind.as_str())
                    .map(|r| r.avg_ms);
                match ms {
                    Some(v) => eprint!(" {v:.4} |"),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();
    }

    eprintln!("## ratio_vs_kiddo (higher = faster than kiddo)");
    eprintln!();
    for &k in KS {
        eprintln!("### k={k}");
        eprintln!();
        eprint!("| n |");
        for kind in ImplKind::all() {
            eprint!(" {} |", kind.as_str());
        }
        eprintln!();
        eprint!("| --- |");
        for _ in ImplKind::all() {
            eprint!(" ---: |");
        }
        eprintln!();

        for &n in LOG_SIZES {
            eprint!("| {n} |");
            for kind in ImplKind::all() {
                let ratio = rows
                    .iter()
                    .find(|r| r.n == n && r.k == k && r.impl_name == kind.as_str())
                    .map(|r| r.ratio_vs_kiddo);
                match ratio {
                    Some(v) => eprint!(" {v:.3} |"),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();
    }
}

fn main() {
    let report_dir = Path::new("reports");
    fs::create_dir_all(report_dir).expect("create reports/");
    let csv_path = report_dir.join("knn_baseline_latency.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");

    let mut stdout = std::io::stdout().lock();
    write_csv_header(&mut stdout);
    write_csv_header(&mut file);

    let mut rows = Vec::new();
    let k_nonzero = |k: usize| NonZero::new(k).expect("k > 0");

    for &n in LOG_SIZES {
        let policy = Policy::Scaled;
        let data = gen_active_data(n, 42);
        let pts = points_f64(&data);
        let metric_pts: Vec<MetricPoint> = pts.iter().map(|&xy| MetricPoint { xy }).collect();

        eprint!("building n={n} mtree... ");
        let t0 = Instant::now();
        let (mtree, ids) = build_tree(policy, n, &data);
        let mtree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{mtree_build_ms:.1} ms");

        eprint!("building n={n} kiddo... ");
        let t0 = Instant::now();
        let kiddo_tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&pts);
        let kiddo_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{kiddo_build_ms:.1} ms");

        eprint!("building n={n} rstar... ");
        let t0 = Instant::now();
        let rstar_tree = RTree::bulk_load(pts.clone());
        let rstar_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{rstar_build_ms:.1} ms");

        eprint!("building n={n} vpsearch... ");
        let t0 = Instant::now();
        let vpsearch_tree = vpsearch::Tree::new(&metric_pts);
        let vpsearch_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{vpsearch_build_ms:.1} ms");

        eprint!("building n={n} vp_tree... ");
        let t0 = Instant::now();
        let vp_tree = VpTree::new(metric_pts.clone());
        let vp_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{vp_tree_build_ms:.1} ms");

        eprint!("building n={n} ball_tree... ");
        let t0 = Instant::now();
        let ball_values = vec![(); n];
        let ball_tree = BallTree::new(pts.clone(), ball_values);
        let ball_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{ball_tree_build_ms:.1} ms");

        let query_ids = sample_query_ids(&ids, QUERY_COUNT, 11);
        let queries = prefetch_queries(&mtree, &query_ids);
        let q_f64 = query_points_f64(&queries);
        let q_metric: Vec<MetricPoint> = q_f64.iter().map(|&xy| MetricPoint { xy }).collect();
        let qn = queries.len();

        let build_ms = |kind: ImplKind| -> f64 {
            match kind {
                ImplKind::MtreeNeedleNone | ImplKind::MtreeNeedleHint => mtree_build_ms,
                ImplKind::Kiddo => kiddo_build_ms,
                ImplKind::Rstar => rstar_build_ms,
                ImplKind::Vpsearch => vpsearch_build_ms,
                ImplKind::VpTree => vp_tree_build_ms,
                ImplKind::BallTree => ball_tree_build_ms,
            }
        };

        for &k in KS {
            let mut kiddo_ms = f64::NAN;
            let mut pending: Vec<(ImplKind, f64)> = Vec::with_capacity(7);

            for kind in ImplKind::all() {
                eprint!("  measure n={n} k={k} impl={}... ", kind.as_str());
                let avg_ms = match kind {
                    ImplKind::MtreeNeedleNone => measure_query_median(qn, TIMED_ROUNDS, || {
                        run_query_batch(&mtree, &queries, k, QueryMode::NeedleNone, false);
                    }),
                    ImplKind::MtreeNeedleHint => measure_query_median(qn, TIMED_ROUNDS, || {
                        run_query_batch(&mtree, &queries, k, QueryMode::NeedleHint, false);
                    }),
                    ImplKind::Kiddo => {
                        let qty = k_nonzero(k);
                        measure_query_median(qn, TIMED_ROUNDS, || {
                            for q in &q_f64 {
                                let hits = kiddo_tree.nearest_n::<SquaredEuclidean>(q, qty);
                                black_box(hits);
                            }
                        })
                    }
                    ImplKind::Rstar => measure_query_median(qn, TIMED_ROUNDS, || {
                        for q in &q_f64 {
                            let hits: Vec<_> =
                                rstar_tree.nearest_neighbor_iter(q).take(k).collect();
                            black_box(hits);
                        }
                    }),
                    ImplKind::Vpsearch => measure_query_median(qn, TIMED_ROUNDS, || {
                        for q in &q_metric {
                            let hits = vpsearch_tree.find_nearest_custom(
                                q,
                                &(),
                                CountBasedNeighborhood::new(k),
                            );
                            black_box(hits);
                        }
                    }),
                    ImplKind::VpTree => {
                        let qspec = Querry::k_nearest_neighbors(k).sorted();
                        measure_query_median(qn, TIMED_ROUNDS, || {
                            for q in &q_metric {
                                let hits = vp_tree.querry(q, &qspec);
                                black_box(hits);
                            }
                        })
                    }
                    ImplKind::BallTree => measure_query_median(qn, TIMED_ROUNDS, || {
                        let mut qctx = ball_tree.query();
                        for q in &q_f64 {
                            let hits: Vec<_> = qctx.nn(q).take(k).collect();
                            black_box(hits);
                        }
                    }),
                };
                eprintln!("{avg_ms:.4} ms/q");
                if matches!(kind, ImplKind::Kiddo) {
                    kiddo_ms = avg_ms;
                }
                pending.push((kind, avg_ms));
            }

            for (kind, avg_ms) in pending {
                let ratio = if avg_ms > 0.0 && kiddo_ms.is_finite() {
                    kiddo_ms / avg_ms
                } else {
                    f64::NAN
                };
                let row = Row {
                    n,
                    k,
                    impl_name: kind.as_str(),
                    avg_ms,
                    build_ms: build_ms(kind),
                    ratio_vs_kiddo: ratio,
                };
                write_csv_row(&mut stdout, &row);
                write_csv_row(&mut file, &row);
                rows.push(row);
            }
        }
    }

    print_markdown_matrices(&rows);
    eprintln!("CSV written to {}", csv_path.display());
}
