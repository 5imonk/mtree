//! Stream-Clustering-Mix: Query + Replace bei n ∈ {50, 200, 1000}.
//!
//! ```bash
//! cargo run --release --example knn_stream_mix_report
//! ```
//!
//! Simuliert den Hotpath aus `mtree` docs/PERF_HOTPATH_STREAM.md: Single-Owner-Loop,
//! pro Zyklus optional k-NN und/oder `update_key`. Schreibt CSV nach
//! `reports/knn_stream_mix.csv` (und stdout) sowie Markdown nach stderr.
//!
//! Methodik: scaled node params, monomorphe Euclid2d, 1 Warmup + 5 timed Rounds
//! à `OPS_PER_ROUND` Zyklen (Median avg ms/Zyklus). Mixes: query_only,
//! replace_only, query_replace; k ∈ {1,10}; Modes needle_none / needle_hint.

use mtree::KnnConfig;
use mtree_experiment::{
    build_tree, gen_active_data, measure_query_median, sample_query_ids, Policy, QueryMode,
    QUERY_COUNT, TIMED_ROUNDS,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;

const STREAM_SIZES: &[usize] = &[50, 200, 1_000];
const STREAM_KS: &[usize] = &[1, 10];
const OPS_PER_ROUND: usize = 2_048;

#[derive(Clone, Copy, Debug)]
enum MixKind {
    QueryOnly,
    ReplaceOnly,
    QueryReplace,
}

impl MixKind {
    fn as_str(self) -> &'static str {
        match self {
            MixKind::QueryOnly => "query_only",
            MixKind::ReplaceOnly => "replace_only",
            MixKind::QueryReplace => "query_replace",
        }
    }

    fn all() -> [MixKind; 3] {
        [
            MixKind::QueryOnly,
            MixKind::ReplaceOnly,
            MixKind::QueryReplace,
        ]
    }
}

struct Row {
    n: usize,
    k: usize,
    mode: &'static str,
    mix: &'static str,
    avg_ms: f64,
    ops_per_s: f64,
    ops: usize,
}

fn write_csv_header(w: &mut impl Write) {
    writeln!(w, "n,k,mode,mix,avg_ms,ops_per_s,ops").unwrap();
}

fn write_csv_row(w: &mut impl Write, r: &Row) {
    writeln!(
        w,
        "{},{},{},{},{:.6},{:.1},{}",
        r.n, r.k, r.mode, r.mix, r.avg_ms, r.ops_per_s, r.ops
    )
    .unwrap();
}

fn fresh_key(rng: &mut StdRng, occupied: &HashSet<(i64, i64)>) -> (i64, i64) {
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

fn print_markdown(rows: &[Row]) {
    eprintln!();
    eprintln!("# Stream-Mix (Query + Replace)");
    eprintln!();
    eprintln!(
        "Median avg ms/cycle · {} ops/round · {} timed rounds",
        OPS_PER_ROUND, TIMED_ROUNDS
    );
    eprintln!();
    for &k in STREAM_KS {
        for mode in ["needle_none", "needle_hint"] {
            eprintln!("## k={k} · {mode}");
            eprintln!();
            eprintln!("| n | query_only | replace_only | query_replace | qr ops/s |");
            eprintln!("| --- | ---: | ---: | ---: | ---: |");
            for &n in STREAM_SIZES {
                let cell = |mix: &str| {
                    rows.iter()
                        .find(|r| r.n == n && r.k == k && r.mode == mode && r.mix == mix)
                        .map(|r| format!("{:.4}", r.avg_ms))
                        .unwrap_or_else(|| "—".into())
                };
                let qr_ops = rows
                    .iter()
                    .find(|r| {
                        r.n == n && r.k == k && r.mode == mode && r.mix == "query_replace"
                    })
                    .map(|r| format!("{:.0}", r.ops_per_s))
                    .unwrap_or_else(|| "—".into());
                eprintln!(
                    "| {n} | {} | {} | {} | {qr_ops} |",
                    cell("query_only"),
                    cell("replace_only"),
                    cell("query_replace"),
                );
            }
            eprintln!();
        }
    }
}

fn main() {
    let report_dir = Path::new("reports");
    fs::create_dir_all(report_dir).expect("create reports/");
    let csv_path = report_dir.join("knn_stream_mix.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");
    let mut stdout = std::io::stdout().lock();
    write_csv_header(&mut stdout);
    write_csv_header(&mut file);

    let mut rows = Vec::new();
    let policy = Policy::Scaled;

    for &n in STREAM_SIZES {
        let data = gen_active_data(n, 42);
        let (mut tree, ids) = build_tree(policy, n, &data);
        let mut occupied: HashSet<(i64, i64)> = data.iter().map(|&(k, _)| k).collect();
        let query_ids = sample_query_ids(&ids, QUERY_COUNT.min(n), 11);
        let mut rng = StdRng::seed_from_u64(99);

        for &k in STREAM_KS {
            for mode in [QueryMode::NeedleNone, QueryMode::NeedleHint] {
                for mix in MixKind::all() {
                    // Restore initial keys so mixes stay comparable.
                    for (i, &(key, _)) in data.iter().enumerate() {
                        let id = ids[i];
                        let cur = tree.key(id).expect("id");
                        if cur != key {
                            occupied.remove(&cur);
                            tree.update_key(id, key).expect("restore key");
                            occupied.insert(key);
                        }
                    }

                    eprint!(
                        "n={n} k={k} mode={} mix={}... ",
                        mode.as_str(),
                        mix.as_str()
                    );

                    let avg_ms = measure_query_median(OPS_PER_ROUND, TIMED_ROUNDS, || {
                        for op in 0..OPS_PER_ROUND {
                            let id = ids[op % ids.len()];
                            match mix {
                                MixKind::QueryOnly => {
                                    let qid = query_ids[op % query_ids.len()];
                                    let key = tree.key(qid).expect("query id");
                                    let hits = match mode {
                                        QueryMode::NeedleNone => tree.knn_search(&key, k, ()),
                                        QueryMode::NeedleHint => tree.knn_search(
                                            &key,
                                            k,
                                            KnnConfig::new().seed(qid),
                                        ),
                                        QueryMode::FromEntry => unreachable!(),
                                    };
                                    black_box(hits);
                                }
                                MixKind::ReplaceOnly => {
                                    let new_key = fresh_key(&mut rng, &occupied);
                                    let old = tree.key(id).expect("id");
                                    occupied.remove(&old);
                                    tree.update_key(id, new_key).expect("update_key");
                                    occupied.insert(new_key);
                                }
                                MixKind::QueryReplace => {
                                    let key = tree.key(id).expect("id");
                                    let hits = match mode {
                                        QueryMode::NeedleNone => tree.knn_search(&key, k, ()),
                                        QueryMode::NeedleHint => tree.knn_search(
                                            &key,
                                            k,
                                            KnnConfig::new().seed(id),
                                        ),
                                        QueryMode::FromEntry => unreachable!(),
                                    };
                                    black_box(hits);
                                    let new_key = fresh_key(&mut rng, &occupied);
                                    occupied.remove(&key);
                                    tree.update_key(id, new_key).expect("update_key");
                                    occupied.insert(new_key);
                                }
                            }
                        }
                    });

                    let ops_per_s = if avg_ms > 0.0 {
                        1000.0 / avg_ms
                    } else {
                        f64::NAN
                    };
                    eprintln!("{avg_ms:.4} ms/op ({ops_per_s:.0} ops/s)");
                    let row = Row {
                        n,
                        k,
                        mode: mode.as_str(),
                        mix: mix.as_str(),
                        avg_ms,
                        ops_per_s,
                        ops: OPS_PER_ROUND,
                    };
                    write_csv_row(&mut stdout, &row);
                    write_csv_row(&mut file, &row);
                    rows.push(row);
                }
            }
        }
    }

    print_markdown(&rows);
    eprintln!("CSV written to {}", csv_path.display());
}

