//! Parametrisierungs-Sweep: Baumgröße `n` × `max_node_size` (min/split abgeleitet).
//!
//! ```bash
//! cargo run --release --example knn_param_sweep_report
//! ```
//!
//! Primärmetrik: k-NN (`k=10`, `filter=off`, `needle_none`). Neben: `build_ms`,
//! insert/erase/update_key (µs/op). Schreibt CSV nach `reports/knn_param_sweep.csv`
//! und Markdown (Top-3 pro `n`, vs Heuristik/fixed) nach stderr.
//!
//! Methodik: 256 Query-EntryIds, 1 Warmup + 5 timed Rounds (Median avg ms/Query).
//! `PARAM_SWEEP_N` (kommagetrennt) überschreibt die Default-`n`-Liste.

use mtree::node_params_for_capacity;
use mtree_experiment::{
    build_tree_with_params_timed, gen_active_data, measure_median, measure_mutate_side,
    params_from_max, prefetch_queries, sample_query_ids, QueryMode, TreeParams, FIXED_PARAMS,
    QUERY_COUNT, TIMED_ROUNDS,
};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

const DEFAULT_SIZES: &[usize] = &[100, 500, 2_000, 10_000, 50_000, 100_000];
const PARAM_MAXES: &[usize] = &[8, 12, 16, 24, 32, 48, 64, 100];
const SWEEP_K: usize = 10;

#[derive(Clone, Debug)]
struct Row {
    n: usize,
    min: usize,
    max: usize,
    split: usize,
    k: usize,
    kind: &'static str,
    avg_ms: f64,
    avg_routing: f64,
    avg_leaves: f64,
    build_ms: f64,
    insert_us: f64,
    erase_us: f64,
    update_us: f64,
    rel_to_best_n: f64,
    rel_to_heuristic: f64,
}

fn sizes_from_env_or_default() -> Vec<usize> {
    match env::var("PARAM_SWEEP_N") {
        Ok(raw) if !raw.trim().is_empty() => raw
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .collect(),
        _ => DEFAULT_SIZES.to_vec(),
    }
}

fn write_csv_header(out: &mut dyn Write) {
    writeln!(
        out,
        "n,min,max,split,k,kind,avg_ms,avg_routing,avg_leaves,build_ms,insert_us,erase_us,update_us,rel_to_best_n,rel_to_heuristic"
    )
    .unwrap();
}

fn write_csv_row(out: &mut dyn Write, row: &Row) {
    writeln!(
        out,
        "{},{},{},{},{},{},{:.6},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.4},{:.4}",
        row.n,
        row.min,
        row.max,
        row.split,
        row.k,
        row.kind,
        row.avg_ms,
        row.avg_routing,
        row.avg_leaves,
        row.build_ms,
        row.insert_us,
        row.erase_us,
        row.update_us,
        row.rel_to_best_n,
        row.rel_to_heuristic
    )
    .unwrap();
}

fn configs_for_n(n: usize) -> Vec<(&'static str, TreeParams)> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for &max in PARAM_MAXES {
        let p = params_from_max(max);
        let key = (p.min_node_size, p.max_node_size, p.split_sampling);
        if seen.insert(key) {
            out.push(("sweep", p));
        }
    }

    let (hmin, hmax, hsplit) = node_params_for_capacity(n);
    let h = TreeParams {
        min_node_size: hmin,
        max_node_size: hmax,
        split_sampling: hsplit,
    };
    let hkey = (h.min_node_size, h.max_node_size, h.split_sampling);
    // Exact √n params when not already a sweep grid point (e.g. max=10 for n=100).
    if seen.insert(hkey) {
        out.push(("heuristic", h));
    }

    // Always measure library default (16,64,16) — may coincide with params_from_max(64).
    out.push(("fixed", FIXED_PARAMS));

    out
}

fn fill_relative_metrics(rows: &mut [Row]) {
    let ns: Vec<usize> = {
        let mut v: Vec<usize> = rows.iter().map(|r| r.n).collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    for n in ns {
        let idxs: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.n == n)
            .map(|(i, _)| i)
            .collect();

        let best_ms = idxs
            .iter()
            .filter(|&&i| rows[i].kind == "sweep" || rows[i].kind == "heuristic")
            .map(|&i| rows[i].avg_ms)
            .fold(f64::INFINITY, f64::min);

        let heur_ms = idxs
            .iter()
            .find(|&&i| {
                let (hmin, hmax, hsplit) = node_params_for_capacity(n);
                rows[i].min == hmin && rows[i].max == hmax && rows[i].split == hsplit
            })
            .map(|&i| rows[i].avg_ms)
            .unwrap_or(f64::NAN);

        for i in idxs {
            rows[i].rel_to_best_n = if best_ms.is_finite() && best_ms > 0.0 {
                rows[i].avg_ms / best_ms
            } else {
                f64::NAN
            };
            rows[i].rel_to_heuristic = if heur_ms.is_finite() && heur_ms > 0.0 {
                rows[i].avg_ms / heur_ms
            } else {
                f64::NAN
            };
        }
    }
}

fn print_markdown(rows: &[Row]) {
    eprintln!();
    eprintln!("# Param sweep · k={SWEEP_K} · needle_none · filter=off");
    eprintln!();
    eprintln!("Top-3 (by avg_ms) per n among sweep+heuristic; vs heuristic & fixed.");
    eprintln!();

    let mut ns: Vec<usize> = rows.iter().map(|r| r.n).collect();
    ns.sort_unstable();
    ns.dedup();

    for n in ns {
        let (hmin, hmax, hsplit) = node_params_for_capacity(n);
        let mut candidates: Vec<&Row> = rows
            .iter()
            .filter(|r| r.n == n && (r.kind == "sweep" || r.kind == "heuristic"))
            .collect();
        candidates.sort_by(|a, b| {
            a.avg_ms
                .partial_cmp(&b.avg_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let heur = rows
            .iter()
            .find(|r| r.n == n && r.min == hmin && r.max == hmax && r.split == hsplit);
        let fixed = rows.iter().find(|r| r.n == n && r.kind == "fixed");

        eprintln!("## n={n} · heuristic=({hmin},{hmax},{hsplit})");
        eprintln!();
        eprintln!("| rank | kind | min | max | split | avg_ms | rel_best | vs_heur | build_ms |");
        eprintln!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for (rank, r) in candidates.iter().take(3).enumerate() {
            let vs_heur = if let Some(h) = heur {
                if r.avg_ms > 0.0 {
                    h.avg_ms / r.avg_ms
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            eprintln!(
                "| {} | {} | {} | {} | {} | {:.4} | {:.3} | {:.3}× | {:.1} |",
                rank + 1,
                r.kind,
                r.min,
                r.max,
                r.split,
                r.avg_ms,
                r.rel_to_best_n,
                vs_heur,
                r.build_ms
            );
        }
        if let (Some(h), Some(f)) = (heur, fixed) {
            let best = candidates.first().copied();
            let speedup_vs_heur = best
                .map(|b| if b.avg_ms > 0.0 { h.avg_ms / b.avg_ms } else { f64::NAN })
                .unwrap_or(f64::NAN);
            let speedup_vs_fixed = best
                .map(|b| if b.avg_ms > 0.0 { f.avg_ms / b.avg_ms } else { f64::NAN })
                .unwrap_or(f64::NAN);
            eprintln!();
            eprintln!(
                "heuristic {:.4} ms/q · fixed {:.4} ms/q · best speedup vs heur {:.3}× · vs fixed {:.3}×",
                h.avg_ms, f.avg_ms, speedup_vs_heur, speedup_vs_fixed
            );
        }
        eprintln!();
    }
}

fn main() {
    let sizes = sizes_from_env_or_default();
    assert!(!sizes.is_empty(), "PARAM_SWEEP_N produced empty size list");

    let report_dir = Path::new("reports");
    fs::create_dir_all(report_dir).expect("create reports/");
    let csv_path = report_dir.join("knn_param_sweep.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");
    let mut stdout = std::io::stdout().lock();
    write_csv_header(&mut stdout);
    write_csv_header(&mut file);

    let mut rows = Vec::new();

    for &n in &sizes {
        let data = gen_active_data(n, 42);
        let configs = configs_for_n(n);
        eprintln!("n={n} · {} configs", configs.len());

        for (kind, params) in configs {
            eprint!(
                "  build kind={kind} min={} max={} split={}... ",
                params.min_node_size, params.max_node_size, params.split_sampling
            );
            let (mut tree, ids, build_ms) = build_tree_with_params_timed(params, &data);
            eprintln!("{build_ms:.1} ms");

            let query_ids = sample_query_ids(&ids, QUERY_COUNT, 11);
            let queries = prefetch_queries(&tree, &query_ids);

            eprint!("    knn k={SWEEP_K}... ");
            let m = measure_median(
                &tree,
                &queries,
                SWEEP_K,
                QueryMode::NeedleNone,
                false,
                TIMED_ROUNDS,
            );
            eprintln!("{:.4} ms/q", m.avg_ms);

            eprint!("    mutate side... ");
            let side = measure_mutate_side(&mut tree, &ids, &data, TIMED_ROUNDS);
            eprintln!(
                "ins={:.1}µs erase={:.1}µs upd={:.1}µs",
                side.insert_us, side.erase_us, side.update_us
            );

            rows.push(Row {
                n,
                min: params.min_node_size,
                max: params.max_node_size,
                split: params.split_sampling,
                k: SWEEP_K,
                kind,
                avg_ms: m.avg_ms,
                avg_routing: m.avg_routing,
                avg_leaves: m.avg_leaves,
                build_ms,
                insert_us: side.insert_us,
                erase_us: side.erase_us,
                update_us: side.update_us,
                rel_to_best_n: f64::NAN,
                rel_to_heuristic: f64::NAN,
            });
        }
    }

    fill_relative_metrics(&mut rows);

    for row in &rows {
        write_csv_row(&mut stdout, row);
        write_csv_row(&mut file, row);
    }

    print_markdown(&rows);
    eprintln!("CSV written to {}", csv_path.display());
}
