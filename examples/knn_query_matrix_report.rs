//! Objective k-NN query matrix: n (log) × k × filter × mode.
//!
//! ```bash
//! cargo run --release --example knn_query_matrix_report
//! ```
//!
//! Writes CSV to stdout and `reports/knn_query_matrix.csv`, plus markdown
//! pivot tables to stderr.
//!
//! Methodology: scaled node params, 256 fixed-seed EntryId queries, 1 warmup +
//! 5 timed rounds (median avg ms/query), ~50% active payload filter.

use mtree_experiment::{
    build_tree, gen_active_data, measure_median, prefetch_queries, sample_query_ids, Policy,
    QueryMode, KS, LOG_SIZES, QUERY_COUNT, TIMED_ROUNDS,
};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone)]
struct Row {
    n: usize,
    k: usize,
    filter: bool,
    mode: &'static str,
    min: usize,
    max: usize,
    split: usize,
    avg_ms: f64,
    avg_routing: f64,
    avg_leaves: f64,
    speedup_vs_baseline: f64,
}

fn filter_label(on: bool) -> &'static str {
    if on {
        "on"
    } else {
        "off"
    }
}

fn write_csv_header(out: &mut dyn Write) {
    writeln!(
        out,
        "n,k,filter,mode,min,max,split,avg_ms,avg_routing,avg_leaves,speedup_vs_needle_none_filter_off"
    )
    .unwrap();
}

fn write_csv_row(out: &mut dyn Write, row: &Row) {
    writeln!(
        out,
        "{},{},{},{},{},{},{},{:.6},{:.2},{:.2},{:.4}",
        row.n,
        row.k,
        filter_label(row.filter),
        row.mode,
        row.min,
        row.max,
        row.split,
        row.avg_ms,
        row.avg_routing,
        row.avg_leaves,
        row.speedup_vs_baseline
    )
    .unwrap();
}

fn print_markdown_matrices(rows: &[Row]) {
    eprintln!();
    eprintln!("# Markdown matrices (median avg ms / query)");
    eprintln!();

    for &k in KS {
        for filter in [false, true] {
            eprintln!(
                "## k={} · filter={}",
                k,
                filter_label(filter)
            );
            eprintln!();
            eprint!("| n |");
            for mode in QueryMode::all() {
                eprint!(" {} |", mode.as_str());
            }
            eprintln!();
            eprint!("| --- |");
            for _ in QueryMode::all() {
                eprint!(" ---: |");
            }
            eprintln!();

            for &n in LOG_SIZES {
                eprint!("| {} |", n);
                for mode in QueryMode::all() {
                    let ms = rows
                        .iter()
                        .find(|r| {
                            r.n == n && r.k == k && r.filter == filter && r.mode == mode.as_str()
                        })
                        .map(|r| r.avg_ms);
                    match ms {
                        Some(v) => eprint!(" {:.4} |", v),
                        None => eprint!(" — |"),
                    }
                }
                eprintln!();
            }
            eprintln!();
        }
    }

    eprintln!("## Speedup vs needle_none / filter=off (higher = faster)");
    eprintln!();
    for &k in KS {
        eprintln!("### k={}", k);
        eprintln!();
        eprint!("| n |");
        for filter in [false, true] {
            for mode in QueryMode::all() {
                eprint!(" {}/{} |", mode.as_str(), filter_label(filter));
            }
        }
        eprintln!();
        eprint!("| --- |");
        for _ in 0..(2 * QueryMode::all().len()) {
            eprint!(" ---: |");
        }
        eprintln!();

        for &n in LOG_SIZES {
            eprint!("| {} |", n);
            for filter in [false, true] {
                for mode in QueryMode::all() {
                    let sp = rows
                        .iter()
                        .find(|r| {
                            r.n == n && r.k == k && r.filter == filter && r.mode == mode.as_str()
                        })
                        .map(|r| r.speedup_vs_baseline);
                    match sp {
                        Some(v) => eprint!(" {:.3} |", v),
                        None => eprint!(" — |"),
                    }
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
    let csv_path = report_dir.join("knn_query_matrix.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");

    let mut stdout = std::io::stdout().lock();
    write_csv_header(&mut stdout);
    write_csv_header(&mut file);

    let mut rows = Vec::new();

    for &n in LOG_SIZES {
        let policy = Policy::Scaled;
        let params = policy.params(n);
        let data = gen_active_data(n, 42);
        eprint!(
            "building n={n} policy={} min={} max={} split={}... ",
            policy.as_str(),
            params.min_node_size,
            params.max_node_size,
            params.split_sampling
        );
        let (tree, ids) = build_tree(policy, n, &data);
        let query_ids = sample_query_ids(&ids, QUERY_COUNT, 11);
        let queries = prefetch_queries(&tree, &query_ids);
        eprintln!("done ({} queries)", queries.len());

        for &k in KS {
            let mut baseline_ms = f64::NAN;

            for filter in [false, true] {
                for mode in QueryMode::all() {
                    eprint!(
                        "  measure n={n} k={k} filter={} mode={}... ",
                        filter_label(filter),
                        mode.as_str()
                    );
                    let m = measure_median(&tree, &queries, k, mode, filter, TIMED_ROUNDS);
                    if matches!(mode, QueryMode::NeedleNone) && !filter {
                        baseline_ms = m.avg_ms;
                    }
                    let speedup = if m.avg_ms > 0.0 && baseline_ms.is_finite() {
                        baseline_ms / m.avg_ms
                    } else {
                        f64::NAN
                    };
                    let row = Row {
                        n,
                        k,
                        filter,
                        mode: mode.as_str(),
                        min: params.min_node_size,
                        max: params.max_node_size,
                        split: params.split_sampling,
                        avg_ms: m.avg_ms,
                        avg_routing: m.avg_routing,
                        avg_leaves: m.avg_leaves,
                        speedup_vs_baseline: speedup,
                    };
                    write_csv_row(&mut stdout, &row);
                    write_csv_row(&mut file, &row);
                    rows.push(row);
                    eprintln!("{:.4} ms/q", m.avg_ms);
                }
            }
        }
    }

    print_markdown_matrices(&rows);
    eprintln!("CSV written to {}", csv_path.display());
}
