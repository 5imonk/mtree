//! RkNN mutate-maintenance cost matrix: n × k × op (insert|erase).
//!
//! ```bash
//! cargo run --release --example rknn_mutate_cost_report
//! ```
//!
//! Writes CSV to stdout and `reports/rknn_mutate_cost.csv`, plus markdown
//! pivot tables to stderr.
//!
//! Methodology (mtree 2.1 opt-in kNN-graph):
//! - Build graph + plain trees with `with_capacity_hint(_knn_graph)`.
//! - Per op, batch of 64 mutations, 1 warmup + 5 timed rounds → median µs/op.
//! - Phases:
//!   - `discover`: insert → `rknn_search` before mutate; erase → reverse via
//!     forward-list scan (`knn_graph_neighbors`)
//!   - `mutate_graph` / `mutate_plain`: geometric insert|erase (± graph maintain)
//!   - `refresh`: forward `knn_from_entry` on each affected (+ `knn_stats`
//!     routing/leaves as distance-work proxy)
//! - `selective_total_us = discover + plain_mutate + refresh` vs `mutate_us`
//!   (built-in graph maintenance).

use mtree_experiment::{
    measure_rknn_mutate_phases, MutatePhaseMetrics, RknnMutateOp, RKNN_KS, RKNN_LOG_SIZES,
    TIMED_ROUNDS,
};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone)]
struct Row {
    n: usize,
    k: usize,
    op: &'static str,
    mutate_us: f64,
    plain_mutate_us: f64,
    discover_us: f64,
    affected_count: f64,
    refresh_us_total: f64,
    avg_routing: f64,
    avg_leaves: f64,
    selective_total_us: f64,
}

fn write_csv_header(out: &mut dyn Write) {
    writeln!(
        out,
        "n,k,op,mutate_us,plain_mutate_us,discover_us,affected_count,refresh_us_total,avg_routing,avg_leaves,selective_total_us"
    )
    .unwrap();
}

fn write_csv_row(out: &mut dyn Write, row: &Row) {
    writeln!(
        out,
        "{},{},{},{:.4},{:.4},{:.4},{:.2},{:.4},{:.2},{:.2},{:.4}",
        row.n,
        row.k,
        row.op,
        row.mutate_us,
        row.plain_mutate_us,
        row.discover_us,
        row.affected_count,
        row.refresh_us_total,
        row.avg_routing,
        row.avg_leaves,
        row.selective_total_us
    )
    .unwrap();
}

fn find_metric<'a>(rows: &'a [Row], n: usize, k: usize, op: &str) -> Option<&'a Row> {
    rows.iter()
        .find(|r| r.n == n && r.k == k && r.op == op)
}

fn print_markdown_matrices(rows: &[Row]) {
    eprintln!();
    eprintln!("# Markdown matrices (median µs / op)");
    eprintln!();

    for op in RknnMutateOp::all() {
        eprintln!("## op={}", op.as_str());
        eprintln!();

        eprintln!("### mutate_us (graph maintain)");
        eprint!("| n |");
        for &k in RKNN_KS {
            eprint!(" k={} |", k);
        }
        eprintln!();
        eprint!("| --- |");
        for _ in RKNN_KS {
            eprint!(" ---: |");
        }
        eprintln!();
        for &n in RKNN_LOG_SIZES {
            eprint!("| {} |", n);
            for &k in RKNN_KS {
                match find_metric(rows, n, k, op.as_str()).map(|r| r.mutate_us) {
                    Some(v) => eprint!(" {:.2} |", v),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();

        eprintln!("### affected_count");
        eprint!("| n |");
        for &k in RKNN_KS {
            eprint!(" k={} |", k);
        }
        eprintln!();
        eprint!("| --- |");
        for _ in RKNN_KS {
            eprint!(" ---: |");
        }
        eprintln!();
        for &n in RKNN_LOG_SIZES {
            eprint!("| {} |", n);
            for &k in RKNN_KS {
                match find_metric(rows, n, k, op.as_str()).map(|r| r.affected_count) {
                    Some(v) => eprint!(" {:.1} |", v),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();

        eprintln!("### refresh_us_total");
        eprint!("| n |");
        for &k in RKNN_KS {
            eprint!(" k={} |", k);
        }
        eprintln!();
        eprint!("| --- |");
        for _ in RKNN_KS {
            eprint!(" ---: |");
        }
        eprintln!();
        for &n in RKNN_LOG_SIZES {
            eprint!("| {} |", n);
            for &k in RKNN_KS {
                match find_metric(rows, n, k, op.as_str()).map(|r| r.refresh_us_total) {
                    Some(v) => eprint!(" {:.2} |", v),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();

        eprintln!("### avg_routing (per affected refresh)");
        eprint!("| n |");
        for &k in RKNN_KS {
            eprint!(" k={} |", k);
        }
        eprintln!();
        eprint!("| --- |");
        for _ in RKNN_KS {
            eprint!(" ---: |");
        }
        eprintln!();
        for &n in RKNN_LOG_SIZES {
            eprint!("| {} |", n);
            for &k in RKNN_KS {
                match find_metric(rows, n, k, op.as_str()).map(|r| r.avg_routing) {
                    Some(v) => eprint!(" {:.1} |", v),
                    None => eprint!(" — |"),
                }
            }
            eprintln!();
        }
        eprintln!();

        eprintln!("### selective_total_us vs mutate_us");
        eprintln!();
        eprint!("| n |");
        for &k in RKNN_KS {
            eprint!(" sel/k={} | graph/k={} |", k, k);
        }
        eprintln!();
        eprint!("| --- |");
        for _ in RKNN_KS {
            eprint!(" ---: | ---: |");
        }
        eprintln!();
        for &n in RKNN_LOG_SIZES {
            eprint!("| {} |", n);
            for &k in RKNN_KS {
                match find_metric(rows, n, k, op.as_str()) {
                    Some(r) => eprint!(" {:.2} | {:.2} |", r.selective_total_us, r.mutate_us),
                    None => eprint!(" — | — |"),
                }
            }
            eprintln!();
        }
        eprintln!();
    }
}

fn row_from(n: usize, k: usize, op: RknnMutateOp, m: MutatePhaseMetrics) -> Row {
    Row {
        n,
        k,
        op: op.as_str(),
        mutate_us: m.mutate_us,
        plain_mutate_us: m.plain_mutate_us,
        discover_us: m.discover_us,
        affected_count: m.affected_count,
        refresh_us_total: m.refresh_us_total,
        avg_routing: m.avg_routing,
        avg_leaves: m.avg_leaves,
        selective_total_us: m.selective_total_us(),
    }
}

fn main() {
    let report_dir = Path::new("reports");
    fs::create_dir_all(report_dir).expect("create reports/");
    let csv_path = report_dir.join("rknn_mutate_cost.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");

    let mut stdout = std::io::stdout().lock();
    write_csv_header(&mut stdout);
    write_csv_header(&mut file);

    let mut rows = Vec::new();

    for &n in RKNN_LOG_SIZES {
        for &k in RKNN_KS {
            for op in RknnMutateOp::all() {
                eprint!("measuring n={} k={} op={} ... ", n, k, op.as_str());
                let _ = std::io::stderr().flush();
                let m = measure_rknn_mutate_phases(op, n, k, TIMED_ROUNDS);
                let row = row_from(n, k, op, m);
                eprintln!(
                    "mutate={:.2}µs affected={:.1} refresh={:.2}µs sel={:.2}µs",
                    row.mutate_us, row.affected_count, row.refresh_us_total, row.selective_total_us
                );
                write_csv_row(&mut stdout, &row);
                write_csv_row(&mut file, &row);
                rows.push(row);
            }
        }
    }

    print_markdown_matrices(&rows);
    eprintln!("wrote {}", csv_path.display());
}
