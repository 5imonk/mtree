//! Export Criterion `estimates.json` files into `reports/criterion_benches.csv`.
//!
//! ```bash
//! cargo bench --bench rknn_graph   # or any bench
//! cargo run --release --example export_criterion_csv
//! ```
//!
//! Walks `target/criterion/**/new/estimates.json` (and Cargo's alternate
//! target dir via `CARGO_TARGET_DIR`).

use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct Estimate {
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
    median: Estimate,
    std_dev: Estimate,
}

#[derive(Debug)]
struct Row {
    group: String,
    bench_id: String,
    mean_ns: f64,
    median_ns: f64,
    stddev_ns: f64,
}

fn criterion_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        roots.push(PathBuf::from(dir).join("criterion"));
    }
    roots.push(PathBuf::from("target/criterion"));
    roots
}

fn collect_estimate_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("report") {
                continue;
            }
            collect_estimate_files(&path, out);
        } else if path.file_name().and_then(|s| s.to_str()) == Some("estimates.json") {
            // Only "new" measurements (skip baseline "base").
            if path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                == Some("new")
            {
                out.push(path);
            }
        }
    }
}

fn row_from_path(path: &Path, root: &Path) -> Option<Row> {
    let rel = path.strip_prefix(root).ok()?;
    // group / bench_id... / new / estimates.json
    let mut comps: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    if comps.len() < 3 {
        return None;
    }
    // drop new/estimates.json
    comps.pop(); // estimates.json
    comps.pop(); // new
    let group = comps.first()?.to_string();
    let bench_id = if comps.len() == 1 {
        group.clone()
    } else {
        comps[1..].join("/")
    };

    let text = fs::read_to_string(path).ok()?;
    let est: Estimates = serde_json::from_str(&text).ok()?;
    Some(Row {
        group,
        bench_id,
        mean_ns: est.mean.point_estimate,
        median_ns: est.median.point_estimate,
        stddev_ns: est.std_dev.point_estimate,
    })
}

fn main() {
    let mut files = Vec::new();
    let mut root_used = PathBuf::new();
    for root in criterion_candidates() {
        if !root.is_dir() {
            continue;
        }
        let mut found = Vec::new();
        collect_estimate_files(&root, &mut found);
        if !found.is_empty() {
            root_used = root;
            files = found;
            break;
        }
    }

    if files.is_empty() {
        eprintln!("no Criterion estimates found under:");
        for root in criterion_candidates() {
            eprintln!("  - {}", root.display());
        }
        eprintln!("Run cargo bench first, or set CARGO_TARGET_DIR.");
        std::process::exit(1);
    }

    files.sort();
    let mut rows: Vec<Row> = files
        .iter()
        .filter_map(|p| row_from_path(p, &root_used))
        .collect();
    rows.sort_by(|a, b| (&a.group, &a.bench_id).cmp(&(&b.group, &b.bench_id)));

    let report_dir = Path::new("reports");
    fs::create_dir_all(report_dir).expect("create reports/");
    let csv_path = report_dir.join("criterion_benches.csv");
    let mut file = fs::File::create(&csv_path).expect("create csv");
    let mut stdout = std::io::stdout().lock();

    let header = "group,bench_id,mean_ns,median_ns,stddev_ns";
    writeln!(stdout, "{header}").unwrap();
    writeln!(file, "{header}").unwrap();
    for row in &rows {
        let line = format!(
            "{},{},{:.6},{:.6},{:.6}",
            row.group, row.bench_id, row.mean_ns, row.median_ns, row.stddev_ns
        );
        writeln!(stdout, "{line}").unwrap();
        writeln!(file, "{line}").unwrap();
    }

    eprintln!(
        "exported {} benches from {} → {}",
        rows.len(),
        root_used.display(),
        csv_path.display()
    );
}
