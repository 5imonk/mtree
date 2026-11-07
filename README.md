# mtree-experiment

Dünnes Harness-Crate: Benchmarks und Reports für [`mtree`](https://github.com/5imonk/mtree).

Erwartet die Library als Sibling-Checkout unter `../mtree` (path-Dependency mit Feature `knn_stats`).

## Query-Matrix-Report (primär)

Objektiv gemessene Matrix: Baumgröße `n` (log), `k`, Filter (~50 % aktiv) × Modus
(`needle_none` / `needle_hint` / `from_entry`).

```bash
cargo run --release --example knn_query_matrix_report
```

Schreibt CSV nach `reports/knn_query_matrix.csv` (und stdout) sowie Markdown-Matrizen
nach stderr. Methodik: scaled node params, 256 feste Query-`EntryId`s, 1 Warmup +
5 timed Rounds (Median avg ms/Query).

Interaktives Dashboard (nach dem Lauf, Daten inline): Canvas
`canvases/knn-query-performance.canvas.tsx` (Symlink auf Cursors Canvas-Ordner)
neben dem Chat öffnen — Varianten vergleichen oder einzelne Kurven betrachten.

## Weitere Reports / Benches

```bash
cargo run --release --example knn_seed_hint_report
cargo run --release --example knn_from_entry_report
cargo bench --bench knn_seed_hint   # inkl. Gruppe knn_query_variants
cargo bench --bench mtree_benches
```

Alternativ per Git statt path — in `Cargo.toml`:

```toml
mtree = { git = "https://github.com/5imonk/mtree.git", branch = "main", features = ["knn_stats"] }
```

Dieser Ordner liegt auf dem Orphan-Branch `experiment/benches` (kein Library-Quellcode).
