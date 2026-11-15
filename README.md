# mtree-experiment

Dünnes Harness-Crate: Benchmarks und Reports für [`mtree`](https://github.com/5imonk/mtree).

Erwartet die Library als Sibling-Checkout unter `../mtree` (path-Dependency mit Feature `knn_stats`).

## Visualisierungs-Hub (standalone)

Cursor-unabhängiges Dashboard über alle Report-CSVs und exportierte Criterion-Benches:

```bash
# aus Repo-Root
python3 -m http.server 8765
# Browser: http://localhost:8765/viz/
```

Hash-Navigation: Query Matrix, Baseline, Stream Mix, Param Sweep, RkNN Mutate,
Seed Hint, From Entry, Criterion Benches. Fehlende CSVs werden mit dem
Erzeugungsbefehl angezeigt.

## Query-Matrix-Report (primär)

Objektiv gemessene Matrix: Baumgröße `n` (log), `k`, Filter (~50 % aktiv) × Modus
(`needle_none` / `needle_hint` / `from_entry`).

```bash
cargo run --release --example knn_query_matrix_report
```

Schreibt CSV nach `reports/knn_query_matrix.csv` (und stdout) sowie Markdown-Matrizen
nach stderr. Methodik: scaled node params, 256 feste Query-`EntryId`s, 1 Warmup +
5 timed Rounds (Median avg ms/Query). Viz: Hub → Query Matrix.

## Baseline-Latency vs. etablierte Libraries

Vergleicht mtree (`needle_none`, plus unfair `needle_hint`) auf derselben
2D-Euclidean-Punktmenge mit:

- **Metrik-Peers (fairer):** `vpsearch`, `vp_tree`, `ball-tree`
- **Euclidean/Spatial-Ceiling:** `kiddo`, `rstar`

Kein Active-Filter.

```bash
cargo run --release --example knn_baseline_latency_report
```

Schreibt CSV nach `reports/knn_baseline_latency.csv` (und stdout) sowie Markdown-Matrizen
nach stderr. Methodik wie oben (256 Queries, 1 Warmup + 5 timed Rounds, Median avg
ms/Query). `build_ms` ist nur Nebenmetrik (einmalige Index-Konstruktion; Bulk vs.
sequentielle Inserts).

Fairness: Headline für Metrikraum = `mtree_needle_none` vs. vpsearch/vp_tree/ball_tree.
`needle_hint` nutzt Seed-Wissen, das die anderen APIs nicht bekommen. kiddo/rstar sind
Euclidean/spatial-Spezialisten (Obergrenze). kiddo 5.3 (ImmutableKdTree) statt 6.x wegen
rustc-MSRV auf diesem Host; `generator` ist auf `0.8.11` gepinnt.

Viz: Hub → Baseline Latency.

## Stream-Mix (Phase 5 / Query+Replace)

Workload-nah für Stream-Clustering: n ∈ {50, 200, 1000}, Mixes `query_only` /
`replace_only` / `query_replace` (`update_key`), Modes `needle_none` /
`needle_hint`, k ∈ {1, 10}. Monomorphe Euclid2d-Distanz.

```bash
cargo run --release --example knn_stream_mix_report
```

Schreibt CSV nach `reports/knn_stream_mix.csv`. Viz: Hub → Stream Mix.

## Parametrisierungs-Sweep (`max` vs Baumgröße)

Systematischer Sweep von `max_node_size` ∈ {8,12,16,24,32,48,64,100} über
`n` ∈ {100,500,2k,10k,50k,100k}; `min`/`split` wie die √n-Heuristik abgeleitet.
Headline: k-NN `k=10`, `needle_none`, filter off. Zusätzlich exakte
`node_params_for_capacity(n)` (Clamp 16…64) und fixed Library-Default
`(16,64,16)`. Nebenmetriken: `build_ms`, insert/erase/update_key (µs/op).

```bash
cargo run --release --example knn_param_sweep_report
# optional: PARAM_SWEEP_N=100,1000,10000 cargo run --release --example knn_param_sweep_report
```

Schreibt CSV nach `reports/knn_param_sweep.csv` (und stdout) sowie Top-3-Markdown
pro `n` nach stderr. Viz: Hub → Param Sweep.

## RkNN Mutate-Cost (mtree 2.1 opt-in Graph)

Matrix über Modellgröße `n` ∈ {100…10k} und `k` ∈ {1,5,10,20} × `insert`/`erase`.
Zerlegt den Maintenance-Aufwand in Phasen:

- **discover** — Insert: `rknn_search` vor der Mutation; Erase: Reverse-Opfer via
  Forward-Listen-Scan (`knn_graph_neighbors`)
- **mutate_graph** / **mutate_plain** — geometrisches Insert/Erase (± eingebaute
  Graph-Maintenance)
- **refresh** — Forward-`knn_from_entry` je Affected (+ `knn_stats` Routing/Leaves
  als Distanz-/Visit-Proxy)
- **selective_total** ≈ discover + plain_mutate + refresh (Vergleich zu
  `mutate_graph`)

```bash
cargo run --release --example rknn_mutate_cost_report
cargo bench --bench rknn_graph
```

Schreibt CSV nach `reports/rknn_mutate_cost.csv` (und stdout) sowie Markdown-Pivots
nach stderr. Methodik: 64 Mutationen/Batch, 1 Warmup + 5 timed Rounds (Median µs/op).
Viz: Hub → RkNN Mutate Cost.

## Seed Hint / From Entry

```bash
cargo run --release --example knn_seed_hint_report   # → reports/knn_seed_hint.csv
cargo run --release --example knn_from_entry_report  # → reports/knn_from_entry.csv
```

Viz: Hub → Seed Hint / From Entry.

## Criterion-Benches → Hub

Nach einem Bench-Lauf die Estimates nach CSV exportieren:

```bash
cargo bench --bench knn_seed_hint
cargo bench --bench mtree_benches
cargo bench --bench rknn_graph
cargo run --release --example export_criterion_csv   # → reports/criterion_benches.csv
```

Viz: Hub → Criterion Benches. Zusätzlich bleibt Criterions eigenes HTML unter
`target/criterion/report/index.html`.

Alternativ per Git statt path — in `Cargo.toml`:

```toml
mtree = { git = "https://github.com/5imonk/mtree.git", branch = "main", features = ["knn_stats"] }
```

Dieser Ordner liegt auf dem Orphan-Branch `experiment/benches` (kein Library-Quellcode).
