# mtree-experiment

Dünnes Harness-Crate: Benchmarks und Reports für [`mtree`](https://github.com/5imonk/mtree).

Erwartet die Library als Sibling-Checkout unter `../mtree` (path-Dependency mit Feature `knn_stats`).

```bash
# lokal (path)
cargo run --release --example knn_seed_hint_report
cargo run --release --example knn_from_entry_report
cargo bench --bench knn_seed_hint
cargo bench --bench mtree_benches
```

Alternativ per Git statt path — in `Cargo.toml`:

```toml
mtree = { git = "https://github.com/5imonk/mtree.git", branch = "main", features = ["knn_stats"] }
```

Dieser Ordner liegt auf dem Orphan-Branch `experiment/benches` (kein Library-Quellcode).
