# M-Tree Rust Implementation

Eine hochperformante Implementierung der M-Tree Datenstruktur in Rust mit SIMD-Optimierungen und paralleler Verarbeitung.

## Features

- **SIMD-optimierte Distanzberechnungen**: Nutzt portable-simd für vektorisierte euklidische Distanzen
- **Parallele Suche**: Rayon-Integration für parallele Traversierung
- **Thread-sicher**: Arc<Mutex<>> für sichere parallele Operationen
- **Effiziente Suchoperationen**: Range-Suche, k-NN-Suche, Nearest-Neighbor

## Verwendung

```rust
use mtree::MTree;

let tree = MTree::new();
tree.insert(vec![1.0, 2.0], "value1");
tree.insert(vec![3.0, 4.0], "value2");

let results = tree.knn_search(&vec![1.5, 2.5], 5);
```

## Lizenz

LGPL-3.0
