# M-Tree Rust Implementation

Eine hochperformante Implementierung der M-Tree Datenstruktur in Rust mit SIMD-Optimierungen und paralleler Verarbeitung.

## Features

- **SIMD-optimierte Distanzberechnungen**: Nutzt portable-simd für vektorisierte euklidische Distanzen
- **Parallele Suche**: Rayon-Integration für parallele Traversierung
- **Thread-sicher**: Arc<Mutex<>> für sichere parallele Operationen
- **Effiziente Suchoperationen**: Range-Suche, k-NN-Suche, Nearest-Neighbor
- **Dual-Index**: Stabile `EntryId` (O(1)) plus `HashMap<K, EntryId>` für Schlüssel-Lookup

## Begriffe

| | Rolle |
|---|--------|
| **Key `K`** | Position im Metrikraum (Koordinaten) — bestimmt Baum und Suche |
| **Value `V`** | Nutzdaten (Label, Struct, …) — nicht für Routing |
| **`EntryId`** | Library-Handle für O(1) `get` / `erase_by_id` / `update_*` |

## Verwendung

```rust
use mtree::{MTree, DuplicateKey};

#[derive(Clone)]
struct Euclid2d;
impl mtree::distance::Distance<(i64, i64)> for Euclid2d {
    type Output = f64;
    fn distance(&self, a: &(i64, i64), b: &(i64, i64)) -> f64 {
        let dx = (a.0 - b.0) as f64;
        let dy = (a.1 - b.1) as f64;
        (dx * dx + dy * dy).sqrt()
    }
    fn clone_box(&self) -> Box<dyn mtree::distance::Distance<(i64, i64), Output = f64> + Send + Sync> {
        Box::new(Euclid2d)
    }
}

let mut tree = MTree::with_distance(Euclid2d);
let id = tree.insert((1, 2), "Raum A".to_string()).unwrap();
let _ = tree.insert((3, 4), "Raum B".to_string()).unwrap();

// Metrische Suche
let nearby = tree.knn_search(&(1, 2), 5);

// Löschen per ID oder Schlüssel
tree.erase_by_id(id);
tree.erase_by_key(&(3, 4));
```

### Key und Value per `EntryId`

`get(id)` liefert `Option<&Arc<ObjectNode<…>>>`. Darauf Key und Value lesen:

```rust
let id = tree.insert((1, 2), "Raum A".to_string()).unwrap();

// Variante 1: if let
if let Some(node) = tree.get(id) {
    let key: (i64, i64) = node.key();    // K — Koordinaten / Metrik-Key
    let value: String = node.payload();  // V — Nutzdaten
}

// Variante 2: map (nach erase liefert get None)
let value = tree.get(id).map(|n| n.payload());
let key   = tree.get(id).map(|n| n.key());

// Nach erase_by_id: ID ungültig (Slot kann später recycelt werden)
tree.erase_by_id(id);
assert!(tree.get(id).is_none());
```

Updates über dieselbe ID:

```rust
let id = tree.insert((1, 2), "alt".to_string()).unwrap();
tree.update_value(id, "neu".to_string());           // nur V ändern
tree.update_key(id, (10, 20)).expect("key free");   // K ändern (Baum-Relokation)
```

Duplikat-Keys werden abgelehnt:

```rust
tree.insert((1, 2), "a".to_string()).unwrap();
assert!(matches!(tree.insert((1, 2), "b".into()), Err(DuplicateKey)));
```

Für Float-Vektoren als Key: `mtree::distance::Point` verwenden (`Hash` + `Eq` auf Bit-Pattern).

## Lizenz

LGPL-3.0
