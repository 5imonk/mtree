# M-Tree Rust Implementation

Eine hochperformante Implementierung der M-Tree Datenstruktur in Rust mit SIMD-Optimierungen und paralleler Verarbeitung.

## Features

- **SIMD-optimierte Distanzberechnungen**: Nutzt portable-simd für vektorisierte euklidische Distanzen
- **Parallele Suche**: Rayon-Integration für parallele Traversierung
- **Thread-sicher**: Arc<Mutex<>> für sichere parallele Operationen
- **Einheitliche Such-API**: `knn_search` / `knn_from_entry` / `search` / `range_search` mit optionalem Filter und Annulus-Config
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

// Alle Einträge (schnell: Scan über internen Index, keine Metrik)
for node in tree.iter() {
    let _ = (node.key(), node.payload());
}

// Metrische Suche — siehe Abschnitt unten
let nearby = tree.knn_search(&(1, 2), 5, ());

// Löschen per ID oder Schlüssel
tree.erase_by_id(id);
tree.erase_by_key(&(3, 4));
```

### Suche

Vier öffentliche Einstiege, optional mit Filter und (bei k-NN / Annulus) Radiusgrenzen:

| Methode | Bedeutung |
|---------|-----------|
| `knn_search` | k nächste Nachbarn; Config optional (`()`, `None` oder `KnnConfig`) |
| `knn_from_entry` | k-NN von einem gespeicherten `EntryId`; `include_self` steuert, ob der Punkt selbst zählt |
| `search` | Annulus `min_radius ≤ dist < max_radius` (`SearchConfig` Pflicht) |
| `range_search` | Kugel `dist ≤ radius` (`RangeSearchConfig` Pflicht) |

```rust
use mtree::{KnnConfig, RangeSearchConfig, SearchConfig};

// Plain k-NN
let nearby = tree.knn_search(&(1, 2), 5, ());

// k-NN von einem Eintrag im Baum (optional ohne den Punkt selbst)
let neighbors = tree.knn_from_entry(id, 5, false, ()).unwrap();

// k-NN mit Filter und Annulus
let hits = tree.knn_search(
    &(1, 2),
    10,
    KnnConfig::new()
        .min_radius(1.0)
        .max_radius(5.0)
        .with_active(|_id, v: &String| v.starts_with("Raum"))
        .include_inactive(false),
);

// Annulus-Iterator
let band: Vec<_> = tree
    .search(&(1, 2), SearchConfig::new(1.0, 5.0))
    .collect();

// Kugel, optional gefiltert
let in_ball: Vec<_> = tree
    .range_search(
        &(1, 2),
        RangeSearchConfig::new(3.0).with_active(|_id, v: &String| !v.is_empty()),
    )
    .collect();
```

- **`include_inactive`** gilt nur für k-NN: bei gesetztem Filter zählen nur aktive Treffer für `k`; inaktive innerhalb der Distanz des k-ten Aktiven können mitgeliefert werden.
- Bei `search` / `range_search` bedeutet `is_active: None` (Default) „alle Treffer“, `Some(f)` nur aktive.

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

// Paralleles Side-Array (optional): dichter Slot-Index ohne unsafe
let slot = id.slot_index();
// side_table[slot] = …
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
