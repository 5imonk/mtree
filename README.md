# M-Tree Rust Implementation

Eine hochperformante Implementierung der M-Tree Datenstruktur in Rust mit SIMD-Optimierungen.

## Features

- **SIMD-optimierte Distanzberechnungen**: Nutzt portable-simd für vektorisierte euklidische Distanzen
- **Ownership-Modell**: Mutation nur unter `&mut MTree`; Queries unter `&self` ohne per-node Locks (Arena + `NodeId`)
- **Parallelität**: Viele parallele Reader (`&MTree`) sind erlaubt und gewollt (z. B. Rayon). Writer nur exklusiv (`&mut MTree`); kein concurrent Mutate desselben Baums — sonst externe Sync oder eigene Bäume/Shards
- **Einheitliche Such-API**: `knn_search` / `knn_from_entry` / `search` / `range_search` mit optionalem Filter und Annulus-Config
- **Dual-Index**: Stabile `EntryId` (O(1)) plus `HashMap<K, EntryId>` für Schlüssel-Lookup

Performance / Stream-Clustering Hotpath (Branch-Plan): [`docs/PERF_HOTPATH_STREAM.md`](docs/PERF_HOTPATH_STREAM.md).  
Fortsetzung (sdo-sicher, nach Phase 1–4): [`docs/PERF_NEXT_SDO_SAFE.md`](docs/PERF_NEXT_SDO_SAFE.md).

## Begriffe

| | Rolle |
|---|--------|
| **Key `K`** | Position im Metrikraum (Koordinaten) — bestimmt Baum und Suche |
| **Identity** | Optionaler Index/Timestamp; wird zu einem winzigen `ε` gehasht. Gleiche `K` + verschiedene Identity = zwei Einträge |
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
}

let mut tree = MTree::with_distance(Euclid2d);
// Oder mit Größenhinweis (wählt min/max/split per √n-Heuristik, Clamp 16…64):
// let mut tree = MTree::with_capacity_hint(10_000, Euclid2d);
let id = tree.insert((1, 2), "Raum A".to_string()).unwrap();
let _ = tree.insert((3, 4), "Raum B".to_string()).unwrap();

// Alle Einträge (schnell: Scan über internen Index, keine Metrik)
let ids: Vec<_> = tree.iter().collect();
for id in ids {
    let _ = (tree.key(id), tree.payload(id));
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
| `knn_search` | k nächste Nachbarn als `(EntryId, Distanz)`; Config optional (`()`, `None` oder `KnnConfig`); mit Radien `min < dist ≤ max`; optional `.seed(EntryId)` als Pruning-Hint |
| `knn_from_entry` | k-NN von einem gespeicherten `EntryId`; `include_self` steuert, ob der Punkt selbst zählt |
| `search` | Annulus `min_radius < dist ≤ max_radius` (`SearchConfig` Pflicht) |
| `range_search` | Kugel `dist ≤ radius` (`RangeSearchConfig` Pflicht) |

```rust
use mtree::{KnnConfig, RangeSearchConfig, SearchConfig};

// Plain k-NN
let nearby = tree.knn_search(&(1, 2), 5, ());

// k-NN mit nahen Eintrag als Pruning-Hint (Ergebnis bleibt exakt)
let nearby_seeded = tree.knn_search(&(1, 2), 5, KnnConfig::new().seed(id));

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
- **`.seed(EntryId)`** (k-NN): besucht zuerst das Leaf des Hints und setzt den Pruning-Radius früher; ungültige ID wird ignoriert. Ergebnis identisch zu unseeded.
- Bei `search` / `range_search` bedeutet `is_active: None` (Default) „alle Treffer“, `Some(f)` nur aktive.

### Key und Value per `EntryId`

`knn_search`, `knn_from_entry`, `search`, `range_search` und `iter` liefern `EntryId`. Key, Nutzdaten und Epsilon liest man am Baum:

```rust
let id = tree.insert((1, 2), "Raum A".to_string()).unwrap();

let key: Option<(i64, i64)> = tree.key(id);
let value: Option<String> = tree.payload(id);
let epsilon: Option<f64> = tree.epsilon(id);

// Nach erase_by_id: ID ungültig (Slot kann später recycelt werden)
tree.erase_by_id(id);
assert!(tree.key(id).is_none());
assert!(tree.payload(id).is_none());

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

Duplikat-Keys ohne Identity werden abgelehnt. Mit Identity dürfen gleiche Koordinaten mehrfach vorkommen:

```rust
tree.insert((1, 2), "a".to_string()).unwrap();
assert!(matches!(tree.insert((1, 2), "b".into()), Err(DuplicateKey)));

let t1 = 1_000u64;
let t2 = 2_000u64;
tree.insert_with_identity((1, 2), "event-a".into(), &t1).unwrap();
tree.insert_with_identity((1, 2), "event-b".into(), &t2).unwrap();

// Dieselbe Quelle bei der Query → Distanz 0 zu genau diesem Eintrag
let hits: Vec<_> = tree
    .range_search(&(1, 2), RangeSearchConfig::new(0.0).identity(&t1))
    .collect();
```

Die Metrik ist überall `d'(a,b) = Distance(K_a, K_b) + |ε_a − ε_b|` mit `ε = epsilon_from(identity)` (Bereich `(0, 2^-40]`, Identity `0` → `ε = 0`). Dasselbe gilt für `knn_search` / `search` über `.identity(&quelle)` bzw. `.epsilon(ε)`. `knn_from_entry` verwendet das gespeicherte `ε` des Eintrags.

`erase_by_key(&K)` löscht nur, wenn genau ein Eintrag mit diesem Key existiert; sonst `erase_by_id` oder `erase_by_key_identity`.

Für Float-Vektoren als Key: `mtree::distance::Point` verwenden (`Hash` + `Eq` auf Bit-Pattern).

## Version

`2.0.0`, auch als `mtree::VERSION`. Ab `2.0` sind Brüche an dieser API `3.0`, kompatible Erweiterungen `2.x`, Fixes `2.0.z`. Das Changelog steht in `CHANGELOG.md`.

## Lizenz

LGPL-3.0
