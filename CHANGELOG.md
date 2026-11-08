# Changelog

## [Unreleased]

## [2.0.0]

Breaking Release: Ownership-Arena, monomorphe Distanz und Query-Hotpath.

### Breaking

- `Distance::clone_box` entfernt; bisherige Overrides kompilieren nicht mehr.
- `MTree` hat einen neuen Typparameter `Dist` (Default `DynDistance`); Constructors speichern die Distanz monomorph statt als `Box<dyn Distance>`.
- Type-erased Distanz über `MTree::with_dyn_distance`; `with_distance` / `with_params` / `with_capacity_hint` sind monomorph.
- `MTree::new` für `Vec<f32>`/`Vec<f64>` nutzt `EuclideanDistance` direkt.
- `MTree: Clone` verlangt jetzt `V: Clone`, `S: Clone`, `Dist: Clone`.
- Default-Knotengrößen `16/64/16` (`DEFAULT_*`); `node_params_for_capacity`-Clamp `16…64` (vorher `5/100/20` bzw. `8…100`).
- `Query` / `RangeQuery` / `FilteredQuery` / `FilteredRangeQuery` leihen `&MTree` (Lifetime) und tragen `Dist`; kein Arc/Mutex-Sharing mehr über Query-Lebensdauer.
- `NodeStats::{add,remove}_descendant` nimmt `(&K, &V)` statt `&(K, V)`.

### Added

- `DynDistance`, `EuclideanDistance`, `QueryScratch`, `DEFAULT_{MIN,MAX}_NODE_SIZE`, `DEFAULT_SPLIT_SAMPLING`.
- `knn_search_with_scratch` / `knn_from_entry_with_scratch` für wiederverwendbaren Query-Scratch.
- Routing-Arena + `NodeId`; Queries unter `&self` ohne per-node Locks (parallele Reader bleiben erlaubt).
- Hotpath-Docs: `docs/PERF_HOTPATH_STREAM.md`, `docs/PERF_NEXT_SDO_SAFE.md`.

## [1.0.0]

Erste stabile API. `knn_search`, `knn_from_entry`, `search`, `range_search` und `iter` liefern `EntryId`. Key, Payload und Epsilon liest man mit `MTree::key`, `MTree::payload` und `MTree::epsilon`. Knoten und `PlaceholderQueue` bleiben intern.
