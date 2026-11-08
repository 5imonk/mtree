# Performance Next: sdo-sichere Hotpath-Fortsetzung

Agent-Handoff für Branch `perf/hotpath-stream-clustering` **nach** Phase 1–4
(Algo / Key / Ownership-Arena / monomorphe Distance).

**Scope:** Code + Docs **nur in diesem Repo** (`mtree`).  
**Nicht hier:** `mtree-experiment` nur zum Messen; `sdo` nur als Consumer-Constraint
(kein sdo-Code in diesem Auftrag, außer optionaler Smoke-Hinweis am Ende).

Vorgänger: [`PERF_HOTPATH_STREAM.md`](PERF_HOTPATH_STREAM.md).  
Herkunft der Priorisierung: Peer-Gap-Analyse + realer Verbrauch in Sibling
`../sdo` (`rust/sdo`, Wrapper `crates/sdo/src/tree_index.rs`).

---

## Agent-Arbeitsanweisung

1. Branch `perf/hotpath-stream-clustering` — **dieses** Repo.
2. Lies dieses Dokument vollständig, dann P1 → P2 → messen.
3. Nach jedem Teilschritt: `cargo test` (ggf. mit Feature `knn_stats`).
4. **Härteste Constraint:** Queries bleiben unter `&self` und müssen **parallele
   Read-only-Zugriffe** auf demselben Baum erlauben (Rayon in sdo).  
   **Niemals** `knn_search` / `knn_from_entry` / `search` / `range_search` auf
   `&mut self` umbiegen.
5. Filter (`with_active`), Seed, `knn_from_entry`, Annulus/Range,
   `insert_with_identity` / ε, `EntryId`, `erase_by_id`, `update_value` bleiben.
6. Bei Unsicherheit: Korrektheit (exact kNN + Filter) vor Mikro-Opts.
7. Messung: Sibling `../mtree-experiment` (`knn_baseline_latency_report`,
   `knn_query_matrix_report`, `knn_stream_mix_report`) und/oder sdo
   `learn_parallel` nach Dependency-Bump — **sdo-Bump ist nicht Teil der
   Pflichtarbeit in diesem Repo**, nur Abnahmekriterium „API bleibt nutzbar“.

---

## Stand nach Phase 1–4 (kurz)

Erledigt:

- LB-PQ + Pop-Skip, Triangle/`parent_distance`, capped k-Hits (Richtung).
- Key außerhalb Payload-Lock (`UnsafeCell`, Write unter `&mut MTree`).
- Routing-Arena + `NodeId`, kein per-node `Mutex` auf dem Query-Pfad.
- Monomorphe `Dist` möglich (`with_distance` / `with_params`); Default weiter
  `DynDistance`.

Restlücke vs. Metrik-Peers (`needle_none`, k=1, experiment Phase‑5-Größenordnung):

| n | vs vpsearch / vp_tree | vs ball_tree |
| --- | ---: | ---: |
| ≤1000 | ~5–12× | ~4–5× |
| 100k | ~70–80× | ~23× |

Stream-Ziel ist **n≤1000**, nicht VP-Parität bei 100k. Nächster Gewinn: Layout/Mikro
unter Beibehaltung Online-API + paralleler `&self`-Queries.

---

## Consumer-Constraint: `../sdo` (rust/sdo)

```text
SdoModel / SdoStreamModel
  └── TreeIndex.tree: MTree<Point, f64>
        ├── &self  Rayon: fit knn, score, stream query/requery/predict
        └── &mut   sequentiell: insert / erase / update_value (Replace=erase+insert)
```

Load-bearing:

| API | Warum |
| --- | --- |
| `&self` Queries, parallel | Fit/Score/Stream Parallel-Mode |
| `KnnConfig::seed` | Requery / Predict |
| `KnnConfig::with_active` | Active-kNN (Filter auf `V=f64`) |
| `knn_from_entry` | Cutoffs, Complementarity |
| Annulus / `range_search` | Graph-Nachbarschaft |
| `insert_with_identity` + ε | gleiche Coords, Distanz `‖·‖+|Δε|` |
| `erase_by_id` + `update_value` | Replace / Counts |
| `EntryId` + Maps | `ObserverIndex ↔ EntryId` |

Nicht load-bearing: `update_key` (sdo nutzt erase+insert), Tree-`Clone`.

Profil (sdo, Aug 2026, **altes** mtree v1.0.0): Query ~55% Wall;
`ObjectNode::key` ~21% Self; **`Mutex::lock_contended` ~11%** unter parallelem
`knn_search` (Value-`RwLock` im Filter-Pfad). Distanz-SIMD ~1.5% → nachrangig.

**Sync-Kontrakt (verbindlich für diesen Handoff):**

- Viele Reader (`&MTree`) gleichzeitig: **erlaubt und gewollt**.
- Writer nur exklusiv (`&mut MTree`); kein concurrent Mutate desselben Baums.
- Key/`parent` in `UnsafeCell`: Write nur unter `&mut`; **Reads unter shared `&`
  sind erlaubt** (Borrow-Checker schließt Writer aus). README + SAFETY-Kommentare
  müssen das so sagen (heutige Formulierungen „keine parallelen Reads“ sind zu eng
  und widersprechen sdo).

---

## Explizit nicht tun

- Queries auf `&mut self` umbiegen (zerstört sdo-Rayon).
- Scratch, der **`&mut MTree`** für jede Query verlangt (gleicher Break).
- ε / `insert_with_identity`, `with_active`, Seed, `knn_from_entry`, Annulus entfernen.
- Default-Compact/Rebuild pro Replace.
- Zu VP-Tree / `move_towards`-Pflicht konvertieren.
- Parität mit vpsearch bei n=100k als Akzeptanzkriterium.

---

## Phasen (umsetzen)

### P1 — Kontrakt festnageln (Docs / Safety)

Dateien: `README.md`, `src/node.rs` (SAFETY an `key`/`set_key`/`parent`), ggf.
`docs/PERF_HOTPATH_STREAM.md` Verweis aktualisieren.

- README Parallelität: **shared read-only Queries OK**; Mutation nur `&mut` /
  externe Sync / Shards.
- SAFETY-Kommentare: parallele Key-Reads unter `&` OK; Write nur `&mut`.
- `cargo test` grün (reine Doc/Kommentar-Änderung).

### P2 — Layout / Mikro (Code)

Reihenfolge empfohlen; nach jedem Block tests.

#### P2.1 Object-Arena / kein `Arc` auf dem Hit-Pfad

Dateien: `src/node.rs`, `src/tree.rs`, `src/search.rs`, `src/filtered/search.rs`,
Tests.

- Leaves analog Routing: stabile Slots / Index; `NodePtr` ohne `Arc<ObjectNode>`
  auf dem Query-Hit-Pfad.
- k-NN-Results / Heaps: **`(EntryId, dist)`**, kein `Arc::clone` pro Hit.
- Public API (`EntryId`, `get`/`payload`/`key`, Filter-Signatur) stabil halten.
- `rust/sdo` hält keine Leaf-Arcs → OK. (Älteres Monolith-`Repos/sdo` mit
  `Arc<ObjectNode>` in Nachbar-Infos ist **nicht** Primär-Consumer.)

#### P2.2 QueryScratch ohne `&mut` Tree

Dateien: `src/search.rs`, filtered analog, ggf. kleines `src/scratch.rs`.

- Wiederverwendbare PQ + Hit-Heaps.
- **Erlaubt:** `thread_local!` und/oder
  `knn_search_with_scratch(&self, …, scratch: &mut QueryScratch)`.
- **Verboten:** Scratch nur über `&mut self` am Tree.
- Kinder-Iteration ohne per-visit `Vec<NodeId>`-Allok (direkt über `children`).

#### P2.3 Ein capped Hit-Heap

- Ein Heap `(EntryId, dist)` statt Distanz-Heap + Arc-Heap + final sort (soweit
  nach P2.1 noch Doppelstruktur übrig).
- API liefert weiter sortierte `Vec<(EntryId, D)>`.

#### P2.4 `V` ohne `RwLock` (Filter-Hebel)

Dateien: `src/node.rs`, `update_value`, filtered search.

- Mutation von `V` nur unter `&mut MTree` (`update_value`).
- Filter/`payload`-Read unter `&self` **ohne** Lock (wie Key).
- `with_active: Fn(EntryId, &V)` bleibt.
- Erwarteter sdo-Gewinn: weniger Contention unter Rayon + filtered kNN.

### P3 — Messen (dieses Ökosystem)

Im Sibling `../mtree-experiment` (path auf diesen Checkout):

```bash
cargo test                          # in mtree
cargo run --release --example knn_baseline_latency_report
cargo run --release --example knn_query_matrix_report
cargo run --release --example knn_stream_mix_report
```

Fokus: n∈{50,200,1000}, `needle_none` / Stream `query_replace`; Visits + ms.

Optional (nicht blockierend für Merge in mtree): `../sdo` Dependency auf diesen
Branch/path, Smoke parallel Fit/Score + `learn_parallel`, Profil vs. tag v1.0.0
(Lock-Self-Time / Query-Wall).

---

## Akzeptanzkriterien

- Alle bestehenden `cargo test` grün (inkl. Filter/Seed/from_entry/Range wo vorhanden).
- Public Query-APIs weiter `&self` und `Sync` für parallele Read-only-Nutzung.
- Kein Zwang zu `&mut` Query; Scratch parallelisierbar (TLS oder externer Scratch).
- Exact kNN / Seed / Filter / Annulus verhaltensgleich.
- Spürbar weniger Allok/Indirection auf dem Hotpath (Stream-Mix / Matrix Visits+ms).
- README/SAFETY beschreiben den Read-many / write-exclusive Kontrakt korrekt.

## Realistische Erwartung

- Stream n≤1000: weitere Annäherung Richtung ball-tree-Gap (~4–5× → ggf. ~2–3×),
  nicht VP-Parität.
- sdo Parallel-Mode: Query billiger + weniger Lock-Tax → Rayon eher wertvoller,
  nicht überflüssig.

```text
P1 Kontrakt → P2.1 Leaf-Arena → P2.2 Scratch(&self) → P2.3 Hit-Heap
  → P2.4 V ohne RwLock → messen (experiment / optional sdo)
```
