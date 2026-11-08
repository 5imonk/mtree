# Performance Hotpath: Stream-Clustering

Agent-Handoff für Branch `perf/hotpath-stream-clustering`.  
**Scope dieses Dokuments:** Anleitung für Code-Arbeit in **diesem** Repo (`mtree`).  
Messung optional über Sibling `../mtree-experiment`.

## Kontext und Zielbild

**Workload (entscheidet Sync):** Stream-Clustering, Baum **n ≈ 50–1000**, pro Punkt Query, regelmäßig Replace, Single-Owner-Loop → **Variante C**: exclusive Ownership, **kein** per-node `Mutex`/`RwLock` auf dem Query-Hotpath.

**Nicht-Ziele:** Kein Umbau zu VP-Tree; keine `move_towards`-Pflicht; Metrik-API, `EntryId`, insert/erase, `update_*`, Filter, Seed, Range/Annulus bleiben.

**Inspiration (Peers):** `vpsearch`, `vp_tree`, `ball-tree` — Contiguous/Index-Layout, monomorphe Distance, lockfreie Query, tau/LB-Pruning. Ball-tree ist konzeptionell am nächsten (Covering Balls).

**Benchmark-Hintergrund (experiment):** bei großen n war mtree `needle_none` ~300–1000× langsamer als Metrik-Peers; Visit-Stats ~96 Routing + ~73 Leaves ohne Seed vs. ~12 + ~7 mit Seed. Bei Stream-n≤1000 dominiert vor allem **Per-Visit-Tax** (Locks/Clone/dyn).

```text
handoff (dieses Doc) → Sync-Ziel C → Phase1 Algo → Phase2 Key → Phase3 Ownership
  → optional monomorphe Distance → messen
```

---

## Agent-Arbeitsanweisung

1. Arbeite **nur** in diesem Repo auf Branch `perf/hotpath-stream-clustering`.
2. Lies dieses Dokument vollständig, dann Phase 1 → 2 → 3.
3. Nach jeder Phase: `cargo test`.
4. Kein Refactor „per-node Mutex optimieren“ — Ziel ist **Entfernung** der per-node Locks.
5. Bei Unsicherheit: Korrektheit (exact kNN) vor Mikro-Opts.
6. `mtree-experiment` nur zum Messen anfassen (path-dependency), nicht als Primär-Repo.

---

## Teil A — Warum heutige Kosten existieren (Pros/Cons)

### A1. Per-node `Arc<Mutex<RoutingNode>>`

- **Warum da:** README „thread-sicher“ / parallele Ops; Interior Mutability für Shared `&MTree` + concurrent Insert/Search an verschiedenen Teilbäumen.
- **Pro:** Feine Sync ohne ganzen Baum zu sperren; theoretisch Concurrent Workloads auf großem Shared Index.
- **Contra:** Jeder Visit zahlt Lock auch single-threaded; Nested Locks; schlechte Lokalität; bei n≤1000 Locks relativ dominant.
- **Stream-Clustering:** **kontraproduktiv** → entfernen zugunsten Ownership-Modell C.

### A2. `RwLock<(K,V)>` + `key()`-Clone

- **Warum da:** `update_value`/`update_key` ohne `&mut` Baum; Key+Value in einem Slot hinter `Arc`.
- **Pro:** Parallele Payload-Reads; Updates ohne exklusiven Baum-Borrow.
- **Contra:** Jede Distanz: Read-Lock + `K`-Clone; Key zahlt Value-Sync mit.
- **Stream:** Key inline/`&K` ohne Lock; `V` über `&mut` oder getrenntes Array by `EntryId`.
- **Ort:** `src/node.rs` (`ObjectNode::key` clont unter `.read()`).

### A3. `Box<dyn Distance>` + `clone_box` pro Query

- **Warum da:** Object-safe, austauschbare Metrik, `Send+Sync` Trait-Objekt am Tree.
- **Pro:** Flexible API, ein Tree-Typ für alle Distances.
- **Contra:** Vtable + Allok pro Query; kein Inlining.
- **Stream:** Generics im Suchkern; dyn nur Facade.
- **Ort:** `src/search.rs` (`knn_search_with_placeholder_queue` ~`clone_box`), `src/tree.rs` (`distance_fn`).

### A4. PQ nach Upper Bound, kein Pop-Skip

- **Warum da:** Vermutlich „sichere“ Bound / Annulus-Denkweise; False-Negative-Angst bei frühem Stop.
- **Pro:** Konservativ.
- **Contra:** Schlechte Best-First-Ordnung; zu viele Expands ohne Seed.
- **Fix:** LB-Priorität + Skip wenn `lb > kth` (korrekt für exact kNN).
- **Ort:** `src/search.rs` / `KnnQueueEntry` in `src/tree.rs`.

### A5. `parent_distance` ungenutzt in Search

- **Warum da:** Klassisches M-Tree-Feld für Triangle-Inequality beim Insert/Split gepflegt.
- **Pro:** Daten für Pruning ohne Extra-Speicher.
- **Contra:** Ungenutzt = verschenkte Distanzersparnis.
- **Fix:** `|d(P,Q)−pd|`-Schranken in Search.

### A6. Unbounded `results` + Sort

- **Warum da:** Einfach sammeln dann top-k.
- **Pro:** Simpel.
- **Contra:** Arc-Clones + Sort O(m log m), m≫k.
- **Fix:** k-Heap `(EntryId, dist)`.

### A7. Covering-Radii / Struktur

- **Warum da:** Dynamische Inserts → M-Tree-Semantik.
- **Pro:** Online Replace ohne Full-Rebuild.
- **Contra:** Lose Kugeln, mehr Visits als VP-Bulk-Build.
- **Stream:** Dynamik behalten; optional seltenes Compact — nicht VP werden.

**Hinweis:** `src/query.rs` und `src/placeholder_queue.rs` liegen **nicht** auf dem f64-kNN-Hotpath (`knn_search_with_placeholder_queue` in `search.rs`).

---

## Teil B — Optimierungspunkte (priorisiert für Stream)

### Sync-Varianten (Entscheidung)

| Variante | Idee | Für Stream-Clustering |
| --- | --- | --- |
| A per-node Mutex | Feine Concurrent Sync | **Nein** — Tax auf jedem Visit |
| B Tree-`RwLock` | Viele Reader, seltene Writer | Nur wenn Multi-Reader und Replace selten |
| **C Ownership `&mut`** | Ein Owner, Query ohne Node-Locks | **Ja — gewählt** |

### P0.1 Sync-Zielbild C

| | |
| --- | --- |
| Warum ändern | A1/A2 schaden dem Stream-Hotpath |
| Pro | Maximal schnelle Query+Replace; Arena möglich |
| Contra | Kein concurrent Mutate desselben Baums; Parallelität = Shards/eigene Bäume |
| Risiko | Med — README/Semantik |

### P0.2 Key aus RwLock; Search mit `&K`

| | |
| --- | --- |
| Pro | Kein Clone/Lock pro Distanz |
| Contra | `update_key` muss Key+Index konsistent schreiben |
| Risiko | Low–Med |

### P0.3 LB-PQ + Pop-Skip

| | |
| --- | --- |
| Pro | Weniger Visits ohne Seed; rein algorithmisch |
| Contra | Sorgfältige Numerik mit `UPPER_BOUND_FACTOR` / `LOWER_BOUND_FACTOR` |
| Risiko | Med (Korrektheit) — **zuerst** |

### P0.4 `parent_distance`-Triangle-Pruning

| | |
| --- | --- |
| Pro | Weniger Distanzcalls; Daten existieren |
| Contra | Edge Cases mit ε |
| Risiko | Low–Med — **früh** mit P0.3 |

### P0.5 Monomorphe Distance + kein `clone_box`

| | |
| --- | --- |
| Pro | Inlining |
| Contra | Typ-Parameter / dual API |
| Risiko | Med API |

### P1 (nach P0)

- Scratch/`Query`-Kontext (wiederverwendbare Heaps) — hoch relevant bei vielen Stream-Queries.
- PQ speichert Indices, nicht `Arc` — braucht Arena/Ownership-Layout.
- Parent-Ptr ohne Mutex.
- Rebuild/Compact nur batched; bei Replace-pro-Punkt **nicht** default.

### P2 — nicht tun

- Zu VP-Tree konvertieren / reines 1-VP-Binary-Layout.
- `move_towards` erzwingen (bricht abstrakte Metriken).
- Squared distance global (bricht ε/allgemeine Metrik); höchstens opt-in Euclid ohne ε.

---

## Teil C — Phasen (Code)

### Phase 1 — algorithmisch (niedriges Risiko)

Dateien: `src/search.rs`, gefilterter Pfad unter `src/filtered/` analog falls gleicher PQ-Bug.

- `KnnQueueEntry`: Ordnungskey = **lower bound** `(center_dist - covering_radius).max(0.0)`.
- Nach `pq.pop()`: wenn LB `> confirmed_kth` → nicht expandieren.
- Triangle-Pruning mit `parent_distance`, wo anwendbar.
- Results: capped k-Struktur; API liefert weiter sortierte `Vec<(EntryId, D)>`.
- `cargo test` grün (kNN, Seed, Filter, Range).

### Phase 2 — Key ohne RwLock

Dateien: `src/node.rs`, Search-Aufrufe.

- `ObjectNode`: `key: K`, `epsilon`, `parent_distance` außerhalb des Payload-Locks.
- Distanzpfad: `&K` ohne Clone/Lock.
- `update_value` / `update_key` anpassen.

### Phase 3 — Ownership ohne per-node Mutex

- `RoutingNode` nicht hinter `Mutex`; `NodePtr::Routing` ohne `Arc<Mutex<…>>` (oder Arena + Index).
- Mutation: `&mut MTree`; Query: `&self` ohne Interior Locks auf Nodes.
- README: Thread-Sicherheit = extern synchronisieren oder ein Owner; Claim „Arc\<Mutex\<\>\> parallel“ entfernen/ersetzen.

### Phase 4 — monomorphe Distance (optional)

- Interner generischer Suchkern; dyn-Wrapper an der API-Grenze ok.

### Phase 5 — messen

- Sibling experiment: `cargo run --release --example knn_baseline_latency_report` (path `../mtree`).
- Mini Stream-Mix: n∈{50,200,1000}, Query+Replace (experiment example oder lokal).

---

## Akzeptanzkriterien

- Alle bestehenden Tests grün.
- Exact kNN (und Seed/Filter/Range) verhaltensgleich.
- Spürbar bessere Latency/Throughput bei n≤1000 Query+Replace.
- Visits bei `needle_none` sinken nach Phase 1 (wenn `knn_stats` genutzt).
- Kein VP-Tree; `EntryId` / `update_*` / Filter / Seed bleiben.

## Realistische Erwartung (Stream n≤1000)

- Phase 1–2: oft deutliche Verbesserung.
- Phase 3: weiterer Sprung (Per-Visit-Kosten).
- Parität mit vpsearch bei n=100k ist kein Ziel; Clustering zählt Query+Replace-Throughput bei kleinem n.
