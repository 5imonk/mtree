# kNN nach erase/insert liefert den falschen k-ten Nachbarn

Stand: `9d71d05` auf `main`. Betrifft `knn_search`, `knn_from_entry` und damit jede Suche, die nach vielen `erase` + `insert` denselben Baum nutzt (SDOstream `replace`).

## Symptom

Nach mehreren hundert Remove/Insert-Zyklen ist die k-te Distanz aus `knn_from_entry` systematisch **zu groß**. Es ist eine echte Distanz zu einem gespeicherten Punkt, aber nicht zum k-ten. Beispiel aus SDOstream (2D, `k = 128`, Rang `x = 6`): Observer 15 lieferte `0.606` statt der sortierten Distanz `0.073`. Dieselbe Suche speist `graph_cutoff` (Rang χ) und `model_density` (Rang x). Die Cutoff-Werte sind deshalb keine χ-/x-ten Nachbarn, obwohl sie zur aktuellen Geometrie passen.

Ein statischer Baum ohne `erase` (wenige `insert`, dann eine Suche) bleibt korrekt. Der Fehler zeigt sich erst, wenn der Baum splittet und Knoten per `donate_child` / `merge_routing_nodes` umhängt.

## Regression

`src/tests.rs`, Test `knn_from_entry_matches_naive_after_replaces`: 128 eindeutige Punkte, 400 `erase_by_id` + `insert_with_identity`, danach für jeden Eintrag `knn_from_entry(id, 6, false)` gegen die sortierte Distanz der übrigen Keys. Toleranz `1e-6` (Identity-ε liegt bei `< 2^-40` und erklärt den Rangfehler nicht).

Auf `9d71d05` fällt dieser Test. Nach dem Fix muss er grün sein, zusammen mit den bestehenden Tests in `src/tests.rs`.

## Ursache

Die kNN-Suche verwirft einen Teilbaum, wenn `dist(query, center) - covering_radius` über dem Pruning-Radius liegt (`src/search.rs`, `knn_search_with_placeholder_queue`). Das ist nur korrekt, solange `covering_radius` jedes Routing-Knotens mindestens die Distanz vom Knotenzentrum zum weitesten Nachkommen ist. `compute_node_radius` (`src/tree.rs`) setzt den Radius aus `dist_to_parent` der Kinder, nicht aus einer frischen Distanz zum aktuellen Zentrum. Stimmt `dist_to_parent` nicht, schrumpft der Radius und nähere Blätter werden abgeschnitten. Der k-te Treffer ist dann zu weit weg.

Drei Stellen lassen die Parent-Distanz veralten:

1. **`donate_child` und `merge_routing_nodes` hängen ObjectNodes um, ohne `set_parent` aufzurufen.** Der Kommentar behauptet, ein `Arc` sei nicht mutierbar. `ObjectNode::set_parent(&self, parent, distance)` existiert und schreibt Parent-Zeiger und `parent_distance`. Routing-Kinder werden aktualisiert, Object-Kinder nicht. Danach rechnet `compute_node_radius` mit der Distanz zum alten Zentrum.

2. **`split` setzt bei einem Nicht-Root die `parent`-Zeiger von `node1`/`node2` nicht.** Es schreibt nur `parent_distance`, und das während `node1` und `node2` schon gelockt sind und danach der Parent gelockt wird. Die Kinder zeigen auf die neuen Knoten, die neuen Knoten zeigen nach oben auf `None`. `knn_from_entry` startet im Blatt (`obj.parent()`). Liegt die Parent-Kette nicht am echten Root, bleiben die Covering-Radien der Vorfahren stehen, auch wenn ein Blatt wächst.

3. **Root-Split setzt `parent_distance` der beiden neuen Root-Kinder auf `0`.** Der Root-Radius aus `parent_distance + child.covering_radius` deckt die Distanz zwischen den Zentren nicht ab.

`tree_insert` vergrößert den Covering-Radius des gewählten Kindes, läuft die Vorfahren aber nicht hoch. Ein korrekt gesetzter Parent-Zeiger nützt erst, wenn ein gewachsener Radius nach oben durchschlägt (`parent_distance + covering_radius` des Kindes, nur vergrößern). Schrumpfen beim Löschen darf die Hülle oben zu groß lassen. Zu klein ist der Fehler.

## Nicht die Ursache

- Die Rangauswahl im Cutoff-Cache von SDOstream. `knn_from_observer` ohne Cache zeigt denselben zu weiten k-ten Nachbarn.
- `|ε_a − ε_b|` mit `ε < 2^-40`. Das verschiebt keine 6-te Distanz auf die ~90-te.
- `ζ`-Mischung der Graph-Cutoffs. Bei `ζ = 1` ist der finale Radius der Graph-Cutoff selbst.

## Fehlgeschlagener Fix, nicht wieder einbauen

Ein Versuch, an den drei Stellen `set_parent` zu ergänzen und `expand_cover_up` die Vorfahren hochzulaufen, ist **hängengeblieben** (Mutex ist nicht reentrant) und wurde auf `9d71d05` zurückgesetzt. Bitte nicht denselben Patch wieder aufsetzen.

Beobachtet wurde:

- `split` hält `node1` und `node2` und lockt danach den Parent. Ist der Parent einer der beiden Knoten, blockiert der Thread.
- `compute_node_radius` läuft unter dem Parent-Lock und lockt die Kinder. Ein Kind, das derselbe Knoten ist, blockiert.
- `expand_cover_up` ohne Zyklusprüfung läuft endlos, sobald `parent` im Kreis zeigt, und vergrößert den Radius bei jedem Schritt.
- Dieselbe Parent-Schleife lässt `split` rekursiv auf demselben Knoten laufen, weil nach dem Split `children.len() > max_node_size` bleibt.

Lock-Reihenfolge: Kind-Guard fallen lassen, bevor der Parent gelockt wird. Nie einen Knoten locken, dessen Mutex dieser Thread schon hält. Parent-Zyklen abbrechen. Bestehende Tests in `src/tests.rs` müssen mitlaufen, nicht nur der neue.

## Abnahme

```bash
cargo test --lib knn_from_entry_matches_naive_after_replaces
cargo test --lib
```

SDOstream hängt per `[patch]` auf diesem Clone (`../mtree` aus dem Workspace `sdo`). Dort müssen danach diese beiden Tests grün sein, ohne die M-Tree-Suche zu umgehen:

- `cargo test -p sdo --test knn_after_replace`
- `cargo test -p sdostream-clust --test cutoffs after_stream_graph_cutoff_is_chi_th_active_and_density_is_xth_live`
