# Gefilterte kNN bleibt nach erase/insert falsch

Stand: `7f6ce49` auf `main` (Merge von `7f92c7b`). Die ungefilterte Suche ist damit in Ordnung. Die gefilterte nicht. Nur diese Schicht anfassen.

## Was schon stimmt

`knn_from_entry` ohne Filter, also `knn_search_plain` in `src/search.rs`, trifft nach vielen `erase_by_id` + `insert_with_identity` den k-ten Nachbarn. In sdo ist das `knn_from_observer` über alle lebenden Observer. Der Test `knn_from_observer_matches_brute_force_after_replaces` ist für Rang 6 grün (Abweichung unter `1e-9`).

Paarweise Distanzen und die Frage, welcher Eintrag das Aktiv-Prädikat erfüllt, sind nicht der Fehler. Dieselbe Distanzfunktion und dasselbe Prädikat benutzt der Brute-Force-Vergleich.

## Was noch falsch ist

`knn_from_entry` mit `KnnConfig::with_active` landet in `knn_search_filtered` (`src/filtered/search.rs`, `knn_search_filtered_with_placeholder_queue`). Dieser Weg liefert nach demselben Replace-Strom den falschen k-ten Nachbarn.

Das gilt auch, wenn das Prädikat jeden Eintrag annimmt. Es ist kein Fehler der Aktiv-Menge.

Beleg aus sdo, derselbe Strom wie der grüne Live-Test (128 Punkte, 400 Remove/Insert, Seed 11, 2D):

- `rho = 0`, alle 128 Observer sind aktiv. `knn_active_from_observer`, Rang 13, Observer 15: geliefert `0.1710826274941235`, sortierte Distanz der übrigen Aktiven `0.15414923705690659`.
- Echte Teilmenge (`rho = 0.15`): derselbe Observer, Rang 13, geliefert `0.2264828170377615` statt `0.20107682659483922`. Die Liste hatte 13 Treffer, der 13. ist zu weit. Ein näherer Aktiver fehlt.

In sdo ist `graph_cutoff` nur diese Liste, Rang χ (`knn_active_from_observer`, dann `hits[χ - 1]`). Ist die Liste kürzer als χ, wird der letzte Treffer genommen. Der kann näher sein als der echte χ-te. Deshalb kann ein Cutoff auch zu klein ausfallen (Observer 303: `0.195` statt `0.288`), ohne dass eine andere Distanzformel im Spiel ist.

## Auftrag

Die gefilterte Suche muss nach demselben erase/insert-Strom dieselben k Distanzen liefern wie die sortierte Liste der Einträge, die das Prädikat bestehen. Zwei Fälle, beide Pflicht:

1. Prädikat ist immer wahr. Das Ergebnis muss mit `knn_from_entry` ohne Filter übereinstimmen. „Alle aktiv“ darf nicht an der gefilterten Implementierung vorbeigehen.
2. Prädikat gilt nur für eine Teilmenge. Der k-te Treffer ist der k-te der Teilmenge, nicht ein fernerer Punkt und nicht ein Inaktiver.

Regression neben `knn_from_entry_matches_naive_after_replaces`: gleiche Punktmenge und dieselben 400 Ersetzungen, einmal mit immer-wahrem Filter, einmal mit einem Filter, der nur einen Teil der Ids durchlässt. Toleranz wie dort (`1e-6`).

Schleife, die dabei zu prüfen ist: in `knn_search_filtered_with_placeholder_queue` wird abgebrochen, sobald `entry.distance_bound` über der bisher k-ten aktiven Distanz liegt. Für Routing-Kinder ist `distance_bound` die obere Schranke `Distanz zum Zentrum + covering_radius`. Eine zu kleine Hülle macht diese Schranke zu klein, und der Zweig mit dem eigentlichen k-ten Nachbarn wird nicht mehr besucht. Die plain Suche in `src/search.rs` schneidet so nicht mehr ab. Die gefilterte Suche muss dasselbe Garantieniveau haben.

`src/search.rs` und die bereits grüne ungefilterte Suche nicht wieder aufmachen. Den alten, hängengebliebenen `expand_cover_up`-Patch aus `docs/bug-knn-after-replace.md` nicht wieder einbauen.
