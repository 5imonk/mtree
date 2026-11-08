// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Wiederverwendbare Arbeitsspeicher-Puffer für k-NN-Abfragen.
//!
//! [`QueryScratch`] hält Priority-Queue und Hit-Heaps, damit wiederholte
//! k-NN-Suchen keine neuen `BinaryHeap`s allozieren. Die Suche selbst bleibt `&self`
//! auf dem Baum; der Scratch wird separat als `&mut QueryScratch` übergeben
//! (siehe [`MTree::knn_search_with_scratch`](crate::tree::MTree::knn_search_with_scratch))
//! oder implizit über einen `thread_local!` in [`MTree::knn_search`](crate::tree::MTree::knn_search).

use crate::entry::EntryId;
use crate::tree::{KnnDistanceKey, KnnQueueEntry};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Max-Heap-Eintrag der k besten Treffer (größte Distanz oben → leichtes Evict).
pub(crate) struct KnnHitEntry {
    pub(crate) dist: f64,
    pub(crate) id: EntryId,
}

impl PartialEq for KnnHitEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for KnnHitEntry {}
impl PartialOrd for KnnHitEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for KnnHitEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Wiederverwendbare Heaps/Vektoren für k-NN-Suchen.
///
/// Ein Scratch kann beliebig oft für Suchen auf verschiedenen Bäumen
/// wiederverwendet werden; jede Suche leert ihn zu Beginn.
#[derive(Default)]
pub struct QueryScratch {
    /// Best-First-Queue über Teilbäume.
    pub(crate) pq: BinaryHeap<KnnQueueEntry>,
    /// Einziger gecappter Max-Heap der k besten Treffer (Plain-/Annulus-k-NN);
    /// `peek()` bei `len() >= k` liefert den Pruning-Radius.
    pub(crate) hits: BinaryHeap<KnnHitEntry>,
    /// Nur für gefilterte k-NN: die k besten *aktiven* Distanzen (`active_distances`).
    /// Wird von Plain-/Annulus-k-NN nicht genutzt.
    pub(crate) best: BinaryHeap<KnnDistanceKey>,
    /// Zwischenergebnisse der gefilterten k-NN (`(id, dist, active)`).
    pub(crate) results: Vec<(EntryId, f64, bool)>,
}

impl QueryScratch {
    /// Leerer Scratch ohne vorallokierten Speicher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Leert alle Puffer, behält die Kapazität.
    pub fn clear(&mut self) {
        self.pq.clear();
        self.hits.clear();
        self.best.clear();
        self.results.clear();
    }
}

thread_local! {
    static TLS_SCRATCH: RefCell<QueryScratch> = RefCell::new(QueryScratch::new());
}

/// Führt `f` mit dem thread-lokalen Scratch aus. Bei Re-Entrancy (z. B. verschachtelte
/// `knn_search` im Filter-Prädikat) wird ein temporärer Scratch verwendet.
pub(crate) fn with_tls_scratch<R>(f: impl FnOnce(&mut QueryScratch) -> R) -> R {
    let mut f = Some(f);
    let res = TLS_SCRATCH.try_with(|cell| {
        cell.try_borrow_mut().ok().map(|mut s| {
            s.clear();
            (f.take().unwrap())(&mut s)
        })
    });
    match res {
        Ok(Some(r)) => r,
        _ => (f.take().unwrap())(&mut QueryScratch::new()),
    }
}
