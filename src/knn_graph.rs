// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Materialisierter kNN-Graph (Forward-Listen + Reverse) für RkNN / Mk-artige Pflege.

use crate::entry::EntryId;

/// Side-Struktur parallel zu `MTree::by_id` (Slot-Index).
#[derive(Clone, Debug)]
pub(crate) struct KnnGraph {
    pub(crate) k_max: usize,
    /// Pro Slot: sortierte Forward-Liste `(neighbor, dist)`, Länge ≤ `k_max`.
    pub(crate) forward: Vec<Option<Vec<(EntryId, f64)>>>,
    /// Pro Slot: wer listet mich in seiner Forward-Liste.
    pub(crate) reverse: Vec<Vec<EntryId>>,
}

impl KnnGraph {
    pub(crate) fn new(k_max: usize) -> Self {
        Self {
            k_max: k_max.max(1),
            forward: Vec::new(),
            reverse: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.forward.clear();
        self.reverse.clear();
    }

    /// Stellt Slot-Kapazität für `slot` sicher (nach `alloc_slot` / Resize von `by_id`).
    pub(crate) fn ensure_slot(&mut self, slot: usize) {
        if self.forward.len() <= slot {
            self.forward.resize_with(slot + 1, || None);
        }
        if self.reverse.len() <= slot {
            self.reverse.resize_with(slot + 1, Vec::new);
        }
    }

    pub(crate) fn release_slot(&mut self, id: EntryId) {
        let slot = id.slot_index();
        if slot < self.forward.len() {
            self.forward[slot] = None;
        }
        if slot < self.reverse.len() {
            self.reverse[slot].clear();
        }
    }

    #[inline]
    pub(crate) fn knn_distance_of(list: &[(EntryId, f64)], k_max: usize) -> f64 {
        if list.len() < k_max {
            f64::INFINITY
        } else {
            list.last().map(|(_, d)| *d).unwrap_or(f64::INFINITY)
        }
    }

    pub(crate) fn knn_distance(&self, id: EntryId) -> f64 {
        let slot = id.slot_index();
        match self.forward.get(slot).and_then(|o| o.as_ref()) {
            Some(list) => Self::knn_distance_of(list, self.k_max),
            None => f64::INFINITY,
        }
    }

    pub(crate) fn rank_distance(&self, id: EntryId, k: usize) -> Option<f64> {
        if k == 0 || k > self.k_max {
            return None;
        }
        let list = self.forward.get(id.slot_index())?.as_ref()?;
        if list.len() < k {
            // Unvollständige Liste: konservativ ∞ (noch Platz für weitere Nachbarn).
            return Some(f64::INFINITY);
        }
        Some(list[k - 1].1)
    }

    pub(crate) fn forward_list(&self, id: EntryId) -> Option<&[(EntryId, f64)]> {
        self.forward
            .get(id.slot_index())
            .and_then(|o| o.as_ref())
            .map(|v| v.as_slice())
    }

    pub(crate) fn reverse_list(&self, id: EntryId) -> &[EntryId] {
        self.reverse
            .get(id.slot_index())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Setzt Forward von `id` und aktualisiert Reverse-Kanten (alte Kanten lösen).
    pub(crate) fn set_forward(&mut self, id: EntryId, mut neighbors: Vec<(EntryId, f64)>) {
        self.ensure_slot(id.slot_index());
        neighbors.truncate(self.k_max);
        neighbors.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
        });
        // Drop reverse edges from previous forward.
        if let Some(old) = self.forward[id.slot_index()].take() {
            for (nb, _) in old {
                self.remove_reverse_edge(nb, id);
            }
        }
        for &(nb, _) in &neighbors {
            self.add_reverse_edge(nb, id);
        }
        self.forward[id.slot_index()] = Some(neighbors);
    }

    fn add_reverse_edge(&mut self, listed: EntryId, owner: EntryId) {
        self.ensure_slot(listed.slot_index());
        let rev = &mut self.reverse[listed.slot_index()];
        if !rev.contains(&owner) {
            rev.push(owner);
        }
    }

    fn remove_reverse_edge(&mut self, listed: EntryId, owner: EntryId) {
        let slot = listed.slot_index();
        if slot < self.reverse.len() {
            self.reverse[slot].retain(|e| *e != owner);
        }
    }

    /// Entfernt `victim` aus Forward von `owner`; gibt gedroppte Distanz-Info nicht zurück.
    pub(crate) fn remove_from_forward(&mut self, owner: EntryId, victim: EntryId) -> bool {
        let slot = owner.slot_index();
        let Some(list) = self.forward.get_mut(slot).and_then(|o| o.as_mut()) else {
            return false;
        };
        let before = list.len();
        list.retain(|(id, _)| *id != victim);
        if list.len() != before {
            self.remove_reverse_edge(victim, owner);
            true
        } else {
            false
        }
    }

    /// Fügt `cand` in Forward von `owner` ein (sortiert); droppt ggf. schlechtesten.
    /// Gibt `(inserted, dropped)` zurück.
    pub(crate) fn try_insert_neighbor(
        &mut self,
        owner: EntryId,
        cand: EntryId,
        dist: f64,
    ) -> (bool, Option<EntryId>) {
        if owner == cand || !dist.is_finite() {
            return (false, None);
        }
        self.ensure_slot(owner.slot_index());
        let k_max = self.k_max;
        if self.forward[owner.slot_index()].is_none() {
            self.forward[owner.slot_index()] = Some(Vec::new());
        }
        let list = self.forward[owner.slot_index()].as_mut().unwrap();
        if let Some(pos) = list.iter().position(|(id, _)| *id == cand) {
            if dist < list[pos].1 {
                list.remove(pos);
            } else {
                return (false, None);
            }
        }
        let knn_dist = Self::knn_distance_of(list, k_max);
        if list.len() >= k_max && dist > knn_dist {
            return (false, None);
        }
        list.push((cand, dist));
        list.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
        });
        let mut dropped = None;
        if list.len() > k_max {
            let (old, _) = list.pop().unwrap();
            dropped = Some(old);
        }
        self.add_reverse_edge(cand, owner);
        if let Some(old) = dropped {
            self.remove_reverse_edge(old, owner);
        }
        (true, dropped)
    }
}
