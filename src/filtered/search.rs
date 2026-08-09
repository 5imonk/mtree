// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Gefilterte Such-APIs für [`MTree`](crate::tree::MTree).

use crate::entry::EntryId;
use crate::filtered::query::{FilteredQuery, FilteredRangeQuery};
use crate::node::{NodePtr, ObjectNode};
use crate::stats::NodeStats;
use crate::tree::{DistanceType, KnnDistanceKey, KnnQueueEntry, MTree, UPPER_BOUND_FACTOR};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::Arc;

impl<K, V, D, S> MTree<K, V, D, S>
where
    K: Clone + Send + Sync + Hash + Eq + Default,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Default,
{
    /// Radius-Suche, die nur aktive Einträge liefert (`is_active(id, value)`).
    pub fn range_search_filtered<F>(
        &self,
        needle: &K,
        radius: D,
        is_active: F,
    ) -> FilteredRangeQuery<K, V, D, S, F>
    where
        F: Fn(EntryId, &V) -> bool,
    {
        FilteredRangeQuery::new(self.range_search(needle, radius), is_active)
    }

    /// Bereichssuche (Annulus), die nur aktive Einträge liefert.
    pub fn search_filtered<F>(
        &self,
        needle: &K,
        min_radius: D,
        max_radius: D,
        is_active: F,
    ) -> FilteredQuery<K, V, D, S, F>
    where
        F: Fn(EntryId, &V) -> bool,
    {
        FilteredQuery::new(self.search(needle, min_radius, max_radius), is_active)
    }

    /// k-NN mit Predicate: sucht bis `k` aktive Treffer gefunden sind.
    ///
    /// `is_active(id, value)` entscheidet, ob ein Eintrag aktiv ist.
    /// `include_inactive`: wenn `true`, enthält das Ergebnis auch inaktive Punkte
    /// innerhalb der Distanz des k-ten aktiven Nachbarn; sonst nur die (bis zu) `k`
    /// nächsten Aktiven.
    pub fn knn_search_filtered<F>(
        &self,
        needle: &K,
        k: usize,
        is_active: F,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)>
    where
        F: Fn(EntryId, &V) -> bool,
    {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_filtered_with_placeholder_queue(
                needle,
                k,
                &is_active,
                include_inactive,
            );
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_filtered_fallback(needle, k, is_active, include_inactive)
    }

    /// Gefilterte k-NN: nur aktive Objekte zählen für k und den Pruning-Radius.
    fn knn_search_filtered_with_placeholder_queue<F>(
        &self,
        needle: &K,
        k: usize,
        is_active: &F,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)>
    where
        F: Fn(EntryId, &V) -> bool,
    {
        let root = self.root.as_ref().unwrap().clone();
        let distance_fn = self.distance_fn.clone_box();

        let root_distance = {
            let root_guard = root.lock().unwrap();
            let d: D = distance_fn.distance(&root_guard.key, needle);
            if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&d) }
            } else {
                0.0
            }
        };

        let mut pq: BinaryHeap<KnnQueueEntry<K, V, S>> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_distance,
        });

        // Max-Heap der Distanzen der bisher besten aktiven Treffer (Größe ≤ k).
        let mut active_distances: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)> = Vec::new();

        while let Some(entry) = pq.pop() {
            let pruning_radius = if active_distances.len() >= k {
                active_distances
                    .peek()
                    .map(|d| d.0)
                    .unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let pruning_radius_relaxed = if pruning_radius.is_finite() {
                pruning_radius * UPPER_BOUND_FACTOR
            } else {
                f64::INFINITY
            };

            if active_distances.len() >= k && entry.distance_bound > pruning_radius_relaxed {
                break;
            }

            let node_guard = entry.node.lock().unwrap();
            if node_guard.is_leaf {
                for child in &node_guard.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        let obj_key = obj_node.key();
                        let dist_d: D = distance_fn.distance(&obj_key, needle);
                        let dist = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&dist_d) }
                        } else {
                            0.0
                        };
                        if dist > pruning_radius_relaxed {
                            continue;
                        }

                        let active = {
                            let guard = obj_node.value.read().unwrap();
                            is_active(obj_node.id, &guard.1)
                        };

                        if active {
                            results.push((obj_node.clone(), dist, true));
                            if active_distances.len() < k {
                                active_distances.push(KnnDistanceKey(dist));
                            } else if let Some(max) = active_distances.peek() {
                                if dist < max.0 {
                                    active_distances.pop();
                                    active_distances.push(KnnDistanceKey(dist));
                                }
                            }
                        } else if include_inactive {
                            results.push((obj_node.clone(), dist, false));
                        }
                    }
                }
                continue;
            }

            for child in &node_guard.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let center_d: D = distance_fn.distance(&child_guard.key, needle);
                    let center_dist = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                        && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                    {
                        unsafe { std::mem::transmute_copy(&center_d) }
                    } else {
                        0.0
                    };
                    let covering_radius = child_guard.covering_radius;
                    let lower_bound = (center_dist - covering_radius).max(0.0);
                    let upper_bound = center_dist + covering_radius;

                    if lower_bound > pruning_radius_relaxed {
                        continue;
                    }

                    pq.push(KnnQueueEntry {
                        node: routing_child.clone(),
                        center_distance: center_dist,
                        distance_bound: upper_bound,
                    });
                }
            }
        }

        let final_radius = if active_distances.len() >= k {
            active_distances
                .peek()
                .map(|d| d.0)
                .unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };

        results.retain(|(_, dist, _)| {
            *dist <= final_radius * UPPER_BOUND_FACTOR || !final_radius.is_finite()
        });
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if include_inactive {
            results.into_iter().map(|(n, d, _)| (n, d)).collect()
        } else {
            results
                .into_iter()
                .filter(|(_, _, active)| *active)
                .take(k)
                .map(|(n, d, _)| (n, d))
                .collect()
        }
    }

    /// Fallback für nicht-f64 Distanztypen: alles sammeln, sortieren, filtern.
    fn knn_search_filtered_fallback<F>(
        &self,
        needle: &K,
        k: usize,
        is_active: F,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)>
    where
        F: Fn(EntryId, &V) -> bool,
    {
        let max_radius = D::infinity();
        let mut all: Vec<_> = self.range_search(needle, max_radius).collect();
        all.sort_by(|a, b| {
            let da = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy::<D, f64>(&a.1) }
            } else {
                0.0
            };
            let db = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy::<D, f64>(&b.1) }
            } else {
                0.0
            };
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });

        let mut out = Vec::new();
        let mut active_count = 0usize;
        let mut kth_active_dist: Option<D> = None;

        for (node, dist) in all {
            if let Some(radius) = kth_active_dist {
                let radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy::<D, f64>(&radius) }
                } else {
                    f64::INFINITY
                };
                let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy::<D, f64>(&dist) }
                } else {
                    0.0
                };
                if dist_f64 > radius_f64 * UPPER_BOUND_FACTOR {
                    break;
                }
            }

            let active = {
                let guard = node.value.read().unwrap();
                is_active(node.id, &guard.1)
            };

            if active {
                out.push((node, dist));
                active_count += 1;
                if active_count == k {
                    kth_active_dist = Some(dist);
                    if !include_inactive {
                        break;
                    }
                }
            } else if include_inactive {
                out.push((node, dist));
            }
        }

        if !include_inactive {
            out.truncate(k.min(out.len()));
        }

        out
    }
}
