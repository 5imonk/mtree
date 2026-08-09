// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Interne gefilterte k-NN-Helfer für [`MTree`](crate::tree::MTree).

use crate::entry::EntryId;
use crate::node::{NodePtr, ObjectNode};
use crate::stats::NodeStats;
use crate::tree::{
    DistanceType, KnnDistanceKey, KnnQueueEntry, MTree, LOWER_BOUND_FACTOR, UPPER_BOUND_FACTOR,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::Arc;

fn dist_as_f64<D: DistanceType>(d: D) -> f64 {
    if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
        && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
    {
        unsafe { std::mem::transmute_copy(&d) }
    } else {
        0.0
    }
}

impl<K, V, D, S> MTree<K, V, D, S>
where
    K: Clone + Send + Sync + Hash + Eq + Default,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Default,
{
    /// k-NN mit Predicate: sucht bis `k` aktive Treffer gefunden sind.
    pub(crate) fn knn_search_filtered(
        &self,
        needle: &K,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_filtered_with_placeholder_queue(
                needle,
                k,
                is_active,
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
    fn knn_search_filtered_with_placeholder_queue(
        &self,
        needle: &K,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
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
    fn knn_search_filtered_fallback(
        &self,
        needle: &K,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let max_radius = D::infinity();
        let mut all: Vec<_> = self.query_ball(needle, max_radius).collect();
        all.sort_by(|a, b| {
            dist_as_f64(a.1)
                .partial_cmp(&dist_as_f64(b.1))
                .unwrap_or(Ordering::Equal)
        });

        let mut out = Vec::new();
        let mut active_count = 0usize;
        let mut kth_active_dist: Option<D> = None;

        for (node, dist) in all {
            if let Some(radius) = kth_active_dist {
                if dist_as_f64(dist) > dist_as_f64(radius) * UPPER_BOUND_FACTOR {
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

    /// k-NN im Annulus mit Predicate: `min_radius ≤ dist < max_radius`, bis `k` Aktive.
    pub(crate) fn knn_search_range_filtered(
        &self,
        needle: &K,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_range_filtered_with_queue(
                needle,
                k,
                min_radius,
                max_radius,
                is_active,
                include_inactive,
            );
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_range_filtered_fallback(
            needle,
            k,
            min_radius,
            max_radius,
            is_active,
            include_inactive,
        )
    }

    fn knn_search_range_filtered_with_queue(
        &self,
        needle: &K,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let root = self.root.as_ref().unwrap().clone();
        let distance_fn = self.distance_fn.clone_box();

        let (root_distance, root_upper) = {
            let root_guard = root.lock().unwrap();
            let d = dist_as_f64(distance_fn.distance(&root_guard.key, needle));
            (d, d + root_guard.covering_radius)
        };

        let mut pq: BinaryHeap<KnnQueueEntry<K, V, S>> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_upper,
        });

        let mut active_distances: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)> = Vec::new();

        while let Some(entry) = pq.pop() {
            let kth = if active_distances.len() >= k {
                active_distances
                    .peek()
                    .map(|d| d.0)
                    .unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            };
            let effective_max = kth.min(max_r);
            let effective_max_relaxed = if effective_max.is_finite() {
                effective_max * UPPER_BOUND_FACTOR
            } else {
                f64::INFINITY
            };
            let min_prune = min_r * LOWER_BOUND_FACTOR;

            if entry.distance_bound < min_prune {
                continue;
            }
            if active_distances.len() >= k && entry.distance_bound > effective_max_relaxed {
                break;
            }

            let node_guard = entry.node.lock().unwrap();
            if node_guard.is_leaf {
                for child in &node_guard.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        let dist = dist_as_f64(distance_fn.distance(&obj_node.key(), needle));
                        if dist < min_r || dist >= max_r {
                            continue;
                        }
                        if dist > effective_max_relaxed {
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
                    let center_dist = dist_as_f64(distance_fn.distance(&child_guard.key, needle));
                    let covering_radius = child_guard.covering_radius;
                    let lower_bound = (center_dist - covering_radius).max(0.0);
                    let upper_bound = center_dist + covering_radius;

                    if lower_bound > effective_max_relaxed {
                        continue;
                    }
                    if upper_bound < min_prune {
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

        let final_cap = if active_distances.len() >= k {
            active_distances
                .peek()
                .map(|d| d.0)
                .unwrap_or(max_r)
                .min(max_r)
        } else {
            max_r
        };

        results.retain(|(_, dist, _)| {
            let within_k =
                active_distances.len() < k || *dist <= final_cap * UPPER_BOUND_FACTOR;
            *dist >= min_r && *dist < max_r && within_k
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

    fn knn_search_range_filtered_fallback(
        &self,
        needle: &K,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let mut all: Vec<_> = self.query_annulus(needle, min_radius, max_radius).collect();
        all.retain(|(_, d)| {
            let dist = dist_as_f64(*d);
            dist >= min_r && dist < max_r
        });
        all.sort_by(|a, b| {
            dist_as_f64(a.1)
                .partial_cmp(&dist_as_f64(b.1))
                .unwrap_or(Ordering::Equal)
        });

        let mut out = Vec::new();
        let mut active_count = 0usize;
        let mut kth_active_dist: Option<f64> = None;

        for (node, dist) in all {
            let dist_f = dist_as_f64(dist);
            if let Some(radius) = kth_active_dist {
                if dist_f > radius * UPPER_BOUND_FACTOR {
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
                    kth_active_dist = Some(dist_f);
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
