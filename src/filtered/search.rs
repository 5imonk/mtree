// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Interne gefilterte k-NN-Helfer für [`MTree`](crate::tree::MTree).

use crate::entry::EntryId;
use crate::node::{NodeId, NodePtr, ObjectNode};
use crate::search::{
    is_seeded_leaf, triangle_lower_bound, visit_seeded_leaf_objects, KnnSeedCtx,
};
use crate::stats::NodeStats;
use crate::tree::{
    metric_f64, DistanceType, KnnDistanceKey, KnnQueueEntry, MTree, LOWER_BOUND_FACTOR,
    UPPER_BOUND_FACTOR,
};
use std::cell::Cell;
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

fn push_active_kth(active_distances: &mut BinaryHeap<KnnDistanceKey>, k: usize, dist: f64) {
    if active_distances.len() < k {
        active_distances.push(KnnDistanceKey(dist));
    } else if let Some(max) = active_distances.peek() {
        if dist < max.0 {
            active_distances.pop();
            active_distances.push(KnnDistanceKey(dist));
        }
    }
}

fn skip_self(from: Option<&KnnSeedCtx>, id: EntryId) -> bool {
    from.map(|c| c.exclude_self() && c.id == id).unwrap_or(false)
}

fn object_dist<K, V, D, S, Dist>(
    obj: &ObjectNode<K, V, S>,
    needle: &K,
    needle_eps: f64,
    from: Option<&KnnSeedCtx>,
    distance_fn: &Dist,
) -> f64
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: crate::distance::Distance<K, Output = D> + ?Sized,
{
    if from.map(|c| c.zero_dist_for(obj.id)).unwrap_or(false) {
        0.0
    } else {
        metric_f64(distance_fn, obj.key(), obj.epsilon(), needle, needle_eps)
    }
}

impl<K, V, D, S, Dist> MTree<K, V, D, S, Dist>
where
    K: Clone + Send + Sync + Hash + Eq + Default,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Default,
    Dist: crate::distance::Distance<K, Output = D>,
{
    /// k-NN mit Predicate: sucht bis `k` aktive Treffer gefunden sind.
    pub(crate) fn knn_search_filtered(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_filtered_with_placeholder_queue(
                needle,
                needle_eps,
                k,
                is_active,
                include_inactive,
                from,
            );
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_filtered_fallback(needle, needle_eps, k, is_active, include_inactive, from)
    }

    /// Gefilterte k-NN: nur aktive Objekte zählen für k und den Pruning-Radius.
    fn knn_search_filtered_with_placeholder_queue(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let root = self.root.unwrap();
        let distance_fn = &self.distance_fn;

        let mut active_distances: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)> = Vec::new();
        let prune_cell = Cell::new(f64::INFINITY);

        let consider =
            |obj: Arc<ObjectNode<K, V, S>>,
             dist: f64,
             pruning_relaxed: f64,
             active_distances: &mut BinaryHeap<KnnDistanceKey>,
             results: &mut Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)>| {
                if dist > pruning_relaxed {
                    return;
                }
                let active = {
                    let guard = obj.value.read().unwrap();
                    is_active(obj.id, &*guard)
                };
                if active {
                    results.push((obj, dist, true));
                    push_active_kth(active_distances, k, dist);
                } else if include_inactive {
                    results.push((obj, dist, false));
                }
            };

        if let Some(ctx) = from {
            if let Some(leaf_id) = ctx.leaf {
                #[cfg(feature = "knn_stats")]
                {
                    crate::knn_stats::record_routing_visit();
                    crate::knn_stats::record_leaf_visit();
                }
                let leaf = &self.nodes[leaf_id.index()];
                visit_seeded_leaf_objects(
                    leaf,
                    needle,
                    needle_eps,
                    ctx,
                    distance_fn,
                    &prune_cell,
                    |obj, dist| {
                        let pruning_radius = if active_distances.len() >= k {
                            active_distances
                                .peek()
                                .map(|d| d.0)
                                .unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        let pruning_relaxed = if pruning_radius.is_finite() {
                            pruning_radius * UPPER_BOUND_FACTOR
                        } else {
                            f64::INFINITY
                        };
                        consider(
                            obj,
                            dist,
                            pruning_relaxed,
                            &mut active_distances,
                            &mut results,
                        );
                        let pruning_radius = if active_distances.len() >= k {
                            active_distances
                                .peek()
                                .map(|d| d.0)
                                .unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        prune_cell.set(pruning_radius);
                    },
                );
                if leaf_id == root {
                    return finish_filtered_knn(results, &active_distances, k, include_inactive);
                }
            }
        }

        let root_node = &self.nodes[root.index()];
        let root_distance = metric_f64(
            distance_fn,
            &root_node.key,
            root_node.epsilon,
            needle,
            needle_eps,
        );
        let r = root_node.covering_radius;
        let root_lb = (root_distance - r).max(0.0);
        let root_ub = root_distance + r;

        let mut pq: BinaryHeap<KnnQueueEntry> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_lb,
            upper_bound: root_ub,
        });

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

            // Pop-Skip auf LB (Best-First): Rest der PQ hat ≥ aktuelle LB.
            if entry.distance_bound > pruning_radius_relaxed {
                break;
            }

            if is_seeded_leaf(from, entry.node) {
                continue;
            }

            #[cfg(feature = "knn_stats")]
            crate::knn_stats::record_routing_visit();

            let node = &self.nodes[entry.node.index()];
            if node.is_leaf {
                #[cfg(feature = "knn_stats")]
                crate::knn_stats::record_leaf_visit();
                for child in &node.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        if skip_self(from, obj_node.id) {
                            continue;
                        }
                        let dist = object_dist(obj_node, needle, needle_eps, from, distance_fn);
                        consider(
                            obj_node.clone(),
                            dist,
                            pruning_radius_relaxed,
                            &mut active_distances,
                            &mut results,
                        );
                    }
                }
                continue;
            }

            let routing_ids: Vec<NodeId> = node
                .children
                .iter()
                .filter_map(|child| child.as_routing_id())
                .collect();

            for child_id in routing_ids {
                let child_node = &self.nodes[child_id.index()];
                let covering_radius = child_node.covering_radius;
                let tri_lb = triangle_lower_bound(
                    entry.center_distance,
                    child_node.parent_distance,
                    covering_radius,
                );
                if tri_lb > pruning_radius_relaxed {
                    continue;
                }

                let center_dist = metric_f64(
                    distance_fn,
                    &child_node.key,
                    child_node.epsilon,
                    needle,
                    needle_eps,
                );
                let lower_bound = (center_dist - covering_radius).max(0.0);
                let upper_bound = center_dist + covering_radius;

                if lower_bound > pruning_radius_relaxed {
                    continue;
                }

                pq.push(KnnQueueEntry {
                    node: child_id,
                    center_distance: center_dist,
                    distance_bound: lower_bound,
                    upper_bound,
                });
            }
        }

        finish_filtered_knn(results, &active_distances, k, include_inactive)
    }

    /// Fallback für nicht-f64 Distanztypen: alles sammeln, sortieren, filtern.
    fn knn_search_filtered_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let max_radius = D::infinity();
        let mut query = self.query_ball(needle, needle_eps, max_radius);
        let mut all = Vec::new();
        while let Some(hit) = query.next_object() {
            all.push(hit);
        }
        if let Some(from) = from {
            if from.exclude_self() {
                all.retain(|(n, _)| n.id != from.id);
            }
        }
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
                is_active(node.id, &*guard)
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

    /// k-NN im Annulus mit Predicate: `min_radius < dist ≤ max_radius`, bis `k` Aktive.
    pub(crate) fn knn_search_range_filtered(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_range_filtered_with_queue(
                needle,
                needle_eps,
                k,
                min_radius,
                max_radius,
                is_active,
                include_inactive,
                from,
            );
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_range_filtered_fallback(
            needle,
            needle_eps,
            k,
            min_radius,
            max_radius,
            is_active,
            include_inactive,
            from,
        )
    }

    fn knn_search_range_filtered_with_queue(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let root = self.root.unwrap();
        let distance_fn = &self.distance_fn;

        let mut active_distances: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)> = Vec::new();
        let prune_cell = Cell::new(f64::INFINITY);

        let consider =
            |obj: Arc<ObjectNode<K, V, S>>,
             dist: f64,
             effective_max_relaxed: f64,
             active_distances: &mut BinaryHeap<KnnDistanceKey>,
             results: &mut Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)>| {
                if dist <= min_r || dist > max_r {
                    return;
                }
                if dist > effective_max_relaxed {
                    return;
                }
                let active = {
                    let guard = obj.value.read().unwrap();
                    is_active(obj.id, &*guard)
                };
                if active {
                    results.push((obj, dist, true));
                    push_active_kth(active_distances, k, dist);
                } else if include_inactive {
                    results.push((obj, dist, false));
                }
            };

        if let Some(ctx) = from {
            if let Some(leaf_id) = ctx.leaf {
                #[cfg(feature = "knn_stats")]
                {
                    crate::knn_stats::record_routing_visit();
                    crate::knn_stats::record_leaf_visit();
                }
                let leaf = &self.nodes[leaf_id.index()];
                visit_seeded_leaf_objects(
                    leaf,
                    needle,
                    needle_eps,
                    ctx,
                    distance_fn,
                    &prune_cell,
                    |obj, dist| {
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
                        consider(
                            obj,
                            dist,
                            effective_max_relaxed,
                            &mut active_distances,
                            &mut results,
                        );
                        let kth = if active_distances.len() >= k {
                            active_distances
                                .peek()
                                .map(|d| d.0)
                                .unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        prune_cell.set(kth.min(max_r));
                    },
                );
                if leaf_id == root {
                    return finish_filtered_range_knn(
                        results,
                        &active_distances,
                        k,
                        include_inactive,
                        min_r,
                        max_r,
                    );
                }
            }
        }

        let root_node = &self.nodes[root.index()];
        let root_distance = metric_f64(
            distance_fn,
            &root_node.key,
            root_node.epsilon,
            needle,
            needle_eps,
        );
        let r = root_node.covering_radius;
        let root_lb = (root_distance - r).max(0.0);
        let root_ub = root_distance + r;

        let mut pq: BinaryHeap<KnnQueueEntry> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_lb,
            upper_bound: root_ub,
        });

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

            if entry.upper_bound < min_prune {
                continue;
            }
            // Auch ohne volle k Aktive: effective_max = max_r — Subtrees jenseits
            // der Annulus-Obergrenze nicht expandieren. LB-Best-First → break.
            if entry.distance_bound > effective_max_relaxed {
                break;
            }

            if is_seeded_leaf(from, entry.node) {
                continue;
            }

            #[cfg(feature = "knn_stats")]
            crate::knn_stats::record_routing_visit();

            let node = &self.nodes[entry.node.index()];
            if node.is_leaf {
                #[cfg(feature = "knn_stats")]
                crate::knn_stats::record_leaf_visit();
                for child in &node.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        if skip_self(from, obj_node.id) {
                            continue;
                        }
                        let dist = object_dist(obj_node, needle, needle_eps, from, distance_fn);
                        consider(
                            obj_node.clone(),
                            dist,
                            effective_max_relaxed,
                            &mut active_distances,
                            &mut results,
                        );
                    }
                }
                continue;
            }

            let routing_ids: Vec<NodeId> = node
                .children
                .iter()
                .filter_map(|child| child.as_routing_id())
                .collect();

            for child_id in routing_ids {
                let child_node = &self.nodes[child_id.index()];
                let covering_radius = child_node.covering_radius;
                let tri_lb = triangle_lower_bound(
                    entry.center_distance,
                    child_node.parent_distance,
                    covering_radius,
                );
                if tri_lb > effective_max_relaxed {
                    continue;
                }

                let center_dist = metric_f64(
                    distance_fn,
                    &child_node.key,
                    child_node.epsilon,
                    needle,
                    needle_eps,
                );
                let lower_bound = (center_dist - covering_radius).max(0.0);
                let upper_bound = center_dist + covering_radius;

                if lower_bound > effective_max_relaxed {
                    continue;
                }
                if upper_bound < min_prune {
                    continue;
                }

                pq.push(KnnQueueEntry {
                    node: child_id,
                    center_distance: center_dist,
                    distance_bound: lower_bound,
                    upper_bound,
                });
            }
        }

        finish_filtered_range_knn(
            results,
            &active_distances,
            k,
            include_inactive,
            min_r,
            max_r,
        )
    }

    fn knn_search_range_filtered_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        is_active: &dyn Fn(EntryId, &V) -> bool,
        include_inactive: bool,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let mut query = self.query_annulus(needle, needle_eps, min_radius, max_radius);
        let mut all = Vec::new();
        while let Some(hit) = query.next_object() {
            all.push(hit);
        }
        if let Some(from) = from {
            if from.exclude_self() {
                all.retain(|(n, _)| n.id != from.id);
            }
        }
        all.retain(|(_, d)| {
            let dist = dist_as_f64(*d);
            dist > min_r && dist <= max_r
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
                is_active(node.id, &*guard)
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

fn finish_filtered_knn<K, V, S>(
    mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)>,
    active_distances: &BinaryHeap<KnnDistanceKey>,
    k: usize,
    include_inactive: bool,
) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)>
where
    S: NodeStats<K, V>,
{
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

fn finish_filtered_range_knn<K, V, S>(
    mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64, bool)>,
    active_distances: &BinaryHeap<KnnDistanceKey>,
    k: usize,
    include_inactive: bool,
    min_r: f64,
    max_r: f64,
) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)>
where
    S: NodeStats<K, V>,
{
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
        let within_k = active_distances.len() < k || *dist <= final_cap * UPPER_BOUND_FACTOR;
        *dist > min_r && *dist <= max_r && within_k
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
