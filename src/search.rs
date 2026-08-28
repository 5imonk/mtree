// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Such-APIs für [`MTree`](crate::tree::MTree).

use crate::distance::Distance;
use crate::entry::EntryId;
use crate::filtered::query::{FilteredQuery, FilteredRangeQuery};
use crate::node::{NodePtr, ObjectNode, RoutingNode};
use crate::placeholder_queue::PlaceholderQueue;
use crate::query::{Query, RangeQuery};
use crate::search_config::{
    IntoKnnConfig, IntoRangeSearchConfig, IntoSearchConfig, KnnConfigResolved,
};
use crate::stats::NodeStats;
use crate::tree::{
    metric_f64, DistanceType, KnnDistanceKey, KnnQueueEntry, KnnTag, MTree, LOWER_BOUND_FACTOR,
    UPPER_BOUND_FACTOR,
};
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Kontext für k-NN, wenn die Needle ein gespeicherter Eintrag ist.
pub(crate) struct KnnFromEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    pub id: EntryId,
    pub include_self: bool,
    pub leaf: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
}

pub(crate) fn is_seeded_leaf<K, V, S>(
    from: Option<&KnnFromEntry<K, V, S>>,
    node: &Arc<Mutex<RoutingNode<K, V, S>>>,
) -> bool
where
    S: NodeStats<K, V>,
{
    from.and_then(|f| f.leaf.as_ref())
        .map(|leaf| Arc::ptr_eq(leaf, node))
        .unwrap_or(false)
}

/// Leaf des Query-Eintrags zuerst bewerten: Self optional mit Distanz 0.
pub(crate) fn visit_seeded_leaf_objects<K, V, D, S>(
    leaf: &RoutingNode<K, V, S>,
    needle: &K,
    needle_eps: f64,
    from: &KnnFromEntry<K, V, S>,
    distance_fn: &dyn Distance<K, Output = D>,
    pruning_radius: &Cell<f64>,
    mut on_object: impl FnMut(Arc<ObjectNode<K, V, S>>, f64),
) where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    for child in &leaf.children {
        if let NodePtr::Object(obj) = child {
            if obj.id == from.id {
                if from.include_self {
                    on_object(obj.clone(), 0.0);
                }
                continue;
            }
            let pr = pruning_radius.get();
            let dist = metric_f64(distance_fn, &obj.key(), obj.epsilon(), needle, needle_eps);
            if pr.is_finite() && dist > pr * UPPER_BOUND_FACTOR {
                continue;
            }
            on_object(obj.clone(), dist);
        }
    }
}

fn retain_exclude_self<K, V, S, D>(
    results: &mut Vec<(Arc<ObjectNode<K, V, S>>, D)>,
    from: Option<&KnnFromEntry<K, V, S>>,
) where
    S: NodeStats<K, V>,
{
    if let Some(from) = from {
        if !from.include_self {
            results.retain(|(n, _)| n.id != from.id);
        }
    }
}

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
    /// Interne Annulus-Query (`min < dist ≤ max`), ohne Filter.
    pub(crate) fn query_annulus(
        &self,
        needle: &K,
        needle_eps: f64,
        min_radius: D,
        max_radius: D,
    ) -> Query<K, V, D, S> {
        if let Some(ref root) = self.root {
            Query::new(
                needle.clone(),
                needle_eps,
                min_radius,
                max_radius,
                root.clone(),
                self.distance_fn.clone_box(),
            )
        } else {
            Query::empty()
        }
    }

    /// Interne Kugel-Query (`dist ≤ radius`), ohne Filter.
    pub(crate) fn query_ball(
        &self,
        needle: &K,
        needle_eps: f64,
        radius: D,
    ) -> RangeQuery<K, V, D, S> {
        if let Some(ref root) = self.root {
            RangeQuery::new(
                needle.clone(),
                needle_eps,
                radius,
                root.clone(),
                self.distance_fn.clone_box(),
            )
        } else {
            RangeQuery::empty()
        }
    }

    /// Bereichssuche (Annulus): `min_radius < dist ≤ max_radius`, optional gefiltert.
    pub fn search<'a>(
        &'a self,
        needle: &K,
        config: impl IntoSearchConfig<'a, D, V>,
    ) -> FilteredQuery<K, V, D, S, Box<dyn Fn(EntryId, &V) -> bool + 'a>> {
        let cfg = config.into_search_config();
        let inner = self.query_annulus(needle, cfg.epsilon, cfg.min_radius, cfg.max_radius);
        let pred: Box<dyn Fn(EntryId, &V) -> bool + 'a> =
            cfg.is_active.unwrap_or_else(|| Box::new(|_, _| true));
        FilteredQuery::new(inner, pred)
    }

    /// Radius-Suche (Kugel `dist ≤ radius`), optional gefiltert.
    pub fn range_search<'a>(
        &'a self,
        needle: &K,
        config: impl IntoRangeSearchConfig<'a, D, V>,
    ) -> FilteredRangeQuery<K, V, D, S, Box<dyn Fn(EntryId, &V) -> bool + 'a>> {
        let cfg = config.into_range_search_config();
        let inner = self.query_ball(needle, cfg.epsilon, cfg.radius);
        let pred: Box<dyn Fn(EntryId, &V) -> bool + 'a> =
            cfg.is_active.unwrap_or_else(|| Box::new(|_, _| true));
        FilteredRangeQuery::new(inner, pred)
    }

    /// k-Nearest-Neighbor-Suche.
    ///
    /// `config` kann `()`, `None` oder [`KnnConfig`](crate::search_config::KnnConfig) sein:
    /// optional Filter, `include_inactive`, sowie Annulus `min_radius`/`max_radius`.
    pub fn knn_search<'a>(
        &'a self,
        needle: &K,
        k: usize,
        config: impl IntoKnnConfig<'a, D, V>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let cfg = config.into_knn_config();
        self.knn_search_dispatch(needle, cfg.epsilon, k, cfg, None)
    }

    /// k-NN mit einem gespeicherten Eintrag als Query.
    ///
    /// `include_self`: wenn `true`, ist der Eintrag selbst (Distanz 0) ein Treffer,
    /// sofern Filter/Annulus ihn zulassen. `k` zählt nach dieser Wahl.
    ///
    /// `None` wenn `id` unbekannt. `config` wie bei [`knn_search`].
    pub fn knn_from_entry<'a>(
        &'a self,
        id: EntryId,
        k: usize,
        include_self: bool,
        config: impl IntoKnnConfig<'a, D, V>,
    ) -> Option<Vec<(Arc<ObjectNode<K, V, S>>, D)>> {
        let obj = self.get(id)?;
        if k == 0 {
            return Some(Vec::new());
        }
        let key = obj.key();
        let needle_eps = obj.epsilon();
        let ctx = KnnFromEntry {
            id,
            include_self,
            leaf: obj.parent(),
        };
        let cfg = config.into_knn_config();
        Some(self.knn_search_dispatch(&key, needle_eps, k, cfg, Some(&ctx)))
    }

    fn knn_search_dispatch(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        cfg: KnnConfigResolved<'_, D, V>,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let has_annulus = cfg.min_radius.is_some() || cfg.max_radius.is_some();
        let min_r = cfg.min_radius.unwrap_or_else(D::zero);
        let max_r = cfg.max_radius.unwrap_or_else(D::infinity);

        match (has_annulus, cfg.is_active) {
            (false, None) => self.knn_search_plain(needle, needle_eps, k, from),
            (false, Some(active)) => self.knn_search_filtered(
                needle,
                needle_eps,
                k,
                active.as_ref(),
                cfg.include_inactive,
                from,
            ),
            (true, None) => self.knn_search_range(needle, needle_eps, k, min_r, max_r, from),
            (true, Some(active)) => self.knn_search_range_filtered(
                needle,
                needle_eps,
                k,
                min_r,
                max_r,
                active.as_ref(),
                cfg.include_inactive,
                from,
            ),
        }
    }

    /// Plain k-NN (PlaceholderQueue + Pruning für D=f64, sonst Fallback)
    pub(crate) fn knn_search_plain(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_with_placeholder_queue(needle, needle_eps, k, from);
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_fallback(needle, needle_eps, k, from)
    }

    /// k-NN mit PlaceholderQueue und dynamischem Pruning (intern, D als f64 verwendet)
    fn knn_search_with_placeholder_queue(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let root = self.root.as_ref().unwrap().clone();
        let distance_fn = self.distance_fn.clone_box();

        let compare = |a: &KnnDistanceKey, b: &KnnDistanceKey| a < b;
        let mut placeholder_queue =
            PlaceholderQueue::new(k, compare, KnnDistanceKey(f64::INFINITY));
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64)> = Vec::new();
        let mut subtree_id: usize = 0;
        let mut object_id: usize = 0;
        let prune_cell = Cell::new(f64::INFINITY);

        if let Some(ctx) = from {
            if let Some(leaf) = &ctx.leaf {
                let leaf_guard = leaf.lock().unwrap();
                visit_seeded_leaf_objects(
                    &leaf_guard,
                    needle,
                    needle_eps,
                    ctx,
                    distance_fn.as_ref(),
                    &prune_cell,
                    |obj, dist| {
                        let pruning_radius = if placeholder_queue.len() >= k {
                            placeholder_queue.get_max_key().0
                        } else {
                            f64::INFINITY
                        };
                        if dist <= pruning_radius {
                            results.push((obj, dist));
                            placeholder_queue.add_placeholder(
                                KnnTag::Object(object_id),
                                KnnDistanceKey(dist),
                                1,
                                KnnTag::Object(object_id),
                            );
                            object_id = object_id.wrapping_add(1);
                            if placeholder_queue.len() >= k {
                                prune_cell.set(placeholder_queue.get_max_key().0);
                            }
                        }
                    },
                );
                drop(leaf_guard);
                if Arc::ptr_eq(leaf, &root) {
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    return results.into_iter().take(k).collect();
                }
            }
        }

        let root_distance = {
            let root_guard = root.lock().unwrap();
            metric_f64(distance_fn.as_ref(), &root_guard.key, root_guard.epsilon, needle, needle_eps)
        };

        let mut pq: BinaryHeap<KnnQueueEntry<K, V, S>> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_distance,
        });

        while let Some(entry) = pq.pop() {
            let pruning_radius = if placeholder_queue.len() >= k {
                placeholder_queue.get_max_key().0
            } else {
                f64::INFINITY
            };
            let pruning_radius_relaxed = if pruning_radius.is_finite() {
                pruning_radius * UPPER_BOUND_FACTOR
            } else {
                f64::INFINITY
            };

            if is_seeded_leaf(from, &entry.node) {
                continue;
            }

            let node_guard = entry.node.lock().unwrap();
            if node_guard.is_leaf {
                for child in &node_guard.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        if let Some(ctx) = from {
                            if obj_node.id == ctx.id && !ctx.include_self {
                                continue;
                            }
                        }
                        let dist = if from.map(|c| c.id == obj_node.id).unwrap_or(false) {
                            0.0
                        } else {
                            metric_f64(
                                distance_fn.as_ref(),
                                &obj_node.key(),
                                obj_node.epsilon(),
                                needle,
                                needle_eps,
                            )
                        };
                        if dist <= pruning_radius {
                            results.push((obj_node.clone(), dist));
                            placeholder_queue.add_placeholder(
                                KnnTag::Object(object_id),
                                KnnDistanceKey(dist),
                                1,
                                KnnTag::Object(object_id),
                            );
                            object_id = object_id.wrapping_add(1);
                        }
                    }
                }
                continue;
            }

            for child in &node_guard.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let center_dist = metric_f64(
                        distance_fn.as_ref(),
                        &child_guard.key,
                        child_guard.epsilon,
                        needle,
                        needle_eps,
                    );
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
                    let tag = KnnTag::Subtree(subtree_id);
                    placeholder_queue.add_placeholder(tag, KnnDistanceKey(upper_bound), 1, tag);
                    subtree_id = subtree_id.wrapping_add(1);
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.into_iter().take(k).collect()
    }

    /// Fallback für nicht-f64 Distanztypen: alles sammeln, sortieren, filtern.
    fn knn_search_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let max_radius = D::infinity();
        let mut results: Vec<_> = self.query_ball(needle, needle_eps, max_radius).collect();
        retain_exclude_self(&mut results, from);
        results.sort_by(|a, b| {
            let da = dist_as_f64(a.1);
            let db = dist_as_f64(b.1);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        results.into_iter().take(k).collect()
    }

    /// k-NN im Annulus: die k nächsten Punkte mit `min_radius < dist ≤ max_radius`.
    pub(crate) fn knn_search_range(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 =
                self.knn_search_range_with_queue(needle, needle_eps, k, min_radius, max_radius, from);
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        self.knn_search_range_fallback(needle, needle_eps, k, min_radius, max_radius, from)
    }

    fn knn_search_range_with_queue(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let root = self.root.as_ref().unwrap().clone();
        let distance_fn = self.distance_fn.clone_box();

        let mut best: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64)> = Vec::new();
        let prune_cell = Cell::new(f64::INFINITY);

        if let Some(ctx) = from {
            if let Some(leaf) = &ctx.leaf {
                let leaf_guard = leaf.lock().unwrap();
                visit_seeded_leaf_objects(
                    &leaf_guard,
                    needle,
                    needle_eps,
                    ctx,
                    distance_fn.as_ref(),
                    &prune_cell,
                    |obj, dist| {
                        if dist <= min_r || dist > max_r {
                            return;
                        }
                        let kth = if best.len() >= k {
                            best.peek().map(|d| d.0).unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        let effective_max = kth.min(max_r);
                        let effective_max_relaxed = if effective_max.is_finite() {
                            effective_max * UPPER_BOUND_FACTOR
                        } else {
                            f64::INFINITY
                        };
                        if dist > effective_max_relaxed {
                            return;
                        }
                        results.push((obj, dist));
                        if best.len() < k {
                            best.push(KnnDistanceKey(dist));
                        } else if let Some(max) = best.peek() {
                            if dist < max.0 {
                                best.pop();
                                best.push(KnnDistanceKey(dist));
                            }
                        }
                        let kth = if best.len() >= k {
                            best.peek().map(|d| d.0).unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        prune_cell.set(kth.min(max_r));
                    },
                );
                drop(leaf_guard);
                if Arc::ptr_eq(leaf, &root) {
                    let final_cap = if best.len() >= k {
                        best.peek().map(|d| d.0).unwrap_or(max_r).min(max_r)
                    } else {
                        max_r
                    };
                    results.retain(|(_, dist)| {
                        let within_k = best.len() < k || *dist <= final_cap * UPPER_BOUND_FACTOR;
                        *dist > min_r && *dist <= max_r && within_k
                    });
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    return results.into_iter().take(k).collect();
                }
            }
        }

        let (root_distance, root_upper) = {
            let root_guard = root.lock().unwrap();
            let d = metric_f64(
                distance_fn.as_ref(),
                &root_guard.key,
                root_guard.epsilon,
                needle,
                needle_eps,
            );
            (d, d + root_guard.covering_radius)
        };

        let mut pq: BinaryHeap<KnnQueueEntry<K, V, S>> = BinaryHeap::new();
        pq.push(KnnQueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_upper,
        });

        while let Some(entry) = pq.pop() {
            let kth = if best.len() >= k {
                best.peek().map(|d| d.0).unwrap_or(f64::INFINITY)
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
            if best.len() >= k && entry.distance_bound > effective_max_relaxed {
                break;
            }

            if is_seeded_leaf(from, &entry.node) {
                continue;
            }

            let node_guard = entry.node.lock().unwrap();
            if node_guard.is_leaf {
                for child in &node_guard.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        if let Some(ctx) = from {
                            if obj_node.id == ctx.id && !ctx.include_self {
                                continue;
                            }
                        }
                        let dist = if from.map(|c| c.id == obj_node.id).unwrap_or(false) {
                            0.0
                        } else {
                            metric_f64(
                                distance_fn.as_ref(),
                                &obj_node.key(),
                                obj_node.epsilon(),
                                needle,
                                needle_eps,
                            )
                        };
                        if dist <= min_r || dist > max_r {
                            continue;
                        }
                        if dist > effective_max_relaxed {
                            continue;
                        }

                        results.push((obj_node.clone(), dist));
                        if best.len() < k {
                            best.push(KnnDistanceKey(dist));
                        } else if let Some(max) = best.peek() {
                            if dist < max.0 {
                                best.pop();
                                best.push(KnnDistanceKey(dist));
                            }
                        }
                    }
                }
                continue;
            }

            for child in &node_guard.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let center_dist = metric_f64(
                        distance_fn.as_ref(),
                        &child_guard.key,
                        child_guard.epsilon,
                        needle,
                        needle_eps,
                    );
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

        let final_cap = if best.len() >= k {
            best.peek().map(|d| d.0).unwrap_or(max_r).min(max_r)
        } else {
            max_r
        };

        results.retain(|(_, dist)| {
            let within_k = best.len() < k || *dist <= final_cap * UPPER_BOUND_FACTOR;
            *dist > min_r && *dist <= max_r && within_k
        });
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.into_iter().take(k).collect()
    }

    fn knn_search_range_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
        from: Option<&KnnFromEntry<K, V, S>>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let mut all: Vec<_> = self.query_annulus(needle, needle_eps, min_radius, max_radius).collect();
        retain_exclude_self(&mut all, from);
        all.retain(|(_, d)| {
            let dist = dist_as_f64(*d);
            dist > min_r && dist <= max_r
        });
        all.sort_by(|a, b| {
            dist_as_f64(a.1)
                .partial_cmp(&dist_as_f64(b.1))
                .unwrap_or(Ordering::Equal)
        });
        all.into_iter().take(k).collect()
    }
}
