// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Such-APIs für [`MTree`](crate::tree::MTree).

use crate::distance::Distance;
use crate::entry::EntryId;
use crate::filtered::query::{FilteredQuery, FilteredRangeQuery};
use crate::node::{NodeId, NodePtr, ObjectNode, RoutingNode};
use crate::query::{Query, RangeQuery};
use crate::search_config::{
    IntoKnnConfig, IntoRangeSearchConfig, IntoSearchConfig, KnnConfigResolved,
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

fn entry_hits<K, V, D, S>(hits: Vec<(Arc<ObjectNode<K, V, S>>, D)>) -> Vec<(EntryId, D)>
where
    S: NodeStats<K, V>,
{
    hits.into_iter().map(|(node, dist)| (node.id, dist)).collect()
}

/// Wie der Seed in der k-NN-Traversierung behandelt wird.
pub(crate) enum KnnSeedMode {
    /// Needle = gespeicherter Eintrag; Distanz 0, optional Self ausschließen.
    FromEntry { include_self: bool },
    /// Beliebige Needle; Seed nur Pruning-Hint mit echter Distanz.
    Hint,
}

/// Seed-Kontext: Leaf zuerst besuchen (FromEntry oder Hint).
pub(crate) struct KnnSeedCtx {
    pub id: EntryId,
    pub leaf: Option<NodeId>,
    pub mode: KnnSeedMode,
}

impl KnnSeedCtx {
    pub(crate) fn exclude_self(&self) -> bool {
        matches!(self.mode, KnnSeedMode::FromEntry { include_self: false })
    }

    /// Distanz 0 nur im FromEntry-Modus (Needle-Key = Seed-Key).
    pub(crate) fn zero_dist_for(&self, id: EntryId) -> bool {
        id == self.id && matches!(self.mode, KnnSeedMode::FromEntry { .. })
    }
}

pub(crate) fn is_seeded_leaf(from: Option<&KnnSeedCtx>, node: NodeId) -> bool {
    from.and_then(|f| f.leaf).map(|leaf| leaf == node).unwrap_or(false)
}

/// Seed-Leaf zuerst bewerten: FromEntry mit Distanz 0, Hint mit echter Distanz.
pub(crate) fn visit_seeded_leaf_objects<K, V, D, S>(
    leaf: &RoutingNode<K, V, S>,
    needle: &K,
    needle_eps: f64,
    from: &KnnSeedCtx,
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
                match from.mode {
                    KnnSeedMode::FromEntry { include_self } => {
                        if include_self {
                            on_object(obj.clone(), 0.0);
                        }
                    }
                    KnnSeedMode::Hint => {
                        let pr = pruning_radius.get();
                        let dist =
                            metric_f64(distance_fn, obj.key(), obj.epsilon(), needle, needle_eps);
                        if !(pr.is_finite() && dist > pr * UPPER_BOUND_FACTOR) {
                            on_object(obj.clone(), dist);
                        }
                    }
                }
                continue;
            }
            let pr = pruning_radius.get();
            let dist = metric_f64(distance_fn, obj.key(), obj.epsilon(), needle, needle_eps);
            if pr.is_finite() && dist > pr * UPPER_BOUND_FACTOR {
                continue;
            }
            on_object(obj.clone(), dist);
        }
    }
}

fn retain_exclude_self<K, V, S, D>(
    results: &mut Vec<(Arc<ObjectNode<K, V, S>>, D)>,
    from: Option<&KnnSeedCtx>,
) where
    S: NodeStats<K, V>,
{
    if let Some(from) = from {
        if from.exclude_self() {
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

/// Max-Heap-Eintrag der k besten Treffer (größte Distanz oben → leichtes Evict).
struct KnnHitEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    dist: f64,
    node: Arc<ObjectNode<K, V, S>>,
}

impl<K, V, S> PartialEq for KnnHitEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl<K, V, S> Eq for KnnHitEntry<K, V, S> where S: NodeStats<K, V> {}
impl<K, V, S> PartialOrd for KnnHitEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl<K, V, S> Ord for KnnHitEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

fn note_knn_hit<K, V, S>(
    hits: &mut BinaryHeap<KnnHitEntry<K, V, S>>,
    best: &mut BinaryHeap<KnnDistanceKey>,
    k: usize,
    node: Arc<ObjectNode<K, V, S>>,
    dist: f64,
) where
    S: NodeStats<K, V>,
{
    if best.len() < k {
        best.push(KnnDistanceKey(dist));
        hits.push(KnnHitEntry { dist, node });
    } else if dist < best.peek().map(|d| d.0).unwrap_or(f64::INFINITY) {
        best.pop();
        best.push(KnnDistanceKey(dist));
        hits.pop();
        hits.push(KnnHitEntry { dist, node });
    }
}

fn finish_knn_hits<K, V, S>(
    hits: BinaryHeap<KnnHitEntry<K, V, S>>,
) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)>
where
    S: NodeStats<K, V>,
{
    let mut out: Vec<_> = hits.into_iter().map(|h| (h.node, h.dist)).collect();
    // Distanz primär; EntryId sekundär für deterministische Gleichstands-Ordnung.
    out.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.id.slot_index().cmp(&b.0.id.slot_index()))
    });
    out
}

/// Triangle-Inequality-LB vor Distanzberechnung zum Kind.
#[inline]
pub(crate) fn triangle_lower_bound(
    parent_center_dist: f64,
    child_parent_distance: f64,
    covering_radius: f64,
) -> f64 {
    ((parent_center_dist - child_parent_distance).abs() - covering_radius).max(0.0)
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
    ) -> Query<'_, K, V, D, S> {
        if let Some(root) = self.root {
            Query::new(
                self,
                needle.clone(),
                needle_eps,
                min_radius,
                max_radius,
                root,
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
    ) -> RangeQuery<'_, K, V, D, S> {
        if let Some(root) = self.root {
            RangeQuery::new(self, needle.clone(), needle_eps, radius, root)
        } else {
            RangeQuery::empty()
        }
    }

    /// Bereichssuche (Annulus): `min_radius < dist ≤ max_radius`, optional gefiltert.
    pub fn search<'a>(
        &'a self,
        needle: &K,
        config: impl IntoSearchConfig<'a, D, V>,
    ) -> FilteredQuery<'a, K, V, D, S, Box<dyn Fn(EntryId, &V) -> bool + 'a>> {
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
    ) -> FilteredRangeQuery<'a, K, V, D, S, Box<dyn Fn(EntryId, &V) -> bool + 'a>> {
        let cfg = config.into_range_search_config();
        let inner = self.query_ball(needle, cfg.epsilon, cfg.radius);
        let pred: Box<dyn Fn(EntryId, &V) -> bool + 'a> =
            cfg.is_active.unwrap_or_else(|| Box::new(|_, _| true));
        FilteredRangeQuery::new(inner, pred)
    }

    /// k-Nearest-Neighbor-Suche.
    ///
    /// `config` kann `()`, `None` oder [`KnnConfig`](crate::search_config::KnnConfig) sein:
    /// optional Filter, `include_inactive`, Annulus `min_radius`/`max_radius`, sowie
    /// `.seed(EntryId)` als Pruning-Hint.
    pub fn knn_search<'a>(
        &'a self,
        needle: &K,
        k: usize,
        config: impl IntoKnnConfig<'a, D, V>,
    ) -> Vec<(EntryId, D)> {
        let cfg = config.into_knn_config();
        let seed_ctx = cfg.seed.and_then(|id| {
            self.get(id).map(|obj| KnnSeedCtx {
                id,
                leaf: obj.parent(),
                mode: KnnSeedMode::Hint,
            })
        });
        entry_hits(self.knn_search_dispatch(needle, cfg.epsilon, k, cfg, seed_ctx.as_ref()))
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
    ) -> Option<Vec<(EntryId, D)>> {
        let obj = self.get(id)?;
        if k == 0 {
            return Some(Vec::new());
        }
        let needle_eps = obj.epsilon();
        let ctx = KnnSeedCtx {
            id,
            leaf: obj.parent(),
            mode: KnnSeedMode::FromEntry { include_self },
        };
        let cfg = config.into_knn_config();
        Some(entry_hits(self.knn_search_dispatch(
            obj.key(),
            needle_eps,
            k,
            cfg,
            Some(&ctx),
        )))
    }

    fn knn_search_dispatch(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        cfg: KnnConfigResolved<'_, D, V>,
        from: Option<&KnnSeedCtx>,
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

    /// Plain k-NN (Pruning über bestätigte Distanzen für D=f64, sonst Fallback)
    pub(crate) fn knn_search_plain(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnSeedCtx>,
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

    /// k-NN mit dynamischem Pruning über die k besten bestätigten Distanzen.
    fn knn_search_with_placeholder_queue(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let root = self.root.unwrap();
        let distance_fn = self.distance_fn.as_ref();

        // Nur bestätigte Objektdistanzen. Teilbaum-Oberschranken zählen nicht als Nachbar,
        // sonst fällt der Pruning-Radius unter die echte k-te Distanz.
        let mut best: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut hits: BinaryHeap<KnnHitEntry<K, V, S>> = BinaryHeap::new();
        let prune_cell = Cell::new(f64::INFINITY);
        let confirmed_kth = |best: &BinaryHeap<KnnDistanceKey>| -> f64 {
            if best.len() >= k {
                best.peek().map(|d| d.0).unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
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
                        let pruning_radius = confirmed_kth(&best);
                        if dist <= pruning_radius {
                            note_knn_hit(&mut hits, &mut best, k, obj, dist);
                            prune_cell.set(confirmed_kth(&best));
                        }
                    },
                );
                if leaf_id == root {
                    return finish_knn_hits(hits);
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
            let pruning_radius = confirmed_kth(&best);
            let pruning_radius_relaxed = if pruning_radius.is_finite() {
                pruning_radius * UPPER_BOUND_FACTOR
            } else {
                f64::INFINITY
            };

            // Pop-Skip: LB der PQ-Priorität über k-ter Distanz → Rest der PQ ebenfalls.
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
                        if let Some(ctx) = from {
                            if ctx.exclude_self() && obj_node.id == ctx.id {
                                continue;
                            }
                        }
                        let dist = if from.map(|c| c.zero_dist_for(obj_node.id)).unwrap_or(false)
                        {
                            0.0
                        } else {
                            metric_f64(
                                distance_fn,
                                obj_node.key(),
                                obj_node.epsilon(),
                                needle,
                                needle_eps,
                            )
                        };
                        if dist <= pruning_radius {
                            note_knn_hit(&mut hits, &mut best, k, obj_node.clone(), dist);
                        }
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

        finish_knn_hits(hits)
    }

    /// Fallback für nicht-f64 Distanztypen: alles sammeln, sortieren, filtern.
    fn knn_search_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let max_radius = D::infinity();
        let mut query = self.query_ball(needle, needle_eps, max_radius);
        let mut results = Vec::new();
        while let Some(hit) = query.next_object() {
            results.push(hit);
        }
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
        from: Option<&KnnSeedCtx>,
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
        from: Option<&KnnSeedCtx>,
    ) -> Vec<(Arc<ObjectNode<K, V, S>>, f64)> {
        let min_r = dist_as_f64(min_radius);
        let max_r = dist_as_f64(max_radius);
        if !(min_r < max_r) {
            return Vec::new();
        }

        let root = self.root.unwrap();
        let distance_fn = self.distance_fn.as_ref();

        let mut best: BinaryHeap<KnnDistanceKey> = BinaryHeap::new();
        let mut hits: BinaryHeap<KnnHitEntry<K, V, S>> = BinaryHeap::new();
        let prune_cell = Cell::new(f64::INFINITY);

        if let Some(ctx) = from {
            if let Some(leaf_id) = ctx.leaf {
                let leaf = &self.nodes[leaf_id.index()];
                visit_seeded_leaf_objects(
                    leaf,
                    needle,
                    needle_eps,
                    ctx,
                    distance_fn,
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
                        note_knn_hit(&mut hits, &mut best, k, obj, dist);
                        let kth = if best.len() >= k {
                            best.peek().map(|d| d.0).unwrap_or(f64::INFINITY)
                        } else {
                            f64::INFINITY
                        };
                        prune_cell.set(kth.min(max_r));
                    },
                );
                if leaf_id == root {
                    return finish_knn_hits(hits);
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

            if entry.upper_bound < min_prune {
                continue;
            }
            // LB-Best-First: alle weiteren Einträge haben ≥ aktuelle LB.
            if best.len() >= k && entry.distance_bound > effective_max_relaxed {
                break;
            }

            if is_seeded_leaf(from, entry.node) {
                continue;
            }

            let node = &self.nodes[entry.node.index()];
            if node.is_leaf {
                for child in &node.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        if let Some(ctx) = from {
                            if ctx.exclude_self() && obj_node.id == ctx.id {
                                continue;
                            }
                        }
                        let dist = if from.map(|c| c.zero_dist_for(obj_node.id)).unwrap_or(false)
                        {
                            0.0
                        } else {
                            metric_f64(
                                distance_fn,
                                obj_node.key(),
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

                        note_knn_hit(&mut hits, &mut best, k, obj_node.clone(), dist);
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

        finish_knn_hits(hits)
    }

    fn knn_search_range_fallback(
        &self,
        needle: &K,
        needle_eps: f64,
        k: usize,
        min_radius: D,
        max_radius: D,
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
