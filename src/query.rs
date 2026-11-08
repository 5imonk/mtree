// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use crate::entry::EntryId;
use crate::node::{NodeId, NodePtr, ObjectNode};
use crate::stats::NodeStats;
use crate::tree::{metric_distance, metric_f64, DistanceType, MTree};
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Queue-Eintrag für Query-Traversierung (min-heap auf Lower Bound).
struct QueueEntry {
    node: NodeId,
    center_distance: f64,
    /// Lower Bound — Ordnungskey der PQ.
    distance_bound: f64,
    /// Upper Bound — Annulus-Pruning.
    upper_bound: f64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance_bound == other.distance_bound
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.distance_bound.partial_cmp(&self.distance_bound)
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance_bound
            .partial_cmp(&self.distance_bound)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Query Iterator für Bereichssuchen
pub struct Query<'a, K, V, D = f64, S = crate::stats::DescendantCounter>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    tree: Option<&'a MTree<K, V, D, S>>,
    needle: K,
    needle_eps: f64,
    min_radius: D,
    max_radius: D,
    queue: BinaryHeap<QueueEntry>,
    current_leaf: Option<NodeId>,
    current_leaf_distance: f64,
    leaf_iterator: usize,
    is_at_end: bool,
}

impl<'a, K, V, D, S> Query<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub fn empty() -> Self {
        Self {
            tree: None,
            needle: unsafe { std::mem::zeroed() },
            needle_eps: 0.0,
            min_radius: D::zero(),
            max_radius: D::zero(),
            queue: BinaryHeap::new(),
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: true,
        }
    }

    pub fn new(
        tree: &'a MTree<K, V, D, S>,
        needle: K,
        needle_eps: f64,
        min_radius: D,
        max_radius: D,
        root: NodeId,
    ) -> Self {
        let mut queue: BinaryHeap<QueueEntry> = BinaryHeap::new();
        let root_node = &tree.nodes[root.index()];
        let root_distance = metric_f64(
            tree.distance_fn.as_ref(),
            &root_node.key,
            root_node.epsilon,
            &needle,
            needle_eps,
        );
        let r = root_node.covering_radius;
        let root_lb = (root_distance - r).max(0.0);
        let root_ub = root_distance + r;
        queue.push(QueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_lb,
            upper_bound: root_ub,
        });
        Self {
            tree: Some(tree),
            needle,
            needle_eps,
            min_radius,
            max_radius,
            queue,
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: false,
        }
    }

    pub fn at_end(&self) -> bool {
        self.is_at_end
    }

    fn find_next_leaf(&mut self) -> bool {
        let tree = match self.tree {
            Some(t) => t,
            None => return false,
        };
        while let Some(entry) = self.queue.pop() {
            let min_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&self.min_radius) }
            } else {
                0.0
            };
            let max_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&self.max_radius) }
            } else {
                f64::INFINITY
            };
            let min_prune = min_radius_f64 * crate::tree::LOWER_BOUND_FACTOR;
            let max_prune = max_radius_f64 * crate::tree::UPPER_BOUND_FACTOR;
            if entry.upper_bound < min_prune || entry.distance_bound >= max_prune {
                continue;
            }

            let node = &tree.nodes[entry.node.index()];

            if node.is_leaf {
                self.current_leaf = Some(entry.node);
                self.current_leaf_distance = entry.center_distance;
                self.leaf_iterator = 0;
                return true;
            }

            let routing_ids: Vec<NodeId> = node
                .children
                .iter()
                .filter_map(|child| child.as_routing_id())
                .collect();

            for child_id in routing_ids {
                let child_node = &tree.nodes[child_id.index()];
                let covering_radius = child_node.covering_radius;
                let tri_lb = crate::search::triangle_lower_bound(
                    entry.center_distance,
                    child_node.parent_distance,
                    covering_radius,
                );
                // Grobe Triangle-LB: wenn schon zu weit für max, Distanz sparen.
                if tri_lb >= max_prune {
                    continue;
                }
                let dist_f64 = metric_f64(
                    tree.distance_fn.as_ref(),
                    &child_node.key,
                    child_node.epsilon,
                    &self.needle,
                    self.needle_eps,
                );
                let upper_bound = dist_f64 + covering_radius;
                let lower_bound = (dist_f64 - covering_radius).max(0.0);
                // Annulus (min, max]: Subtree kann schneiden wenn upper > min und lower <= max
                if upper_bound >= min_prune && lower_bound < max_prune {
                    self.queue.push(QueueEntry {
                        node: child_id,
                        center_distance: dist_f64,
                        distance_bound: lower_bound,
                        upper_bound,
                    });
                }
            }
        }
        false
    }
}

impl<'a, K, V, D, S> Query<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub(crate) fn next_object(&mut self) -> Option<(Arc<ObjectNode<K, V, S>>, D)> {
        if self.is_at_end {
            return None;
        }

        loop {
            if let Some(leaf_id) = self.current_leaf {
                let tree = self.tree?;
                let leaf = &tree.nodes[leaf_id.index()];

                while self.leaf_iterator < leaf.children.len() {
                    if let NodePtr::Object(ref obj_node) = &leaf.children[self.leaf_iterator] {
                        let dist = metric_distance(
                            tree.distance_fn.as_ref(),
                            obj_node.key(),
                            obj_node.epsilon(),
                            &self.needle,
                            self.needle_eps,
                        );
                        let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&dist) }
                        } else {
                            0.0
                        };

                        let min_radius_f64 = if std::mem::size_of::<D>()
                            == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&self.min_radius) }
                        } else {
                            0.0
                        };
                        let max_radius_f64 = if std::mem::size_of::<D>()
                            == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&self.max_radius) }
                        } else {
                            f64::INFINITY
                        };

                        self.leaf_iterator += 1;

                        if dist_f64 > min_radius_f64 && dist_f64 <= max_radius_f64 {
                            return Some((obj_node.clone(), dist));
                        }
                    } else {
                        self.leaf_iterator += 1;
                    }
                }
            }

            // Kein weiteres Element im aktuellen Leaf, suche nächsten
            if !self.find_next_leaf() {
                self.is_at_end = true;
                return None;
            }
        }
    }
}

impl<'a, K, V, D, S> Iterator for Query<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_object().map(|(node, dist)| (node.id, dist))
    }
}

/// RangeQuery Iterator für Radius-Suchen
pub struct RangeQuery<'a, K, V, D = f64, S = crate::stats::DescendantCounter>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    tree: Option<&'a MTree<K, V, D, S>>,
    needle: K,
    needle_eps: f64,
    radius: D,
    queue: BinaryHeap<QueueEntry>,
    current_leaf: Option<NodeId>,
    current_leaf_distance: f64,
    leaf_iterator: usize,
    is_at_end: bool,
}

impl<'a, K, V, D, S> RangeQuery<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub fn empty() -> Self {
        Self {
            tree: None,
            needle: unsafe { std::mem::zeroed() },
            needle_eps: 0.0,
            radius: D::zero(),
            queue: BinaryHeap::new(),
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: true,
        }
    }

    pub fn new(
        tree: &'a MTree<K, V, D, S>,
        needle: K,
        needle_eps: f64,
        radius: D,
        root: NodeId,
    ) -> Self {
        let mut queue: BinaryHeap<QueueEntry> = BinaryHeap::new();
        let root_node = &tree.nodes[root.index()];
        let root_distance = metric_f64(
            tree.distance_fn.as_ref(),
            &root_node.key,
            root_node.epsilon,
            &needle,
            needle_eps,
        );
        let r = root_node.covering_radius;
        let root_lb = (root_distance - r).max(0.0);
        let root_ub = root_distance + r;
        queue.push(QueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_lb,
            upper_bound: root_ub,
        });
        Self {
            tree: Some(tree),
            needle,
            needle_eps,
            radius,
            queue,
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: false,
        }
    }

    pub fn at_end(&self) -> bool {
        self.is_at_end
    }

    fn find_next_leaf(&mut self) -> bool {
        let tree = match self.tree {
            Some(t) => t,
            None => return false,
        };
        while let Some(entry) = self.queue.pop() {
            let radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&self.radius) }
            } else {
                f64::INFINITY
            };
            let radius_prune = radius_f64 * crate::tree::UPPER_BOUND_FACTOR;
            // LB-Pop-Skip; upper_bound bleibt für Annulus-Query relevant / Layout-Parität.
            if entry.distance_bound > radius_prune {
                continue;
            }
            let _ = entry.upper_bound;

            let node = &tree.nodes[entry.node.index()];

            if node.is_leaf {
                self.current_leaf = Some(entry.node);
                self.current_leaf_distance = entry.center_distance;
                self.leaf_iterator = 0;
                return true;
            }

            let routing_ids: Vec<NodeId> = node
                .children
                .iter()
                .filter_map(|child| child.as_routing_id())
                .collect();

            for child_id in routing_ids {
                let child_node = &tree.nodes[child_id.index()];
                let covering_radius = child_node.covering_radius;
                let tri_lb = crate::search::triangle_lower_bound(
                    entry.center_distance,
                    child_node.parent_distance,
                    covering_radius,
                );
                if tri_lb > radius_prune {
                    continue;
                }
                let dist_f64 = metric_f64(
                    tree.distance_fn.as_ref(),
                    &child_node.key,
                    child_node.epsilon,
                    &self.needle,
                    self.needle_eps,
                );
                let lower_bound = (dist_f64 - covering_radius).max(0.0);
                let upper_bound = dist_f64 + covering_radius;
                if lower_bound <= radius_prune {
                    self.queue.push(QueueEntry {
                        node: child_id,
                        center_distance: dist_f64,
                        distance_bound: lower_bound,
                        upper_bound,
                    });
                }
            }
        }
        false
    }
}

impl<'a, K, V, D, S> RangeQuery<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub(crate) fn next_object(&mut self) -> Option<(Arc<ObjectNode<K, V, S>>, D)> {
        if self.is_at_end {
            return None;
        }

        loop {
            if let Some(leaf_id) = self.current_leaf {
                let tree = self.tree?;
                let leaf = &tree.nodes[leaf_id.index()];

                while self.leaf_iterator < leaf.children.len() {
                    if let NodePtr::Object(ref obj_node) = &leaf.children[self.leaf_iterator] {
                        let dist = metric_distance(
                            tree.distance_fn.as_ref(),
                            obj_node.key(),
                            obj_node.epsilon(),
                            &self.needle,
                            self.needle_eps,
                        );
                        let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&dist) }
                        } else {
                            0.0
                        };

                        let radius_f64 = if std::mem::size_of::<D>()
                            == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&self.radius) }
                        } else {
                            f64::INFINITY
                        };

                        self.leaf_iterator += 1;

                        if dist_f64 <= radius_f64 {
                            return Some((obj_node.clone(), dist));
                        }
                    } else {
                        self.leaf_iterator += 1;
                    }
                }
            }

            // Kein weiteres Element im aktuellen Leaf, suche nächsten
            if !self.find_next_leaf() {
                self.is_at_end = true;
                return None;
            }
        }
    }
}

impl<'a, K, V, D, S> Iterator for RangeQuery<'a, K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_object().map(|(node, dist)| (node.id, dist))
    }
}
