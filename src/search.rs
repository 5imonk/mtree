// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Ungefilterte Such-APIs für [`MTree`](crate::tree::MTree).

use crate::node::{NodePtr, ObjectNode};
use crate::placeholder_queue::PlaceholderQueue;
use crate::query::{Query, RangeQuery};
use crate::stats::NodeStats;
use crate::tree::{DistanceType, KnnDistanceKey, KnnQueueEntry, KnnTag, MTree, UPPER_BOUND_FACTOR};
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
    /// Bereichssuche
    pub fn search(&self, needle: &K, min_radius: D, max_radius: D) -> Query<K, V, D, S> {
        if let Some(ref root) = self.root {
            Query::new(
                needle.clone(),
                min_radius,
                max_radius,
                root.clone(),
                self.distance_fn.clone_box(),
            )
        } else {
            Query::empty()
        }
    }

    /// Radius-Suche
    pub fn range_search(&self, needle: &K, radius: D) -> RangeQuery<K, V, D, S> {
        if let Some(ref root) = self.root {
            RangeQuery::new(
                needle.clone(),
                radius,
                root.clone(),
                self.distance_fn.clone_box(),
            )
        } else {
            RangeQuery::empty()
        }
    }

    /// k-Nearest-Neighbor Suche (PlaceholderQueue + Pruning für D=f64, sonst Fallback)
    pub fn knn_search(&self, needle: &K, k: usize) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            let results_f64 = self.knn_search_with_placeholder_queue(needle, k);
            return results_f64
                .into_iter()
                .map(|(arc, d)| (arc, unsafe { std::mem::transmute_copy(&d) }))
                .collect();
        }
        return self.knn_search_fallback(needle, k);
    }

    /// k-NN mit PlaceholderQueue und dynamischem Pruning (intern, D als f64 verwendet)
    fn knn_search_with_placeholder_queue(
        &self,
        needle: &K,
        k: usize,
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

        let compare = |a: &KnnDistanceKey, b: &KnnDistanceKey| a < b;
        let mut placeholder_queue =
            PlaceholderQueue::new(k, compare, KnnDistanceKey(f64::INFINITY));
        let mut results: Vec<(Arc<ObjectNode<K, V, S>>, f64)> = Vec::new();
        let mut subtree_id: usize = 0;
        let mut object_id: usize = 0;

        while let Some(entry) = pq.pop() {
            let pruning_radius = placeholder_queue.get_max_key().0;
            let pruning_radius_relaxed = pruning_radius * UPPER_BOUND_FACTOR;

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
    fn knn_search_fallback(&self, needle: &K, k: usize) -> Vec<(Arc<ObjectNode<K, V, S>>, D)> {
        let max_radius = D::infinity();
        let mut results: Vec<_> = self.range_search(needle, max_radius).collect();
        results.sort_by(|a, b| {
            let da = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&a.1) }
            } else {
                0.0
            };
            let db = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&b.1) }
            } else {
                0.0
            };
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });
        results.into_iter().take(k).collect()
    }
}
