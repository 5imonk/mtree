// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Reverse-kNN Query und Insert-Maintenance-Walk (preInsert-Äquivalent).

use crate::entry::EntryId;
use crate::knn_graph::KnnGraph;
use crate::node::{NodeId, NodePtr};
use crate::tree::{dist_value_as_f64, DistanceType, MTree};
use crate::distance::Distance;
use crate::stats::NodeStats;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::hash::Hash;

#[derive(Clone)]
struct MaintHeapEntry {
    dist: f64,
    id: EntryId,
}

impl PartialEq for MaintHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for MaintHeapEntry {}
impl PartialOrd for MaintHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaintHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by dist (worst of knn candidates on top); tie-break by id.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.slot_index().cmp(&other.id.slot_index()))
    }
}

impl<K, V, D, S, Dist> MTree<K, V, D, S, Dist>
where
    K: Clone + Send + Sync + Hash + Eq + Default,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Default + Clone,
    Dist: Distance<K, Output = D>,
{
    /// Ob der kNN-Graph aktiv ist.
    pub fn knn_graph_enabled(&self) -> bool {
        self.knn_graph.is_some()
    }

    /// `k_max` des Graphen, falls aktiv.
    pub fn knn_graph_k_max(&self) -> Option<usize> {
        self.knn_graph.as_ref().map(|g| g.k_max)
    }

    /// Forward-kNN-Liste eines Eintrags aus dem Graphen (falls aktiv und belegt).
    pub fn knn_graph_neighbors(&self, id: EntryId) -> Option<&[(EntryId, f64)]> {
        self.knn_graph.as_ref()?.forward_list(id)
    }

    /// Reverse-kNN: alle Einträge `o` mit `d(needle, o) ≤ knnDist_k(o)`.
    ///
    /// `None` wenn Graph aus, `k == 0` oder `k > k_max`.
    pub fn rknn_search(&self, needle: &K, k: usize) -> Option<Vec<(EntryId, D)>> {
        self.rknn_search_eps(needle, 0.0, k)
    }

    /// Wie [`rknn_search`], mit Needle-Epsilon (Identity-Metrik).
    pub fn rknn_search_eps(&self, needle: &K, needle_eps: f64, k: usize) -> Option<Vec<(EntryId, D)>> {
        let graph = self.knn_graph.as_ref()?;
        if k == 0 || k > graph.k_max {
            return None;
        }
        let root = self.root?;
        let mut out = Vec::new();
        self.rknn_visit(root, needle, needle_eps, k, graph, &mut out);
        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
        });
        Some(out)
    }

    /// Reverse-kNN von einem gespeicherten Eintrag (ohne Self).
    pub fn rknn_from_entry(&self, id: EntryId, k: usize) -> Option<Vec<(EntryId, D)>> {
        let (key, eps) = {
            let node = self.get(id)?;
            (node.key().clone(), node.epsilon())
        };
        let mut hits = self.rknn_search_eps(&key, eps, k)?;
        hits.retain(|(oid, _)| *oid != id);
        Some(hits)
    }

    fn rknn_visit(
        &self,
        node_id: NodeId,
        needle: &K,
        needle_eps: f64,
        k: usize,
        graph: &KnnGraph,
        out: &mut Vec<(EntryId, D)>,
    ) {
        let node = &self.nodes[node_id.index()];
        if node.is_leaf {
            for child in &node.children {
                let NodePtr::Object(oid) = child else { continue };
                let obj = self.object(*oid);
                let dist = self.metric(needle, needle_eps, obj.key(), obj.epsilon());
                let dist_f = dist_value_as_f64(dist);
                let Some(rank_d) = graph.rank_distance(*oid, k) else {
                    continue;
                };
                if dist_f <= rank_d {
                    out.push((*oid, dist));
                }
            }
            return;
        }

        for child in &node.children {
            match child {
                NodePtr::Object(oid) => {
                    let obj = self.object(*oid);
                    let dist = self.metric(needle, needle_eps, obj.key(), obj.epsilon());
                    let dist_f = dist_value_as_f64(dist);
                    if let Some(rank_d) = graph.rank_distance(*oid, k) {
                        if dist_f <= rank_d {
                            out.push((*oid, dist));
                        }
                    }
                }
                NodePtr::Routing(cid) => {
                    let child_node = &self.nodes[cid.index()];
                    let center_dist = dist_value_as_f64(self.metric(
                        needle,
                        needle_eps,
                        &child_node.key,
                        child_node.epsilon,
                    ));
                    let min_dist = (center_dist - child_node.covering_radius).max(0.0);
                    if min_dist <= child_node.max_knn_distance {
                        self.rknn_visit(*cid, needle, needle_eps, k, graph, out);
                    }
                }
            }
        }
    }

    /// Nach geometrischem Insert: Graph-Maintenance (preInsert-Äquivalent).
    pub(crate) fn knn_graph_maintain_insert(
        &mut self,
        q: EntryId,
        preset_hits: Option<&[(EntryId, f64)]>,
    ) {
        if self.knn_graph.is_none() {
            return;
        }
        let k_max = self.knn_graph.as_ref().unwrap().k_max;
        self.knn_graph.as_mut().unwrap().ensure_slot(q.slot_index());

        let (q_key, q_eps) = {
            let obj = self.object(q);
            (obj.key().clone(), obj.epsilon())
        };

        let mut forward_hits: Vec<(EntryId, f64)> = if let Some(hits) = preset_hits {
            hits.iter()
                .copied()
                .filter(|(id, d)| *id != q && d.is_finite())
                .take(k_max)
                .collect()
        } else {
            Vec::new()
        };
        let build_forward = preset_hits.is_none();

        let mut forward_heap: BinaryHeap<MaintHeapEntry> = BinaryHeap::new();
        let mut touched: Vec<EntryId> = Vec::new();

        // Combined / victim walk from root. Do not refresh max_knn mid-walk
        // (shrinking max would risk false pruning of later victims).
        if let Some(root) = self.root {
            let mut stack = vec![root];
            while let Some(nid) = stack.pop() {
                let is_leaf = self.nodes[nid.index()].is_leaf;
                let children: Vec<NodePtr> = self.nodes[nid.index()].children.clone();
                if is_leaf {
                    for child in children {
                        let NodePtr::Object(pid) = child else { continue };
                        if pid == q {
                            continue;
                        }
                        let (p_key, p_eps) = {
                            let obj = self.object(pid);
                            (obj.key().clone(), obj.epsilon())
                        };
                        let dist = dist_value_as_f64(self.metric(&q_key, q_eps, &p_key, p_eps));

                        if build_forward {
                            if forward_heap.len() < k_max {
                                forward_heap.push(MaintHeapEntry { dist, id: pid });
                            } else if let Some(top) = forward_heap.peek() {
                                if dist < top.dist {
                                    forward_heap.pop();
                                    forward_heap.push(MaintHeapEntry { dist, id: pid });
                                }
                            }
                        }

                        let knn_d = self.knn_graph.as_ref().unwrap().knn_distance(pid);
                        if dist <= knn_d {
                            let (ins, _) = self
                                .knn_graph
                                .as_mut()
                                .unwrap()
                                .try_insert_neighbor(pid, q, dist);
                            if ins {
                                touched.push(pid);
                            }
                        }
                    }
                } else {
                    for child in children {
                        match child {
                            NodePtr::Object(pid) => {
                                if pid == q {
                                    continue;
                                }
                                let (p_key, p_eps) = {
                                    let obj = self.object(pid);
                                    (obj.key().clone(), obj.epsilon())
                                };
                                let dist =
                                    dist_value_as_f64(self.metric(&q_key, q_eps, &p_key, p_eps));
                                if build_forward {
                                    if forward_heap.len() < k_max {
                                        forward_heap.push(MaintHeapEntry { dist, id: pid });
                                    } else if let Some(top) = forward_heap.peek() {
                                        if dist < top.dist {
                                            forward_heap.pop();
                                            forward_heap.push(MaintHeapEntry { dist, id: pid });
                                        }
                                    }
                                }
                                let knn_d = self.knn_graph.as_ref().unwrap().knn_distance(pid);
                                if dist <= knn_d {
                                    let (ins, _) = self
                                        .knn_graph
                                        .as_mut()
                                        .unwrap()
                                        .try_insert_neighbor(pid, q, dist);
                                    if ins {
                                        touched.push(pid);
                                    }
                                }
                            }
                            NodePtr::Routing(cid) => {
                                let child_node = &self.nodes[cid.index()];
                                let center_dist = dist_value_as_f64(self.metric(
                                    &q_key,
                                    q_eps,
                                    &child_node.key,
                                    child_node.epsilon,
                                ));
                                let min_dist = (center_dist - child_node.covering_radius).max(0.0);
                                let fwd_bound = if build_forward {
                                    if forward_heap.len() < k_max {
                                        f64::INFINITY
                                    } else {
                                        forward_heap.peek().map(|e| e.dist).unwrap_or(f64::INFINITY)
                                    }
                                } else {
                                    0.0
                                };
                                // Victim-Seite: kein max_knn-Pruning während Mutation
                                // (Splits lassen Max ggf. stale/0). Forward darf prunen.
                                let need_forward = build_forward && min_dist <= fwd_bound;
                                let need_victim = true;
                                if need_forward || need_victim {
                                    stack.push(cid);
                                }
                            }
                        }
                    }
                }
            }
        }

        if build_forward {
            forward_hits = forward_heap
                .into_sorted_vec()
                .into_iter()
                .map(|e| (e.id, e.dist))
                .collect();
        } else {
            forward_hits.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
            });
            forward_hits.truncate(k_max);
        }

        self.knn_graph
            .as_mut()
            .unwrap()
            .set_forward(q, forward_hits);
        touched.push(q);
        let _ = touched;
        self.recompute_all_max_knn();
    }

    /// Vor geometrischem Erase: Graph-Opfer patchen und Kanten von `q` lösen.
    pub(crate) fn knn_graph_maintain_erase(&mut self, q: EntryId) {
        if self.knn_graph.is_none() {
            return;
        }
        let victims: Vec<EntryId> = self
            .knn_graph
            .as_ref()
            .unwrap()
            .reverse_list(q)
            .to_vec();

        for owner in &victims {
            self.knn_graph
                .as_mut()
                .unwrap()
                .remove_from_forward(*owner, q);
        }

        // Drop q's forward reverse-edges, then clear q's lists.
        if let Some(list) = self
            .knn_graph
            .as_ref()
            .unwrap()
            .forward_list(q)
            .map(|s| s.to_vec())
        {
            for (nb, _) in list {
                self.knn_graph
                    .as_mut()
                    .unwrap()
                    .remove_from_forward(q, nb);
            }
        }
        self.knn_graph.as_mut().unwrap().release_slot(q);

        // Refill after q is removed from all lists (still geometrically present —
        // exclude q explicitly so knn_search cannot re-add it).
        for owner in victims {
            self.refill_forward_after_erase(owner, q);
        }
        self.recompute_all_max_knn();
    }

    fn refill_forward_after_erase(&mut self, owner: EntryId, exclude: EntryId) {
        let k_max = self.knn_graph.as_ref().unwrap().k_max;
        let current_len = self
            .knn_graph
            .as_ref()
            .unwrap()
            .forward_list(owner)
            .map(|l| l.len())
            .unwrap_or(0);
        if current_len >= k_max {
            return;
        }
        let need = k_max;
        let existing: Vec<EntryId> = self
            .knn_graph
            .as_ref()
            .unwrap()
            .forward_list(owner)
            .map(|l| l.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default();

        let (key, eps) = {
            let obj = self.object(owner);
            (obj.key().clone(), obj.epsilon())
        };
        let hits = self.knn_search_eps_raw(&key, eps, need + 2 + existing.len());
        let mut additions = Vec::new();
        for (id, dist) in hits {
            if id == owner || id == exclude {
                continue;
            }
            if existing.contains(&id) {
                continue;
            }
            additions.push((id, dist_value_as_f64(dist)));
            if existing.len() + additions.len() >= k_max {
                break;
            }
        }
        for (id, dist) in additions {
            let _ = self
                .knn_graph
                .as_mut()
                .unwrap()
                .try_insert_neighbor(owner, id, dist);
        }
    }

    /// kNN ohne Graph-Config; interne Hilfe für Refill (plain, f64-Ausgabe als D).
    fn knn_search_eps_raw(&self, needle: &K, needle_eps: f64, k: usize) -> Vec<(EntryId, D)> {
        self.knn_search_with_scratch(
            needle,
            k,
            crate::search_config::KnnConfig::new().epsilon(needle_eps),
            &mut crate::scratch::QueryScratch::default(),
        )
    }

    /// Bottom-up Max-Aggregation über den ganzen Baum (nach Split/Mutation).
    pub(crate) fn recompute_all_max_knn(&mut self) {
        if self.knn_graph.is_none() {
            return;
        }
        let Some(root) = self.root else {
            return;
        };
        self.recompute_max_knn_subtree(root);
    }

    fn recompute_max_knn_subtree(&mut self, node_id: NodeId) -> f64 {
        let children: Vec<NodePtr> = self.nodes[node_id.index()].children.clone();
        let mut m = 0.0_f64;
        for child in children {
            match child {
                NodePtr::Object(oid) => {
                    let d = self.knn_graph.as_ref().unwrap().knn_distance(oid);
                    if !d.is_finite() {
                        self.nodes[node_id.index()].max_knn_distance = f64::INFINITY;
                        return f64::INFINITY;
                    }
                    m = m.max(d);
                }
                NodePtr::Routing(cid) => {
                    let d = self.recompute_max_knn_subtree(cid);
                    if !d.is_finite() {
                        self.nodes[node_id.index()].max_knn_distance = f64::INFINITY;
                        return f64::INFINITY;
                    }
                    m = m.max(d);
                }
            }
        }
        self.nodes[node_id.index()].max_knn_distance = m;
        m
    }
}
