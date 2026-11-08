// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use crate::distance::{Distance, DynDistance, EuclideanDistance};
use crate::entry::{DuplicateKey, EntryId, UpdateKeyError};
use crate::node::{NodeId, NodePtr, ObjectNode, RoutingNode};
use crate::stats::{DescendantCounter, NodeStats};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

/// M-Tree Datenstruktur für effiziente Bereichs- und Nearest-Neighbor-Suche.
///
/// Routing-Knoten liegen in einer Arena (`nodes`); Blätter (`ObjectNode`) in `by_id`.
/// Distanz ist monomorph (`Dist`); Default [`DynDistance`] hält `MTree<K, V>` typlöschbar.
pub struct MTree<K, V, D = f64, S = DescendantCounter, Dist = DynDistance<K, D>>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
{
    min_node_size: usize,
    max_node_size: usize,
    split_sampling: usize,
    /// Arena aller Routing-Knoten. Indizes bleiben stabil; gesplittete Knoten
    /// bleiben als tote Einträge liegen (wie zuvor verwaiste Arcs).
    pub(crate) nodes: Vec<RoutingNode<K, V, S>>,
    pub(crate) root: Option<NodeId>,
    pub(crate) by_id: Vec<Option<ObjectNode<K, V, S>>>,
    free_list: Vec<usize>,
    key_index: HashMap<K, HashMap<u64, EntryId>>,
    entry_count: usize,
    pub(crate) distance_fn: Dist,
}

impl<K, V, D, S, Dist> Clone for MTree<K, V, D, S, Dist>
where
    K: Clone + Send + Sync + Hash + Eq,
    V: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Clone,
    Dist: Distance<K, Output = D> + Clone,
{
    /// Klont Routing-Arena und Objekt-Arena (`by_id`) tief (`V: Clone`).
    /// Parent-`NodeId`s bleiben gültig, solange die Arena 1:1 geklont wird.
    fn clone(&self) -> Self {
        Self {
            min_node_size: self.min_node_size,
            max_node_size: self.max_node_size,
            split_sampling: self.split_sampling,
            nodes: self.nodes.clone(),
            root: self.root,
            by_id: self
                .by_id
                .iter()
                .map(|slot| slot.as_ref().map(|node| node.clone_node()))
                .collect(),
            free_list: self.free_list.clone(),
            key_index: self.key_index.clone(),
            entry_count: self.entry_count,
            distance_fn: self.distance_fn.clone(),
        }
    }
}

/// Trait für Distanztypen
pub trait DistanceType: Copy + Send + Sync + PartialOrd + 'static {
    fn infinity() -> Self;
    fn zero() -> Self;
    fn sqrt(self) -> Self;
}

impl DistanceType for f32 {
    fn infinity() -> Self {
        f32::INFINITY
    }
    fn zero() -> Self {
        0.0
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl DistanceType for f64 {
    fn infinity() -> Self {
        f64::INFINITY
    }
    fn zero() -> Self {
        0.0
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

pub(crate) fn dist_value_as_f64<D: DistanceType>(d: D) -> f64 {
    if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
        && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
    {
        unsafe { std::mem::transmute_copy(&d) }
    } else {
        0.0
    }
}

pub(crate) fn dist_value_from_f64<D: DistanceType>(x: f64) -> D {
    if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
        && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
    {
        unsafe { std::mem::transmute_copy(&x) }
    } else {
        D::zero()
    }
}

/// Erweiterte Metrik `d(K,K) + |εa − εb|`.
pub(crate) fn metric_distance<K, D, Dist>(
    distance_fn: &Dist,
    a: &K,
    eps_a: f64,
    b: &K,
    eps_b: f64,
) -> D
where
    D: DistanceType,
    Dist: Distance<K, Output = D> + ?Sized,
{
    let base = dist_value_as_f64(distance_fn.distance(a, b));
    dist_value_from_f64(base + (eps_a - eps_b).abs())
}

pub(crate) fn metric_f64<K, D, Dist>(
    distance_fn: &Dist,
    a: &K,
    eps_a: f64,
    b: &K,
    eps_b: f64,
) -> f64
where
    D: DistanceType,
    Dist: Distance<K, Output = D> + ?Sized,
{
    dist_value_as_f64(metric_distance(distance_fn, a, eps_a, b, eps_b))
}

/// Objekt-Lookup nur über das `by_id`-Feld (erlaubt gleichzeitigen `&mut nodes`-Borrow).
#[inline]
pub(crate) fn obj_in<K, V, S>(
    by_id: &[Option<ObjectNode<K, V, S>>],
    id: EntryId,
) -> &ObjectNode<K, V, S>
where
    S: NodeStats<K, V>,
{
    by_id[id.index()]
        .as_ref()
        .expect("EntryId verweist auf belegten Slot")
}

/// Zählt einen Nachkommen bei `start` und dessen Vorfahren. Eine Parent-Schleife bricht ab.
fn adjust_ancestors_in<K, V, S, F>(nodes: &mut [RoutingNode<K, V, S>], start: NodeId, mut adjust: F)
where
    S: NodeStats<K, V>,
    F: FnMut(&mut S),
{
    let mut current = start;
    let mut seen: Vec<NodeId> = Vec::new();
    loop {
        if seen.contains(&current) {
            return;
        }
        seen.push(current);
        let parent = {
            let node = &mut nodes[current.index()];
            adjust(&mut node.stats);
            node.parent
        };
        let Some(parent) = parent else {
            return;
        };
        if parent == current || seen.contains(&parent) {
            return;
        }
        current = parent;
    }
}

/// Vergrößert die Hülle der Vorfahren auf `parent_distance + covering_radius`.
/// Schrumpft nie. Stoppt, sobald ein Vorfahr schon abdeckt oder die Parent-Kette kreist.
fn expand_cover_up_in<K, V, S>(nodes: &mut [RoutingNode<K, V, S>], start: NodeId)
where
    S: NodeStats<K, V>,
{
    let mut current = start;
    let mut seen: Vec<NodeId> = Vec::new();

    loop {
        if seen.contains(&current) {
            return;
        }
        seen.push(current);

        let (parent, needed) = {
            let node = &nodes[current.index()];
            let Some(parent) = node.parent else {
                return;
            };
            if parent == current || seen.contains(&parent) {
                return;
            }
            (parent, node.parent_distance + node.covering_radius)
        };

        let grew = {
            let parent_node = &mut nodes[parent.index()];
            if needed > parent_node.covering_radius {
                parent_node.covering_radius = needed;
                true
            } else {
                false
            }
        };
        if !grew {
            return;
        }
        current = parent;
    }
}

/// Leicht kleinere Schranke beim Pruning, um numerische Fehler zu vermeiden (Untergrenze).
pub const LOWER_BOUND_FACTOR: f64 = 0.999;
/// Leicht größere Schranke beim Pruning, um numerische Fehler zu vermeiden (Obergrenze).
pub const UPPER_BOUND_FACTOR: f64 = 1.001;

/// Wrapper für f64 als Ord-Key (der Heap der k besten Distanzen verlangt Key: Ord)
#[derive(Clone, Copy)]
pub(crate) struct KnnDistanceKey(pub(crate) f64);
impl Default for KnnDistanceKey {
    fn default() -> Self {
        KnnDistanceKey(f64::INFINITY)
    }
}
impl PartialEq for KnnDistanceKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for KnnDistanceKey {}
impl PartialOrd for KnnDistanceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for KnnDistanceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Queue-Eintrag für k-NN-Traversierung (min-heap: kleinste Lower Bound zuerst).
pub(crate) struct KnnQueueEntry {
    pub(crate) node: NodeId,
    /// Distanz Query → Routing-Zentrum (für Triangle-Pruning der Kinder).
    pub(crate) center_distance: f64,
    /// Lower Bound `(center - covering_radius).max(0)` — Ordnungskey der PQ.
    pub(crate) distance_bound: f64,
    /// Upper Bound `center + covering_radius` — Annulus-/Range-Pruning.
    pub(crate) upper_bound: f64,
}
impl PartialEq for KnnQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance_bound == other.distance_bound
    }
}
impl Eq for KnnQueueEntry {}
impl PartialOrd for KnnQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance_bound.partial_cmp(&self.distance_bound)
    }
}
impl Ord for KnnQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_bound
            .partial_cmp(&self.distance_bound)
            .unwrap_or(Ordering::Equal)
    }
}

/// Empfohlene Knotenparameter `(min_node_size, max_node_size, split_sampling)`
/// für einen Baum mit etwa `expected_entries` Einträgen.
///
/// Heuristik: `max ≈ √n` (geclampt auf 8…100), `min = max(2, max/4)`,
/// `split = clamp(max/4, 5, 20)`. Bei `expected_entries == 0` wie `n = 1`.
pub fn node_params_for_capacity(expected_entries: usize) -> (usize, usize, usize) {
    let n = expected_entries.max(1);
    let max = ((n as f64).sqrt().round() as usize).clamp(8, 100);
    let min = (max / 4).max(2);
    let split = (max / 4).clamp(5, 20);
    (min, max, split)
}

impl<K, V> MTree<K, V, f64, DescendantCounter, DynDistance<K, f64>>
where
    K: Clone + Send + Sync + Hash + Eq + 'static,
    V: Send + Sync,
{
    /// Wie [`Self::with_distance`], aber Distanz typgelöscht (`DynDistance`).
    /// Nützlich wenn der konkrete Distanztyp in Signaturen nicht erscheinen soll.
    pub fn with_dyn_distance<F>(distance_fn: F) -> Self
    where
        F: Distance<K, Output = f64> + Send + Sync + 'static,
    {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            nodes: Vec::new(),
            root: None,
            by_id: Vec::new(),
            free_list: Vec::new(),
            key_index: HashMap::new(),
            entry_count: 0,
            distance_fn: DynDistance::new(distance_fn),
        }
    }
}

impl<K, V, F> MTree<K, V, f64, DescendantCounter, F>
where
    K: Clone + Send + Sync + Hash + Eq,
    V: Send + Sync,
    F: Distance<K, Output = f64>,
{
    /// Erstellt einen neuen M-Tree mit benutzerdefinierter Distanzfunktion (monomorph).
    pub fn with_distance(distance_fn: F) -> Self {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            nodes: Vec::new(),
            root: None,
            by_id: Vec::new(),
            free_list: Vec::new(),
            key_index: HashMap::new(),
            entry_count: 0,
            distance_fn,
        }
    }

    /// Erstellt einen neuen M-Tree mit benutzerdefinierten Parametern (monomorph).
    pub fn with_params(
        min_node_size: usize,
        max_node_size: usize,
        split_sampling: usize,
        distance_fn: F,
    ) -> Self {
        Self {
            min_node_size,
            max_node_size,
            split_sampling,
            nodes: Vec::new(),
            root: None,
            by_id: Vec::new(),
            free_list: Vec::new(),
            key_index: HashMap::new(),
            entry_count: 0,
            distance_fn,
        }
    }

    /// Wie [`with_params`], mit Knotengrößen aus [`node_params_for_capacity`].
    ///
    /// `expected_entries` ist ein Hinweis auf die geplante Datenmenge; die
    /// Parameter bleiben danach fest (kein automatisches Nachskalieren).
    pub fn with_capacity_hint(expected_entries: usize, distance_fn: F) -> Self {
        let (min, max, split) = node_params_for_capacity(expected_entries);
        Self::with_params(min, max, split, distance_fn)
    }
}

/// Spezielle Implementierung für Vec<f32> mit euklidischer Distanz
impl<V> MTree<Vec<f32>, V, f64, DescendantCounter, EuclideanDistance>
where
    V: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            nodes: Vec::new(),
            root: None,
            by_id: Vec::new(),
            free_list: Vec::new(),
            key_index: HashMap::new(),
            entry_count: 0,
            distance_fn: EuclideanDistance,
        }
    }
}

/// Spezielle Implementierung für Vec<f64> mit euklidischer Distanz
impl<V> MTree<Vec<f64>, V, f64, DescendantCounter, EuclideanDistance>
where
    V: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            nodes: Vec::new(),
            root: None,
            by_id: Vec::new(),
            free_list: Vec::new(),
            key_index: HashMap::new(),
            entry_count: 0,
            distance_fn: EuclideanDistance,
        }
    }
}

impl<K, V, D, S, Dist> MTree<K, V, D, S, Dist>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
{
    /// Iteriert über alle Eintrags-IDs (O(Kapazität von `by_id`), keine Distanzberechnung).
    ///
    /// Reihenfolge: aufsteigend nach Slot-Index, nicht Insert-Reihenfolge.
    /// Leere Slots nach `erase` werden übersprungen.
    pub fn iter(&self) -> impl Iterator<Item = EntryId> + '_ {
        self.by_id
            .iter()
            .filter_map(|slot| slot.as_ref().map(|node| node.id))
    }

    /// Key des Eintrags, falls `id` noch belegt ist.
    pub fn key(&self, id: EntryId) -> Option<K> {
        self.get(id).map(|node| node.key().clone())
    }

    /// Nutzdaten des Eintrags, falls `id` noch belegt ist.
    pub fn payload(&self, id: EntryId) -> Option<V>
    where
        V: Clone,
    {
        self.get(id).map(|node| node.payload())
    }

    /// Identity-Epsilon des Eintrags, falls `id` noch belegt ist.
    pub fn epsilon(&self, id: EntryId) -> Option<f64> {
        self.get(id).map(|node| node.epsilon())
    }

    /// Objekt zu einer ID, die aus einem Leaf-`NodePtr::Object` stammt (muss belegt sein).
    #[inline]
    pub(crate) fn object(&self, id: EntryId) -> &ObjectNode<K, V, S> {
        obj_in(&self.by_id, id)
    }

    /// Liefert den Eintrag zur stabilen ID (O(1)).
    pub(crate) fn get(&self, id: EntryId) -> Option<&ObjectNode<K, V, S>> {
        self.by_id.get(id.index())?.as_ref()
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
    fn alloc_slot(&mut self) -> EntryId {
        let index = if let Some(index) = self.free_list.pop() {
            index
        } else {
            let index = self.by_id.len();
            self.by_id.push(None);
            index
        };
        EntryId::from_index(index)
    }

    fn release_slot(&mut self, id: EntryId) {
        let index = id.index();
        self.by_id[index] = None;
        self.free_list.push(index);
        self.entry_count -= 1;
    }

    fn alloc_routing(&mut self, node: RoutingNode<K, V, S>) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn metric(&self, a: &K, eps_a: f64, b: &K, eps_b: f64) -> D {
        metric_distance(&self.distance_fn, a, eps_a, b, eps_b)
    }

    fn child_key(&self, child: &NodePtr) -> K {
        match child {
            NodePtr::Object(id) => obj_in(&self.by_id, *id).key().clone(),
            NodePtr::Routing(id) => self.nodes[id.index()].key.clone(),
        }
    }

    fn child_epsilon(&self, child: &NodePtr) -> f64 {
        match child {
            NodePtr::Object(id) => obj_in(&self.by_id, *id).epsilon(),
            NodePtr::Routing(id) => self.nodes[id.index()].epsilon,
        }
    }

    fn child_identity_hash(&self, child: &NodePtr) -> u64 {
        match child {
            NodePtr::Object(id) => obj_in(&self.by_id, *id).identity_hash(),
            NodePtr::Routing(id) => self.nodes[id.index()].identity_hash,
        }
    }

    fn index_try_insert(
        &mut self,
        key: K,
        identity_hash: u64,
        id: EntryId,
    ) -> Result<(), DuplicateKey> {
        let inner = self.key_index.entry(key).or_insert_with(HashMap::new);
        if inner.contains_key(&identity_hash) {
            return Err(DuplicateKey);
        }
        inner.insert(identity_hash, id);
        Ok(())
    }

    fn index_remove(&mut self, key: &K, identity_hash: u64) {
        if let Some(inner) = self.key_index.get_mut(key) {
            inner.remove(&identity_hash);
            if inner.is_empty() {
                self.key_index.remove(key);
            }
        }
    }

    /// Fügt einen Eintrag in den Baum ein (Identity-Hash `0`, `ε = 0`).
    pub fn insert(&mut self, key: K, value: V) -> Result<EntryId, DuplicateKey> {
        self.insert_with_identity_hash(key, value, 0)
    }

    /// Insert mit Identitätsquelle (Index, Timestamp, …). Gleiche Koordinaten sind erlaubt,
    /// solange die gehashte Quelle verschieden ist.
    pub fn insert_with_identity<I: Hash>(
        &mut self,
        key: K,
        value: V,
        identity: &I,
    ) -> Result<EntryId, DuplicateKey> {
        self.insert_with_identity_hash(key, value, crate::distance::identity_hash(identity))
    }

    fn insert_with_identity_hash(
        &mut self,
        key: K,
        value: V,
        identity_hash: u64,
    ) -> Result<EntryId, DuplicateKey> {
        if self
            .key_index
            .get(&key)
            .map(|inner| inner.contains_key(&identity_hash))
            .unwrap_or(false)
        {
            return Err(DuplicateKey);
        }

        let epsilon = crate::distance::epsilon_from_hash(identity_hash);
        let id = self.alloc_slot();
        let entry = ObjectNode::with_identity(id, key.clone(), value, identity_hash, epsilon);

        if self.index_try_insert(key, identity_hash, id).is_err() {
            self.free_list.push(id.index());
            return Err(DuplicateKey);
        }
        self.entry_count += 1;

        self.by_id[id.index()] = Some(entry);

        if let Some(root) = self.root {
            self.tree_insert(id, root);
        } else {
            self.install_first_root(id);
        }

        Ok(id)
    }

    /// Legt den Wurzel-Leaf für den ersten Eintrag an.
    fn install_first_root(&mut self, id: EntryId) {
        let (key, epsilon, identity_hash) = {
            let entry = obj_in(&self.by_id, id);
            (entry.key().clone(), entry.epsilon(), entry.identity_hash())
        };
        let new_root = RoutingNode::with_routing(key, epsilon, identity_hash, true);
        let root_id = self.alloc_routing(new_root);
        let MTree { by_id, nodes, .. } = self;
        let entry = obj_in(by_id, id);
        entry.set_parent(Some(root_id), 0.0);
        let root = &mut nodes[root_id.index()];
        {
            let entry_value = entry.value();
            root.stats.add_descendant((entry.key(), entry_value));
        }
        root.children.push(NodePtr::Object(id));
        self.root = Some(root_id);
    }

    fn tree_insert(&mut self, id: EntryId, node_id: NodeId) {
        let mut current = node_id;

        loop {
            let is_leaf = self.nodes[current.index()].is_leaf;
            let entry = obj_in(&self.by_id, id);
            let key = entry.key();
            let entry_eps = entry.epsilon();

            if is_leaf {
                let distance = self.metric(
                    &self.nodes[current.index()].key,
                    self.nodes[current.index()].epsilon,
                    key,
                    entry_eps,
                );

                let ancestor = self.nodes[current.index()].parent;
                {
                    let entry_value = entry.value();
                    self.nodes[current.index()]
                        .stats
                        .add_descendant((key, entry_value));
                }

                let distance_f64 = dist_value_as_f64(distance);
                let radius_grew = distance_f64 > self.nodes[current.index()].covering_radius;
                if radius_grew {
                    self.nodes[current.index()].covering_radius = distance_f64;
                    self.nodes[current.index()].furthest_descendant = Some(id);
                }

                self.nodes[current.index()]
                    .children
                    .push(NodePtr::Object(id));

                entry.set_parent(Some(current), distance_f64);

                let too_big = self.nodes[current.index()].children.len() > self.max_node_size;
                if too_big {
                    self.split(current);
                } else if radius_grew {
                    self.expand_cover_up(current);
                }

                if let Some(ancestor) = ancestor {
                    let entry = obj_in(&self.by_id, id);
                    let entry_value = entry.value();
                    adjust_ancestors_in(&mut self.nodes, ancestor, |stats| {
                        stats.add_descendant((entry.key(), entry_value))
                    });
                }
                break;
            } else {
                let routing_children: Vec<NodeId> = self.nodes[current.index()]
                    .children
                    .iter()
                    .filter_map(|c| match c {
                        NodePtr::Routing(id) => Some(*id),
                        _ => None,
                    })
                    .collect();

                let mut nearest: Option<NodeId> = None;
                let mut nearest_distance = D::infinity();
                let mut within_covering_radius = false;
                let mut radius_increase = D::infinity();

                for child_id in routing_children {
                    let child = &self.nodes[child_id.index()];
                    let dist = self.metric(&child.key, child.epsilon, key, entry_eps);
                    let dist_f64 = dist_value_as_f64(dist);
                    let covering_radius = child.covering_radius;

                    if !within_covering_radius && dist_f64 <= covering_radius {
                        within_covering_radius = true;
                        nearest = Some(child_id);
                        nearest_distance = dist;
                    } else if within_covering_radius {
                        if dist_f64 <= covering_radius && dist < nearest_distance {
                            nearest_distance = dist;
                            nearest = Some(child_id);
                        }
                    } else {
                        let increase_f64 = dist_f64 - covering_radius;
                        let increase = dist_value_from_f64(increase_f64);
                        if dist < radius_increase
                            || (dist == radius_increase && dist < nearest_distance)
                        {
                            nearest = Some(child_id);
                            nearest_distance = dist;
                            radius_increase = increase;
                        }
                    }
                }

                if let Some(next) = nearest {
                    if !within_covering_radius {
                        let nearest_dist_f64 = dist_value_as_f64(nearest_distance);
                        let radius_grew = {
                            let next_node = &mut self.nodes[next.index()];
                            let grew = nearest_dist_f64 > next_node.covering_radius;
                            if grew {
                                next_node.covering_radius = nearest_dist_f64;
                            }
                            grew
                        };
                        if radius_grew {
                            self.expand_cover_up(next);
                        }
                    }
                    current = next;
                } else {
                    break;
                }
            }
        }
    }

    fn expand_cover_up(&mut self, start: NodeId) {
        expand_cover_up_in(&mut self.nodes, start);
    }

    /// Covering-Radius für einen Arena-Knoten neu berechnen.
    fn compute_node_radius(&mut self, node_id: NodeId) {
        let is_leaf = self.nodes[node_id.index()].is_leaf;
        if is_leaf {
            let mut max_distance = 0.0f64;
            let mut furthest: Option<EntryId> = None;

            for child in &self.nodes[node_id.index()].children {
                if let NodePtr::Object(oid) = child {
                    let pd = obj_in(&self.by_id, *oid).dist_to_parent();
                    if pd > max_distance {
                        max_distance = pd;
                        furthest = Some(*oid);
                    }
                }
            }

            if let Some(furthest_id) = furthest {
                let node = &mut self.nodes[node_id.index()];
                node.covering_radius = max_distance;
                node.furthest_descendant = Some(furthest_id);
            }
        } else {
            let routing_ids: Vec<NodeId> = self.nodes[node_id.index()]
                .children
                .iter()
                .filter_map(|c| match c {
                    NodePtr::Routing(id) if *id != node_id => Some(*id),
                    _ => None,
                })
                .collect();

            let mut max_radius = 0.0f64;
            let mut furthest: Option<EntryId> = None;

            for child_id in routing_ids {
                let child = &self.nodes[child_id.index()];
                let total_radius = child.parent_distance + child.covering_radius;
                if total_radius > max_radius {
                    max_radius = total_radius;
                    furthest = child.furthest_descendant;
                }
            }

            if let Some(furthest_id) = furthest {
                let node = &mut self.nodes[node_id.index()];
                node.covering_radius = max_radius;
                node.furthest_descendant = Some(furthest_id);
            }
        }
    }

    /// Covering-Radius für einen noch nicht allokierten Split-Knoten.
    fn compute_node_radius_pending(&self, node: &mut RoutingNode<K, V, S>) {
        if node.is_leaf {
            let mut max_distance = 0.0f64;
            let mut furthest: Option<EntryId> = None;

            for child in &node.children {
                if let NodePtr::Object(oid) = child {
                    let pd = obj_in(&self.by_id, *oid).dist_to_parent();
                    if pd > max_distance {
                        max_distance = pd;
                        furthest = Some(*oid);
                    }
                }
            }

            if let Some(furthest_id) = furthest {
                node.covering_radius = max_distance;
                node.furthest_descendant = Some(furthest_id);
            }
        } else {
            let mut max_radius = 0.0f64;
            let mut furthest: Option<EntryId> = None;

            for child in &node.children {
                if let NodePtr::Routing(child_id) = child {
                    let child_node = &self.nodes[child_id.index()];
                    let total_radius = child_node.parent_distance + child_node.covering_radius;
                    if total_radius > max_radius {
                        max_radius = total_radius;
                        furthest = child_node.furthest_descendant;
                    }
                }
            }

            if let Some(furthest_id) = furthest {
                node.covering_radius = max_radius;
                node.furthest_descendant = Some(furthest_id);
            }
        }
    }

    /// Kinder auf den neuen Split-Knoten zeigen lassen.
    fn retarget_split_children(&mut self, node_id: NodeId) {
        let children = self.nodes[node_id.index()].children.clone();
        for child in children {
            match child {
                NodePtr::Object(oid) => {
                    let obj_node = obj_in(&self.by_id, oid);
                    obj_node.set_parent(Some(node_id), obj_node.dist_to_parent());
                }
                NodePtr::Routing(routing_id) => {
                    if routing_id == node_id {
                        continue;
                    }
                    self.nodes[routing_id.index()].parent = Some(node_id);
                }
            }
        }
    }

    fn split(&mut self, node_id: NodeId) {
        let children = std::mem::take(&mut self.nodes[node_id.index()].children);
        let is_leaf = self.nodes[node_id.index()].is_leaf;
        let parent = self.nodes[node_id.index()].parent;
        let is_root = self.root == Some(node_id);

        let mut node1 = RoutingNode::new(is_leaf);
        let mut node2 = RoutingNode::new(is_leaf);

        self.promote_and_partition(children, &mut node1, &mut node2);

        let node1_id = self.alloc_routing(node1);
        let node2_id = self.alloc_routing(node2);
        self.retarget_split_children(node1_id);
        self.retarget_split_children(node2_id);

        if is_root {
            let (k1, e1, cr1) = {
                let n = &self.nodes[node1_id.index()];
                (n.key.clone(), n.epsilon, n.covering_radius)
            };
            let (k2, e2, cr2) = {
                let n = &self.nodes[node2_id.index()];
                (n.key.clone(), n.epsilon, n.covering_radius)
            };
            let mut new_root: RoutingNode<K, V, S> = RoutingNode::new(false);
            let d1 = dist_value_as_f64(self.metric(&new_root.key, new_root.epsilon, &k1, e1));
            let d2 = dist_value_as_f64(self.metric(&new_root.key, new_root.epsilon, &k2, e2));
            new_root.covering_radius = (d1 + cr1).max(d2 + cr2);
            {
                let stats1 = self.nodes[node1_id.index()].stats.clone();
                new_root.stats.add_descendants(&stats1);
            }
            {
                let stats2 = self.nodes[node2_id.index()].stats.clone();
                new_root.stats.add_descendants(&stats2);
            }
            new_root.children.push(NodePtr::Routing(node1_id));
            new_root.children.push(NodePtr::Routing(node2_id));
            let new_root_id = self.alloc_routing(new_root);
            self.nodes[node1_id.index()].parent = Some(new_root_id);
            self.nodes[node1_id.index()].parent_distance = d1;
            self.nodes[node2_id.index()].parent = Some(new_root_id);
            self.nodes[node2_id.index()].parent_distance = d2;
            self.root = Some(new_root_id);
        } else if let Some(parent_id) = parent {
            if parent_id == node1_id || parent_id == node2_id {
                return;
            }

            let (pkey, peps) = {
                let p = &self.nodes[parent_id.index()];
                (p.key.clone(), p.epsilon)
            };
            let (k1, e1) = {
                let n = &self.nodes[node1_id.index()];
                (n.key.clone(), n.epsilon)
            };
            let (k2, e2) = {
                let n = &self.nodes[node2_id.index()];
                (n.key.clone(), n.epsilon)
            };
            let d1 = dist_value_as_f64(self.metric(&pkey, peps, &k1, e1));
            let d2 = dist_value_as_f64(self.metric(&pkey, peps, &k2, e2));
            self.nodes[node1_id.index()].parent = Some(parent_id);
            self.nodes[node1_id.index()].parent_distance = d1;
            self.nodes[node2_id.index()].parent = Some(parent_id);
            self.nodes[node2_id.index()].parent_distance = d2;

            let too_big = {
                let parent_node = &mut self.nodes[parent_id.index()];
                parent_node.children.retain(|child| match child {
                    NodePtr::Routing(id) => *id != node_id,
                    _ => true,
                });
                parent_node.children.push(NodePtr::Routing(node1_id));
                parent_node.children.push(NodePtr::Routing(node2_id));
                parent_node.children.len() > self.max_node_size
            };

            self.expand_cover_up(node1_id);
            self.expand_cover_up(node2_id);

            if too_big && parent_id != node_id && parent_id != node1_id && parent_id != node2_id {
                self.split(parent_id);
            }
        }
    }

    fn promote_and_partition(
        &mut self,
        children: Vec<NodePtr>,
        node1: &mut RoutingNode<K, V, S>,
        node2: &mut RoutingNode<K, V, S>,
    ) {
        let mut best_key_1: Option<K> = None;
        let mut best_key_2: Option<K> = None;
        let mut best_eps_1 = 0.0;
        let mut best_eps_2 = 0.0;
        let mut best_id_1 = 0u64;
        let mut best_id_2 = 0u64;
        let mut best_av_radius = D::infinity();

        for _ in 0..self.split_sampling {
            let (key1, eps1, id1, key2, eps2, id2) = self.promote(&children);
            node1.key = key1.clone();
            node1.epsilon = eps1;
            node1.identity_hash = id1;
            node2.key = key2.clone();
            node2.epsilon = eps2;
            node2.identity_hash = id2;

            let estimated_av_radius =
                self.partition(node1.is_leaf, &children, node1, node2, best_av_radius);

            if estimated_av_radius < best_av_radius {
                best_key_1 = Some(key1);
                best_key_2 = Some(key2);
                best_eps_1 = eps1;
                best_eps_2 = eps2;
                best_id_1 = id1;
                best_id_2 = id2;
                best_av_radius = estimated_av_radius;
            }
        }

        if let (Some(k1), Some(k2)) = (best_key_1, best_key_2) {
            node1.key = k1;
            node1.epsilon = best_eps_1;
            node1.identity_hash = best_id_1;
            node2.key = k2;
            node2.epsilon = best_eps_2;
            node2.identity_hash = best_id_2;
            self.partition(node1.is_leaf, &children, node1, node2, D::infinity());

            if node1.is_leaf {
                for child in &node1.children {
                    if let NodePtr::Object(oid) = child {
                        let obj_node = obj_in(&self.by_id, *oid);
                        let v = obj_node.value();
                        node1.stats.add_descendant((obj_node.key(), v));
                    }
                }
                for child in &node2.children {
                    if let NodePtr::Object(oid) = child {
                        let obj_node = obj_in(&self.by_id, *oid);
                        let v = obj_node.value();
                        node2.stats.add_descendant((obj_node.key(), v));
                    }
                }
            } else {
                let ids1: Vec<NodeId> = node1
                    .children
                    .iter()
                    .filter_map(|c| c.as_routing_id())
                    .collect();
                let ids2: Vec<NodeId> = node2
                    .children
                    .iter()
                    .filter_map(|c| c.as_routing_id())
                    .collect();
                for id in ids1 {
                    let child_stats = self.nodes[id.index()].stats.clone();
                    node1.stats.add_descendants(&child_stats);
                }
                for id in ids2 {
                    let child_stats = self.nodes[id.index()].stats.clone();
                    node2.stats.add_descendants(&child_stats);
                }
            }

            self.compute_node_radius_pending(node1);
            self.compute_node_radius_pending(node2);
        }
    }

    fn promote(&self, children: &[NodePtr]) -> (K, f64, u64, K, f64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        let seed = hasher.finish();

        let a_idx = (seed as usize) % children.len();
        let b_idx = ((seed >> 32) as usize) % (children.len() - 1);
        let b_idx = if b_idx >= a_idx { b_idx + 1 } else { b_idx };

        (
            self.child_key(&children[a_idx]),
            self.child_epsilon(&children[a_idx]),
            self.child_identity_hash(&children[a_idx]),
            self.child_key(&children[b_idx]),
            self.child_epsilon(&children[b_idx]),
            self.child_identity_hash(&children[b_idx]),
        )
    }

    fn partition(
        &mut self,
        from_leaf: bool,
        from: &[NodePtr],
        to_1: &mut RoutingNode<K, V, S>,
        to_2: &mut RoutingNode<K, V, S>,
        estimated_radius_bound: D,
    ) -> D {
        struct Child {
            index: usize,
            from_1: f64,
            from_2: f64,
            quotient: f64,
        }

        let mut distances: Vec<Child> = Vec::new();
        for (i, child) in from.iter().enumerate() {
            let key = self.child_key(child);
            let eps = self.child_epsilon(child);
            let d1 = self.metric(&to_1.key, to_1.epsilon, &key, eps);
            let d2 = self.metric(&to_2.key, to_2.epsilon, &key, eps);
            let d1_f64 = dist_value_as_f64(d1);
            let d2_f64 = dist_value_as_f64(d2);
            let quotient = if d2_f64 != 0.0 {
                d1_f64 / d2_f64
            } else {
                f64::INFINITY
            };
            distances.push(Child {
                index: i,
                from_1: d1_f64,
                from_2: d2_f64,
                quotient,
            });
        }

        distances.sort_by(|a, b| {
            a.quotient
                .partial_cmp(&b.quotient)
                .unwrap_or(Ordering::Equal)
        });

        let mut boundary = from.len() / 2;
        if distances[boundary].quotient > 1.0 {
            while distances[boundary].quotient > 1.0 && boundary > self.min_node_size {
                boundary -= 1;
            }
        } else {
            while distances[boundary].quotient < 1.0 && boundary < from.len() - self.min_node_size {
                boundary += 1;
            }
        }

        let mut estimated_radius_1 = 0.0f64;
        let mut estimated_radius_2 = 0.0f64;

        for (i, dist) in distances.iter().enumerate() {
            let node = &from[dist.index];
            let parent_distance = if i < boundary {
                dist.from_1
            } else {
                dist.from_2
            };
            let estimated_radius = if from_leaf {
                parent_distance
            } else if let NodePtr::Routing(id) = node {
                parent_distance + self.nodes[id.index()].covering_radius
            } else {
                parent_distance
            };

            if i < boundary {
                estimated_radius_1 = estimated_radius_1.max(estimated_radius);
            } else {
                estimated_radius_2 = estimated_radius_2.max(estimated_radius);
            }
        }

        let total_radius = estimated_radius_1 + estimated_radius_2;
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>() {
            let bound_f64: f64 = unsafe { std::mem::transmute_copy(&estimated_radius_bound) };
            if total_radius >= bound_f64 {
                return estimated_radius_bound;
            }
        }

        to_1.children.clear();
        to_2.children.clear();

        for (i, dist) in distances.iter().enumerate() {
            let node = from[dist.index];
            let parent_distance = if i < boundary {
                dist.from_1
            } else {
                dist.from_2
            };

            match node {
                NodePtr::Object(oid) => {
                    let obj_node = obj_in(&self.by_id, oid);
                    obj_node.set_parent(obj_node.parent(), parent_distance);
                }
                NodePtr::Routing(routing_id) => {
                    self.nodes[routing_id.index()].parent_distance = parent_distance;
                }
            }

            if i < boundary {
                to_1.children.push(node);
            } else {
                to_2.children.push(node);
            }
        }

        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            unsafe { std::mem::transmute_copy(&total_radius) }
        } else {
            estimated_radius_bound
        }
    }

    /// Entfernt einen Eintrag anhand seiner stabilen ID.
    pub fn erase_by_id(&mut self, id: EntryId) -> bool {
        let (key, identity_hash) = match self.by_id.get(id.index()).and_then(|o| o.as_ref()) {
            Some(e) => (e.key().clone(), e.identity_hash()),
            None => return false,
        };

        self.tree_erase(id);
        self.index_remove(&key, identity_hash);
        self.release_slot(id);
        true
    }

    /// Entfernt den Eintrag zu `key`, falls genau einer mit diesem Key existiert.
    /// Bei mehreren Identitäten (gleiche Koordinaten) `false` — dann `erase_by_id`
    /// oder [`erase_by_key_identity`] verwenden.
    pub fn erase_by_key(&mut self, key: &K) -> bool {
        let id = match self.key_index.get(key) {
            Some(inner) if inner.len() == 1 => *inner.values().next().unwrap(),
            _ => return false,
        };
        self.erase_by_id(id)
    }

    /// Entfernt den Eintrag zu `key` und Identitätsquelle.
    pub fn erase_by_key_identity<I: Hash>(&mut self, key: &K, identity: &I) -> bool {
        let h = crate::distance::identity_hash(identity);
        let id = match self.key_index.get(key).and_then(|inner| inner.get(&h)) {
            Some(&id) => id,
            None => return false,
        };
        self.erase_by_id(id)
    }

    /// Entfernt einen Eintrag anhand seines Schlüssels.
    #[deprecated(note = "use erase_by_key instead")]
    pub fn erase(&mut self, key: &K) -> bool {
        self.erase_by_key(key)
    }

    /// Aktualisiert nur die Nutzdaten (O(1), Baum unverändert).
    pub fn update_value(&mut self, id: EntryId, value: V) -> bool {
        let slot = match self.by_id.get(id.index()).and_then(|o| o.as_ref()) {
            Some(node) => node,
            None => return false,
        };
        slot.set_value(value);
        true
    }

    /// Verschiebt den Eintrag im Metrikraum (erase + reinsert, gleiche EntryId).
    pub fn update_key(&mut self, id: EntryId, new_key: K) -> Result<(), UpdateKeyError> {
        let (old_key, identity_hash) = match self.by_id.get(id.index()).and_then(|o| o.as_ref()) {
            Some(e) => (e.key().clone(), e.identity_hash()),
            None => return Err(UpdateKeyError::NotFound),
        };

        if new_key == old_key {
            return Ok(());
        }

        if let Some(&existing) = self
            .key_index
            .get(&new_key)
            .and_then(|inner| inner.get(&identity_hash))
        {
            if existing != id {
                return Err(UpdateKeyError::DuplicateKey);
            }
        }

        self.tree_erase(id);

        self.by_id
            .get(id.index())
            .and_then(|o| o.as_ref())
            .ok_or(UpdateKeyError::NotFound)?
            .set_key(new_key.clone());

        self.index_remove(&old_key, identity_hash);
        self.index_try_insert(new_key, identity_hash, id)
            .map_err(|_| UpdateKeyError::DuplicateKey)?;

        if self.by_id.get(id.index()).and_then(|o| o.as_ref()).is_none() {
            return Err(UpdateKeyError::NotFound);
        }

        if let Some(root) = self.root {
            self.tree_insert(id, root);
        } else {
            self.install_first_root(id);
        }

        Ok(())
    }

    fn tree_erase(&mut self, id: EntryId) {
        let entry = obj_in(&self.by_id, id);
        if let Some(parent_id) = entry.parent() {
            let entry_id = id;
            self.nodes[parent_id.index()].children.retain(|child| match child {
                NodePtr::Object(oid) => *oid != entry_id,
                _ => true,
            });

            let ancestor = self.nodes[parent_id.index()].parent;
            let entry_value = entry.value();
            self.nodes[parent_id.index()]
                .stats
                .remove_descendant((entry.key(), entry_value));

            let needs_radius_recompute =
                self.nodes[parent_id.index()].furthest_descendant == Some(entry_id);

            if let Some(ancestor) = ancestor {
                if ancestor != parent_id {
                    adjust_ancestors_in(&mut self.nodes, ancestor, |stats| {
                        stats.remove_descendant((entry.key(), entry_value))
                    });
                }
            }

            if needs_radius_recompute {
                self.compute_node_radius(parent_id);
            }

            let mut current_parent = parent_id;
            loop {
                let children_len = self.nodes[current_parent.index()].children.len();

                if children_len >= self.min_node_size {
                    break;
                }

                let is_root = self.root == Some(current_parent);

                if is_root {
                    self.pull_up_root();
                    break;
                }

                let next_parent = self.nodes[current_parent.index()].parent;

                self.rebalance_node(current_parent);

                if let Some(new_parent) = next_parent {
                    current_parent = new_parent;
                } else {
                    break;
                }
            }
        }
    }

    fn pull_up_root(&mut self) {
        if let Some(root_id) = self.root {
            let children_len = self.nodes[root_id.index()].children.len();
            if children_len == 1 {
                if let Some(NodePtr::Routing(only_child)) =
                    self.nodes[root_id.index()].children.first().cloned()
                {
                    self.nodes[only_child.index()].parent = None;
                    self.nodes[only_child.index()].parent_distance = 0.0;
                    self.root = Some(only_child);
                }
            } else if children_len == 0 {
                self.root = None;
            }
        }
    }

    fn rebalance_node(&mut self, node_id: NodeId) {
        let parent_id = self.nodes[node_id.index()].parent;

        if let Some(parent_id) = parent_id {
            let sibling_ids: Vec<NodeId> = self.nodes[parent_id.index()]
                .children
                .iter()
                .filter_map(|c| match c {
                    NodePtr::Routing(id) if *id != node_id => Some(*id),
                    _ => None,
                })
                .collect();

            let node_key = (
                self.nodes[node_id.index()].key.clone(),
                self.nodes[node_id.index()].epsilon,
            );

            let mut nearest_donor: Option<NodeId> = None;
            let mut nearest_donor_distance = D::infinity();
            let mut nearest_merge: Option<NodeId> = None;
            let mut nearest_merge_distance = D::infinity();

            for sibling_id in sibling_ids {
                let sibling = &self.nodes[sibling_id.index()];
                let dist =
                    self.metric(&node_key.0, node_key.1, &sibling.key, sibling.epsilon);

                if sibling.children.len() > self.min_node_size {
                    if dist < nearest_donor_distance {
                        nearest_donor_distance = dist;
                        nearest_donor = Some(sibling_id);
                    }
                } else if dist < nearest_merge_distance {
                    nearest_merge_distance = dist;
                    nearest_merge = Some(sibling_id);
                }
            }

            if let Some(donor) = nearest_donor {
                self.donate_child(donor, node_id);
            } else if let Some(merge_target) = nearest_merge {
                let merge_dist = dist_value_as_f64(nearest_merge_distance);
                self.merge_routing_nodes(node_id, merge_target, merge_dist);
            }
        }
    }

    fn donate_child(&mut self, from_id: NodeId, to_id: NodeId) {
        let (to_key, to_eps, is_leaf) = {
            let to = &self.nodes[to_id.index()];
            (to.key.clone(), to.epsilon, to.is_leaf)
        };

        let from_children = self.nodes[from_id.index()].children.clone();
        let mut best_idx = 0;
        let mut best_dist = D::infinity();
        for (i, grandchild) in from_children.iter().enumerate() {
            let grandchild_key = self.child_key(grandchild);
            let eps = self.child_epsilon(grandchild);
            let dist = self.metric(&to_key, to_eps, &grandchild_key, eps);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        let nearest_dist_f64 = dist_value_as_f64(best_dist);

        let (grandchild, needs_radius_recompute) = {
            let from = &mut self.nodes[from_id.index()];
            let grandchild = from.children.remove(best_idx);

            let needs_recompute = if let Some(furthest_id) = from.furthest_descendant {
                match grandchild {
                    NodePtr::Object(oid) => oid == furthest_id,
                    NodePtr::Routing(_) => true,
                }
            } else {
                false
            };

            if is_leaf {
                if let NodePtr::Object(oid) = grandchild {
                    let obj_node = obj_in(&self.by_id, oid);
                    let v = obj_node.value();
                    from.stats.remove_descendant((obj_node.key(), v));
                }
            }

            (grandchild, needs_recompute)
        };

        if !is_leaf {
            if let NodePtr::Routing(routing_id) = &grandchild {
                let child_stats = self.nodes[routing_id.index()].stats.clone();
                self.nodes[from_id.index()]
                    .stats
                    .remove_descendants(&child_stats);
            }
        }

        match grandchild {
            NodePtr::Object(oid) => {
                obj_in(&self.by_id, oid).set_parent(Some(to_id), nearest_dist_f64);
            }
            NodePtr::Routing(routing_id) => {
                if routing_id != to_id && routing_id != from_id {
                    self.nodes[routing_id.index()].parent = Some(to_id);
                    self.nodes[routing_id.index()].parent_distance = nearest_dist_f64;
                }
            }
        }

        if is_leaf {
            if let NodePtr::Object(oid) = grandchild {
                let obj_node = obj_in(&self.by_id, oid);
                let v = obj_node.value();
                self.nodes[to_id.index()]
                    .stats
                    .add_descendant((obj_node.key(), v));
            }
        } else if let NodePtr::Routing(routing_id) = grandchild {
            if routing_id != to_id {
                let child_stats = self.nodes[routing_id.index()].stats.clone();
                self.nodes[to_id.index()]
                    .stats
                    .add_descendants(&child_stats);
            }
        }

        let covering_before = self.nodes[to_id.index()].covering_radius;
        {
            let to = &mut self.nodes[to_id.index()];
            let is_furthest = is_leaf && nearest_dist_f64 > covering_before;
            to.children.push(grandchild);
            if is_furthest {
                if let NodePtr::Object(oid) = grandchild {
                    to.covering_radius = nearest_dist_f64;
                    to.furthest_descendant = Some(oid);
                }
            }
        }

        if !is_leaf {
            self.compute_node_radius(to_id);
        }

        if self.nodes[to_id.index()].covering_radius > covering_before {
            self.expand_cover_up(to_id);
        }

        if needs_radius_recompute {
            self.compute_node_radius(from_id);
        }
    }

    fn merge_routing_nodes(&mut self, from_id: NodeId, to_id: NodeId, from_to_distance: f64) {
        if from_id == to_id {
            return;
        }

        let (is_leaf, from_covering_radius, to_covering_radius, to_key, to_eps) = {
            let from = &self.nodes[from_id.index()];
            let to = &self.nodes[to_id.index()];
            (
                from.is_leaf,
                from.covering_radius,
                to.covering_radius,
                to.key.clone(),
                to.epsilon,
            )
        };

        let needs_recompute = if !is_leaf {
            from_to_distance + from_covering_radius > LOWER_BOUND_FACTOR * to_covering_radius
        } else {
            false
        };

        let children_to_move = std::mem::take(&mut self.nodes[from_id.index()].children);

        for child in children_to_move {
            let child_key = self.child_key(&child);
            let child_eps = self.child_epsilon(&child);
            let new_parent_distance = self.metric(&to_key, to_eps, &child_key, child_eps);
            let new_parent_distance_f64 = dist_value_as_f64(new_parent_distance);

            match child {
                NodePtr::Object(oid) => {
                    obj_in(&self.by_id, oid).set_parent(Some(to_id), new_parent_distance_f64);
                }
                NodePtr::Routing(routing_id) => {
                    if routing_id != to_id && routing_id != from_id {
                        self.nodes[routing_id.index()].parent = Some(to_id);
                        self.nodes[routing_id.index()].parent_distance = new_parent_distance_f64;
                    }
                }
            }

            let is_furthest = if is_leaf {
                matches!(child, NodePtr::Object(_))
                    && new_parent_distance_f64 > self.nodes[to_id.index()].covering_radius
            } else {
                false
            };

            if is_leaf {
                if let NodePtr::Object(oid) = child {
                    let obj_node = obj_in(&self.by_id, oid);
                    let v = obj_node.value();
                    self.nodes[to_id.index()]
                        .stats
                        .add_descendant((obj_node.key(), v));
                }
            }

            self.nodes[to_id.index()].children.push(child);

            if is_furthest {
                if let NodePtr::Object(id) = child {
                    let to = &mut self.nodes[to_id.index()];
                    to.covering_radius = new_parent_distance_f64;
                    to.furthest_descendant = Some(id);
                }
            }
        }

        if !is_leaf {
            let from_stats = self.nodes[from_id.index()].stats.clone();
            self.nodes[to_id.index()]
                .stats
                .add_descendants(&from_stats);
        }

        if needs_recompute {
            self.compute_node_radius(to_id);
        }
        let radius_grew = self.nodes[to_id.index()].covering_radius > to_covering_radius;
        if radius_grew {
            self.expand_cover_up(to_id);
        }

        let parent_id = self.nodes[to_id.index()].parent;

        if let Some(parent_id) = parent_id {
            self.nodes[parent_id.index()]
                .children
                .retain(|child| match child {
                    NodePtr::Routing(id) => *id != from_id,
                    _ => true,
                });
        }
    }

    /// Leert den Baum
    pub fn clear(&mut self) {
        self.root = None;
        self.nodes.clear();
        self.by_id.clear();
        self.free_list.clear();
        self.key_index.clear();
        self.entry_count = 0;
    }

    /// Anzahl der Einträge
    pub fn size(&self) -> usize {
        self.entry_count
    }

    /// Prüft ob der Baum leer ist
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }
}
