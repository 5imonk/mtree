// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use crate::distance::{Distance, EuclideanDistance};
use crate::node::{NodePtr, ObjectNode, RoutingNode};
use crate::placeholder_queue::PlaceholderQueue;
use crate::query::{Query, RangeQuery};
use crate::stats::{DescendantCounter, NodeStats};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// M-Tree Datenstruktur für effiziente Bereichs- und Nearest-Neighbor-Suche
pub struct MTree<K, V, D = f64, S = DescendantCounter>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    min_node_size: usize,
    max_node_size: usize,
    split_sampling: usize,
    root: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
    entries: Vec<Arc<ObjectNode<K, V, S>>>,
    distance_fn: Box<dyn Distance<K, Output = D> + Send + Sync>,
    root_mutex: Mutex<()>,
}

impl<K, V, D, S> Clone for MTree<K, V, D, S>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    fn clone(&self) -> Self {
        Self {
            min_node_size: self.min_node_size,
            max_node_size: self.max_node_size,
            split_sampling: self.split_sampling,
            root: self.root.clone(),  // Arc wird geklont (shared ownership)
            entries: self.entries.clone(),  // Vec und Arcs werden geklont
            distance_fn: self.distance_fn.clone_box(),  // Distance-Funktion wird geklont
            root_mutex: Mutex::new(()),  // Neuer Mutex für den Klon
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

// f64 implementiert bereits Into<f64> und From<f64> implizit

/// Leicht kleinere Schranke beim Pruning, um numerische Fehler zu vermeiden (Untergrenze).
pub const LOWER_BOUND_FACTOR: f64 = 0.999;
/// Leicht größere Schranke beim Pruning, um numerische Fehler zu vermeiden (Obergrenze).
pub const UPPER_BOUND_FACTOR: f64 = 1.001;

/// Wrapper für f64 als Ord-Key (PlaceholderQueue verlangt Key: Ord)
#[derive(Clone, Copy)]
struct KnnDistanceKey(f64);
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

/// Tag für PlaceholderQueue bei k-NN (eindeutig pro Subtree/Objekt)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum KnnTag {
    Subtree(usize),
    Object(usize),
}

/// Queue-Eintrag für k-NN-Traversierung (min-heap: kleinster distance_bound zuerst)
struct KnnQueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    node: Arc<Mutex<RoutingNode<K, V, S>>>,
    #[allow(dead_code)]
    center_distance: f64,
    distance_bound: f64,
}
impl<K, V, S> PartialEq for KnnQueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance_bound == other.distance_bound
    }
}
impl<K, V, S> Eq for KnnQueueEntry<K, V, S> where S: NodeStats<K, V> {}
impl<K, V, S> PartialOrd for KnnQueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance_bound.partial_cmp(&self.distance_bound)
    }
}
impl<K, V, S> Ord for KnnQueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_bound
            .partial_cmp(&self.distance_bound)
            .unwrap_or(Ordering::Equal)
    }
}

impl<K, V> MTree<K, V, f64>
where
    K: Clone + Send + Sync + Hash + Eq,
    V: Send + Sync,
{
    /// Erstellt einen neuen M-Tree mit benutzerdefinierter Distanzfunktion
    pub fn with_distance<F>(distance_fn: F) -> Self
    where
        F: Distance<K, Output = f64> + Send + Sync + 'static,
    {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            root: None,
            entries: Vec::new(),
            distance_fn: Box::new(distance_fn),
            root_mutex: Mutex::new(()),
        }
    }

    /// Erstellt einen neuen M-Tree mit benutzerdefinierten Parametern
    pub fn with_params<F>(
        min_node_size: usize,
        max_node_size: usize,
        split_sampling: usize,
        distance_fn: F,
    ) -> Self
    where
        F: Distance<K, Output = f64> + Send + Sync + 'static,
    {
        Self {
            min_node_size,
            max_node_size,
            split_sampling,
            root: None,
            entries: Vec::new(),
            distance_fn: Box::new(distance_fn),
            root_mutex: Mutex::new(()),
        }
    }
}

/// Spezielle Implementierung für Vec<f32> mit euklidischer Distanz
impl<V> MTree<Vec<f32>, V, f64>
where
    V: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            root: None,
            entries: Vec::new(),
            distance_fn: Box::new(EuclideanDistance),
            root_mutex: Mutex::new(()),
        }
    }
}

/// Spezielle Implementierung für Vec<f64> mit euklidischer Distanz
impl<V> MTree<Vec<f64>, V, f64>
where
    V: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            min_node_size: 5,
            max_node_size: 100,
            split_sampling: 20,
            root: None,
            entries: Vec::new(),
            distance_fn: Box::new(EuclideanDistance),
            root_mutex: Mutex::new(()),
        }
    }
}

impl<K, V, D, S> MTree<K, V, D, S>
where
    K: Clone + Send + Sync + Hash + Eq + Default,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V> + Default,
{
    /// Fügt einen Eintrag in den Baum ein
    pub fn insert(&mut self, key: K, value: V) {
        let _lock = self.root_mutex.lock().unwrap();

        let mut entry = Arc::new(ObjectNode::new(key.clone(), value));

        if let Some(root) = self.root.clone() {
            let entry_clone = entry.clone();
            self.entries.push(entry_clone);
            drop(_lock); // Lock freigeben vor tree_insert
            self.tree_insert(entry, root);
        } else {
            // Erstelle neuen Root; parent setzen, bevor entry irgendwo anders referenziert wird
            let new_root = RoutingNode::with_key(key.clone(), true);
            let root_arc = Arc::new(Mutex::new(new_root));
            if let Some(entry_mut) = Arc::get_mut(&mut entry) {
                entry_mut.parent = Some(root_arc.clone());
                entry_mut.parent_distance = 0.0;
            }
            {
                let mut root_guard = root_arc.lock().unwrap();
                root_guard.children.push(NodePtr::Object(entry.clone()));
            }
            self.entries.push(entry);
            self.root = Some(root_arc);
        }
    }

    fn tree_insert(
        &mut self,
        entry: Arc<ObjectNode<K, V, S>>,
        node: Arc<Mutex<RoutingNode<K, V, S>>>,
    ) {
        let key = entry.value.0.clone();
        let mut current = node;

        loop {
            let mut node_guard = current.lock().unwrap();

            if node_guard.is_leaf {
                // Füge zu diesem Blatt hinzu
                let distance = self.distance_fn.distance(&node_guard.key, &key);

                // Parent-Referenz und parent_distance setzen
                // Da Arc nicht mut ist, müssen wir parent später setzen oder eine andere Lösung finden
                // Für jetzt: parent wird beim Erstellen gesetzt oder über eine andere Methode
                // entry.parent wird später aktualisiert wenn möglich

                // Stats aktualisieren
                node_guard.stats.add_descendant(&entry.value);

                // Covering-Radius aktualisieren wenn nötig
                let distance_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy(&distance) }
                } else {
                    0.0
                };

                if distance_f64 > node_guard.covering_radius {
                    node_guard.covering_radius = distance_f64;
                    node_guard.furthest_descendant =
                        Some(entry.as_ref() as *const ObjectNode<K, V, S>);
                }

                node_guard.children.push(NodePtr::Object(entry.clone()));

                if node_guard.children.len() > self.max_node_size {
                    drop(node_guard);
                    self.split(current.clone());
                }
                break;
            } else {
                // Finde nächsten Knoten - C++ Logik nachbilden
                let mut nearest: Option<Arc<Mutex<RoutingNode<K, V, S>>>> = None;
                let mut nearest_distance = D::infinity();
                let mut within_covering_radius = false;
                let mut radius_increase = D::infinity();

                for child in &node_guard.children {
                    if let NodePtr::Routing(ref routing_child) = child {
                        let child_guard = routing_child.lock().unwrap();
                        let dist = self.distance_fn.distance(&child_guard.key, &key);
                        let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&dist) }
                        } else {
                            0.0
                        };
                        let covering_radius = child_guard.covering_radius;

                        if !within_covering_radius && dist_f64 <= covering_radius {
                            within_covering_radius = true;
                            nearest = Some(routing_child.clone());
                            nearest_distance = dist;
                        } else if within_covering_radius {
                            if dist_f64 <= covering_radius {
                                if dist < nearest_distance {
                                    nearest_distance = dist;
                                    nearest = Some(routing_child.clone());
                                }
                            }
                        } else {
                            // dist_f64 > covering_radius
                            let increase_f64 = dist_f64 - covering_radius;
                            let increase = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                            {
                                unsafe { std::mem::transmute_copy(&increase_f64) }
                            } else {
                                D::infinity()
                            };
                            if dist < radius_increase
                                || (dist == radius_increase && dist < nearest_distance)
                            {
                                nearest = Some(routing_child.clone());
                                nearest_distance = dist;
                                radius_increase = increase;
                            }
                        }
                    }
                }

                drop(node_guard);

                if let Some(next) = nearest {
                    // Aktualisiere covering_radius wenn nötig
                    if !within_covering_radius {
                        let mut next_guard = next.lock().unwrap();
                        let nearest_dist_f64 = if std::mem::size_of::<D>()
                            == std::mem::size_of::<f64>()
                            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                        {
                            unsafe { std::mem::transmute_copy(&nearest_distance) }
                        } else {
                            0.0
                        };
                        if nearest_dist_f64 > next_guard.covering_radius {
                            next_guard.covering_radius = nearest_dist_f64;
                            // furthest_descendant wird später gesetzt wenn entry eingefügt ist
                        }
                        drop(next_guard);
                    }

                    current = next;
                } else {
                    break;
                }
            }
        }
    }

    fn compute_node_radius(&self, node: &mut RoutingNode<K, V, S>) {
        // C++ Referenz: computeNodeRadius()
        if node.is_leaf {
            // Für Blattknoten: Finde maximale parent_distance
            let mut max_distance = 0.0f64;
            let mut furthest: Option<*const ObjectNode<K, V, S>> = None;

            for child in &node.children {
                if let NodePtr::Object(obj_node) = child {
                    if obj_node.parent_distance > max_distance {
                        max_distance = obj_node.parent_distance;
                        furthest = Some(obj_node.as_ref() as *const ObjectNode<K, V, S>);
                    }
                }
            }

            if let Some(furthest_ptr) = furthest {
                node.covering_radius = max_distance;
                node.furthest_descendant = Some(furthest_ptr);
            }
        } else {
            // Für innere Knoten: distance_to_child + child.covering_radius
            let mut max_radius = 0.0f64;
            let mut furthest: Option<*const ObjectNode<K, V, S>> = None;

            for child in &node.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let child_distance = child.parent_distance();
                    let total_radius = child_distance + child_guard.covering_radius;

                    if total_radius > max_radius {
                        max_radius = total_radius;
                        furthest = child_guard.furthest_descendant;
                    }
                }
            }

            if let Some(furthest_ptr) = furthest {
                node.covering_radius = max_radius;
                node.furthest_descendant = Some(furthest_ptr);
            }
        }
    }

    fn split(&mut self, node: Arc<Mutex<RoutingNode<K, V, S>>>) {
        let mut node_guard = node.lock().unwrap();
        let children = std::mem::take(&mut node_guard.children);
        let is_leaf = node_guard.is_leaf;
        let parent = node_guard.parent.clone();
        let is_root = if let Some(ref root) = self.root {
            Arc::ptr_eq(&node, root)
        } else {
            false
        };
        drop(node_guard);

        // Erstelle zwei neue Knoten
        let mut node1 = RoutingNode::new(is_leaf);
        let mut node2 = RoutingNode::new(is_leaf);

        self.promote_and_partition(children, &mut node1, &mut node2);

        // Parent-Referenzen für alle Kinder aktualisieren
        let node1_arc = Arc::new(Mutex::new(node1));
        let node2_arc = Arc::new(Mutex::new(node2));

        // Aktualisiere Parent-Referenzen für node1 Kinder
        {
            let mut node1_guard = node1_arc.lock().unwrap();
            for child in &mut node1_guard.children {
                match child {
                    NodePtr::Object(obj_node) => {
                        if let Some(obj_mut) = Arc::get_mut(obj_node) {
                            obj_mut.parent = Some(node1_arc.clone());
                        }
                    }
                    NodePtr::Routing(routing_node) => {
                        let mut routing_guard = routing_node.lock().unwrap();
                        routing_guard.parent = Some(node1_arc.clone());
                    }
                }
            }
        }

        // Aktualisiere Parent-Referenzen für node2 Kinder
        {
            let mut node2_guard = node2_arc.lock().unwrap();
            for child in &mut node2_guard.children {
                match child {
                    NodePtr::Object(obj_node) => {
                        if let Some(obj_mut) = Arc::get_mut(obj_node) {
                            obj_mut.parent = Some(node2_arc.clone());
                        }
                    }
                    NodePtr::Routing(routing_node) => {
                        let mut routing_guard = routing_node.lock().unwrap();
                        routing_guard.parent = Some(node2_arc.clone());
                    }
                }
            }
        }

        // Berechne parent_distance für beide Knoten
        {
            let mut node1_guard = node1_arc.lock().unwrap();
            let mut node2_guard = node2_arc.lock().unwrap();

            if let Some(ref parent_arc) = parent {
                let parent_guard = parent_arc.lock().unwrap();
                let dist1 = self
                    .distance_fn
                    .distance(&parent_guard.key, &node1_guard.key);
                let dist2 = self
                    .distance_fn
                    .distance(&parent_guard.key, &node2_guard.key);
                node1_guard.parent_distance = if std::mem::size_of::<D>()
                    == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy(&dist1) }
                } else {
                    0.0
                };
                node2_guard.parent_distance = if std::mem::size_of::<D>()
                    == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy(&dist2) }
                } else {
                    0.0
                };
            }
        }

        // Root-Split behandeln oder Parent aktualisieren
        if is_root {
            // Erstelle neuen Root
            let mut new_root = RoutingNode::new(false);
            new_root.children.push(NodePtr::Routing(node1_arc.clone()));
            new_root.children.push(NodePtr::Routing(node2_arc.clone()));

            let new_root_arc = Arc::new(Mutex::new(new_root));

            // Setze Parent-Referenzen für neue Knoten
            {
                let mut node1_guard = node1_arc.lock().unwrap();
                let mut node2_guard = node2_arc.lock().unwrap();
                node1_guard.parent = Some(new_root_arc.clone());
                node2_guard.parent = Some(new_root_arc.clone());
                node1_guard.parent_distance = 0.0;
                node2_guard.parent_distance = 0.0;
            }

            self.root = Some(new_root_arc);
        } else if let Some(ref parent_arc) = parent {
            // Füge beide neuen Knoten zum Parent hinzu
            let mut parent_guard = parent_arc.lock().unwrap();

            // Entferne alten Knoten aus Parent
            parent_guard.children.retain(|child| {
                if let NodePtr::Routing(ref routing_child) = child {
                    !Arc::ptr_eq(routing_child, &node)
                } else {
                    true
                }
            });

            // Füge neue Knoten hinzu
            parent_guard
                .children
                .push(NodePtr::Routing(node1_arc.clone()));
            parent_guard
                .children
                .push(NodePtr::Routing(node2_arc.clone()));

            // Prüfe ob Parent zu viele Kinder hat
            if parent_guard.children.len() > self.max_node_size {
                drop(parent_guard);
                self.split(parent_arc.clone());
            }
        }
    }

    fn promote_and_partition(
        &self,
        children: Vec<NodePtr<K, V, S>>,
        node1: &mut RoutingNode<K, V, S>,
        node2: &mut RoutingNode<K, V, S>,
    ) {
        let mut best_key_1: Option<K> = None;
        let mut best_key_2: Option<K> = None;
        let mut best_av_radius = D::infinity();

        // Sampling-basierte Promotion
        for _ in 0..self.split_sampling {
            let (key1, key2) = self.promote(&children);
            node1.key = key1.clone();
            node2.key = key2.clone();

            let estimated_av_radius =
                self.partition(node1.is_leaf, &children, node1, node2, best_av_radius);

            if estimated_av_radius < best_av_radius {
                best_key_1 = Some(key1);
                best_key_2 = Some(key2);
                best_av_radius = estimated_av_radius;
            }
        }

        if let (Some(k1), Some(k2)) = (best_key_1, best_key_2) {
            node1.key = k1;
            node2.key = k2;
            self.partition(node1.is_leaf, &children, node1, node2, D::infinity());

            // Stats aktualisieren nach Partitionierung
            if node1.is_leaf {
                for child in &node1.children {
                    if let NodePtr::Object(obj_node) = child {
                        node1.stats.add_descendant(&obj_node.value);
                    }
                }
                for child in &node2.children {
                    if let NodePtr::Object(obj_node) = child {
                        node2.stats.add_descendant(&obj_node.value);
                    }
                }
            } else {
                for child in &node1.children {
                    if let NodePtr::Routing(routing_node) = child {
                        let routing_guard = routing_node.lock().unwrap();
                        node1.stats.add_descendants(&routing_guard.stats);
                    }
                }
                for child in &node2.children {
                    if let NodePtr::Routing(routing_node) = child {
                        let routing_guard = routing_node.lock().unwrap();
                        node2.stats.add_descendants(&routing_guard.stats);
                    }
                }
            }

            // Compute node radius für beide Knoten
            self.compute_node_radius(node1);
            self.compute_node_radius(node2);
        }
    }

    fn promote(&self, children: &[NodePtr<K, V, S>]) -> (K, K) {
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

        let key1 = children[a_idx].get_key();
        let key2 = children[b_idx].get_key();
        (key1, key2)
    }

    fn partition(
        &self,
        from_leaf: bool,
        from: &[NodePtr<K, V, S>],
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
            let key = child.get_key();
            let d1 = self.distance_fn.distance(&to_1.key, &key);
            let d2 = self.distance_fn.distance(&to_2.key, &key);
            // Für f64 direkt verwenden, sonst konvertieren
            let d1_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() {
                unsafe { std::mem::transmute_copy(&d1) }
            } else {
                // Fallback: verwende 0.0 wenn nicht f64
                0.0f64
            };
            let d2_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() {
                unsafe { std::mem::transmute_copy(&d2) }
            } else {
                0.0f64
            };
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
            } else {
                // Hole covering_radius aus RoutingNode
                if let NodePtr::Routing(ref routing_node) = node {
                    let routing_guard = routing_node.lock().unwrap();
                    parent_distance + routing_guard.covering_radius
                } else {
                    parent_distance
                }
            };

            if i < boundary {
                estimated_radius_1 = estimated_radius_1.max(estimated_radius);
            } else {
                estimated_radius_2 = estimated_radius_2.max(estimated_radius);
            }
        }

        let total_radius = estimated_radius_1 + estimated_radius_2;
        // Vergleich nur für f64
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>() {
            let bound_f64: f64 = unsafe { std::mem::transmute_copy(&estimated_radius_bound) };
            if total_radius >= bound_f64 {
                return estimated_radius_bound;
            }
        }

        to_1.children.clear();
        to_2.children.clear();

        for (i, dist) in distances.iter().enumerate() {
            let mut node = from[dist.index].clone();
            let parent_distance = if i < boundary {
                dist.from_1
            } else {
                dist.from_2
            };

            // Setze parent_distance für das Kind
            match &mut node {
                NodePtr::Object(obj_node) => {
                    // Für ObjectNodes: parent_distance direkt setzen
                    if let Some(obj_mut) = Arc::get_mut(obj_node) {
                        obj_mut.parent_distance = parent_distance;
                    }
                }
                NodePtr::Routing(routing_node) => {
                    // Für RoutingNodes: über Mutex setzen
                    let mut routing_guard = routing_node.lock().unwrap();
                    routing_guard.parent_distance = parent_distance;
                }
            }

            if i < boundary {
                to_1.children.push(node);
            } else {
                to_2.children.push(node);
            }
        }

        // Konvertiere zurück zu D - nur für f64
        if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            unsafe { std::mem::transmute_copy(&total_radius) }
        } else {
            estimated_radius_bound
        }
    }

    /// Entfernt einen Eintrag aus dem Baum
    pub fn erase(&mut self, key: &K) {
        let _lock = self.root_mutex.lock().unwrap();

        // Finde Eintrag in entries (key ist &K, vergleiche mit value.0)
        let entry_pos = self.entries.iter().position(|entry| entry.value.0 == *key);
        if let Some(pos) = entry_pos {
            let entry = self.entries[pos].clone();
            drop(_lock); // Lock freigeben vor tree_erase
            self.tree_erase(entry);
            let _lock2 = self.root_mutex.lock().unwrap();
            self.entries.remove(pos);
        }
    }

    fn tree_erase(&mut self, entry: Arc<ObjectNode<K, V, S>>) {
        // C++ Referenz: treeErase()
        if let Some(ref parent_arc) = entry.parent {
            let mut parent_guard = parent_arc.lock().unwrap();

            // Entferne aus parent.children
            let entry_ptr = entry.as_ref() as *const ObjectNode<K, V, S>;
            parent_guard.children.retain(|child| match child {
                NodePtr::Object(obj_node) => {
                    obj_node.as_ref() as *const ObjectNode<K, V, S> != entry_ptr
                }
                _ => true,
            });

            // Aktualisiere Stats
            parent_guard.stats.remove_descendant(&entry.value);

            // Prüfe ob furthest_descendant entfernt wurde
            let needs_radius_recompute =
                if let Some(furthest_ptr) = parent_guard.furthest_descendant {
                    furthest_ptr == entry_ptr
                } else {
                    false
                };

            drop(parent_guard);

            // Covering-Radius neu berechnen wenn nötig
            if needs_radius_recompute {
                let mut parent_guard = parent_arc.lock().unwrap();
                self.compute_node_radius(&mut parent_guard);
                drop(parent_guard);
            }

            // Rebalancing wenn nötig
            let mut current_parent = parent_arc.clone();
            loop {
                let children_len = {
                    let parent_guard = current_parent.lock().unwrap();
                    parent_guard.children.len()
                };

                if children_len >= self.min_node_size {
                    break;
                }

                // Prüfe ob Root
                let is_root = if let Some(ref root) = self.root {
                    Arc::ptr_eq(&current_parent, root)
                } else {
                    false
                };

                if is_root {
                    self.pull_up_root();
                    break;
                }

                let next_parent = {
                    let parent_guard = current_parent.lock().unwrap();
                    parent_guard.parent.clone()
                };

                self.rebalance_node(current_parent.clone());

                // Hole neuen Parent für nächste Iteration
                if let Some(new_parent) = next_parent {
                    current_parent = new_parent;
                } else {
                    break;
                }
            }
        }
    }

    fn pull_up_root(&mut self) {
        // C++ Referenz: pullUpRoot()
        if let Some(ref root_arc) = self.root {
            let root_guard = root_arc.lock().unwrap();

            if root_guard.children.len() == 1 {
                if let Some(NodePtr::Routing(ref only_child)) = root_guard.children.first() {
                    let child_arc = only_child.clone();
                    drop(root_guard);

                    let mut child_guard = child_arc.lock().unwrap();
                    child_guard.parent = None;
                    child_guard.parent_distance = 0.0;
                    drop(child_guard);

                    self.root = Some(child_arc);
                }
            } else if root_guard.children.is_empty() {
                drop(root_guard);
                self.root = None;
            }
        }
    }

    fn rebalance_node(&mut self, node: Arc<Mutex<RoutingNode<K, V, S>>>) {
        // C++ Referenz: rebalanceNode()
        let node_guard = node.lock().unwrap();
        let parent_arc = node_guard.parent.clone();
        drop(node_guard);

        if let Some(ref parent_arc) = parent_arc {
            let parent_guard = parent_arc.lock().unwrap();

            // Finde Spender oder Merge-Kandidaten
            let mut nearest_donor: Option<Arc<Mutex<RoutingNode<K, V, S>>>> = None;
            let mut nearest_donor_distance = D::infinity();
            let mut nearest_merge: Option<Arc<Mutex<RoutingNode<K, V, S>>>> = None;
            let mut nearest_merge_distance = D::infinity();

            let node_key = {
                let node_guard = node.lock().unwrap();
                node_guard.key.clone()
            };

            for child in &parent_guard.children {
                if let NodePtr::Routing(ref sibling) = child {
                    if Arc::ptr_eq(sibling, &node) {
                        continue;
                    }

                    let sibling_guard = sibling.lock().unwrap();
                    let dist = self.distance_fn.distance(&node_key, &sibling_guard.key);

                    if sibling_guard.children.len() > self.min_node_size {
                        if dist < nearest_donor_distance {
                            nearest_donor_distance = dist;
                            nearest_donor = Some(sibling.clone());
                        }
                    } else {
                        if dist < nearest_merge_distance {
                            nearest_merge_distance = dist;
                            nearest_merge = Some(sibling.clone());
                        }
                    }
                }
            }

            drop(parent_guard);

            if let Some(donor) = nearest_donor {
                self.donate_child(donor, node.clone());
            } else if let Some(merge_target) = nearest_merge {
                let merge_dist = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                    && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
                {
                    unsafe { std::mem::transmute_copy(&nearest_merge_distance) }
                } else {
                    0.0
                };
                self.merge_routing_nodes(node, merge_target, merge_dist);
            }
        }
    }

    fn donate_child(
        &mut self,
        from: Arc<Mutex<RoutingNode<K, V, S>>>,
        to: Arc<Mutex<RoutingNode<K, V, S>>>,
    ) {
        // C++ Referenz: donateChild()
        let (to_key, is_leaf) = {
            let to_guard = to.lock().unwrap();
            (to_guard.key.clone(), to_guard.is_leaf)
        };

        // Finde nächsten Grandchild zu to
        let (nearest_grandchild_idx, nearest_distance) = {
            let from_guard = from.lock().unwrap();
            let mut best_idx = 0;
            let mut best_dist = D::infinity();

            for (i, grandchild) in from_guard.children.iter().enumerate() {
                let grandchild_key = grandchild.get_key();
                let dist = self.distance_fn.distance(&to_key, &grandchild_key);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            (best_idx, best_dist)
        };

        let nearest_dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
        {
            unsafe { std::mem::transmute_copy(&nearest_distance) }
        } else {
            0.0
        };

        // Entferne Grandchild von from
        let (grandchild, needs_radius_recompute) = {
            let mut from_guard = from.lock().unwrap();
            let grandchild = from_guard.children.remove(nearest_grandchild_idx);

            // Prüfe ob furthest_descendant betroffen ist
            let needs_recompute = if let Some(furthest_ptr) = from_guard.furthest_descendant {
                match &grandchild {
                    NodePtr::Object(obj_node) => {
                        obj_node.as_ref() as *const ObjectNode<K, V, S> == furthest_ptr
                    }
                    NodePtr::Routing(_) => {
                        // Vereinfacht: immer neu berechnen wenn RoutingNode
                        true
                    }
                }
            } else {
                false
            };

            // Aktualisiere Stats
            if is_leaf {
                if let NodePtr::Object(ref obj_node) = grandchild {
                    from_guard.stats.remove_descendant(&obj_node.value);
                }
            } else {
                if let NodePtr::Routing(ref routing_node) = grandchild {
                    let routing_guard = routing_node.lock().unwrap();
                    from_guard.stats.remove_descendants(&routing_guard.stats);
                }
            }

            drop(from_guard);
            (grandchild, needs_recompute)
        };

        // Setze parent und parent_distance für grandchild
        match &grandchild {
            NodePtr::Object(_obj_node) => {
                // Für ObjectNodes: parent kann nicht direkt gesetzt werden wenn Arc nicht mut ist
            }
            NodePtr::Routing(routing_node) => {
                let mut routing_guard = routing_node.lock().unwrap();
                routing_guard.parent = Some(to.clone());
                routing_guard.parent_distance = nearest_dist_f64;
            }
        }

        // Aktualisiere Stats für to
        if is_leaf {
            if let NodePtr::Object(ref obj_node) = grandchild {
                let mut to_guard = to.lock().unwrap();
                to_guard.stats.add_descendant(&obj_node.value);
                drop(to_guard);
            }
        } else {
            if let NodePtr::Routing(ref routing_node) = grandchild {
                let routing_guard = routing_node.lock().unwrap();
                let mut to_guard = to.lock().unwrap();
                to_guard.stats.add_descendants(&routing_guard.stats);
                drop(to_guard);
            }
        }

        // Füge zu to hinzu
        {
            let mut to_guard = to.lock().unwrap();
            let current_covering_radius = to_guard.covering_radius;
            let is_furthest = is_leaf && nearest_dist_f64 > current_covering_radius;

            to_guard.children.push(grandchild);

            // Aktualisiere covering_radius von to wenn nötig
            if !is_leaf {
                // Für innere Knoten: computeNonLeafRadius
                self.compute_node_radius(&mut to_guard);
            } else if is_furthest {
                // Für Blattknoten: prüfe ob covering_radius aktualisiert werden muss
                // grandchild wurde bereits gepusht, hole letzten Eintrag
                let furthest_ptr =
                    if let Some(NodePtr::Object(ref obj_node)) = to_guard.children.last() {
                        Some(obj_node.as_ref() as *const ObjectNode<K, V, S>)
                    } else {
                        None
                    };
                if let Some(ptr) = furthest_ptr {
                    to_guard.covering_radius = nearest_dist_f64;
                    to_guard.furthest_descendant = Some(ptr);
                }
            }
        }

        // Aktualisiere Radien von from wenn nötig
        if needs_radius_recompute {
            let mut from_guard = from.lock().unwrap();
            self.compute_node_radius(&mut from_guard);
        }
    }

    fn merge_routing_nodes(
        &mut self,
        from: Arc<Mutex<RoutingNode<K, V, S>>>,
        to: Arc<Mutex<RoutingNode<K, V, S>>>,
        from_to_distance: f64,
    ) {
        // C++ Referenz: mergeRoutingNodes()
        let (is_leaf, from_covering_radius, to_covering_radius, to_key) = {
            let from_guard = from.lock().unwrap();
            let to_guard = to.lock().unwrap();
            (
                from_guard.is_leaf,
                from_guard.covering_radius,
                to_guard.covering_radius,
                to_guard.key.clone(),
            )
        };

        // Prüfe ob covering_radius aktualisiert werden muss
        let needs_recompute = if !is_leaf {
            from_to_distance + from_covering_radius > LOWER_BOUND_FACTOR * to_covering_radius
        } else {
            false
        };

        // Verschiebe alle Kinder von from zu to
        let children_to_move = {
            let mut from_guard = from.lock().unwrap();
            std::mem::take(&mut from_guard.children)
        };

        // Stats werden später aktualisiert, nicht geklont
        // (S muss Clone implementieren, aber wir können es auch anders machen)

        for mut child in children_to_move {
            // Berechne neue parent_distance
            let child_key = child.get_key();
            let new_parent_distance = self.distance_fn.distance(&to_key, &child_key);
            let new_parent_distance_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>()
                && std::mem::align_of::<D>() == std::mem::align_of::<f64>()
            {
                unsafe { std::mem::transmute_copy(&new_parent_distance) }
            } else {
                0.0
            };

            // Setze parent und parent_distance
            match &mut child {
                NodePtr::Object(_obj_node) => {
                    // Für ObjectNodes: parent kann nicht direkt gesetzt werden wenn Arc nicht mut ist
                    // Parent wird später gesetzt wenn möglich
                }
                NodePtr::Routing(routing_node) => {
                    let mut routing_guard = routing_node.lock().unwrap();
                    routing_guard.parent = Some(to.clone());
                    routing_guard.parent_distance = new_parent_distance_f64;
                }
            }

            let mut to_guard = to.lock().unwrap();

            // Aktualisiere covering_radius wenn nötig
            let is_furthest = if is_leaf {
                if let NodePtr::Object(ref _obj_node) = child {
                    let mut is_furthest = false;
                    if new_parent_distance_f64 > to_guard.covering_radius {
                        is_furthest = true;
                    }
                    is_furthest
                } else {
                    false
                }
            } else {
                false
            };

            to_guard.children.push(child);

            if is_furthest {
                // child wurde bereits gepusht, hole letzten Eintrag
                let furthest_ptr =
                    if let Some(NodePtr::Object(ref obj_node)) = to_guard.children.last() {
                        Some(obj_node.as_ref() as *const ObjectNode<K, V, S>)
                    } else {
                        None
                    };
                if let Some(ptr) = furthest_ptr {
                    to_guard.covering_radius = new_parent_distance_f64;
                    to_guard.furthest_descendant = Some(ptr);
                }
            }

            drop(to_guard);
        }

        // Aktualisiere Stats - durchlaufe alle Kinder von from
        {
            let mut to_guard = to.lock().unwrap();
            let from_guard = from.lock().unwrap();
            if is_leaf {
                // Für Blattknoten: zähle ObjectNodes
                for child in &from_guard.children {
                    if let NodePtr::Object(ref obj_node) = child {
                        to_guard.stats.add_descendant(&obj_node.value);
                    }
                }
            } else {
                // Für innere Knoten: addiere Stats
                to_guard.stats.add_descendants(&from_guard.stats);
            }
        }

        if needs_recompute {
            // computeNonLeafRadius - vereinfacht: compute_node_radius
            let mut to_guard = to.lock().unwrap();
            self.compute_node_radius(&mut to_guard);
        }

        // Entferne from aus Parent
        let parent_arc = {
            let to_guard = to.lock().unwrap();
            to_guard.parent.clone()
        };

        if let Some(ref parent_arc) = parent_arc {
            let mut parent_guard = parent_arc.lock().unwrap();
            parent_guard.children.retain(|child| {
                if let NodePtr::Routing(ref routing_child) = child {
                    !Arc::ptr_eq(routing_child, &from)
                } else {
                    true
                }
            });
        }
    }

    /// Leert den Baum
    pub fn clear(&mut self) {
        self.root = None;
        self.entries.clear();
    }

    /// Anzahl der Einträge
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Prüft ob der Baum leer ist
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

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
                        let dist_d: D = distance_fn.distance(&obj_node.value.0, needle);
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
                    placeholder_queue.add_placeholder(
                        tag,
                        KnnDistanceKey(upper_bound),
                        1,
                        tag,
                    );
                    subtree_id = subtree_id.wrapping_add(1);
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.into_iter().take(k).collect()
    }
}
