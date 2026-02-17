// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::sync::{Arc, Mutex};
use std::collections::BinaryHeap;
use crate::node::{ObjectNode, RoutingNode, NodePtr};
use crate::stats::NodeStats;
use crate::tree::DistanceType;

/// Queue-Eintrag für Query-Traversierung
struct QueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    node: Arc<Mutex<RoutingNode<K, V, S>>>,
    center_distance: f64,
    distance_bound: f64,
}

impl<K, V, S> PartialEq for QueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance_bound == other.distance_bound
    }
}

impl<K, V, S> Eq for QueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
}

impl<K, V, S> PartialOrd for QueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.distance_bound.partial_cmp(&self.distance_bound)
    }
}

impl<K, V, S> Ord for QueueEntry<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance_bound.partial_cmp(&self.distance_bound).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Query Iterator für Bereichssuchen
pub struct Query<K, V, D = f64, S = crate::stats::DescendantCounter>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    needle: K,
    min_radius: D,
    max_radius: D,
    queue: BinaryHeap<QueueEntry<K, V, S>>,
    current_leaf: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
    current_leaf_distance: f64,
    leaf_iterator: usize,
    is_at_end: bool,
    distance_fn: Option<Box<dyn crate::distance::Distance<K, Output = D> + Send + Sync>>,
}

impl<K, V, D, S> Query<K, V, D, S>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub fn empty() -> Self {
        Self {
            needle: unsafe { std::mem::zeroed() },
            min_radius: D::zero(),
            max_radius: D::zero(),
            queue: BinaryHeap::new(),
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: true,
            distance_fn: None,
        }
    }
    
    pub fn new(
        needle: K,
        min_radius: D,
        max_radius: D,
        root: Arc<Mutex<RoutingNode<K, V, S>>>,
        distance_fn: Box<dyn crate::distance::Distance<K, Output = D> + Send + Sync>,
    ) -> Self {
        let mut queue: BinaryHeap<QueueEntry<K, V, S>> = BinaryHeap::new();
        let root_distance = {
            let root_guard = root.lock().unwrap();
            let dist = distance_fn.distance(&root_guard.key, &needle);
            if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                unsafe { std::mem::transmute_copy(&dist) }
            } else {
                0.0
            }
        };
        queue.push(QueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_distance,
        });
        Self {
            needle,
            min_radius,
            max_radius,
            queue,
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: false,
            distance_fn: Some(distance_fn),
        }
    }
    
    pub fn at_end(&self) -> bool {
        self.is_at_end
    }
    
    fn find_next_leaf(&mut self) -> bool {
        while let Some(entry) = self.queue.pop() {
            let node_guard = entry.node.lock().unwrap();
            
            if node_guard.is_leaf {
                self.current_leaf = Some(entry.node.clone());
                self.current_leaf_distance = entry.center_distance;
                self.leaf_iterator = 0;
                return true;
            }
            
            let distance_fn = match &self.distance_fn {
                Some(df) => df,
                None => return false,
            };
            for child in &node_guard.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let dist = distance_fn.distance(&child_guard.key, &self.needle);
                    let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                        unsafe { std::mem::transmute_copy(&dist) }
                    } else {
                        0.0
                    };
                    let distance_bound = dist_f64 + child_guard.covering_radius;
                    
                    let min_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                        unsafe { std::mem::transmute_copy(&self.min_radius) }
                    } else {
                        0.0
                    };
                    let max_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                        unsafe { std::mem::transmute_copy(&self.max_radius) }
                    } else {
                        f64::INFINITY
                    };
                    let min_prune = min_radius_f64 * crate::tree::LOWER_BOUND_FACTOR;
                    let max_prune = max_radius_f64 * crate::tree::UPPER_BOUND_FACTOR;
                    if distance_bound >= min_prune && dist_f64 <= max_prune + child_guard.covering_radius {
                        self.queue.push(QueueEntry {
                            node: routing_child.clone(),
                            center_distance: dist_f64,
                            distance_bound,
                        });
                    }
                }
            }
        }
        false
    }
}

impl<K, V, D, S> Iterator for Query<K, V, D, S>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    type Item = (Arc<ObjectNode<K, V, S>>, D);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_at_end {
            return None;
        }
        
        loop {
            if let Some(ref leaf) = self.current_leaf {
                let leaf_guard = leaf.lock().unwrap();
                
                while self.leaf_iterator < leaf_guard.children.len() {
                    if let NodePtr::Object(ref obj_node) = &leaf_guard.children[self.leaf_iterator] {
                        let dist = match &self.distance_fn {
                            Some(df) => df.distance(&obj_node.value.0, &self.needle),
                            None => break,
                        };
                        let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                            unsafe { std::mem::transmute_copy(&dist) }
                        } else {
                            0.0
                        };
                        
                        let min_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                            unsafe { std::mem::transmute_copy(&self.min_radius) }
                        } else {
                            0.0
                        };
                        let max_radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                            unsafe { std::mem::transmute_copy(&self.max_radius) }
                        } else {
                            f64::INFINITY
                        };
                        
                        self.leaf_iterator += 1;
                        
                        if dist_f64 >= min_radius_f64 && dist_f64 <= max_radius_f64 {
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

/// RangeQuery Iterator für Radius-Suchen
pub struct RangeQuery<K, V, D = f64, S = crate::stats::DescendantCounter>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    needle: K,
    radius: D,
    queue: BinaryHeap<QueueEntry<K, V, S>>,
    current_leaf: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
    current_leaf_distance: f64,
    leaf_iterator: usize,
    is_at_end: bool,
    distance_fn: Option<Box<dyn crate::distance::Distance<K, Output = D> + Send + Sync>>,
}

impl<K, V, D, S> RangeQuery<K, V, D, S>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    pub fn empty() -> Self {
        Self {
            needle: unsafe { std::mem::zeroed() },
            radius: D::zero(),
            queue: BinaryHeap::new(),
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: true,
            distance_fn: None,
        }
    }
    
    pub fn new(
        needle: K,
        radius: D,
        root: Arc<Mutex<RoutingNode<K, V, S>>>,
        distance_fn: Box<dyn crate::distance::Distance<K, Output = D> + Send + Sync>,
    ) -> Self {
        let mut queue: BinaryHeap<QueueEntry<K, V, S>> = BinaryHeap::new();
        let root_distance = {
            let root_guard = root.lock().unwrap();
            let dist = distance_fn.distance(&root_guard.key, &needle);
            if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                unsafe { std::mem::transmute_copy(&dist) }
            } else {
                0.0
            }
        };
        queue.push(QueueEntry {
            node: root,
            center_distance: root_distance,
            distance_bound: root_distance,
        });
        Self {
            needle,
            radius,
            queue,
            current_leaf: None,
            current_leaf_distance: 0.0,
            leaf_iterator: 0,
            is_at_end: false,
            distance_fn: Some(distance_fn),
        }
    }
    
    pub fn at_end(&self) -> bool {
        self.is_at_end
    }
    
    fn find_next_leaf(&mut self) -> bool {
        while let Some(entry) = self.queue.pop() {
            let node_guard = entry.node.lock().unwrap();
            
            if node_guard.is_leaf {
                self.current_leaf = Some(entry.node.clone());
                self.current_leaf_distance = entry.center_distance;
                self.leaf_iterator = 0;
                return true;
            }
            
            let distance_fn = match &self.distance_fn {
                Some(df) => df,
                None => return false,
            };
            for child in &node_guard.children {
                if let NodePtr::Routing(ref routing_child) = child {
                    let child_guard = routing_child.lock().unwrap();
                    let dist = distance_fn.distance(&child_guard.key, &self.needle);
                    let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                        unsafe { std::mem::transmute_copy(&dist) }
                    } else {
                        0.0
                    };
                    let distance_bound = dist_f64 + child_guard.covering_radius;
                    
                    let radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                        unsafe { std::mem::transmute_copy(&self.radius) }
                    } else {
                        f64::INFINITY
                    };
                    let radius_prune = radius_f64 * crate::tree::UPPER_BOUND_FACTOR;
                    if dist_f64 <= radius_prune + child_guard.covering_radius {
                        self.queue.push(QueueEntry {
                            node: routing_child.clone(),
                            center_distance: dist_f64,
                            distance_bound,
                        });
                    }
                }
            }
        }
        false
    }
}

impl<K, V, D, S> Iterator for RangeQuery<K, V, D, S>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
{
    type Item = (Arc<ObjectNode<K, V, S>>, D);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_at_end {
            return None;
        }
        
        loop {
            if let Some(ref leaf) = self.current_leaf {
                let leaf_guard = leaf.lock().unwrap();
                
                while self.leaf_iterator < leaf_guard.children.len() {
                    if let NodePtr::Object(ref obj_node) = &leaf_guard.children[self.leaf_iterator] {
                        let dist = match &self.distance_fn {
                            Some(df) => df.distance(&obj_node.value.0, &self.needle),
                            None => break,
                        };
                        let dist_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
                            unsafe { std::mem::transmute_copy(&dist) }
                        } else {
                            0.0
                        };
                        
                        let radius_f64 = if std::mem::size_of::<D>() == std::mem::size_of::<f64>() && std::mem::align_of::<D>() == std::mem::align_of::<f64>() {
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
