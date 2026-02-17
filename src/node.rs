// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::sync::{Arc, Mutex};
use crate::stats::NodeStats;

/// Basis-Trait für Knoten
pub trait Node<K>
where
    K: Send + Sync,
{
    fn get_key(&self) -> &K;
    fn parent_distance(&self) -> f64;
}

/// ObjectNode repräsentiert einen Eintrag im Baum (Blattknoten)
pub struct ObjectNode<K, V, S = crate::stats::DescendantCounter>
where
    S: crate::stats::NodeStats<K, V>,
{
    pub value: (K, V),
    pub parent: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
    pub parent_distance: f64,
}

unsafe impl<K: Send, V: Send, S: Send> Send for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}

impl<K, V, S> ObjectNode<K, V, S>
where
    S: crate::stats::NodeStats<K, V> + Default,
{
    pub fn new(key: K, value: V) -> Self {
        Self {
            value: (key, value),
            parent: None,
            parent_distance: 0.0,
        }
    }
}

impl<K, V, S> Node<K> for ObjectNode<K, V, S>
where
    K: Send + Sync,
    S: crate::stats::NodeStats<K, V>,
{
    fn get_key(&self) -> &K {
        &self.value.0
    }
    
    fn parent_distance(&self) -> f64 {
        self.parent_distance
    }
}

/// RoutingNode ist ein innerer Knoten im Baum
pub struct RoutingNode<K, V, S = crate::stats::DescendantCounter>
where
    S: NodeStats<K, V>,
{
    pub children: Vec<NodePtr<K, V, S>>,
    pub is_leaf: bool,
    pub key: K,
    pub stats: S,
    pub covering_radius: f64,
    pub furthest_descendant: Option<*const ObjectNode<K, V, S>>,
    pub parent: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
    pub parent_distance: f64,
}

unsafe impl<K: Send, V: Send, S: Send> Send for RoutingNode<K, V, S> where S: NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for RoutingNode<K, V, S> where S: NodeStats<K, V> {}

impl<K, V, S> RoutingNode<K, V, S>
where
    S: NodeStats<K, V> + Default,
    K: Default,
{
    pub fn new(is_leaf: bool) -> Self {
        Self {
            children: Vec::new(),
            is_leaf,
            key: K::default(),
            stats: S::default(),
            covering_radius: 0.0,
            furthest_descendant: None,
            parent: None,
            parent_distance: 0.0,
        }
    }
    
    pub fn with_key(key: K, is_leaf: bool) -> Self {
        Self {
            children: Vec::new(),
            is_leaf,
            key,
            stats: S::default(),
            covering_radius: 0.0,
            furthest_descendant: None,
            parent: None,
            parent_distance: 0.0,
        }
    }
}

impl<K, V, S> Node<K> for RoutingNode<K, V, S>
where
    S: NodeStats<K, V>,
    K: Send + Sync,
{
    fn get_key(&self) -> &K {
        &self.key
    }
    
    fn parent_distance(&self) -> f64 {
        self.parent_distance
    }
}

/// Zeiger auf einen Knoten (entweder ObjectNode oder RoutingNode)
pub enum NodePtr<K, V, S = crate::stats::DescendantCounter>
where
    S: NodeStats<K, V>,
{
    Object(Arc<ObjectNode<K, V, S>>),
    Routing(Arc<Mutex<RoutingNode<K, V, S>>>),
}

impl<K, V, S> Clone for NodePtr<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn clone(&self) -> Self {
        match self {
            NodePtr::Object(node) => NodePtr::Object(node.clone()),
            NodePtr::Routing(node) => NodePtr::Routing(node.clone()),
        }
    }
}

unsafe impl<K: Send, V: Send, S: Send> Send for NodePtr<K, V, S> where S: NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for NodePtr<K, V, S> where S: NodeStats<K, V> {}

impl<K, V, S> NodePtr<K, V, S>
where
    S: NodeStats<K, V>,
{
    pub fn get_key(&self) -> K
    where
        K: Clone,
    {
        match self {
            NodePtr::Object(node) => node.value.0.clone(),
            NodePtr::Routing(node) => {
                let node = node.lock().unwrap();
                node.key.clone()
            }
        }
    }
    
    pub fn parent_distance(&self) -> f64 {
        match self {
            NodePtr::Object(node) => node.parent_distance,
            NodePtr::Routing(node) => {
                let node = node.lock().unwrap();
                node.parent_distance
            }
        }
    }
    
    pub fn as_object(&self) -> Option<&Arc<ObjectNode<K, V, S>>> {
        match self {
            NodePtr::Object(node) => Some(node),
            _ => None,
        }
    }
    
    pub fn as_routing(&self) -> Option<&Arc<Mutex<RoutingNode<K, V, S>>>> {
        match self {
            NodePtr::Routing(node) => Some(node),
            _ => None,
        }
    }
}
