// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::sync::{Arc, Mutex, RwLock};
use crate::entry::EntryId;
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
    pub id: EntryId,
    pub value: RwLock<(K, V)>,
    identity_hash: u64,
    epsilon: f64,
    parent: Mutex<Option<Arc<Mutex<RoutingNode<K, V, S>>>>>,
    parent_distance: Mutex<f64>,
}

unsafe impl<K: Send, V: Send, S: Send> Send for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}

impl<K, V, S> ObjectNode<K, V, S>
where
    S: crate::stats::NodeStats<K, V> + Default,
{
    pub fn new(id: EntryId, key: K, value: V) -> Self {
        Self::with_identity(id, key, value, 0, 0.0)
    }

    pub fn with_identity(id: EntryId, key: K, value: V, identity_hash: u64, epsilon: f64) -> Self {
        Self {
            id,
            value: RwLock::new((key, value)),
            identity_hash,
            epsilon,
            parent: Mutex::new(None),
            parent_distance: Mutex::new(0.0),
        }
    }
}

impl<K, V, S> ObjectNode<K, V, S>
where
    S: crate::stats::NodeStats<K, V>,
{
    pub fn parent(&self) -> Option<Arc<Mutex<RoutingNode<K, V, S>>>> {
        self.parent.lock().unwrap().clone()
    }

    pub fn set_parent(
        &self,
        parent: Option<Arc<Mutex<RoutingNode<K, V, S>>>>,
        distance: f64,
    ) {
        *self.parent.lock().unwrap() = parent;
        *self.parent_distance.lock().unwrap() = distance;
    }

    pub fn dist_to_parent(&self) -> f64 {
        *self.parent_distance.lock().unwrap()
    }

    pub fn key(&self) -> K
    where
        K: Clone,
    {
        self.value.read().unwrap().0.clone()
    }

    pub fn payload(&self) -> V
    where
        V: Clone,
    {
        self.value.read().unwrap().1.clone()
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn identity_hash(&self) -> u64 {
        self.identity_hash
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
    pub epsilon: f64,
    pub identity_hash: u64,
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
            epsilon: 0.0,
            identity_hash: 0,
            stats: S::default(),
            covering_radius: 0.0,
            furthest_descendant: None,
            parent: None,
            parent_distance: 0.0,
        }
    }
    
    pub fn with_key(key: K, is_leaf: bool) -> Self {
        Self::with_routing(key, 0.0, 0, is_leaf)
    }

    pub fn with_routing(key: K, epsilon: f64, identity_hash: u64, is_leaf: bool) -> Self {
        Self {
            children: Vec::new(),
            is_leaf,
            key,
            epsilon,
            identity_hash,
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
            NodePtr::Object(node) => node.key(),
            NodePtr::Routing(node) => {
                let node = node.lock().unwrap();
                node.key.clone()
            }
        }
    }

    pub fn epsilon(&self) -> f64 {
        match self {
            NodePtr::Object(node) => node.epsilon(),
            NodePtr::Routing(node) => node.lock().unwrap().epsilon,
        }
    }

    pub fn identity_hash(&self) -> u64 {
        match self {
            NodePtr::Object(node) => node.identity_hash(),
            NodePtr::Routing(node) => node.lock().unwrap().identity_hash,
        }
    }
    
    pub fn parent_distance(&self) -> f64 {
        match self {
            NodePtr::Object(node) => node.dist_to_parent(),
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
