// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::cell::UnsafeCell;
use std::sync::{Arc, RwLock};
use crate::entry::EntryId;
use crate::stats::NodeStats;

/// Index in die Routing-Arena von [`crate::tree::MTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        NodeId(u32::try_from(index).expect("routing arena index fits in u32"))
    }
}

/// Basis-Trait für Knoten
#[allow(dead_code)]
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
    /// Such-Key außerhalb des Payload-Locks (Distanz-Hotpath ohne Clone/Lock).
    /// Mutation nur über [`ObjectNode::set_key`] unter exklusivem `MTree`-Borrow.
    key: UnsafeCell<K>,
    pub value: RwLock<V>,
    identity_hash: u64,
    epsilon: f64,
    /// Leaf-Parent in der Routing-Arena.
    parent: UnsafeCell<Option<NodeId>>,
    parent_distance: UnsafeCell<f64>,
    _stats: std::marker::PhantomData<S>,
}

unsafe impl<K: Send, V: Send, S: Send> Send for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for ObjectNode<K, V, S> where S: crate::stats::NodeStats<K, V> {}

impl<K, V, S> ObjectNode<K, V, S>
where
    S: crate::stats::NodeStats<K, V> + Default,
{
    #[allow(dead_code)]
    pub fn new(id: EntryId, key: K, value: V) -> Self {
        Self::with_identity(id, key, value, 0, 0.0)
    }

    pub fn with_identity(id: EntryId, key: K, value: V, identity_hash: u64, epsilon: f64) -> Self {
        Self {
            id,
            key: UnsafeCell::new(key),
            value: RwLock::new(value),
            identity_hash,
            epsilon,
            parent: UnsafeCell::new(None),
            parent_distance: UnsafeCell::new(0.0),
            _stats: std::marker::PhantomData,
        }
    }
}

impl<K, V, S> ObjectNode<K, V, S>
where
    S: crate::stats::NodeStats<K, V>,
{
    pub fn parent(&self) -> Option<NodeId> {
        // SAFETY: Parent wird nur unter `&mut MTree` geschrieben.
        unsafe { *self.parent.get() }
    }

    pub fn set_parent(&self, parent: Option<NodeId>, distance: f64) {
        // SAFETY: Aufrufer hält `&mut MTree`.
        unsafe {
            *self.parent.get() = parent;
            *self.parent_distance.get() = distance;
        }
    }

    pub fn dist_to_parent(&self) -> f64 {
        // SAFETY: siehe `parent`.
        unsafe { *self.parent_distance.get() }
    }

    pub fn key(&self) -> &K {
        // SAFETY: Key wird nur unter `&mut MTree` geschrieben; parallele Reads sind API-seitig ausgeschlossen.
        unsafe { &*self.key.get() }
    }

    /// Ersetzt den Key. Nur unter exklusivem `MTree`-Borrow aufrufen (z. B. `update_key`).
    pub(crate) fn set_key(&self, new_key: K) {
        // SAFETY: Aufrufer hält `&mut MTree`; keine parallelen `key()`-Reader.
        unsafe {
            *self.key.get() = new_key;
        }
    }

    pub fn payload(&self) -> V
    where
        V: Clone,
    {
        self.value.read().unwrap().clone()
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
    pub furthest_descendant: Option<EntryId>,
    pub parent: Option<NodeId>,
    pub parent_distance: f64,
}

unsafe impl<K: Send, V: Send, S: Send> Send for RoutingNode<K, V, S> where S: NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for RoutingNode<K, V, S> where S: NodeStats<K, V> {}

impl<K, V, S> Clone for RoutingNode<K, V, S>
where
    K: Clone,
    S: NodeStats<K, V> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            children: self.children.clone(),
            is_leaf: self.is_leaf,
            key: self.key.clone(),
            epsilon: self.epsilon,
            identity_hash: self.identity_hash,
            stats: self.stats.clone(),
            covering_radius: self.covering_radius,
            furthest_descendant: self.furthest_descendant,
            parent: self.parent,
            parent_distance: self.parent_distance,
        }
    }
}

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

    #[allow(dead_code)]
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

/// Zeiger auf einen Knoten (Object in `by_id`, Routing in der Arena)
pub enum NodePtr<K, V, S = crate::stats::DescendantCounter>
where
    S: NodeStats<K, V>,
{
    Object(Arc<ObjectNode<K, V, S>>),
    Routing(NodeId),
}

impl<K, V, S> Clone for NodePtr<K, V, S>
where
    S: NodeStats<K, V>,
{
    fn clone(&self) -> Self {
        match self {
            NodePtr::Object(node) => NodePtr::Object(node.clone()),
            NodePtr::Routing(id) => NodePtr::Routing(*id),
        }
    }
}

unsafe impl<K: Send, V: Send, S: Send> Send for NodePtr<K, V, S> where S: NodeStats<K, V> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for NodePtr<K, V, S> where S: NodeStats<K, V> {}

impl<K, V, S> NodePtr<K, V, S>
where
    S: NodeStats<K, V>,
{
    #[allow(dead_code)]
    pub fn as_object(&self) -> Option<&Arc<ObjectNode<K, V, S>>> {
        match self {
            NodePtr::Object(node) => Some(node),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn as_routing_id(&self) -> Option<NodeId> {
        match self {
            NodePtr::Routing(id) => Some(*id),
            _ => None,
        }
    }
}
