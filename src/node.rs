// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::cell::UnsafeCell;
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
    /// Payload ohne Lock. Mutation nur über [`ObjectNode::set_value`] unter exklusivem
    /// `MTree`-Borrow; parallele Reads unter `&MTree` sind erlaubt.
    value: UnsafeCell<V>,
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
            value: UnsafeCell::new(value),
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
        // SAFETY: Write nur unter `&mut MTree`. Parallele Reads unter shared `&MTree`
        // sind erlaubt — der Borrow-Checker schließt Writer aus.
        unsafe { *self.parent.get() }
    }

    pub fn set_parent(&self, parent: Option<NodeId>, distance: f64) {
        // SAFETY: Aufrufer hält `&mut MTree` (exklusiv); keine parallelen Reader möglich.
        unsafe {
            *self.parent.get() = parent;
            *self.parent_distance.get() = distance;
        }
    }

    pub fn dist_to_parent(&self) -> f64 {
        // SAFETY: siehe `parent` — shared Reads OK, Write nur unter `&mut MTree`.
        unsafe { *self.parent_distance.get() }
    }

    pub fn key(&self) -> &K {
        // SAFETY: Write nur unter `&mut MTree`. Parallele Key-Reads unter shared `&MTree`
        // sind erlaubt und gewollt (Query-Hotpath); Borrow-Checker schließt Writer aus.
        unsafe { &*self.key.get() }
    }

    /// Ersetzt den Key. Nur unter exklusivem `MTree`-Borrow aufrufen (z. B. `update_key`).
    pub(crate) fn set_key(&self, new_key: K) {
        // SAFETY: Aufrufer hält `&mut MTree` (exklusiv); keine parallelen `key()`-Reader möglich.
        unsafe {
            *self.key.get() = new_key;
        }
    }

    pub fn value(&self) -> &V {
        // SAFETY: Write nur unter `&mut MTree`. Parallele Reads unter shared `&MTree`
        // sind erlaubt (Filter-/Payload-Hotpath); Borrow-Checker schließt Writer aus.
        unsafe { &*self.value.get() }
    }

    /// Ersetzt das Payload. Nur unter exklusivem `MTree`-Borrow aufrufen (`update_value`).
    pub(crate) fn set_value(&self, new_value: V) {
        // SAFETY: Aufrufer hält `&mut MTree` (exklusiv); keine parallelen `value()`-Reader möglich.
        unsafe {
            *self.value.get() = new_value;
        }
    }

    pub fn payload(&self) -> V
    where
        V: Clone,
    {
        self.value().clone()
    }

    /// Tiefer Klon (Key, Payload, Parent-Metadaten). Nur für `MTree::clone`.
    pub(crate) fn clone_node(&self) -> Self
    where
        K: Clone,
        V: Clone,
    {
        Self {
            id: self.id,
            key: UnsafeCell::new(self.key().clone()),
            value: UnsafeCell::new(self.value().clone()),
            identity_hash: self.identity_hash,
            epsilon: self.epsilon,
            parent: UnsafeCell::new(self.parent()),
            parent_distance: UnsafeCell::new(self.dist_to_parent()),
            _stats: std::marker::PhantomData,
        }
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
    pub children: Vec<NodePtr>,
    pub is_leaf: bool,
    pub key: K,
    pub epsilon: f64,
    pub identity_hash: u64,
    pub stats: S,
    pub covering_radius: f64,
    pub furthest_descendant: Option<EntryId>,
    pub parent: Option<NodeId>,
    pub parent_distance: f64,
    _value: std::marker::PhantomData<fn() -> V>,
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
            _value: std::marker::PhantomData,
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
            _value: std::marker::PhantomData,
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
            _value: std::marker::PhantomData,
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

/// Zeiger auf einen Knoten (Object-ID in `by_id`, Routing-ID in der Routing-Arena).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodePtr {
    Object(EntryId),
    Routing(NodeId),
}

impl NodePtr {
    #[inline]
    #[allow(dead_code)]
    pub fn as_object_id(&self) -> Option<EntryId> {
        match self {
            NodePtr::Object(id) => Some(*id),
            _ => None,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn as_routing_id(&self) -> Option<NodeId> {
        match self {
            NodePtr::Routing(id) => Some(*id),
            _ => None,
        }
    }
}
