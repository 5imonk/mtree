// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use crate::entry::EntryId;
use crate::query::{Query, RangeQuery};
use crate::stats::NodeStats;
use crate::tree::DistanceType;

/// Bereichssuche (Annulus), die nur aktive Einträge liefert.
pub struct FilteredQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    inner: Query<K, V, D, S>,
    is_active: F,
}

impl<K, V, D, S, F> FilteredQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    pub fn new(inner: Query<K, V, D, S>, is_active: F) -> Self {
        Self { inner, is_active }
    }

    pub fn at_end(&self) -> bool {
        self.inner.at_end()
    }
}

impl<K, V, D, S, F> Iterator for FilteredQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, dist)) = self.inner.next_object() {
            let active = {
                let guard = node.value.read().unwrap();
                (self.is_active)(node.id, &guard.1)
            };
            if active {
                return Some((node.id, dist));
            }
        }
        None
    }
}

/// Radius-Suche, die nur aktive Einträge liefert.
pub struct FilteredRangeQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    inner: RangeQuery<K, V, D, S>,
    is_active: F,
}

impl<K, V, D, S, F> FilteredRangeQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    pub fn new(inner: RangeQuery<K, V, D, S>, is_active: F) -> Self {
        Self { inner, is_active }
    }

    pub fn at_end(&self) -> bool {
        self.inner.at_end()
    }
}

impl<K, V, D, S, F> Iterator for FilteredRangeQuery<K, V, D, S, F>
where
    K: Clone + Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    F: Fn(EntryId, &V) -> bool,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, dist)) = self.inner.next_object() {
            let active = {
                let guard = node.value.read().unwrap();
                (self.is_active)(node.id, &guard.1)
            };
            if active {
                return Some((node.id, dist));
            }
        }
        None
    }
}
