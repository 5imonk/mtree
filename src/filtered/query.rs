// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use crate::distance::{Distance, DynDistance};
use crate::entry::EntryId;
use crate::query::{Query, RangeQuery};
use crate::stats::NodeStats;
use crate::tree::DistanceType;

/// Bereichssuche (Annulus), die nur aktive Einträge liefert.
pub struct FilteredQuery<'a, K, V, D = f64, S = crate::stats::DescendantCounter, Dist = DynDistance<K, D>, F = fn(EntryId, &V) -> bool>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    inner: Query<'a, K, V, D, S, Dist>,
    is_active: F,
}

impl<'a, K, V, D, S, Dist, F> FilteredQuery<'a, K, V, D, S, Dist, F>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    pub fn new(inner: Query<'a, K, V, D, S, Dist>, is_active: F) -> Self {
        Self { inner, is_active }
    }

    pub fn at_end(&self) -> bool {
        self.inner.at_end()
    }
}

impl<'a, K, V, D, S, Dist, F> Iterator for FilteredQuery<'a, K, V, D, S, Dist, F>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        let tree = self.inner.tree()?;
        while let Some((id, dist)) = self.inner.next_object() {
            let active = {
                let guard = tree.object(id).value();
                (self.is_active)(id, guard)
            };
            if active {
                return Some((id, dist));
            }
        }
        None
    }
}

/// Radius-Suche, die nur aktive Einträge liefert.
pub struct FilteredRangeQuery<'a, K, V, D = f64, S = crate::stats::DescendantCounter, Dist = DynDistance<K, D>, F = fn(EntryId, &V) -> bool>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    inner: RangeQuery<'a, K, V, D, S, Dist>,
    is_active: F,
}

impl<'a, K, V, D, S, Dist, F> FilteredRangeQuery<'a, K, V, D, S, Dist, F>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    pub fn new(inner: RangeQuery<'a, K, V, D, S, Dist>, is_active: F) -> Self {
        Self { inner, is_active }
    }

    pub fn at_end(&self) -> bool {
        self.inner.at_end()
    }
}

impl<'a, K, V, D, S, Dist, F> Iterator for FilteredRangeQuery<'a, K, V, D, S, Dist, F>
where
    K: Clone + Send + Sync,
    V: Send + Sync,
    D: DistanceType,
    S: NodeStats<K, V>,
    Dist: Distance<K, Output = D>,
    F: Fn(EntryId, &V) -> bool,
{
    type Item = (EntryId, D);

    fn next(&mut self) -> Option<Self::Item> {
        let tree = self.inner.tree()?;
        while let Some((id, dist)) = self.inner.next_object() {
            let active = {
                let guard = tree.object(id).value();
                (self.is_active)(id, guard)
            };
            if active {
                return Some((id, dist));
            }
        }
        None
    }
}
