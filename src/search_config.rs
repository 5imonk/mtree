// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Konfiguration für die einheitlichen Such-APIs.
//!
//! # Übersicht
//!
//! | Typ | Verwendung |
//! |-----|------------|
//! | [`KnnConfig`] | Optionale Optionen für [`crate::tree::MTree::knn_search`] |
//! | [`SearchConfig`] | Pflicht-Config für Annulus-Suche [`crate::tree::MTree::search`] |
//! | [`RangeSearchConfig`] | Pflicht-Config für Kugel-Suche [`crate::tree::MTree::range_search`] |
//!
//! Ohne Filter reicht `KnnConfig::new()` bzw. `SearchConfig::new(min, max)` /
//! `RangeSearchConfig::new(radius)`. Filter setzen mit `.with_active(f)`.
//!
//! Annulus-Semantik bei k-NN und `search`: `min_radius < dist ≤ max_radius`
//! (`None` in `KnnConfig` → 0 bzw. ∞). `range_search` bleibt die Kugel `dist ≤ radius`.

use crate::distance::{epsilon_from_hash, identity_hash};
use crate::entry::EntryId;
use crate::tree::DistanceType;
use std::hash::Hash;

/// Marker: kein Filter gesetzt (Default für Config-Typen).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoFilter;

/// Optionen für [`crate::tree::MTree::knn_search`].
///
/// - `min_radius` / `max_radius`: Annulus `min < dist ≤ max` (`None` → 0 bzw. ∞)
/// - `is_active`: optionaler Filter; `include_inactive` nur wirksam wenn `is_active` gesetzt
pub struct KnnConfig<D, F = NoFilter> {
    pub is_active: Option<F>,
    pub include_inactive: bool,
    pub min_radius: Option<D>,
    pub max_radius: Option<D>,
    /// Needle-ε (Default 0). Setzen über [`KnnConfig::identity`] oder [`KnnConfig::epsilon`].
    pub epsilon: f64,
}

impl<D: DistanceType> KnnConfig<D, NoFilter> {
    pub fn new() -> Self {
        Self {
            is_active: None,
            include_inactive: false,
            min_radius: None,
            max_radius: None,
            epsilon: 0.0,
        }
    }
}

impl<D: DistanceType> Default for KnnConfig<D, NoFilter> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, F> KnnConfig<D, F> {
    pub fn with_active<F2>(self, is_active: F2) -> KnnConfig<D, F2> {
        KnnConfig {
            is_active: Some(is_active),
            include_inactive: self.include_inactive,
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            epsilon: self.epsilon,
        }
    }

    pub fn include_inactive(mut self, yes: bool) -> Self {
        self.include_inactive = yes;
        self
    }

    pub fn min_radius(mut self, r: D) -> Self {
        self.min_radius = Some(r);
        self
    }

    pub fn max_radius(mut self, r: D) -> Self {
        self.max_radius = Some(r);
        self
    }

    /// Needle-ε aus Index/Timestamp (gleiche Quelle wie beim Insert).
    pub fn identity<T: Hash>(mut self, identity: &T) -> Self {
        self.epsilon = epsilon_from_hash(identity_hash(identity));
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Aufgelöstes k-NN-Config (intern / Trait-Objekt für Filter).
pub struct KnnConfigResolved<'a, D, V> {
    pub is_active: Option<Box<dyn Fn(EntryId, &V) -> bool + 'a>>,
    pub include_inactive: bool,
    pub min_radius: Option<D>,
    pub max_radius: Option<D>,
    pub epsilon: f64,
}

/// Ermöglicht `knn_search(needle, k, ())` und `knn_search(needle, k, KnnConfig { ... })`.
pub trait IntoKnnConfig<'a, D, V> {
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V>;
}

impl<'a, D, V> IntoKnnConfig<'a, D, V> for () {
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V> {
        KnnConfigResolved {
            is_active: None,
            include_inactive: false,
            min_radius: None,
            max_radius: None,
            epsilon: 0.0,
        }
    }
}

impl<'a, D, V> IntoKnnConfig<'a, D, V> for KnnConfig<D, NoFilter>
where
    D: 'a,
{
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V> {
        KnnConfigResolved {
            is_active: None,
            include_inactive: self.include_inactive,
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            epsilon: self.epsilon,
        }
    }
}

impl<'a, D, V, F> IntoKnnConfig<'a, D, V> for KnnConfig<D, F>
where
    D: 'a,
    F: Fn(EntryId, &V) -> bool + 'a,
{
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V> {
        KnnConfigResolved {
            is_active: self
                .is_active
                .map(|f| Box::new(f) as Box<dyn Fn(EntryId, &V) -> bool + 'a>),
            include_inactive: self.include_inactive,
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            epsilon: self.epsilon,
        }
    }
}

impl<'a, D, V> IntoKnnConfig<'a, D, V> for Option<KnnConfig<D, NoFilter>>
where
    D: 'a,
{
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V> {
        match self {
            None => ().into_knn_config(),
            Some(cfg) => cfg.into_knn_config(),
        }
    }
}

impl<'a, D, V, F> IntoKnnConfig<'a, D, V> for Option<KnnConfig<D, F>>
where
    D: 'a,
    F: Fn(EntryId, &V) -> bool + 'a,
{
    fn into_knn_config(self) -> KnnConfigResolved<'a, D, V> {
        match self {
            None => ().into_knn_config(),
            Some(cfg) => cfg.into_knn_config(),
        }
    }
}

/// Config für Annulus-Bereichssuche (`min < dist ≤ max`).
pub struct SearchConfig<D, F = NoFilter> {
    pub min_radius: D,
    pub max_radius: D,
    pub is_active: Option<F>,
    pub epsilon: f64,
}

impl<D> SearchConfig<D, NoFilter> {
    pub fn new(min_radius: D, max_radius: D) -> Self {
        Self {
            min_radius,
            max_radius,
            is_active: None,
            epsilon: 0.0,
        }
    }
}

impl<D, F> SearchConfig<D, F> {
    pub fn with_active<F2>(self, is_active: F2) -> SearchConfig<D, F2> {
        SearchConfig {
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            is_active: Some(is_active),
            epsilon: self.epsilon,
        }
    }

    pub fn identity<T: Hash>(mut self, identity: &T) -> Self {
        self.epsilon = epsilon_from_hash(identity_hash(identity));
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Aufgelöste Annulus-Suche (intern).
pub struct SearchConfigResolved<'a, D, V> {
    pub min_radius: D,
    pub max_radius: D,
    pub is_active: Option<Box<dyn Fn(EntryId, &V) -> bool + 'a>>,
    pub epsilon: f64,
}

pub trait IntoSearchConfig<'a, D, V> {
    fn into_search_config(self) -> SearchConfigResolved<'a, D, V>;
}

impl<'a, D, V> IntoSearchConfig<'a, D, V> for SearchConfig<D, NoFilter>
where
    D: 'a,
{
    fn into_search_config(self) -> SearchConfigResolved<'a, D, V> {
        SearchConfigResolved {
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            is_active: None,
            epsilon: self.epsilon,
        }
    }
}

impl<'a, D, V, F> IntoSearchConfig<'a, D, V> for SearchConfig<D, F>
where
    D: 'a,
    F: Fn(EntryId, &V) -> bool + 'a,
{
    fn into_search_config(self) -> SearchConfigResolved<'a, D, V> {
        SearchConfigResolved {
            min_radius: self.min_radius,
            max_radius: self.max_radius,
            is_active: self
                .is_active
                .map(|f| Box::new(f) as Box<dyn Fn(EntryId, &V) -> bool + 'a>),
            epsilon: self.epsilon,
        }
    }
}

/// Config für Radius-Suche (Kugel); `min` ist implizit 0.
pub struct RangeSearchConfig<D, F = NoFilter> {
    pub radius: D,
    pub is_active: Option<F>,
    pub epsilon: f64,
}

impl<D> RangeSearchConfig<D, NoFilter> {
    pub fn new(radius: D) -> Self {
        Self {
            radius,
            is_active: None,
            epsilon: 0.0,
        }
    }
}

impl<D, F> RangeSearchConfig<D, F> {
    pub fn with_active<F2>(self, is_active: F2) -> RangeSearchConfig<D, F2> {
        RangeSearchConfig {
            radius: self.radius,
            is_active: Some(is_active),
            epsilon: self.epsilon,
        }
    }

    pub fn identity<T: Hash>(mut self, identity: &T) -> Self {
        self.epsilon = epsilon_from_hash(identity_hash(identity));
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Aufgelöste Radius-Suche (intern).
pub struct RangeSearchConfigResolved<'a, D, V> {
    pub radius: D,
    pub is_active: Option<Box<dyn Fn(EntryId, &V) -> bool + 'a>>,
    pub epsilon: f64,
}

pub trait IntoRangeSearchConfig<'a, D, V> {
    fn into_range_search_config(self) -> RangeSearchConfigResolved<'a, D, V>;
}

impl<'a, D, V> IntoRangeSearchConfig<'a, D, V> for RangeSearchConfig<D, NoFilter>
where
    D: 'a,
{
    fn into_range_search_config(self) -> RangeSearchConfigResolved<'a, D, V> {
        RangeSearchConfigResolved {
            radius: self.radius,
            is_active: None,
            epsilon: self.epsilon,
        }
    }
}

impl<'a, D, V, F> IntoRangeSearchConfig<'a, D, V> for RangeSearchConfig<D, F>
where
    D: 'a,
    F: Fn(EntryId, &V) -> bool + 'a,
{
    fn into_range_search_config(self) -> RangeSearchConfigResolved<'a, D, V> {
        RangeSearchConfigResolved {
            radius: self.radius,
            is_active: self
                .is_active
                .map(|f| Box::new(f) as Box<dyn Fn(EntryId, &V) -> bool + 'a>),
            epsilon: self.epsilon,
        }
    }
}
