// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

//! Visit counters for plain k-NN (feature `knn_stats` only).
//!
//! Compiled out of default builds — no hot-path overhead without the feature.

use std::cell::Cell;

/// Snapshot of nodes visited during the last instrumented k-NN search(es).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KnnVisitStats {
    /// Routing nodes actually processed from the priority queue (seeded-leaf skips excluded).
    pub routing_nodes: u64,
    /// Leaf routing nodes whose objects were scanned.
    pub leaves: u64,
}

thread_local! {
    static ROUTING_NODES: Cell<u64> = const { Cell::new(0) };
    static LEAVES: Cell<u64> = const { Cell::new(0) };
}

/// Reset per-thread visit counters to zero.
pub fn reset_knn_visit_stats() {
    ROUTING_NODES.with(|c| c.set(0));
    LEAVES.with(|c| c.set(0));
}

/// Read current per-thread visit counters without clearing them.
pub fn knn_visit_stats() -> KnnVisitStats {
    KnnVisitStats {
        routing_nodes: ROUTING_NODES.with(|c| c.get()),
        leaves: LEAVES.with(|c| c.get()),
    }
}

/// Read and clear per-thread visit counters.
pub fn take_knn_visit_stats() -> KnnVisitStats {
    let stats = knn_visit_stats();
    reset_knn_visit_stats();
    stats
}

#[inline]
pub(crate) fn record_routing_visit() {
    ROUTING_NODES.with(|c| c.set(c.get().wrapping_add(1)));
}

#[inline]
pub(crate) fn record_leaf_visit() {
    LEAVES.with(|c| c.set(c.get().wrapping_add(1)));
}
