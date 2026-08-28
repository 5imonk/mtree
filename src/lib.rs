// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

pub mod distance;
pub mod entry;
pub mod filtered;
pub mod node;
pub mod query;
pub mod placeholder_queue;
mod search;
pub mod search_config;
pub mod stats;
pub mod tree;

#[cfg(test)]
mod tests;

pub use entry::{DuplicateKey, EntryId, UpdateKeyError};
pub use filtered::{FilteredQuery, FilteredRangeQuery};
pub use search_config::{KnnConfig, RangeSearchConfig, SearchConfig};
pub use tree::MTree;
pub use query::{Query, RangeQuery};
pub use stats::NodeStats;
pub use placeholder_queue::PlaceholderQueue;
pub use distance::{AsCoordinates, Point, epsilon_from, epsilon_from_hash, identity_hash, EPSILON_MAX};
