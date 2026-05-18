// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::fmt;
use std::num::NonZeroU32;

/// Stabiler Handle auf einen Eintrag im M-Tree (Slot-Index; nach `erase_by_id` ungültig).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntryId(NonZeroU32);

impl EntryId {
    pub(crate) fn from_index(index: usize) -> Self {
        let raw = u32::try_from(index)
            .ok()
            .and_then(|i| i.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("entry index fits in EntryId");
        EntryId(raw)
    }

    pub(crate) fn index(self) -> usize {
        self.slot_index()
    }

    /// Dichter Slot-Index in `MTree::by_id` (0-basiert).
    ///
    /// Nach `erase_by_id` ist die `EntryId` ungültig; bei Slot-Recycle kann dieselbe
    /// `EntryId` einen anderen Eintrag bezeichnen — für parallele Side-Tables nur mit
    /// Vorsicht oder später mit Generation nutzen.
    pub fn slot_index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Ein Schlüssel ist bereits im Baum vorhanden.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DuplicateKey;

impl fmt::Display for DuplicateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("duplicate key")
    }
}

impl std::error::Error for DuplicateKey {}

/// Fehler bei `update_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateKeyError {
    NotFound,
    DuplicateKey,
}

impl fmt::Display for UpdateKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpdateKeyError::NotFound => f.write_str("entry not found"),
            UpdateKeyError::DuplicateKey => f.write_str("duplicate key"),
        }
    }
}

impl std::error::Error for UpdateKeyError {}
