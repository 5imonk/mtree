// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

/// Trait für Knotenstatistiken
pub trait NodeStats<K, V>: Send + Sync {
    /// Anzahl der Nachkommen
    fn get_descendant_count(&self) -> usize;
    
    /// Fügt einen Nachkommen hinzu
    fn add_descendant(&mut self, value: &(K, V));
    
    /// Fügt mehrere Nachkommen hinzu
    fn add_descendants(&mut self, from: &Self);
    
    /// Entfernt mehrere Nachkommen
    fn remove_descendants(&mut self, from: &Self);
    
    /// Entfernt einen Nachkommen
    fn remove_descendant(&mut self, value: &(K, V));
    
    /// Serialisierung (optional)
    fn serialize(&self, _os: &mut dyn std::io::Write) -> std::io::Result<()> {
        Ok(())
    }
    
    /// Deserialisierung (optional)
    fn unserialize(&mut self, _is: &mut dyn std::io::Read) -> std::io::Result<()> {
        Ok(())
    }
}

/// Einfacher Zähler für Nachkommen
#[derive(Clone, Default)]
pub struct DescendantCounter {
    descendants: usize,
}

impl DescendantCounter {
    pub fn new() -> Self {
        Self { descendants: 0 }
    }
}

impl<K, V> NodeStats<K, V> for DescendantCounter {
    fn get_descendant_count(&self) -> usize {
        self.descendants
    }
    
    fn add_descendant(&mut self, _value: &(K, V)) {
        self.descendants += 1;
    }
    
    fn add_descendants(&mut self, from: &Self) {
        self.descendants += from.descendants;
    }
    
    fn remove_descendants(&mut self, from: &Self) {
        self.descendants -= from.descendants;
    }
    
    fn remove_descendant(&mut self, _value: &(K, V)) {
        self.descendants -= 1;
    }
}
