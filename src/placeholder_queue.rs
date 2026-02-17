// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

/// Placeholder für k-NN-Optimierung
/// Verwaltet obere Schranken für Distanzen mit Tag-basierten Updates
pub struct PlaceholderQueue<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
    heap: BinaryHeap<Placeholder<Key, Tag>>,
    tag_table: HashMap<Tag, usize>, // Tag -> Index im Heap (vereinfacht: wir verwenden Handle)
    max_size: usize,
    compare: Box<dyn Fn(&Key, &Key) -> bool>,
    max_key: Key,
}

/// Placeholder-Eintrag im Heap
struct Placeholder<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
    key: Key,
    tag: Tag,
    #[allow(dead_code)]
    multiplicity: usize,
}

impl<Key, Tag> PartialEq for Placeholder<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<Key, Tag> Eq for Placeholder<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
}

impl<Key, Tag> PartialOrd for Placeholder<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Key, Tag> Ord for Placeholder<Key, Tag>
where
    Key: Ord + Clone,
    Tag: std::hash::Hash + Eq + Clone,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-Heap: größter Key zuerst
        self.key.cmp(&other.key)
    }
}

impl<Key, Tag> PlaceholderQueue<Key, Tag>
where
    Key: Ord + Clone + Default,
    Tag: std::hash::Hash + Eq + Clone,
{
    /// Erstellt eine neue PlaceholderQueue
    pub fn new(max_size: usize, compare: impl Fn(&Key, &Key) -> bool + 'static, max_key: Key) -> Self {
        Self {
            heap: BinaryHeap::new(),
            tag_table: HashMap::new(),
            max_size,
            compare: Box::new(compare),
            max_key,
        }
    }
    
    /// Fügt einen Placeholder hinzu oder aktualisiert einen bestehenden
    pub fn add_placeholder(&mut self, _from_placeholder: Tag, upper_bound: Key, multiplicity: usize, tag: Tag) {
        // Prüfe ob Tag bereits existiert
        if let Some(_existing_idx) = self.tag_table.get(&tag) {
            // Tag existiert bereits - aktualisiere wenn upper_bound kleiner ist
            // Da BinaryHeap keine direkte Update-Operation hat, müssen wir neu aufbauen
            // Vereinfacht: entferne alten Eintrag und füge neuen hinzu
            // Für echte Optimierung würde man einen IndexedHeap verwenden
            let mut new_heap = BinaryHeap::new();
            let mut found = false;
            let upper_bound_clone = upper_bound.clone();
            for placeholder in self.heap.drain() {
                if placeholder.tag == tag {
                    // Aktualisiere wenn neue upper_bound kleiner ist
                    if (self.compare)(&upper_bound_clone, &placeholder.key) {
                        new_heap.push(Placeholder {
                            key: upper_bound_clone.clone(),
                            tag: tag.clone(),
                            multiplicity,
                        });
                        found = true;
                    } else {
                        new_heap.push(placeholder);
                    }
                } else {
                    new_heap.push(placeholder);
                }
            }
            self.heap = new_heap;
            if !found {
                // Tag nicht gefunden, füge neuen hinzu
                self.heap.push(Placeholder {
                    key: upper_bound_clone.clone(),
                    tag: tag.clone(),
                    multiplicity,
                });
                self.tag_table.insert(tag, self.heap.len() - 1);
            }
        } else {
            // Neuer Tag
            if self.heap.len() < self.max_size {
                self.heap.push(Placeholder {
                    key: upper_bound.clone(),
                    tag: tag.clone(),
                    multiplicity,
                });
                self.tag_table.insert(tag, self.heap.len() - 1);
            } else {
                // Prüfe ob upper_bound kleiner als max_key ist
                if (self.compare)(&upper_bound, &self.max_key) {
                    // Entferne größten Eintrag
                    if let Some(old_max) = self.heap.pop() {
                        self.tag_table.remove(&old_max.tag);
                    }
                    self.heap.push(Placeholder {
                        key: upper_bound,
                        tag: tag.clone(),
                        multiplicity,
                    });
                    self.tag_table.insert(tag, self.heap.len() - 1);
                    // Aktualisiere max_key
                    if let Some(new_max) = self.heap.peek() {
                        self.max_key = new_max.key.clone();
                    }
                }
            }
        }
    }
    
    /// Gibt den größten Key zurück.
    /// Bei k-NN-Nutzung entspricht dies dem aktuellen Pruning-Radius (k‑te kleinste obere Schranke).
    pub fn get_max_key(&self) -> Key {
        if let Some(max_placeholder) = self.heap.peek() {
            max_placeholder.key.clone()
        } else {
            self.max_key.clone()
        }
    }
    
    /// Gibt die Anzahl der Placeholder zurück
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    
    /// Prüft ob die Queue leer ist
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}
