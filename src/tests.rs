#[cfg(test)]
mod tests {
    use crate::MTree;
    use crate::distance::Distance;
    use crate::DuplicateKey;
    use crate::{KnnConfig, RangeSearchConfig, SearchConfig};
    use std::collections::BTreeSet;

    /// Euklidische Distanz für (i64, i64) – implementiert clone_box für search/range_search
    #[derive(Clone)]
    struct Euclid2d;
    impl Distance<(i64, i64)> for Euclid2d {
        type Output = f64;
        fn distance(&self, a: &(i64, i64), b: &(i64, i64)) -> f64 {
            let dx = (a.0 - b.0) as f64;
            let dy = (a.1 - b.1) as f64;
            (dx * dx + dy * dy).sqrt()
        }
        fn clone_box(&self) -> Box<dyn Distance<(i64, i64), Output = f64> + Send + Sync> {
            Box::new(Euclid2d)
        }
    }

    fn new_tree_i64() -> MTree<(i64, i64), String> {
        MTree::with_params(5, 100, 20, Euclid2d)
    }

    // ---- Naive Referenz-Implementierungen (gleiche Distanz wie MTree) ----

    fn naive_range_search(
        data: &[((i64, i64), String)],
        needle: &(i64, i64),
        radius: f64,
    ) -> Vec<((i64, i64), String)> {
        let dist = Euclid2d;
        data.iter()
            .filter(|(k, _)| dist.distance(k, needle) <= radius)
            .cloned()
            .collect()
    }

    fn naive_search_min_max(
        data: &[((i64, i64), String)],
        needle: &(i64, i64),
        min_radius: f64,
        max_radius: f64,
    ) -> Vec<((i64, i64), String)> {
        let dist = Euclid2d;
        data.iter()
            .filter(|(k, _)| {
                let d = dist.distance(k, needle);
                d >= min_radius && d < max_radius
            })
            .cloned()
            .collect()
    }

    fn naive_knn_search(
        data: &[((i64, i64), String)],
        needle: &(i64, i64),
        k: usize,
    ) -> Vec<((i64, i64), String, f64)> {
        let dist = Euclid2d;
        let mut with_dist: Vec<_> = data
            .iter()
            .map(|(k, v)| {
                let d = dist.distance(k, needle);
                (*k, v.clone(), d)
            })
            .collect();
        with_dist.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        with_dist.into_iter().take(k).collect()
    }

    #[test]
    fn test_basic_insert() {
        let mut tree = new_tree_i64();
        tree.insert((1, 2), "test1".to_string()).unwrap();
        tree.insert((3, 4), "test2".to_string()).unwrap();
        assert_eq!(tree.size(), 2);
    }

    #[test]
    fn test_empty_tree() {
        let tree: MTree<(i64, i64), String> = MTree::with_params(5, 100, 20, Euclid2d);
        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn test_range_search() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "origin".to_string()).unwrap();
        tree.insert((1, 0), "right".to_string()).unwrap();
        tree.insert((10, 10), "far".to_string()).unwrap();
        let results: Vec<_> = tree.range_search(&(0, 0), RangeSearchConfig::new(2.0)).collect();
        assert_eq!(results.len(), 2);
        let keys: Vec<_> = results.iter().map(|(n, _)| n.key()).collect();
        assert!(keys.contains(&(0, 0)));
        assert!(keys.contains(&(1, 0)));
        assert!(!keys.contains(&(10, 10)));
    }

    #[test]
    fn test_search_min_max_radius() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((1, 0), "b".to_string()).unwrap();
        tree.insert((2, 0), "c".to_string()).unwrap();
        let results: Vec<_> = tree.search(&(0, 0), SearchConfig::new(0.5, 1.5)).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.payload(), "b");
    }

    #[test]
    fn test_erase() {
        let mut tree = new_tree_i64();
        let k1: (i64, i64) = (1, 2);
        let k2: (i64, i64) = (3, 4);
        tree.insert(k1, "x".to_string()).unwrap();
        tree.insert(k2, "y".to_string()).unwrap();
        assert_eq!(tree.size(), 2);
        assert!(tree.erase_by_key(&k1));
        assert_eq!(tree.size(), 1);
        let results: Vec<_> = tree.range_search(&(0, 0), RangeSearchConfig::new(100.0)).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.payload(), "y");
    }

    #[test]
    fn test_knn_search() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "origin".to_string()).unwrap();
        tree.insert((1, 0), "one".to_string()).unwrap();
        tree.insert((2, 0), "two".to_string()).unwrap();
        tree.insert((3, 0), "three".to_string()).unwrap();
        let knn = tree.knn_search(&(0, 0), 2, ());
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0.payload(), "origin");
        assert_eq!(knn[1].0.payload(), "one");
    }

    #[test]
    fn test_knn_from_entry_include_self() {
        let mut tree = new_tree_i64();
        let origin = tree.insert((0, 0), "origin".to_string()).unwrap();
        tree.insert((1, 0), "one".to_string()).unwrap();
        tree.insert((2, 0), "two".to_string()).unwrap();
        tree.insert((3, 0), "three".to_string()).unwrap();

        let with_self = tree.knn_from_entry(origin, 2, true, ()).unwrap();
        let by_key = tree.knn_search(&(0, 0), 2, ());
        assert_eq!(with_self.len(), 2);
        assert_eq!(with_self[0].0.id, origin);
        assert_eq!(with_self[0].1, 0.0);
        assert_eq!(with_self[0].0.payload(), by_key[0].0.payload());
        assert_eq!(with_self[1].0.payload(), by_key[1].0.payload());
        assert!((with_self[1].1 - by_key[1].1).abs() < 1e-9);

        let without = tree.knn_from_entry(origin, 2, false, ()).unwrap();
        assert_eq!(without.len(), 2);
        assert!(without.iter().all(|(n, _)| n.id != origin));
        assert_eq!(without[0].0.payload(), "one");
        assert_eq!(without[1].0.payload(), "two");
    }

    #[test]
    fn test_knn_from_entry_unknown_and_k_zero() {
        let mut tree = new_tree_i64();
        let id = tree.insert((0, 0), "a".to_string()).unwrap();
        assert!(tree.knn_from_entry(id, 0, true, ()).unwrap().is_empty());
        assert!(tree.erase_by_id(id));
        assert!(tree.knn_from_entry(id, 3, false, ()).is_none());
    }

    #[test]
    fn test_knn_from_entry_vs_knn_search() {
        let data = test_data();
        let mut small = MTree::with_params(2, 4, 3, Euclid2d);
        let mut wide = new_tree_i64();
        let mut ids = Vec::new();
        for (k, v) in &data {
            ids.push(small.insert(*k, v.clone()).unwrap());
            wide.insert(*k, v.clone()).unwrap();
        }

        for id in &ids {
            let key = small.get(*id).unwrap().key();
            for k in [1usize, 3, 5, 10] {
                let with_self = small.knn_from_entry(*id, k, true, ()).unwrap();
                assert!(!with_self.is_empty());
                assert_eq!(with_self[0].0.id, *id);
                assert_eq!(with_self[0].1, 0.0);
                let without = small.knn_from_entry(*id, k, false, ()).unwrap();
                assert!(without.iter().all(|(n, _)| n.id != *id));
                assert!(without.windows(2).all(|w| w[0].1 <= w[1].1));

                let from_wide = wide.knn_from_entry(*id, k, true, ()).unwrap();
                let by_key = wide.knn_search(&key, k, ());
                assert_eq!(from_wide.len(), by_key.len());
                for (a, b) in from_wide.iter().zip(by_key.iter()) {
                    assert_eq!(a.0.key(), b.0.key());
                    assert!((a.1 - b.1).abs() < 1e-9);
                }

                let without_wide = wide.knn_from_entry(*id, k, false, ()).unwrap();
                let naive = naive_knn_search(&data, &key, data.len());
                let expected: Vec<_> = naive
                    .into_iter()
                    .filter(|(nk, _, _)| *nk != key)
                    .take(k)
                    .collect();
                assert_eq!(without_wide.len(), expected.len());
                for (a, b) in without_wide.iter().zip(expected.iter()) {
                    assert_eq!(a.0.key(), b.0);
                    assert!((a.1 - b.2).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_knn_from_entry_filter_and_annulus() {
        let mut tree = new_tree_i64();
        let origin = tree.insert((0, 0), "active:origin".to_string()).unwrap();
        tree.insert((1, 0), "inactive:one".to_string()).unwrap();
        tree.insert((2, 0), "active:two".to_string()).unwrap();
        tree.insert((3, 0), "active:three".to_string()).unwrap();

        let is_active = |_id, v: &String| v.starts_with("active:");
        let hits = tree
            .knn_from_entry(
                origin,
                2,
                false,
                KnnConfig::new()
                    .with_active(is_active)
                    .include_inactive(false),
            )
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.payload(), "active:two");
        assert_eq!(hits[1].0.payload(), "active:three");

        let band = tree
            .knn_from_entry(origin, 10, true, KnnConfig::new().min_radius(1.0).max_radius(3.0))
            .unwrap();
        let payloads: Vec<_> = band.iter().map(|(n, _)| n.payload()).collect();
        assert!(!payloads.iter().any(|p| p.contains("origin")));
        assert!(payloads.contains(&"inactive:one".to_string()) || payloads.contains(&"active:two".to_string()));
        let keys: Vec<_> = band.iter().map(|(n, _)| n.key()).collect();
        assert!(keys.contains(&(1, 0)));
        assert!(keys.contains(&(2, 0)));
        assert!(!keys.contains(&(0, 0)));
        assert!(!keys.contains(&(3, 0)));
    }

    #[test]
    fn test_clear() {
        let mut tree = new_tree_i64();
        tree.insert((1, 2), "a".to_string()).unwrap();
        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn test_empty_range_search() {
        let tree = new_tree_i64();
        let results: Vec<_> = tree.range_search(&(0, 0), RangeSearchConfig::new(1.0)).collect();
        assert!(results.is_empty());
    }

    // ---- Korrektheitstests: MTree vs. naiv ----

    /// Gemeinsame Testdaten (20–50 Punkte)
    fn test_data() -> Vec<((i64, i64), String)> {
        vec![
            ((0, 0), "origin".to_string()),
            ((1, 0), "r1".to_string()),
            ((2, 0), "r2".to_string()),
            ((3, 0), "r3".to_string()),
            ((0, 1), "u1".to_string()),
            ((1, 1), "d1".to_string()),
            ((2, 1), "d2".to_string()),
            ((0, 2), "u2".to_string()),
            ((1, 2), "d3".to_string()),
            ((10, 10), "far".to_string()),
            ((-1, 0), "l1".to_string()),
            ((-2, 0), "l2".to_string()),
            ((0, -1), "d4".to_string()),
            ((5, 5), "mid".to_string()),
            ((4, 0), "r4".to_string()),
            ((0, 4), "u4".to_string()),
        ]
    }

    fn build_tree_and_vec(
        data: &[((i64, i64), String)],
    ) -> (MTree<(i64, i64), String>, Vec<((i64, i64), String)>) {
        let mut tree = new_tree_i64();
        let vec_data: Vec<_> = data.to_vec();
        for (k, v) in data {
            tree.insert(*k, v.clone()).unwrap();
        }
        (tree, vec_data)
    }

    #[test]
    fn test_range_search_vs_naive() {
        let data = test_data();
        let (tree, vec_data) = build_tree_and_vec(&data);
        let needles = [(0, 0), (1, 1), (10, 10), (5, 5)];
        let radii = [0.5, 1.5, 3.0, 5.0, 100.0];
        for needle in needles {
            for radius in radii {
                let mtree_results: Vec<_> = tree.range_search(&needle, RangeSearchConfig::new(radius)).collect();
                let naive_results = naive_range_search(&vec_data, &needle, radius);
                let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(n, _)| n.key()).collect();
                let naive_keys: BTreeSet<_> = naive_results.iter().map(|(k, _)| *k).collect();
                assert_eq!(
                    mtree_keys.len(),
                    naive_keys.len(),
                    "needle={:?} radius={}",
                    needle,
                    radius
                );
                assert_eq!(mtree_keys, naive_keys, "needle={:?} radius={}", needle, radius);
            }
        }
    }

    #[test]
    fn test_search_min_max_vs_naive() {
        let data = test_data();
        let (tree, vec_data) = build_tree_and_vec(&data);
        let needles = [(0, 0), (1, 1), (5, 5)];
        let ranges = [(0.0, 1.0), (0.5, 2.0), (1.0, 4.0), (2.0, 10.0)];
        for needle in needles {
            for (min_r, max_r) in ranges {
                let mtree_results: Vec<_> = tree.search(&needle, SearchConfig::new(min_r, max_r)).collect();
                let naive_results = naive_search_min_max(&vec_data, &needle, min_r, max_r);
                let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(n, _)| n.key()).collect();
                let naive_keys: BTreeSet<_> = naive_results.iter().map(|(k, _)| *k).collect();
                assert_eq!(mtree_keys.len(), naive_keys.len());
                assert_eq!(mtree_keys, naive_keys);
            }
        }
    }

    #[test]
    fn test_knn_search_vs_naive() {
        let data = test_data();
        let (tree, vec_data) = build_tree_and_vec(&data);
        let needles = [(0, 0), (1, 1), (10, 10)];
        let k_values = [1, 3, 5, 10, 100];
        for needle in needles {
            for k in k_values {
                let mtree_results = tree.knn_search(&needle, k, ());
                let naive_results = naive_knn_search(&vec_data, &needle, k);
                assert_eq!(
                    mtree_results.len(),
                    naive_results.len(),
                    "needle={:?} k={}",
                    needle,
                    k
                );
                for (i, ((node, dist), (key, _, naive_dist))) in
                    mtree_results.iter().zip(naive_results.iter()).enumerate()
                {
                    assert_eq!(node.key(), *key, "needle={:?} k={} i={}", needle, k, i);
                    assert!(
                        (dist - naive_dist).abs() < 1e-9,
                        "needle={:?} k={} i={} dists {:?} vs {:?}",
                        needle,
                        k,
                        i,
                        dist,
                        naive_dist
                    );
                }
            }
        }
    }

    #[test]
    fn test_erase_vs_naive() {
        // Small tree (2 entries): erase one, compare MTree range result vs naive.
        let data = [((1, 2), "x".to_string()), ((3, 4), "y".to_string())];
        let mut tree = new_tree_i64();
        for (k, v) in &data {
            tree.insert(*k, v.clone()).unwrap();
        }
        let key = (1, 2);
        tree.erase_by_key(&key);
        let vec_data: Vec<_> = data.iter().filter(|(k, _)| *k != key).cloned().collect();
        assert_eq!(tree.size(), vec_data.len());
        let needle = (0, 0);
        let radius = 100.0;
        let mtree_results: Vec<_> = tree.range_search(&needle, RangeSearchConfig::new(radius)).collect();
        let naive_results = naive_range_search(&vec_data, &needle, radius);
        let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(n, _)| n.key()).collect();
        let naive_keys: BTreeSet<_> = naive_results.iter().map(|(k, _)| *k).collect();
        assert_eq!(mtree_keys, naive_keys);
    }

    #[test]
    fn test_insert_many_then_range() {
        let data = test_data();
        let mut tree = new_tree_i64();
        let mut vec_data = Vec::new();
        for (i, (k, v)) in data.iter().enumerate() {
            tree.insert(*k, v.clone()).unwrap();
            vec_data.push((*k, v.clone()));
            assert_eq!(tree.size(), i + 1);
            let needle = (0, 0);
            let mtree_results: Vec<_> = tree.range_search(&needle, RangeSearchConfig::new(20.0)).collect();
            let naive_results = naive_range_search(&vec_data, &needle, 20.0);
            let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(n, _)| n.key()).collect();
            let naive_keys: BTreeSet<_> = naive_results.iter().map(|(k, _)| *k).collect();
            assert_eq!(mtree_keys, naive_keys, "after insert {:?}", k);
        }
    }

    #[test]
    fn test_clear_empties_everything() {
        let mut tree = new_tree_i64();
        for (k, v) in &test_data() {
            tree.insert(*k, v.clone()).unwrap();
        }
        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
        let results: Vec<_> = tree.range_search(&(0, 0), RangeSearchConfig::new(1000.0)).collect();
        assert!(results.is_empty());
    }

    // ---- Tests für Point mit SIMD-Optimierung und erase() ----

    use crate::distance::{EuclideanDistance, Point};

    fn new_tree_point() -> MTree<Point, String, f64> {
        MTree::with_distance(EuclideanDistance)
    }

    #[test]
    fn test_point_simd_and_erase() {
        let mut tree = new_tree_point();
        
        // Füge einige Punkte hinzu
        let p1 = Point::new(vec![1.0, 2.0]);
        let p2 = Point::new(vec![3.0, 4.0]);
        let p3 = Point::new(vec![5.0, 6.0]);
        
        tree.insert(p1.clone(), "point1".to_string()).unwrap();
        tree.insert(p2.clone(), "point2".to_string()).unwrap();
        tree.insert(p3.clone(), "point3".to_string()).unwrap();

        assert_eq!(tree.size(), 3);

        let results: Vec<_> = tree.range_search(&Point::new(vec![1.5, 2.5]), RangeSearchConfig::new(1.0)).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.payload(), "point1");

        assert!(tree.erase_by_key(&p2));
        assert_eq!(tree.size(), 2);

        let results: Vec<_> = tree.range_search(&Point::new(vec![1.5, 2.5]), RangeSearchConfig::new(1.0)).collect();
        assert_eq!(results.len(), 1);

        let results: Vec<_> = tree.range_search(&Point::new(vec![5.5, 6.5]), RangeSearchConfig::new(1.0)).collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_point_knn_search() {
        let mut tree = new_tree_point();
        
        // Füge Punkte in einem 2D-Grid hinzu
        for x in 0..5 {
            for y in 0..5 {
                let p = Point::new(vec![x as f64, y as f64]);
                tree.insert(p, format!("point_{}_{}", x, y)).unwrap();
            }
        }
        
        assert_eq!(tree.size(), 25);
        
        // Test k-NN Suche mit SIMD-Optimierung
        let query = Point::new(vec![2.1, 2.1]);
        let results = tree.knn_search(&query, 3, ());
        
        assert_eq!(results.len(), 3);
        
        // Die nächsten 3 Punkte sollten (2,2), (2,3), (3,2) oder ähnlich sein
        // (je nach exakter Distanzberechnung)
        let mut distances: Vec<f64> = results.iter().map(|(_, d)| *d).collect();
        distances.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
        
        // Erste Distanz sollte sehr klein sein (nahe bei (2,2))
        assert!(distances[0] < 0.2);
        // Alle Distanzen sollten endlich sein
        assert!(distances.iter().all(|d: &f64| d.is_finite()));
    }

    #[test]
    fn test_entry_id_get() {
        let mut tree = new_tree_i64();
        let id = tree.insert((1, 2), "a".to_string()).unwrap();
        let node = tree.get(id).unwrap();
        assert_eq!(node.id, id);
        assert_eq!(node.key(), (1, 2));
        assert_eq!(node.payload(), "a");
    }

    #[test]
    fn test_iter_all_entries() {
        let mut tree = new_tree_i64();
        let id_a = tree.insert((1, 2), "a".to_string()).unwrap();
        let id_b = tree.insert((3, 4), "b".to_string()).unwrap();
        let payloads: Vec<_> = tree.iter().map(|n| n.payload()).collect();
        assert_eq!(payloads.len(), 2);
        assert!(payloads.contains(&"a".to_string()));
        assert!(payloads.contains(&"b".to_string()));
        assert!(tree.erase_by_id(id_a));
        assert_eq!(tree.iter().count(), 1);
        assert_eq!(tree.iter().next().unwrap().id, id_b);
    }

    #[test]
    fn test_entry_id_slot_index() {
        let mut tree = new_tree_i64();
        let id0 = tree.insert((0, 0), "a".to_string()).unwrap();
        let id1 = tree.insert((1, 0), "b".to_string()).unwrap();
        assert_eq!(id0.slot_index(), 0);
        assert_eq!(id1.slot_index(), 1);
        assert!(tree.erase_by_id(id0));
        let id2 = tree.insert((2, 0), "c".to_string()).unwrap();
        assert_eq!(id2.slot_index(), 0); // slot recycled
    }

    #[test]
    fn test_duplicate_insert() {
        let mut tree = new_tree_i64();
        tree.insert((1, 2), "a".to_string()).unwrap();
        assert!(matches!(
            tree.insert((1, 2), "b".to_string()),
            Err(DuplicateKey)
        ));
    }

    #[test]
    fn test_erase_by_id() {
        let mut tree = new_tree_i64();
        let id1 = tree.insert((1, 2), "x".to_string()).unwrap();
        tree.insert((3, 4), "y".to_string()).unwrap();
        assert!(tree.erase_by_id(id1));
        assert!(tree.get(id1).is_none());
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn test_slot_reuse() {
        let mut tree = new_tree_i64();
        let old_id = tree.insert((1, 2), "a".to_string()).unwrap();
        assert!(tree.erase_by_id(old_id));
        assert!(tree.get(old_id).is_none());
        let new_id = tree.insert((9, 9), "b".to_string()).unwrap();
        // Slot may be recycled — same EntryId, new payload
        assert_eq!(new_id, old_id);
        assert_eq!(tree.get(old_id).unwrap().payload(), "b");
    }

    #[test]
    fn test_update_value() {
        let mut tree = new_tree_i64();
        let id = tree.insert((0, 0), "old".to_string()).unwrap();
        assert!(tree.update_value(id, "new".to_string()));
        assert_eq!(tree.get(id).unwrap().payload(), "new");
        let knn = tree.knn_search(&(0, 0), 1, ());
        assert_eq!(knn[0].0.payload(), "new");
    }

    #[test]
    fn test_update_key_vs_naive() {
        let mut tree = new_tree_i64();
        let id = tree.insert((0, 0), "origin".to_string()).unwrap();
        tree.update_key(id, (10, 10)).unwrap();
        let results: Vec<_> = tree.range_search(&(10, 10), RangeSearchConfig::new(0.5)).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.key(), (10, 10));
        let far: Vec<_> = tree.range_search(&(0, 0), RangeSearchConfig::new(0.5)).collect();
        assert!(far.is_empty());
    }

    #[test]
    fn test_knn_search_config_filtered_by_value() {
        let mut tree = new_tree_i64();
        // Distances from (0,0): 0, 1, 2, 3 — mark "one" and "three" inactive via payload prefix
        tree.insert((0, 0), "active:origin".to_string()).unwrap();
        tree.insert((1, 0), "inactive:one".to_string()).unwrap();
        tree.insert((2, 0), "active:two".to_string()).unwrap();
        tree.insert((3, 0), "inactive:three".to_string()).unwrap();
        tree.insert((4, 0), "active:four".to_string()).unwrap();

        let is_active = |_id, v: &String| v.starts_with("active:");

        // k=2 Aktive, nur Aktive
        let only_active = tree.knn_search(&(0, 0), 2, KnnConfig::new().with_active(is_active).include_inactive(false));
        assert_eq!(only_active.len(), 2);
        assert_eq!(only_active[0].0.payload(), "active:origin");
        assert_eq!(only_active[1].0.payload(), "active:two");

        // k=2 Aktive, inkl. Inaktive innerhalb des Radius (Distanz des 2. Aktiven = 2)
        let with_inactive = tree.knn_search(&(0, 0), 2, KnnConfig::new().with_active(is_active).include_inactive(true));
        let payloads: Vec<_> = with_inactive.iter().map(|(n, _)| n.payload()).collect();
        assert!(payloads.contains(&"active:origin".to_string()));
        assert!(payloads.contains(&"inactive:one".to_string()));
        assert!(payloads.contains(&"active:two".to_string()));
        assert!(!payloads.contains(&"inactive:three".to_string()));
        assert!(!payloads.contains(&"active:four".to_string()));
    }

    #[test]
    fn test_knn_search_config_filtered_by_entry_id() {
        let mut tree = new_tree_i64();
        let id0 = tree.insert((0, 0), "origin".to_string()).unwrap();
        let _id1 = tree.insert((1, 0), "one".to_string()).unwrap();
        let id2 = tree.insert((2, 0), "two".to_string()).unwrap();
        let _id3 = tree.insert((3, 0), "three".to_string()).unwrap();
        let id4 = tree.insert((4, 0), "four".to_string()).unwrap();

        let active_ids = [id0, id2, id4];
        let is_active = |id, _v: &String| active_ids.contains(&id);

        let hits = tree.knn_search(&(0, 0), 2, KnnConfig::new().with_active(is_active).include_inactive(false));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.id, id0);
        assert_eq!(hits[1].0.id, id2);

        let with_inactive = tree.knn_search(&(0, 0), 2, KnnConfig::new().with_active(is_active).include_inactive(true));
        let ids: Vec<_> = with_inactive.iter().map(|(n, _)| n.id).collect();
        assert!(ids.contains(&id0));
        assert!(ids.contains(&_id1));
        assert!(ids.contains(&id2));
        assert!(!ids.contains(&_id3));
    }

    #[test]
    fn test_knn_search_config_filtered_skips_nearby_inactive() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "off".to_string()).unwrap();
        tree.insert((1, 0), "off".to_string()).unwrap();
        tree.insert((2, 0), "on".to_string()).unwrap();
        tree.insert((3, 0), "on".to_string()).unwrap();

        let hits = tree.knn_search(&(0, 0), 2, KnnConfig::new().with_active(|_id, v: &String| v == "on").include_inactive(false));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.payload(), "on");
        assert_eq!(hits[0].0.key(), (2, 0));
        assert_eq!(hits[1].0.payload(), "on");
        assert_eq!(hits[1].0.key(), (3, 0));
    }

    #[test]
    fn test_knn_search_config_filtered_edge_cases() {
        let mut tree = new_tree_i64();
        assert!(tree
            .knn_search(&(0, 0), 1, KnnConfig::new().with_active(|_, _: &String| true).include_inactive(false))
            .is_empty());

        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((1, 0), "b".to_string()).unwrap();

        assert!(tree
            .knn_search(&(0, 0), 0, KnnConfig::new().with_active(|_, _: &String| true).include_inactive(false))
            .is_empty());

        // Weniger Aktive als k
        let hits = tree.knn_search(&(0, 0), 5, KnnConfig::new().with_active(|_id, v: &String| v == "a").include_inactive(false));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0.payload(), "a");

        let with_inactive = tree.knn_search(&(0, 0), 5, KnnConfig::new().with_active(|_id, v: &String| v == "a").include_inactive(true));
        assert_eq!(with_inactive.len(), 2);
    }

    #[test]
    fn test_range_search_config_filtered_by_value() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "active:origin".to_string()).unwrap();
        tree.insert((1, 0), "inactive:one".to_string()).unwrap();
        tree.insert((2, 0), "active:two".to_string()).unwrap();
        tree.insert((10, 0), "active:far".to_string()).unwrap();

        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(3.0).with_active(|_id, v: &String| v.starts_with("active:")))
            .collect();
        let payloads: Vec<_> = results.iter().map(|(n, _)| n.payload()).collect();
        assert_eq!(results.len(), 2);
        assert!(payloads.contains(&"active:origin".to_string()));
        assert!(payloads.contains(&"active:two".to_string()));
        assert!(!payloads.contains(&"inactive:one".to_string()));
        assert!(!payloads.contains(&"active:far".to_string()));
    }

    #[test]
    fn test_range_search_config_filtered_by_entry_id() {
        let mut tree = new_tree_i64();
        let id0 = tree.insert((0, 0), "origin".to_string()).unwrap();
        let _id1 = tree.insert((1, 0), "one".to_string()).unwrap();
        let id2 = tree.insert((2, 0), "two".to_string()).unwrap();

        let active = [id0, id2];
        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(5.0).with_active(|id, _: &String| active.contains(&id)))
            .collect();
        let ids: Vec<_> = results.iter().map(|(n, _)| n.id).collect();
        assert_eq!(results.len(), 2);
        assert!(ids.contains(&id0));
        assert!(ids.contains(&id2));
        assert!(!ids.contains(&_id1));
    }

    #[test]
    fn test_search_config_annulus() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "active:a".to_string()).unwrap();
        tree.insert((1, 0), "inactive:b".to_string()).unwrap();
        tree.insert((2, 0), "active:c".to_string()).unwrap();
        tree.insert((3, 0), "active:d".to_string()).unwrap();

        let results: Vec<_> = tree
            .search(&(0, 0), SearchConfig::new(0.5, 2.5).with_active(|_id, v: &String| v.starts_with("active:")))
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.payload(), "active:c");
    }

    #[test]
    fn test_knn_search_config_range_annulus_bounds() {
        let mut tree = new_tree_i64();
        // Distances from (0,0): 0, 1, 2, 3, 4
        tree.insert((0, 0), "d0".to_string()).unwrap();
        tree.insert((1, 0), "d1".to_string()).unwrap();
        tree.insert((2, 0), "d2".to_string()).unwrap();
        tree.insert((3, 0), "d3".to_string()).unwrap();
        tree.insert((4, 0), "d4".to_string()).unwrap();

        // min ≤ dist < max: 1 ≤ d < 3 → d1, d2 (d3 bei max ausgeschlossen)
        let hits = tree.knn_search(&(0, 0), 10, KnnConfig::new().min_radius(1.0).max_radius(3.0));
        let payloads: Vec<_> = hits.iter().map(|(n, _)| n.payload()).collect();
        assert_eq!(hits.len(), 2);
        assert_eq!(payloads, vec!["d1".to_string(), "d2".to_string()]);

        // k begrenzt
        let top1 = tree.knn_search(&(0, 0), 1, KnnConfig::new().min_radius(1.0).max_radius(3.0));
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0.payload(), "d1");

        // dist == max_radius ausgeschlossen; dist == min_radius eingeschlossen
        let band = tree.knn_search(&(0, 0), 5, KnnConfig::new().min_radius(2.0).max_radius(4.0));
        let keys: Vec<_> = band.iter().map(|(n, _)| n.key()).collect();
        assert!(keys.contains(&(2, 0)));
        assert!(keys.contains(&(3, 0)));
        assert!(!keys.contains(&(4, 0)));
    }

    #[test]
    fn test_knn_search_config_range_fewer_than_k() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((5, 0), "b".to_string()).unwrap();
        // 0 ≤ d < 10 → a und b
        let hits = tree.knn_search(&(0, 0), 5, KnnConfig::new().min_radius(0.0).max_radius(10.0));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.payload(), "a");
        assert_eq!(hits[1].0.payload(), "b");
    }

    #[test]
    fn test_knn_search_config_range_filtered() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "active:a".to_string()).unwrap();
        tree.insert((1, 0), "inactive:b".to_string()).unwrap();
        tree.insert((2, 0), "active:c".to_string()).unwrap();
        tree.insert((3, 0), "inactive:d".to_string()).unwrap();
        tree.insert((4, 0), "active:e".to_string()).unwrap();

        let is_active = |_id, v: &String| v.starts_with("active:");

        // 0.5 ≤ d < 4: b(1), c(2), d(3); e(4) ausgeschlossen. Aktive: c
        let only = tree.knn_search(&(0, 0), 2, KnnConfig::new().min_radius(0.5).max_radius(4.0).with_active(is_active).include_inactive(false));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].0.payload(), "active:c");

        let with_inactive =
            tree.knn_search(&(0, 0), 2, KnnConfig::new().min_radius(0.5).max_radius(4.0).with_active(is_active).include_inactive(true));
        let payloads: Vec<_> = with_inactive.iter().map(|(n, _)| n.payload()).collect();
        assert!(payloads.contains(&"inactive:b".to_string()));
        assert!(payloads.contains(&"active:c".to_string()));
        assert!(payloads.contains(&"inactive:d".to_string()));
        assert!(!payloads.contains(&"active:e".to_string()));
        assert!(!payloads.contains(&"active:a".to_string()));
    }
}
