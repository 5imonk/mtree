#[cfg(test)]
mod tests {
    use crate::distance::Distance;
    use crate::node_params_for_capacity;
    use crate::DuplicateKey;
    use crate::MTree;
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
                d > min_radius && d <= max_radius
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
    fn test_node_params_for_capacity() {
        assert_eq!(node_params_for_capacity(0), node_params_for_capacity(1));
        let (_min, max_100, _split) = node_params_for_capacity(100);
        assert_eq!(max_100, 10);
        let (min, max, split) = node_params_for_capacity(100);
        assert_eq!(min, (max / 4).max(2));
        assert_eq!(split, (max / 4).clamp(5, 20));
        let (_, max_big, _) = node_params_for_capacity(1_000_000);
        assert_eq!(max_big, 100);
    }

    #[test]
    fn test_with_capacity_hint_inserts() {
        let mut tree = MTree::with_capacity_hint(100, Euclid2d);
        for i in 0..50 {
            tree.insert((i, 0), format!("v{i}")).unwrap();
        }
        assert_eq!(tree.size(), 50);
        let knn = tree.knn_search(&(0, 0), 3, ());
        assert_eq!(knn.len(), 3);
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
        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(2.0))
            .collect();
        assert_eq!(results.len(), 2);
        let keys: Vec<_> = results
            .iter()
            .map(|(id, _)| tree.key(*id).unwrap())
            .collect();
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
        assert_eq!(tree.payload(results[0].0).unwrap(), "b");
        // min exklusiv, max inklusiv: 1 < d <= 2 → nur c
        let closed_max: Vec<_> = tree.search(&(0, 0), SearchConfig::new(1.0, 2.0)).collect();
        assert_eq!(closed_max.len(), 1);
        assert_eq!(tree.payload(closed_max[0].0).unwrap(), "c");
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
        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(100.0))
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(tree.payload(results[0].0).unwrap(), "y");
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
        assert_eq!(tree.payload(knn[0].0).unwrap(), "origin");
        assert_eq!(tree.payload(knn[1].0).unwrap(), "one");
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
        assert_eq!(with_self[0].0, origin);
        assert_eq!(with_self[0].1, 0.0);
        assert_eq!(tree.payload(with_self[0].0), tree.payload(by_key[0].0));
        assert_eq!(tree.payload(with_self[1].0), tree.payload(by_key[1].0));
        assert!((with_self[1].1 - by_key[1].1).abs() < 1e-9);

        let without = tree.knn_from_entry(origin, 2, false, ()).unwrap();
        assert_eq!(without.len(), 2);
        assert!(without.iter().all(|(id, _)| *id != origin));
        assert_eq!(tree.payload(without[0].0).unwrap(), "one");
        assert_eq!(tree.payload(without[1].0).unwrap(), "two");
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
                assert_eq!(with_self[0].0, *id);
                assert_eq!(with_self[0].1, 0.0);
                let without = small.knn_from_entry(*id, k, false, ()).unwrap();
                assert!(without.iter().all(|(hit, _)| *hit != *id));
                assert!(without.windows(2).all(|w| w[0].1 <= w[1].1));

                let from_wide = wide.knn_from_entry(*id, k, true, ()).unwrap();
                let by_key = wide.knn_search(&key, k, ());
                assert_eq!(from_wide.len(), by_key.len());
                for (a, b) in from_wide.iter().zip(by_key.iter()) {
                    assert_eq!(wide.key(a.0), wide.key(b.0));
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
                    assert_eq!(wide.key(a.0).unwrap(), b.0);
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
        assert_eq!(tree.payload(hits[0].0).unwrap(), "active:two");
        assert_eq!(tree.payload(hits[1].0).unwrap(), "active:three");

        let band = tree
            .knn_from_entry(
                origin,
                10,
                true,
                KnnConfig::new().min_radius(1.0).max_radius(3.0),
            )
            .unwrap();
        let payloads: Vec<_> = band
            .iter()
            .map(|(id, _)| tree.payload(*id).unwrap())
            .collect();
        assert!(!payloads.iter().any(|p| p.contains("origin")));
        let keys: Vec<_> = band.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
        // min < dist <= max: 1 < d <= 3 → (2,0), (3,0); (1,0) auf min ausgeschlossen
        assert!(!keys.contains(&(1, 0)));
        assert!(keys.contains(&(2, 0)));
        assert!(keys.contains(&(3, 0)));
        assert!(!keys.contains(&(0, 0)));
    }

    #[test]
    fn test_knn_search_seed_matches_unseeded() {
        let data = test_data();
        let mut tree = new_tree_i64();
        let mut ids = Vec::new();
        for (k, v) in &data {
            ids.push(tree.insert(*k, v.clone()).unwrap());
        }

        let needle = (5, 5);
        for k in [1usize, 3, 5, 10] {
            let plain = tree.knn_search(&needle, k, ());
            for seed in &ids {
                let seeded = tree.knn_search(&needle, k, KnnConfig::new().seed(*seed));
                assert_eq!(plain.len(), seeded.len());
                for (a, b) in plain.iter().zip(seeded.iter()) {
                    assert_eq!(a.0, b.0);
                    assert!((a.1 - b.1).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_knn_search_seed_invalid_and_nearest() {
        let mut tree = new_tree_i64();
        let near = tree.insert((1, 0), "near".to_string()).unwrap();
        let far = tree.insert((100, 0), "far".to_string()).unwrap();
        tree.insert((2, 0), "mid".to_string()).unwrap();
        let gone = tree.insert((50, 0), "gone".to_string()).unwrap();
        assert!(tree.erase_by_id(gone));

        let needle = (0, 0);
        let plain = tree.knn_search(&needle, 2, ());
        let bad_seed = tree.knn_search(&needle, 2, KnnConfig::new().seed(gone));
        assert_eq!(plain.len(), bad_seed.len());
        for (a, b) in plain.iter().zip(bad_seed.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-9);
        }

        let top1 = tree.knn_search(&needle, 1, KnnConfig::new().seed(near));
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0, near);

        let top1_far_seed = tree.knn_search(&needle, 1, KnnConfig::new().seed(far));
        assert_eq!(top1_far_seed[0].0, near);
    }

    #[test]
    fn test_knn_search_seed_annulus_excludes_seed() {
        let mut tree = new_tree_i64();
        let seed = tree.insert((0, 0), "seed".to_string()).unwrap();
        tree.insert((2, 0), "in".to_string()).unwrap();
        tree.insert((5, 0), "out".to_string()).unwrap();

        let hits = tree.knn_search(
            &(0, 0),
            5,
            KnnConfig::new().seed(seed).min_radius(1.0).max_radius(3.0),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(tree.payload(hits[0].0).unwrap(), "in");
        assert!(hits.iter().all(|(id, _)| *id != seed));
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
        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(1.0))
            .collect();
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
                let mtree_results: Vec<_> = tree
                    .range_search(&needle, RangeSearchConfig::new(radius))
                    .collect();
                let naive_results = naive_range_search(&vec_data, &needle, radius);
                let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
                let naive_keys: BTreeSet<_> = naive_results.iter().map(|(k, _)| *k).collect();
                assert_eq!(
                    mtree_keys.len(),
                    naive_keys.len(),
                    "needle={:?} radius={}",
                    needle,
                    radius
                );
                assert_eq!(
                    mtree_keys, naive_keys,
                    "needle={:?} radius={}",
                    needle, radius
                );
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
                let mtree_results: Vec<_> = tree
                    .search(&needle, SearchConfig::new(min_r, max_r))
                    .collect();
                let naive_results = naive_search_min_max(&vec_data, &needle, min_r, max_r);
                let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
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
                    assert_eq!(
                        tree.key(*node).unwrap(),
                        *key,
                        "needle={:?} k={} i={}",
                        needle,
                        k,
                        i
                    );
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
        let mtree_results: Vec<_> = tree
            .range_search(&needle, RangeSearchConfig::new(radius))
            .collect();
        let naive_results = naive_range_search(&vec_data, &needle, radius);
        let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
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
            let mtree_results: Vec<_> = tree
                .range_search(&needle, RangeSearchConfig::new(20.0))
                .collect();
            let naive_results = naive_range_search(&vec_data, &needle, 20.0);
            let mtree_keys: BTreeSet<_> = mtree_results.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
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
        let results: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(1000.0))
            .collect();
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

        let results: Vec<_> = tree
            .range_search(&Point::new(vec![1.5, 2.5]), RangeSearchConfig::new(1.0))
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(tree.payload(results[0].0).unwrap(), "point1");

        assert!(tree.erase_by_key(&p2));
        assert_eq!(tree.size(), 2);

        let results: Vec<_> = tree
            .range_search(&Point::new(vec![1.5, 2.5]), RangeSearchConfig::new(1.0))
            .collect();
        assert_eq!(results.len(), 1);

        let results: Vec<_> = tree
            .range_search(&Point::new(vec![5.5, 6.5]), RangeSearchConfig::new(1.0))
            .collect();
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
        let ids: Vec<_> = tree.iter().collect();
        let payloads: Vec<_> = ids
            .iter()
            .map(|id| tree.payload(*id).unwrap())
            .collect();
        assert_eq!(payloads.len(), 2);
        assert!(payloads.contains(&"a".to_string()));
        assert!(payloads.contains(&"b".to_string()));
        assert!(tree.erase_by_id(id_a));
        assert_eq!(tree.iter().count(), 1);
        assert_eq!(tree.iter().next().unwrap(), id_b);
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
    fn test_same_coords_different_identity() {
        use crate::{epsilon_from, EPSILON_MAX};
        let mut tree = new_tree_i64();
        let t1 = 1_000u64;
        let t2 = 2_000u64;
        tree.insert_with_identity((5, 5), "first".to_string(), &t1)
            .unwrap();
        tree.insert_with_identity((5, 5), "second".to_string(), &t2)
            .unwrap();
        assert!(matches!(
            tree.insert_with_identity((5, 5), "again".to_string(), &t1),
            Err(DuplicateKey)
        ));
        assert_eq!(tree.size(), 2);

        let e1 = epsilon_from(&t1);
        let e2 = epsilon_from(&t2);
        assert!(e1 > 0.0 && e1 <= EPSILON_MAX);
        assert!(e2 > 0.0 && e2 <= EPSILON_MAX);
        assert_ne!(e1, e2);

        let hits: Vec<_> = tree
            .range_search(&(5, 5), RangeSearchConfig::new(0.0).identity(&t1))
            .collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(tree.payload(hits[0].0).unwrap(), "first");
        assert_eq!(hits[0].1, 0.0);

        let other: Vec<_> = tree
            .range_search(
                &(5, 5),
                RangeSearchConfig::new((e1 - e2).abs()).identity(&t1),
            )
            .collect();
        assert_eq!(other.len(), 2);

        let knn = tree.knn_search(&(5, 5), 1, KnnConfig::new().identity(&t2));
        assert_eq!(knn.len(), 1);
        assert_eq!(tree.payload(knn[0].0).unwrap(), "second");
        assert_eq!(knn[0].1, 0.0);

        assert!(!tree.erase_by_key(&(5, 5)));
        assert!(tree.erase_by_key_identity(&(5, 5), &t1));
        assert_eq!(tree.size(), 1);
        assert!(tree.erase_by_key(&(5, 5)));
        assert!(tree.is_empty());
    }

    #[test]
    fn test_identity_query_outside_tiny_ball() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "plain".to_string()).unwrap();
        tree.insert_with_identity((0, 0), "id".to_string(), &42u64)
            .unwrap();
        let far = tree.insert((10, 0), "far".to_string()).unwrap();
        let _ = far;

        let in_ball: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(0.5))
            .collect();
        assert_eq!(in_ball.len(), 2);

        let knn = tree.knn_search(&(0, 0), 2, ());
        assert_eq!(knn.len(), 2);
        assert!(knn.iter().all(|(id, _)| tree.key(*id).unwrap() == (0, 0)));

        let ids: Vec<_> = tree.iter().collect();
        let id_hit = ids
            .into_iter()
            .find(|id| tree.payload(*id).as_deref() == Some("id"))
            .unwrap();
        let from_entry = tree.knn_from_entry(id_hit, 1, true, ()).unwrap();
        assert_eq!(tree.payload(from_entry[0].0).unwrap(), "id");
        assert_eq!(from_entry[0].1, 0.0);
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
        assert_eq!(tree.payload(knn[0].0).unwrap(), "new");
    }

    #[test]
    fn test_update_key_vs_naive() {
        let mut tree = new_tree_i64();
        let id = tree.insert((0, 0), "origin".to_string()).unwrap();
        tree.update_key(id, (10, 10)).unwrap();
        let results: Vec<_> = tree
            .range_search(&(10, 10), RangeSearchConfig::new(0.5))
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(tree.key(results[0].0).unwrap(), (10, 10));
        let far: Vec<_> = tree
            .range_search(&(0, 0), RangeSearchConfig::new(0.5))
            .collect();
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
        let only_active = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .with_active(is_active)
                .include_inactive(false),
        );
        assert_eq!(only_active.len(), 2);
        assert_eq!(tree.payload(only_active[0].0).unwrap(), "active:origin");
        assert_eq!(tree.payload(only_active[1].0).unwrap(), "active:two");

        // k=2 Aktive, inkl. Inaktive innerhalb des Radius (Distanz des 2. Aktiven = 2)
        let with_inactive = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .with_active(is_active)
                .include_inactive(true),
        );
        let payloads: Vec<_> = with_inactive.iter().map(|(id, _)| tree.payload(*id).unwrap()).collect();
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

        let hits = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .with_active(is_active)
                .include_inactive(false),
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, id0);
        assert_eq!(hits[1].0, id2);

        let with_inactive = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .with_active(is_active)
                .include_inactive(true),
        );
        let ids: Vec<_> = with_inactive.iter().map(|(id, _)| *id).collect();
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

        let hits = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .with_active(|_id, v: &String| v == "on")
                .include_inactive(false),
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(tree.payload(hits[0].0).unwrap(), "on");
        assert_eq!(tree.key(hits[0].0).unwrap(), (2, 0));
        assert_eq!(tree.payload(hits[1].0).unwrap(), "on");
        assert_eq!(tree.key(hits[1].0).unwrap(), (3, 0));
    }

    #[test]
    fn test_knn_search_config_filtered_edge_cases() {
        let mut tree = new_tree_i64();
        assert!(tree
            .knn_search(
                &(0, 0),
                1,
                KnnConfig::new()
                    .with_active(|_, _: &String| true)
                    .include_inactive(false)
            )
            .is_empty());

        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((1, 0), "b".to_string()).unwrap();

        assert!(tree
            .knn_search(
                &(0, 0),
                0,
                KnnConfig::new()
                    .with_active(|_, _: &String| true)
                    .include_inactive(false)
            )
            .is_empty());

        // Weniger Aktive als k
        let hits = tree.knn_search(
            &(0, 0),
            5,
            KnnConfig::new()
                .with_active(|_id, v: &String| v == "a")
                .include_inactive(false),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(tree.payload(hits[0].0).unwrap(), "a");

        let with_inactive = tree.knn_search(
            &(0, 0),
            5,
            KnnConfig::new()
                .with_active(|_id, v: &String| v == "a")
                .include_inactive(true),
        );
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
            .range_search(
                &(0, 0),
                RangeSearchConfig::new(3.0).with_active(|_id, v: &String| v.starts_with("active:")),
            )
            .collect();
        let payloads: Vec<_> = results.iter().map(|(id, _)| tree.payload(*id).unwrap()).collect();
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
            .range_search(
                &(0, 0),
                RangeSearchConfig::new(5.0).with_active(|id, _: &String| active.contains(&id)),
            )
            .collect();
        let ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
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
            .search(
                &(0, 0),
                SearchConfig::new(0.5, 2.5).with_active(|_id, v: &String| v.starts_with("active:")),
            )
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(tree.payload(results[0].0).unwrap(), "active:c");
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

        // min < dist <= max: 1 < d <= 3 → d2, d3 (d1 bei min ausgeschlossen, d3 bei max dabei)
        let hits = tree.knn_search(
            &(0, 0),
            10,
            KnnConfig::new().min_radius(1.0).max_radius(3.0),
        );
        let payloads: Vec<_> = hits.iter().map(|(id, _)| tree.payload(*id).unwrap()).collect();
        assert_eq!(hits.len(), 2);
        assert_eq!(payloads, vec!["d2".to_string(), "d3".to_string()]);

        // k begrenzt
        let top1 = tree.knn_search(&(0, 0), 1, KnnConfig::new().min_radius(1.0).max_radius(3.0));
        assert_eq!(top1.len(), 1);
        assert_eq!(tree.payload(top1[0].0).unwrap(), "d2");

        // dist == min_radius ausgeschlossen; dist == max_radius eingeschlossen
        let band = tree.knn_search(&(0, 0), 5, KnnConfig::new().min_radius(2.0).max_radius(4.0));
        let keys: Vec<_> = band.iter().map(|(id, _)| tree.key(*id).unwrap()).collect();
        assert!(!keys.contains(&(2, 0)));
        assert!(keys.contains(&(3, 0)));
        assert!(keys.contains(&(4, 0)));
    }

    #[test]
    fn test_knn_search_config_range_fewer_than_k() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((5, 0), "b".to_string()).unwrap();
        // 0 < d <= 10 → nur b (Ursprung bei min=0 ausgeschlossen)
        let hits = tree.knn_search(
            &(0, 0),
            5,
            KnnConfig::new().min_radius(0.0).max_radius(10.0),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(tree.payload(hits[0].0).unwrap(), "b");
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

        // 0.5 < d <= 4: b(1), c(2), d(3), e(4). Aktive: c, e
        let only = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .min_radius(0.5)
                .max_radius(4.0)
                .with_active(is_active)
                .include_inactive(false),
        );
        assert_eq!(only.len(), 2);
        assert_eq!(tree.payload(only[0].0).unwrap(), "active:c");
        assert_eq!(tree.payload(only[1].0).unwrap(), "active:e");

        let with_inactive = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new()
                .min_radius(0.5)
                .max_radius(4.0)
                .with_active(is_active)
                .include_inactive(true),
        );
        let payloads: Vec<_> = with_inactive.iter().map(|(id, _)| tree.payload(*id).unwrap()).collect();
        assert!(payloads.contains(&"inactive:b".to_string()));
        assert!(payloads.contains(&"active:c".to_string()));
        assert!(payloads.contains(&"inactive:d".to_string()));
        assert!(payloads.contains(&"active:e".to_string()));
        assert!(!payloads.contains(&"active:a".to_string()));
    }

    /// Nach vielen erase/insert muss die k-te Distanz die sortierte Distanz der übrigen Keys sein.
    /// Siehe `docs/bug-knn-after-replace.md`.
    #[test]
    fn knn_from_entry_matches_naive_after_replaces() {
        let mut tree = MTree::with_capacity_hint(128, Euclid2d);
        let mut ids = Vec::new();
        for i in 0..128i64 {
            let id = tree
                .insert_with_identity((i, 0), format!("v{i}"), &i)
                .unwrap();
            ids.push(id);
        }
        for step in 0..400i64 {
            let victim = (step as usize) % ids.len();
            assert!(tree.erase_by_id(ids[victim]));
            ids[victim] = tree
                .insert_with_identity((step, step * 3), format!("n{step}"), &(1_000 + step))
                .unwrap();
        }

        let dist = Euclid2d;
        let ids: Vec<_> = tree.iter().collect();
        let entries: Vec<_> = ids
            .iter()
            .map(|id| (tree.key(*id).unwrap(), *id))
            .collect();
        for (key, id) in &entries {
            let hits = tree.knn_from_entry(*id, 6, false, ()).unwrap();
            assert_eq!(hits.len(), 6);
            let mut brute: Vec<f64> = entries
                .iter()
                .filter(|(_, other)| other != id)
                .map(|(other, _)| dist.distance(key, other))
                .collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (hit, expected) in hits.iter().zip(brute.iter()) {
                assert!(
                    (hit.1 - *expected).abs() < 1e-6,
                    "key={key:?} knn={} brute={expected}",
                    hit.1
                );
            }
        }
    }

    /// Dieselbe Punktmenge und dieselben 400 Ersetzungen, gefiltert.
    /// Immer-wahr bleibt auf der gefilterten Bahn und trifft die ungefilterte Suche.
    /// Je ohne Hint, mit Hint und über `knn_from_entry`.
    #[test]
    fn knn_from_entry_filtered_matches_naive_after_replaces() {
        use crate::EntryId;

        const K: usize = 13;
        const TOL: f64 = 1e-6;

        fn even_slot(id: EntryId, _: &String) -> bool {
            id.slot_index() % 2 == 0
        }

        fn assert_dists(got: &[f64], expected: &[f64], what: &str) {
            assert_eq!(
                got.len(),
                expected.len(),
                "{what} len {} vs {}",
                got.len(),
                expected.len()
            );
            for (hit, exp) in got.iter().zip(expected.iter()) {
                assert!((hit - exp).abs() < TOL, "{what} knn={hit} brute={exp}");
            }
        }

        let mut tree = MTree::with_capacity_hint(128, Euclid2d);
        let mut ids = Vec::new();
        for i in 0..128i64 {
            let id = tree
                .insert_with_identity((i, 0), format!("v{i}"), &i)
                .unwrap();
            ids.push(id);
        }
        for step in 0..400i64 {
            let victim = (step as usize) % ids.len();
            assert!(tree.erase_by_id(ids[victim]));
            ids[victim] = tree
                .insert_with_identity((step, step * 3), format!("n{step}"), &(1_000 + step))
                .unwrap();
        }

        let dist = Euclid2d;
        let ids: Vec<_> = tree.iter().collect();
        let entries: Vec<_> = ids
            .iter()
            .map(|id| (tree.key(*id).unwrap(), *id))
            .collect();

        let reference = |key: &(i64, i64), skip: Option<EntryId>, even_only: bool| {
            let mut brute: Vec<f64> = entries
                .iter()
                .filter(|(_, other)| skip != Some(*other))
                .filter(|(_, other)| !even_only || even_slot(*other, &String::new()))
                .map(|(other, _)| dist.distance(key, other))
                .collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            brute.truncate(K);
            brute
        };

        for (idx, (key, id)) in entries.iter().enumerate() {
            let hint = entries[(idx + 1) % entries.len()].1;

            let plain = tree.knn_search(key, K, ());
            let plain_d: Vec<f64> = plain.iter().map(|(_, d)| *d).collect();
            let all_on =
                tree.knn_search(key, K, KnnConfig::new().with_active(|_, _: &String| true));
            let all_d: Vec<f64> = all_on.iter().map(|(_, d)| *d).collect();
            assert_dists(&all_d, &plain_d, &format!("search key={key:?}"));

            let subset = tree.knn_search(key, K, KnnConfig::new().with_active(even_slot));
            assert!(subset.iter().all(|(id, _)| even_slot(*id, &String::new())));
            let subset_d: Vec<f64> = subset.iter().map(|(_, d)| *d).collect();
            assert_dists(
                &subset_d,
                &reference(key, None, true),
                &format!("search-half key={key:?}"),
            );

            let plain_h = tree.knn_search(key, K, KnnConfig::new().seed(hint));
            let plain_hd: Vec<f64> = plain_h.iter().map(|(_, d)| *d).collect();
            assert_dists(&plain_hd, &plain_d, &format!("hint-plain key={key:?}"));
            let all_h = tree.knn_search(
                key,
                K,
                KnnConfig::new()
                    .seed(hint)
                    .with_active(|_, _: &String| true),
            );
            let all_hd: Vec<f64> = all_h.iter().map(|(_, d)| *d).collect();
            assert_dists(&all_hd, &plain_hd, &format!("hint key={key:?}"));

            let subset_h =
                tree.knn_search(key, K, KnnConfig::new().seed(hint).with_active(even_slot));
            assert!(subset_h
                .iter()
                .all(|(id, _)| even_slot(*id, &String::new())));
            let subset_hd: Vec<f64> = subset_h.iter().map(|(_, d)| *d).collect();
            assert_dists(
                &subset_hd,
                &reference(key, None, true),
                &format!("hint-half key={key:?}"),
            );

            let plain_e = tree.knn_from_entry(*id, K, false, ()).unwrap();
            let plain_ed: Vec<f64> = plain_e.iter().map(|(_, d)| *d).collect();
            let all_e = tree
                .knn_from_entry(
                    *id,
                    K,
                    false,
                    KnnConfig::new().with_active(|_, _: &String| true),
                )
                .unwrap();
            let all_ed: Vec<f64> = all_e.iter().map(|(_, d)| *d).collect();
            assert_dists(&all_ed, &plain_ed, &format!("entry key={key:?}"));
            assert_dists(
                &plain_ed,
                &reference(key, Some(*id), false),
                &format!("entry-plain key={key:?}"),
            );

            let subset_e = tree
                .knn_from_entry(*id, K, false, KnnConfig::new().with_active(even_slot))
                .unwrap();
            assert!(subset_e
                .iter()
                .all(|(hit, _)| *hit != *id && even_slot(*hit, &String::new())));
            let subset_ed: Vec<f64> = subset_e.iter().map(|(_, d)| *d).collect();
            assert_dists(
                &subset_ed,
                &reference(key, Some(*id), true),
                &format!("entry-half key={key:?}"),
            );
        }
    }

    /// Zufällige insert/erase, ein Thread (`&mut self`): kNN und Radius bleiben bei der Referenz.
    /// Seed `0x5A17_5EED` (xorshift64). Knoten 2/4, damit Split, Donate und Merge oft laufen.
    #[test]
    fn knn_matches_naive_under_random_replaces() {
        use crate::EntryId;

        const SEED: u64 = 0x5A17_5EED;
        const K: usize = 5;
        const TOL: f64 = 1e-6;

        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        // *3-Gitter: Distanz exakt 25 ist unmöglich, Identity-ε verschiebt die Kugel nicht.
        fn draw_key(state: &mut u64, live: &[(EntryId, (i64, i64))]) -> (i64, i64) {
            loop {
                let x = ((xorshift64(state) % 81) as i64 - 40) * 3;
                let y = ((xorshift64(state) % 81) as i64 - 40) * 3;
                if live.iter().all(|(_, key)| *key != (x, y)) {
                    return (x, y);
                }
            }
        }

        fn assert_knn(
            tree: &MTree<(i64, i64), String>,
            live: &[(EntryId, (i64, i64))],
            id: EntryId,
            key: (i64, i64),
            op: u64,
        ) {
            let hits = tree.knn_from_entry(id, K, false, ()).unwrap_or_else(|| {
                panic!("seed={SEED:#x} op={op} key={key:?} knn=fehlt brute=fehlt");
            });
            let mut brute: Vec<f64> = live
                .iter()
                .filter(|(other, _)| *other != id)
                .map(|(_, other)| Euclid2d.distance(&key, other))
                .collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(
                hits.len() == K && brute.len() >= K,
                "seed={SEED:#x} op={op} key={key:?} knn={} brute={}",
                hits.first().map(|hit| hit.1).unwrap_or(f64::NAN),
                brute.first().copied().unwrap_or(f64::NAN)
            );
            for (hit, expected) in hits.iter().zip(brute.iter()) {
                assert!(
                    (hit.1 - *expected).abs() < TOL,
                    "seed={SEED:#x} op={op} key={key:?} knn={} brute={expected}",
                    hit.1
                );
            }
        }

        // Alle 100 Operationen nur fünf Ids, nicht der ganze Bestand.
        fn spot_check(
            tree: &MTree<(i64, i64), String>,
            live: &[(EntryId, (i64, i64))],
            rng: &mut u64,
            op: u64,
        ) {
            for _ in 0..5 {
                let idx = (xorshift64(rng) as usize) % live.len();
                let (id, key) = live[idx];
                assert_knn(tree, live, id, key, op);
            }
        }

        fn insert_one(
            tree: &mut MTree<(i64, i64), String>,
            live: &mut Vec<(EntryId, (i64, i64))>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let key = draw_key(rng, live);
            *op += 1;
            let id = tree
                .insert_with_identity(key, format!("v{}", *op), &*op)
                .unwrap();
            live.push((id, key));
        }

        fn erase_one(
            tree: &mut MTree<(i64, i64), String>,
            live: &mut Vec<(EntryId, (i64, i64))>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, key) = live.swap_remove(idx);
            *op += 1;
            assert!(
                tree.erase_by_id(id),
                "seed={SEED:#x} op={op} key={key:?} knn=erase brute=fehlt"
            );
        }

        let mut rng = SEED;
        let mut op = 0u64;
        let mut tree = MTree::with_params(2, 4, 3, Euclid2d);
        let mut live: Vec<(EntryId, (i64, i64))> = Vec::new();

        // Zuerst 200 eindeutige Keys, danach 2000 gemischte Operationen (Bestand etwa 80 bis 200).
        while live.len() < 200 {
            insert_one(&mut tree, &mut live, &mut rng, &mut op);
            if op % 100 == 0 {
                spot_check(&tree, &live, &mut rng, op);
            }
        }
        for _ in 0..2_000 {
            if live.len() > 200 {
                erase_one(&mut tree, &mut live, &mut rng, &mut op);
            } else if live.len() < 80 {
                insert_one(&mut tree, &mut live, &mut rng, &mut op);
            } else if xorshift64(&mut rng) & 1 == 0 {
                insert_one(&mut tree, &mut live, &mut rng, &mut op);
            } else {
                erase_one(&mut tree, &mut live, &mut rng, &mut op);
            }
            if op % 100 == 0 {
                spot_check(&tree, &live, &mut rng, op);
            }
        }

        let q_idx = (xorshift64(&mut rng) as usize) % live.len();
        let qkey = live[q_idx].1;
        assert_eq!(
            tree.size(),
            live.len(),
            "seed={SEED:#x} op={op} key={qkey:?} size {} != {}",
            tree.size(),
            live.len()
        );
        for &(id, key) in &live {
            assert_knn(&tree, &live, id, key, op);
        }

        let got_ids: Vec<_> = tree
            .range_search(&qkey, RangeSearchConfig::new(25.0))
            .map(|(id, _)| id)
            .collect();
        let got: BTreeSet<_> = got_ids
            .iter()
            .map(|id| tree.key(*id).unwrap())
            .collect();
        let naive_rows: Vec<((i64, i64), String)> =
            live.iter().map(|(_, key)| (*key, String::new())).collect();
        let expect: BTreeSet<_> = naive_range_search(&naive_rows, &qkey, 25.0)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(
            got,
            expect,
            "seed={SEED:#x} op={op} key={qkey:?} radius=25.0 treffer {} vs {}",
            got.len(),
            expect.len()
        );
    }

    /// Wie `knn_matches_naive_under_random_replaces`, plus umschaltbare Aktiv-Menge.
    /// Jede Stichprobe: ohne Hint, mit Hint, und `knn_from_entry`.
    #[test]
    fn knn_filtered_matches_naive_under_random_replaces() {
        use crate::EntryId;
        use std::collections::HashSet;

        const SEED: u64 = 0x5A17_5EED;
        const K: usize = 5;
        const TOL: f64 = 1e-6;

        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        fn draw_key(state: &mut u64, live: &[(EntryId, (i64, i64))]) -> (i64, i64) {
            loop {
                let x = ((xorshift64(state) % 81) as i64 - 40) * 3;
                let y = ((xorshift64(state) % 81) as i64 - 40) * 3;
                if live.iter().all(|(_, key)| *key != (x, y)) {
                    return (x, y);
                }
            }
        }

        fn reference(
            live: &[(EntryId, (i64, i64))],
            key: (i64, i64),
            self_id: EntryId,
            exclude_self: bool,
            keep: impl Fn(EntryId) -> bool,
        ) -> Vec<f64> {
            let mut brute: Vec<f64> = live
                .iter()
                .filter(|(other, _)| !exclude_self || *other != self_id)
                .filter(|(other, _)| keep(*other))
                .map(|(_, other)| Euclid2d.distance(&key, other))
                .collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            brute.truncate(K);
            brute
        }

        fn assert_dists(got: &[f64], expected: &[f64], what: &str) {
            assert_eq!(
                got.len(),
                expected.len(),
                "{what} len {} vs {}",
                got.len(),
                expected.len()
            );
            for (hit, exp) in got.iter().zip(expected.iter()) {
                assert!((hit - exp).abs() < TOL, "{what} knn={hit} brute={exp}");
            }
        }

        fn hit_dists<N>(hits: &[(N, f64)]) -> Vec<f64> {
            hits.iter().map(|(_, d)| *d).collect()
        }

        fn check_id(
            tree: &MTree<(i64, i64), String>,
            live: &[(EntryId, (i64, i64))],
            active: &HashSet<EntryId>,
            idx: usize,
            op: u64,
        ) {
            let (id, key) = live[idx];
            let hint = live
                .iter()
                .find(|(other, _)| *other != id && !active.contains(other))
                .map(|(other, _)| *other)
                .unwrap_or(live[(idx + 1) % live.len()].0);
            let tag = format!("seed={SEED:#x} op={op} key={key:?}");

            let plain = tree.knn_search(&key, K, ());
            let expect_all = reference(live, key, id, false, |_| true);
            assert_dists(&hit_dists(&plain), &expect_all, &format!("{tag} search"));

            let all_on =
                tree.knn_search(&key, K, KnnConfig::new().with_active(|_, _: &String| true));
            assert_dists(
                &hit_dists(&all_on),
                &hit_dists(&plain),
                &format!("{tag} search-all"),
            );

            let subset = tree.knn_search(
                &key,
                K,
                KnnConfig::new().with_active(|oid: EntryId, _: &String| active.contains(&oid)),
            );
            assert!(subset.iter().all(|(id, _)| active.contains(id)));
            let expect_sub = reference(live, key, id, false, |oid| active.contains(&oid));
            assert_dists(
                &hit_dists(&subset),
                &expect_sub,
                &format!("{tag} search-sub"),
            );

            let plain_h = tree.knn_search(&key, K, KnnConfig::new().seed(hint));
            assert_dists(&hit_dists(&plain_h), &expect_all, &format!("{tag} hint"));
            let all_h = tree.knn_search(
                &key,
                K,
                KnnConfig::new()
                    .seed(hint)
                    .with_active(|_, _: &String| true),
            );
            assert_dists(
                &hit_dists(&all_h),
                &hit_dists(&plain_h),
                &format!("{tag} hint-all"),
            );
            let subset_h = tree.knn_search(
                &key,
                K,
                KnnConfig::new()
                    .seed(hint)
                    .with_active(|oid: EntryId, _: &String| active.contains(&oid)),
            );
            assert!(subset_h.iter().all(|(id, _)| active.contains(id)));
            assert_dists(
                &hit_dists(&subset_h),
                &expect_sub,
                &format!("{tag} hint-sub"),
            );

            let plain_e = tree.knn_from_entry(id, K, false, ()).unwrap_or_else(|| {
                panic!("{tag} entry knn=fehlt");
            });
            let expect_entry = reference(live, key, id, true, |_| true);
            assert_dists(&hit_dists(&plain_e), &expect_entry, &format!("{tag} entry"));
            let all_e = tree
                .knn_from_entry(
                    id,
                    K,
                    false,
                    KnnConfig::new().with_active(|_, _: &String| true),
                )
                .unwrap_or_else(|| panic!("{tag} entry-all knn=fehlt"));
            assert_dists(
                &hit_dists(&all_e),
                &hit_dists(&plain_e),
                &format!("{tag} entry-all"),
            );
            let subset_e = tree
                .knn_from_entry(
                    id,
                    K,
                    false,
                    KnnConfig::new().with_active(|oid: EntryId, _: &String| active.contains(&oid)),
                )
                .unwrap_or_else(|| panic!("{tag} entry-sub knn=fehlt"));
            assert!(subset_e
                .iter()
                .all(|(hit, _)| *hit != id && active.contains(hit)));
            let expect_entry_sub = reference(live, key, id, true, |oid| active.contains(&oid));
            assert_dists(
                &hit_dists(&subset_e),
                &expect_entry_sub,
                &format!("{tag} entry-sub"),
            );
        }

        fn spot_check(
            tree: &MTree<(i64, i64), String>,
            live: &[(EntryId, (i64, i64))],
            active: &HashSet<EntryId>,
            rng: &mut u64,
            op: u64,
        ) {
            for _ in 0..5 {
                let idx = (xorshift64(rng) as usize) % live.len();
                check_id(tree, live, active, idx, op);
            }
        }

        fn insert_one(
            tree: &mut MTree<(i64, i64), String>,
            live: &mut Vec<(EntryId, (i64, i64))>,
            active: &mut HashSet<EntryId>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let key = draw_key(rng, live);
            *op += 1;
            let id = tree
                .insert_with_identity(key, format!("v{}", *op), &*op)
                .unwrap();
            if xorshift64(rng) & 1 == 0 {
                active.insert(id);
            }
            live.push((id, key));
        }

        fn erase_one(
            tree: &mut MTree<(i64, i64), String>,
            live: &mut Vec<(EntryId, (i64, i64))>,
            active: &mut HashSet<EntryId>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, key) = live.swap_remove(idx);
            *op += 1;
            active.remove(&id);
            assert!(
                tree.erase_by_id(id),
                "seed={SEED:#x} op={op} key={key:?} knn=erase brute=fehlt"
            );
        }

        fn flip_one(
            live: &[(EntryId, (i64, i64))],
            active: &mut HashSet<EntryId>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let id = live[idx].0;
            *op += 1;
            if !active.remove(&id) {
                active.insert(id);
            }
        }

        let mut rng = SEED;
        let mut op = 0u64;
        let mut tree = MTree::with_params(2, 4, 3, Euclid2d);
        let mut live: Vec<(EntryId, (i64, i64))> = Vec::new();
        let mut active: HashSet<EntryId> = HashSet::new();

        while live.len() < 200 {
            insert_one(&mut tree, &mut live, &mut active, &mut rng, &mut op);
            if op % 100 == 0 {
                spot_check(&tree, &live, &active, &mut rng, op);
            }
        }
        for _ in 0..2_000 {
            if xorshift64(&mut rng) % 3 == 0 {
                flip_one(&live, &mut active, &mut rng, &mut op);
            } else if live.len() > 200 {
                erase_one(&mut tree, &mut live, &mut active, &mut rng, &mut op);
            } else if live.len() < 80 {
                insert_one(&mut tree, &mut live, &mut active, &mut rng, &mut op);
            } else if xorshift64(&mut rng) & 1 == 0 {
                insert_one(&mut tree, &mut live, &mut active, &mut rng, &mut op);
            } else {
                erase_one(&mut tree, &mut live, &mut active, &mut rng, &mut op);
            }
            if op % 100 == 0 {
                spot_check(&tree, &live, &active, &mut rng, op);
            }
        }

        for idx in 0..live.len() {
            check_id(&tree, &live, &active, idx, op);
        }
    }
}
