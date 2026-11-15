#[cfg(test)]
mod tests {
    use crate::distance::Distance;
    use crate::node_params_for_capacity;
    use crate::DuplicateKey;
    use crate::MTree;
    use crate::{KnnConfig, RangeSearchConfig, SearchConfig};
    use std::collections::{BTreeSet, HashSet};

    /// Euklidische Distanz für (i64, i64)
    #[derive(Clone)]
    struct Euclid2d;
    impl Distance<(i64, i64)> for Euclid2d {
        type Output = f64;
        fn distance(&self, a: &(i64, i64), b: &(i64, i64)) -> f64 {
            let dx = (a.0 - b.0) as f64;
            let dy = (a.1 - b.1) as f64;
            (dx * dx + dy * dy).sqrt()
        }
    }

    type TreeI64 = MTree<(i64, i64), String, f64, crate::stats::DescendantCounter, Euclid2d>;

    fn new_tree_i64() -> TreeI64 {
        MTree::with_params(5, 100, 20, Euclid2d)
    }

    /// Erweiterte Metrik wie im Baum: `d(K,K) + |εa − εb|`.
    fn metric_d(a: &(i64, i64), eps_a: f64, b: &(i64, i64), eps_b: f64) -> f64 {
        Euclid2d.distance(a, b) + (eps_a - eps_b).abs()
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

    /// Exact-kNN mit Toleranz für Gleichstände an der k-Grenze: Distanzfolgen müssen
    /// übereinstimmen; Keys mit strikt kleinerer Distanz als k-te müssen identisch sein.
    fn assert_knn_matches_naive_allowing_ties(
        got: &[((i64, i64), f64)],
        expected: &[((i64, i64), f64)],
    ) {
        assert_eq!(got.len(), expected.len());
        let mut got_sorted = got.to_vec();
        let mut exp_sorted = expected.to_vec();
        got_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        exp_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (g, e)) in got_sorted.iter().zip(exp_sorted.iter()).enumerate() {
            assert!(
                (g.1 - e.1).abs() < 1e-9,
                "dist mismatch at i={i}: got {} vs {}",
                g.1,
                e.1
            );
        }
        if expected.is_empty() {
            return;
        }
        let kth = exp_sorted.last().unwrap().1;
        let got_nearer: BTreeSet<_> = got_sorted
            .iter()
            .filter(|(_, d)| *d < kth - 1e-12)
            .map(|(k, _)| *k)
            .collect();
        let exp_nearer: BTreeSet<_> = exp_sorted
            .iter()
            .filter(|(_, d)| *d < kth - 1e-12)
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(
            got_nearer, exp_nearer,
            "strictly nearer key sets differ"
        );
    }

    #[test]
    fn test_node_params_for_capacity() {
        assert_eq!(node_params_for_capacity(0), node_params_for_capacity(1));
        let (_min, max_100, _split) = node_params_for_capacity(100);
        // √100 = 10 → floor clamp 16
        assert_eq!(max_100, 16);
        let (min, max, split) = node_params_for_capacity(100);
        assert_eq!(min, (max / 4).max(2));
        assert_eq!(split, (max / 4).clamp(5, 20));
        let (_, max_big, _) = node_params_for_capacity(1_000_000);
        assert_eq!(max_big, 64);
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
        let tree: TreeI64 = MTree::with_params(5, 100, 20, Euclid2d);
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
            let key = small.get(*id).unwrap().key().clone();
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
                assert_knn_matches_naive_allowing_ties(
                    &without_wide
                        .iter()
                        .map(|(id, d)| (wide.key(*id).unwrap(), *d))
                        .collect::<Vec<_>>(),
                    &expected
                        .iter()
                        .map(|(nk, _, d)| (*nk, *d))
                        .collect::<Vec<_>>(),
                );
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
    ) -> (TreeI64, Vec<((i64, i64), String)>) {
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
                let got: Vec<_> = mtree_results
                    .iter()
                    .map(|(id, d)| (tree.key(*id).unwrap(), *d))
                    .collect();
                let expected: Vec<_> = naive_results
                    .iter()
                    .map(|(nk, _, d)| (*nk, *d))
                    .collect();
                assert_knn_matches_naive_allowing_ties(&got, &expected);
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

    fn new_tree_point() -> MTree<Point, String, f64, crate::stats::DescendantCounter, EuclideanDistance> {
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
        assert_eq!(*node.key(), (1, 2));
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
    #[cfg(feature = "knn_stats")]
    fn test_knn_stats_counts_routing_nodes_when_filtered() {
        let mut tree = new_tree_i64();
        tree.insert((0, 0), "a".to_string()).unwrap();
        tree.insert((1, 0), "b".to_string()).unwrap();
        tree.insert((2, 0), "c".to_string()).unwrap();

        crate::reset_knn_visit_stats();
        let _hits = tree.knn_search(
            &(0, 0),
            2,
            KnnConfig::new().with_active(|_id, v: &String| v != "b"),
        );
        let stats = crate::knn_visit_stats();
        assert!(stats.routing_nodes > 0);
        assert!(stats.leaves > 0);
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

        let ids: Vec<_> = tree.iter().collect();
        let entries: Vec<_> = ids
            .iter()
            .map(|id| {
                (
                    tree.key(*id).unwrap(),
                    *id,
                    tree.epsilon(*id).unwrap(),
                )
            })
            .collect();
        for (key, id, eps) in &entries {
            let hits = tree.knn_from_entry(*id, 6, false, ()).unwrap();
            assert_eq!(hits.len(), 6);
            let mut brute: Vec<f64> = entries
                .iter()
                .filter(|(_, other, _)| other != id)
                .map(|(other, _, oeps)| metric_d(key, *eps, other, *oeps))
                .collect();
            brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (hit, expected) in hits.iter().zip(brute.iter()) {
                assert!(
                    (hit.1 - *expected).abs() < 1e-9,
                    "key={key:?} knn={} brute={expected}",
                    hit.1
                );
            }
        }
    }

    /// Dieselbe Punktmenge und dieselben 400 Ersetzungen, gefiltert.
    /// Immer-wahr bleibt auf der gefilterten Bahn und trifft die ungefilterte Suche.
    /// Je ohne Hint, mit Hint und über `knn_from_entry`. Naive: `d+|εa−εb|`.
    #[test]
    fn knn_from_entry_filtered_matches_naive_after_replaces() {
        use crate::EntryId;

        const K: usize = 13;
        const TOL: f64 = 1e-9;

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

        let ids: Vec<_> = tree.iter().collect();
        let entries: Vec<_> = ids
            .iter()
            .map(|id| {
                (
                    tree.key(*id).unwrap(),
                    *id,
                    tree.epsilon(*id).unwrap(),
                )
            })
            .collect();

        // needle_eps: 0 für knn_search, Eintrags-ε für knn_from_entry.
        let reference =
            |key: &(i64, i64), needle_eps: f64, skip: Option<EntryId>, even_only: bool| {
                let mut brute: Vec<f64> = entries
                    .iter()
                    .filter(|(_, other, _)| skip != Some(*other))
                    .filter(|(_, other, _)| !even_only || even_slot(*other, &String::new()))
                    .map(|(other, _, oeps)| metric_d(key, needle_eps, other, *oeps))
                    .collect();
                brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
                brute.truncate(K);
                brute
            };

        for (idx, (key, id, eps)) in entries.iter().enumerate() {
            let hint = entries[(idx + 1) % entries.len()].1;

            let plain = tree.knn_search(key, K, ());
            let plain_d: Vec<f64> = plain.iter().map(|(_, d)| *d).collect();
            assert_dists(
                &plain_d,
                &reference(key, 0.0, None, false),
                &format!("search key={key:?}"),
            );
            let all_on =
                tree.knn_search(key, K, KnnConfig::new().with_active(|_, _: &String| true));
            let all_d: Vec<f64> = all_on.iter().map(|(_, d)| *d).collect();
            assert_dists(&all_d, &plain_d, &format!("search-all key={key:?}"));

            let subset = tree.knn_search(key, K, KnnConfig::new().with_active(even_slot));
            assert!(subset.iter().all(|(id, _)| even_slot(*id, &String::new())));
            let subset_d: Vec<f64> = subset.iter().map(|(_, d)| *d).collect();
            assert_dists(
                &subset_d,
                &reference(key, 0.0, None, true),
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
                &reference(key, 0.0, None, true),
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
                &reference(key, *eps, Some(*id), false),
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
                &reference(key, *eps, Some(*id), true),
                &format!("entry-half key={key:?}"),
            );
        }
    }

    /// Zufällige insert/erase, ein Thread (`&mut self`): kNN und Radius bleiben bei der Referenz.
    /// Seed `0x5A17_5EED` (xorshift64). Knoten 2/4, damit Split, Donate und Merge oft laufen.
    /// Inserts mit Identity-ε; Naive nutzt `d+|εa−εb|`.
    #[test]
    fn knn_matches_naive_under_random_replaces() {
        use crate::{epsilon_from, EntryId};

        const SEED: u64 = 0x5A17_5EED;
        const K: usize = 5;
        const TOL: f64 = 1e-9;

        type Live = (EntryId, (i64, i64), f64);

        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        // *3-Gitter: Distanz exakt 25 ist unmöglich; ε verschiebt die Radius-Kugel nicht über die Grenze.
        fn draw_key(state: &mut u64, live: &[Live]) -> (i64, i64) {
            loop {
                let x = ((xorshift64(state) % 81) as i64 - 40) * 3;
                let y = ((xorshift64(state) % 81) as i64 - 40) * 3;
                if live.iter().all(|(_, key, _)| *key != (x, y)) {
                    return (x, y);
                }
            }
        }

        fn assert_knn(tree: &TreeI64, live: &[Live], id: EntryId, key: (i64, i64), eps: f64, op: u64) {
            let hits = tree.knn_from_entry(id, K, false, ()).unwrap_or_else(|| {
                panic!("seed={SEED:#x} op={op} key={key:?} knn=fehlt brute=fehlt");
            });
            let mut brute: Vec<f64> = live
                .iter()
                .filter(|(other, _, _)| *other != id)
                .map(|(_, other, oeps)| metric_d(&key, eps, other, *oeps))
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

        fn spot_check(tree: &TreeI64, live: &[Live], rng: &mut u64, op: u64) {
            for _ in 0..5 {
                let idx = (xorshift64(rng) as usize) % live.len();
                let (id, key, eps) = live[idx];
                assert_knn(tree, live, id, key, eps, op);
            }
        }

        fn insert_one(tree: &mut TreeI64, live: &mut Vec<Live>, rng: &mut u64, op: &mut u64) {
            let key = draw_key(rng, live);
            *op += 1;
            let id = tree
                .insert_with_identity(key, format!("v{}", *op), &*op)
                .unwrap();
            live.push((id, key, epsilon_from(&*op)));
        }

        fn erase_one(tree: &mut TreeI64, live: &mut Vec<Live>, rng: &mut u64, op: &mut u64) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, key, _) = live.swap_remove(idx);
            *op += 1;
            assert!(
                tree.erase_by_id(id),
                "seed={SEED:#x} op={op} key={key:?} knn=erase brute=fehlt"
            );
        }

        let mut rng = SEED;
        let mut op = 0u64;
        let mut tree = MTree::with_params(2, 4, 3, Euclid2d);
        let mut live: Vec<Live> = Vec::new();

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
        for &(id, key, eps) in &live {
            assert_knn(&tree, &live, id, key, eps, op);
        }

        // range_search Default-Needle-ε=0 (nicht Eintrags-ε).
        let got_ids: Vec<_> = tree
            .range_search(&qkey, RangeSearchConfig::new(25.0))
            .map(|(id, _)| id)
            .collect();
        let got: BTreeSet<_> = got_ids
            .iter()
            .map(|id| tree.key(*id).unwrap())
            .collect();
        let expect: BTreeSet<_> = live
            .iter()
            .filter(|(_, key, eps)| metric_d(&qkey, 0.0, key, *eps) <= 25.0)
            .map(|(_, key, _)| *key)
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
    /// Naive nutzt `d+|εa−εb|` (`knn_search` Needle-ε=0, `knn_from_entry` Eintrags-ε).
    #[test]
    fn knn_filtered_matches_naive_under_random_replaces() {
        use crate::{epsilon_from, EntryId};
        use std::collections::HashSet;

        const SEED: u64 = 0x5A17_5EED;
        const K: usize = 5;
        const TOL: f64 = 1e-9;

        type Live = (EntryId, (i64, i64), f64);

        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        fn draw_key(state: &mut u64, live: &[Live]) -> (i64, i64) {
            loop {
                let x = ((xorshift64(state) % 81) as i64 - 40) * 3;
                let y = ((xorshift64(state) % 81) as i64 - 40) * 3;
                if live.iter().all(|(_, key, _)| *key != (x, y)) {
                    return (x, y);
                }
            }
        }

        fn reference(
            live: &[Live],
            key: (i64, i64),
            needle_eps: f64,
            self_id: EntryId,
            exclude_self: bool,
            keep: impl Fn(EntryId) -> bool,
        ) -> Vec<f64> {
            let mut brute: Vec<f64> = live
                .iter()
                .filter(|(other, _, _)| !exclude_self || *other != self_id)
                .filter(|(other, _, _)| keep(*other))
                .map(|(_, other, oeps)| metric_d(&key, needle_eps, other, *oeps))
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
            tree: &TreeI64,
            live: &[Live],
            active: &HashSet<EntryId>,
            idx: usize,
            op: u64,
        ) {
            let (id, key, eps) = live[idx];
            let hint = live
                .iter()
                .find(|(other, _, _)| *other != id && !active.contains(other))
                .map(|(other, _, _)| *other)
                .unwrap_or(live[(idx + 1) % live.len()].0);
            let tag = format!("seed={SEED:#x} op={op} key={key:?}");

            // knn_search: Default-Needle-ε=0
            let plain = tree.knn_search(&key, K, ());
            let expect_all = reference(live, key, 0.0, id, false, |_| true);
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
            let expect_sub = reference(live, key, 0.0, id, false, |oid| active.contains(&oid));
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

            // knn_from_entry: Needle-ε = Eintrags-ε
            let plain_e = tree.knn_from_entry(id, K, false, ()).unwrap_or_else(|| {
                panic!("{tag} entry knn=fehlt");
            });
            let expect_entry = reference(live, key, eps, id, true, |_| true);
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
            let expect_entry_sub = reference(live, key, eps, id, true, |oid| active.contains(&oid));
            assert_dists(
                &hit_dists(&subset_e),
                &expect_entry_sub,
                &format!("{tag} entry-sub"),
            );
        }

        fn spot_check(
            tree: &TreeI64,
            live: &[Live],
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
            tree: &mut TreeI64,
            live: &mut Vec<Live>,
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
            live.push((id, key, epsilon_from(&*op)));
        }

        fn erase_one(
            tree: &mut TreeI64,
            live: &mut Vec<Live>,
            active: &mut HashSet<EntryId>,
            rng: &mut u64,
            op: &mut u64,
        ) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, key, _) = live.swap_remove(idx);
            *op += 1;
            active.remove(&id);
            assert!(
                tree.erase_by_id(id),
                "seed={SEED:#x} op={op} key={key:?} knn=erase brute=fehlt"
            );
        }

        fn flip_one(live: &[Live], active: &mut HashSet<EntryId>, rng: &mut u64, op: &mut u64) {
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
        let mut live: Vec<Live> = Vec::new();
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

    // ---- kNN-Graph / RkNN (A1–A5) ----

    fn new_tree_i64_graph(k_max: usize) -> TreeI64 {
        MTree::with_knn_graph(k_max, Euclid2d)
    }

    fn naive_rknn(
        data: &[((i64, i64), crate::EntryId)],
        needle: &(i64, i64),
        k: usize,
    ) -> Vec<(crate::EntryId, f64)> {
        let dist = Euclid2d;
        let mut out = Vec::new();
        for &(key_o, id_o) in data {
            let mut dists: Vec<f64> = data
                .iter()
                .filter(|(_, id)| *id != id_o)
                .map(|(key_p, _)| dist.distance(&key_o, key_p))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let knn_d = if dists.len() < k {
                f64::INFINITY
            } else {
                dists[k - 1]
            };
            let d = dist.distance(needle, &key_o);
            if d <= knn_d {
                out.push((id_o, d));
            }
        }
        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
        });
        out
    }

    fn assert_graph_invariants(tree: &TreeI64) {
        let k_max = tree.knn_graph_k_max().expect("graph on");
        let ids: Vec<_> = tree.iter().collect();
        for &id in &ids {
            let list = tree.knn_graph_neighbors(id).unwrap_or(&[]);
            assert!(list.len() <= k_max, "forward longer than k_max");
            for i in 1..list.len() {
                assert!(
                    list[i - 1].1 <= list[i].1 + 1e-12,
                    "forward not sorted by dist"
                );
            }
            for &(nb, _) in list {
                assert!(tree.key(nb).is_some(), "forward neighbor missing");
            }
        }
    }

    #[test]
    fn rknn_disabled_returns_none() {
        let mut tree = new_tree_i64();
        let id = tree.insert((0, 0), "a".into()).unwrap();
        assert!(tree.rknn_search(&(0, 0), 1).is_none());
        assert!(tree.rknn_from_entry(id, 1).is_none());
        assert!(!tree.knn_graph_enabled());
    }

    #[test]
    fn rknn_matches_naive_after_inserts() {
        for &k_max in &[1usize, 5, 10] {
            let mut tree = new_tree_i64_graph(k_max);
            let mut data = Vec::new();
            for i in 0..40 {
                let key = (i as i64 % 7, i as i64 / 7);
                let id = tree.insert(key, format!("v{i}")).unwrap();
                data.push((key, id));
            }
            assert_graph_invariants(&tree);
            for k in 1..=k_max {
                for &(needle, _) in &data {
                    let got = tree.rknn_search(&needle, k).unwrap();
                    let expect = naive_rknn(&data, &needle, k);
                    let got_ids: HashSet<_> = got.iter().map(|(id, _)| *id).collect();
                    let exp_ids: HashSet<_> = expect.iter().map(|(id, _)| *id).collect();
                    assert_eq!(
                        got_ids, exp_ids,
                        "k_max={k_max} k={k} needle={needle:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rknn_graph_survives_erase_and_replace() {
        let k_max = 5;
        let mut tree = new_tree_i64_graph(k_max);
        let mut data = Vec::new();
        for i in 0..30 {
            let key = (i, i * 2);
            let id = tree.insert(key, format!("{i}")).unwrap();
            data.push((key, id));
        }
        // Erase a few.
        for _ in 0..5 {
            let (_, id) = data.pop().unwrap();
            assert!(tree.erase_by_id(id));
        }
        assert_graph_invariants(&tree);
        let mut live: Vec<_> = data
            .iter()
            .copied()
            .filter(|(_, id)| tree.key(*id).is_some())
            .collect();
        for &(needle, _) in &live {
            let got = tree.rknn_search(&needle, k_max).unwrap();
            let expect = naive_rknn(&live, &needle, k_max);
            let got_ids: HashSet<_> = got.iter().map(|(id, _)| *id).collect();
            let exp_ids: HashSet<_> = expect.iter().map(|(id, _)| *id).collect();
            assert_eq!(got_ids, exp_ids);
        }
        // Replace-equivalent: erase + insert.
        let (old_key, old_id) = live.pop().unwrap();
        assert!(tree.erase_by_id(old_id));
        let new_key = (old_key.0 + 100, old_key.1 + 100);
        let new_id = tree.insert(new_key, "repl".into()).unwrap();
        live.push((new_key, new_id));
        assert_graph_invariants(&tree);
        let got = tree.rknn_search(&new_key, k_max).unwrap();
        let expect = naive_rknn(&live, &new_key, k_max);
        let got_ids: HashSet<_> = got.iter().map(|(id, _)| *id).collect();
        let exp_ids: HashSet<_> = expect.iter().map(|(id, _)| *id).collect();
        assert_eq!(got_ids, exp_ids);
    }

    #[test]
    fn rknn_update_key_keeps_invariants() {
        let mut tree = new_tree_i64_graph(4);
        let mut ids = Vec::new();
        for i in 0..20 {
            ids.push(tree.insert((i, 0), format!("{i}")).unwrap());
        }
        tree.update_key(ids[3], (3, 50)).unwrap();
        assert_graph_invariants(&tree);
        let mut data = Vec::new();
        for &id in &ids {
            let key = tree.key(id).unwrap();
            data.push((key, id));
        }
        let needle = tree.key(ids[3]).unwrap();
        let got = tree.rknn_search(&needle, 4).unwrap();
        let expect = naive_rknn(&data, &needle, 4);
        let got_ids: HashSet<_> = got.iter().map(|(id, _)| *id).collect();
        let exp_ids: HashSet<_> = expect.iter().map(|(id, _)| *id).collect();
        assert_eq!(got_ids, exp_ids);
    }

    #[test]
    fn rknn_slot_reuse_clears_stale_reverse() {
        let mut tree = new_tree_i64_graph(3);
        let a = tree.insert((0, 0), "a".into()).unwrap();
        let b = tree.insert((1, 0), "b".into()).unwrap();
        let _c = tree.insert((2, 0), "c".into()).unwrap();
        assert!(tree.erase_by_id(a));
        let a2 = tree.insert((10, 10), "a2".into()).unwrap();
        assert_eq!(a.slot_index(), a2.slot_index());
        assert_graph_invariants(&tree);
        // Neighbors of b must not list the recycled identity incorrectly as old coords.
        if let Some(nbs) = tree.knn_graph_neighbors(b) {
            for &(nb, _) in nbs {
                assert!(tree.key(nb).is_some());
            }
        }
        let _ = b;
    }

    #[test]
    fn rknn_from_entry_excludes_self() {
        let mut tree = new_tree_i64_graph(5);
        let mut ids = Vec::new();
        for i in 0..15 {
            ids.push(tree.insert((i, i), format!("{i}")).unwrap());
        }
        let hits = tree.rknn_from_entry(ids[0], 5).unwrap();
        assert!(hits.iter().all(|(id, _)| *id != ids[0]));
    }

    #[test]
    fn insert_with_knn_hits_matches_plain_insert() {
        let mut plain = new_tree_i64_graph(5);
        let mut with_hits = new_tree_i64_graph(5);
        let mut ids_plain = Vec::new();
        for i in 0..12 {
            let key = (i, 0);
            ids_plain.push(plain.insert(key, format!("{i}")).unwrap());
        }
        let mut ids_hits = Vec::new();
        for i in 0..12 {
            let key = (i, 0);
            let hits: Vec<_> = with_hits
                .knn_search(&key, 5, ())
                .into_iter()
                .map(|(id, d)| (id, d))
                .collect();
            ids_hits.push(
                with_hits
                    .insert_with_knn_hits(key, format!("{i}"), &hits)
                    .unwrap(),
            );
        }
        for i in 0..12 {
            let ka = plain.key(ids_plain[i]).unwrap();
            let kb = with_hits.key(ids_hits[i]).unwrap();
            assert_eq!(ka, kb);
            let ra = plain.rknn_search(&ka, 5).unwrap();
            let rb = with_hits.rknn_search(&kb, 5).unwrap();
            let sa: BTreeSet<_> = ra.iter().map(|(id, _)| plain.key(*id).unwrap()).collect();
            let sb: BTreeSet<_> = rb.iter().map(|(id, _)| with_hits.key(*id).unwrap()).collect();
            assert_eq!(sa, sb);
        }
    }

    /// Stress wie `knn_matches_naive_under_random_replaces`, aber mit kNN-Graph:
    /// Invarianten, Forward-Listen vs. Naive-kNN, `rknn_search` vs. Naive-RkNN.
    /// Inserts mit Identity-ε ≠ 0; Naive nutzt dieselbe Metrik `d+|εa−εb|`.
    #[test]
    fn rknn_matches_naive_under_random_replaces() {
        use crate::{epsilon_from, EntryId};

        const SEED: u64 = 0x2B17_5EED;
        const K_MAX: usize = 5;
        const TOL: f64 = 1e-9;

        /// Live-Eintrag: id, key, deterministisches Identity-ε.
        type Live = (EntryId, (i64, i64), f64);

        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        fn draw_key(state: &mut u64, live: &[Live]) -> (i64, i64) {
            loop {
                let x = ((xorshift64(state) % 81) as i64 - 40) * 3;
                let y = ((xorshift64(state) % 81) as i64 - 40) * 3;
                if live.iter().all(|(_, key, _)| *key != (x, y)) {
                    return (x, y);
                }
            }
        }

        fn naive_forward(live: &[Live], id: EntryId, key: (i64, i64), eps: f64) -> Vec<(EntryId, f64)> {
            let mut brute: Vec<(EntryId, f64)> = live
                .iter()
                .filter(|(other, _, _)| *other != id)
                .map(|(oid, okey, oeps)| (*oid, metric_d(&key, eps, okey, *oeps)))
                .collect();
            brute.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
            });
            brute.truncate(K_MAX.min(brute.len()));
            brute
        }

        fn naive_rknn_eps(
            live: &[Live],
            needle: &(i64, i64),
            needle_eps: f64,
        ) -> Vec<(EntryId, f64)> {
            let mut out = Vec::new();
            for &(id_o, key_o, eps_o) in live {
                let mut neighbors: Vec<(EntryId, f64)> = live
                    .iter()
                    .filter(|(id, _, _)| *id != id_o)
                    .map(|(id_p, key_p, eps_p)| (*id_p, metric_d(&key_o, eps_o, key_p, *eps_p)))
                    .collect();
                neighbors.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
                });
                let knn_d = if neighbors.len() < K_MAX {
                    f64::INFINITY
                } else {
                    neighbors[K_MAX - 1].1
                };
                let d = metric_d(needle, needle_eps, &key_o, eps_o);
                if d <= knn_d {
                    out.push((id_o, d));
                }
            }
            out.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
            });
            out
        }

        fn assert_forward(tree: &TreeI64, live: &[Live], id: EntryId, key: (i64, i64), eps: f64, op: u64) {
            let list = tree.knn_graph_neighbors(id).unwrap_or(&[]);
            let expect = naive_forward(live, id, key, eps);
            assert_eq!(
                list.len(),
                expect.len(),
                "seed={SEED:#x} op={op} key={key:?} forward len {} vs {}",
                list.len(),
                expect.len()
            );
            for (got, exp) in list.iter().zip(expect.iter()) {
                assert_eq!(
                    got.0, exp.0,
                    "seed={SEED:#x} op={op} key={key:?} forward id mismatch"
                );
                assert!(
                    (got.1 - exp.1).abs() < TOL,
                    "seed={SEED:#x} op={op} key={key:?} forward dist {} vs {}",
                    got.1,
                    exp.1
                );
            }
        }

        fn assert_rknn(
            tree: &TreeI64,
            live: &[Live],
            needle: (i64, i64),
            needle_eps: f64,
            op: u64,
        ) {
            let got = tree
                .rknn_search_eps(&needle, needle_eps, K_MAX)
                .unwrap_or_else(|| {
                    panic!("seed={SEED:#x} op={op} key={needle:?} rknn=None");
                });
            let expect = naive_rknn_eps(live, &needle, needle_eps);
            let got_ids: HashSet<_> = got.iter().map(|(id, _)| *id).collect();
            let exp_ids: HashSet<_> = expect.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                got_ids, exp_ids,
                "seed={SEED:#x} op={op} key={needle:?} eps={needle_eps} rknn set mismatch got={} exp={}",
                got_ids.len(),
                exp_ids.len()
            );
            for (id, d) in &got {
                let exp_d = expect
                    .iter()
                    .find(|(eid, _)| eid == id)
                    .map(|(_, ed)| *ed)
                    .unwrap();
                assert!(
                    (d - exp_d).abs() < TOL,
                    "seed={SEED:#x} op={op} key={needle:?} rknn dist {d} vs {exp_d}"
                );
            }
        }

        fn spot_check(tree: &TreeI64, live: &[Live], rng: &mut u64, op: u64) {
            assert_graph_invariants(tree);
            for _ in 0..5 {
                let idx = (xorshift64(rng) as usize) % live.len();
                let (id, key, eps) = live[idx];
                assert_forward(tree, live, id, key, eps, op);
                // Query mit dem ε des gespeicherten Eintrags (nicht ε=0).
                assert_rknn(tree, live, key, eps, op);
            }
        }

        fn insert_one(tree: &mut TreeI64, live: &mut Vec<Live>, rng: &mut u64, op: &mut u64) {
            let key = draw_key(rng, live);
            *op += 1;
            let id = tree
                .insert_with_identity(key, format!("v{}", *op), &*op)
                .unwrap();
            let eps = epsilon_from(&*op);
            debug_assert!((tree.get(id).unwrap().epsilon() - eps).abs() < 1e-18);
            live.push((id, key, eps));
        }

        fn erase_one(tree: &mut TreeI64, live: &mut Vec<Live>, rng: &mut u64, op: &mut u64) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, key, _) = live.swap_remove(idx);
            *op += 1;
            assert!(
                tree.erase_by_id(id),
                "seed={SEED:#x} op={op} key={key:?} erase failed"
            );
        }

        fn update_one(tree: &mut TreeI64, live: &mut Vec<Live>, rng: &mut u64, op: &mut u64) {
            let idx = (xorshift64(rng) as usize) % live.len();
            let (id, _, eps) = live[idx];
            let new_key = draw_key(rng, live);
            *op += 1;
            tree.update_key(id, new_key).unwrap_or_else(|e| {
                panic!("seed={SEED:#x} op={op} update_key={e:?}");
            });
            // Identity-Hash (und damit ε) bleibt bei update_key erhalten.
            live[idx].1 = new_key;
            debug_assert!((tree.get(id).unwrap().epsilon() - eps).abs() < 1e-18);
        }

        let mut rng = SEED;
        let mut op = 0u64;
        // Kleine Nodes → viele Splits; Graph explizit aktivieren.
        let mut tree = MTree::with_params(2, 4, 3, Euclid2d);
        tree.knn_graph = Some(crate::knn_graph::KnnGraph::new(K_MAX));
        let mut live: Vec<Live> = Vec::new();

        while live.len() < 200 {
            insert_one(&mut tree, &mut live, &mut rng, &mut op);
            if op % 100 == 0 {
                spot_check(&tree, &live, &mut rng, op);
            }
        }
        for _ in 0..2_000 {
            let roll = xorshift64(&mut rng) % 5;
            if live.len() > 200 {
                erase_one(&mut tree, &mut live, &mut rng, &mut op);
            } else if live.len() < 80 {
                insert_one(&mut tree, &mut live, &mut rng, &mut op);
            } else if roll == 0 {
                update_one(&mut tree, &mut live, &mut rng, &mut op);
            } else if roll <= 2 {
                insert_one(&mut tree, &mut live, &mut rng, &mut op);
            } else {
                erase_one(&mut tree, &mut live, &mut rng, &mut op);
            }
            if op % 100 == 0 {
                spot_check(&tree, &live, &mut rng, op);
            }
        }

        assert_eq!(tree.size(), live.len(), "seed={SEED:#x} op={op} size mismatch");
        assert_graph_invariants(&tree);
        for &(id, key, eps) in &live {
            assert_forward(&tree, &live, id, key, eps, op);
            assert_rknn(&tree, &live, key, eps, op);
        }
        // Auch RkNN von außen (Needle-ε=0) und mit fremdem ε.
        let outside = draw_key(&mut rng, &live);
        assert_rknn(&tree, &live, outside, 0.0, op);
        assert_rknn(&tree, &live, outside, epsilon_from(&999_999u64), op);
    }

    // ---- Active kNN-Graph ----

    fn new_tree_i64_graphs(k_live: usize, k_active: usize) -> TreeI64 {
        MTree::with_knn_graphs(k_live, k_active, Euclid2d)
    }

    fn naive_rknn_active(
        data: &[((i64, i64), crate::EntryId)],
        active: &HashSet<crate::EntryId>,
        needle: &(i64, i64),
        k: usize,
    ) -> Vec<(crate::EntryId, f64)> {
        let dist = Euclid2d;
        let mut out = Vec::new();
        for &(key_o, id_o) in data {
            if !active.contains(&id_o) {
                continue;
            }
            let mut dists: Vec<f64> = data
                .iter()
                .filter(|(_, id)| *id != id_o && active.contains(id))
                .map(|(key_p, _)| dist.distance(&key_o, key_p))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let knn_d = if dists.len() < k {
                f64::INFINITY
            } else {
                dists[k - 1]
            };
            let d = dist.distance(needle, &key_o);
            if d <= knn_d {
                out.push((id_o, d));
            }
        }
        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.slot_index().cmp(&b.0.slot_index()))
        });
        out
    }

    fn assert_active_graph_invariants(tree: &TreeI64) {
        let k_max = tree.knn_graph_active_k_max().expect("active graph on");
        for id in tree.iter() {
            if !tree.is_active_member(id) {
                assert!(tree.knn_graph_active_neighbors(id).is_none());
                continue;
            }
            let list = tree.knn_graph_active_neighbors(id).unwrap_or(&[]);
            assert!(list.len() <= k_max);
            for &(nb, _) in list {
                assert!(tree.is_active_member(nb), "idle neighbor in active forward");
                assert!(tree.key(nb).is_some());
            }
        }
    }

    #[test]
    fn rknn_active_disabled_returns_none() {
        let mut tree = new_tree_i64_graph(3);
        let id = tree.insert((0, 0), "a".into()).unwrap();
        assert!(!tree.knn_graph_active_enabled());
        assert!(tree.rknn_active_search(&(0, 0), 1).is_none());
        assert!(tree.knn_graph_active_neighbors(id).is_none());
    }

    #[test]
    fn rknn_active_rebuild_matches_naive() {
        let k = 4usize;
        let mut tree = new_tree_i64_graphs(k, k);
        let mut data = Vec::new();
        for i in 0..35 {
            let key = (i as i64 % 6, i as i64 / 6);
            let id = tree.insert(key, format!("v{i}")).unwrap();
            data.push((key, id));
        }
        let active: HashSet<_> = data.iter().step_by(2).map(|(_, id)| *id).collect();
        let active_vec: Vec<_> = active.iter().copied().collect();
        tree.rebuild_active_graph(&active_vec);
        assert_active_graph_invariants(&tree);
        for &(needle, id) in &data {
            if !active.contains(&id) {
                continue;
            }
            let got = tree.rknn_active_search(&needle, k).unwrap();
            let expect = naive_rknn_active(&data, &active, &needle, k);
            let got_ids: HashSet<_> = got.iter().map(|(i, _)| *i).collect();
            let exp_ids: HashSet<_> = expect.iter().map(|(i, _)| *i).collect();
            assert_eq!(got_ids, exp_ids, "needle={needle:?}");
        }
    }

    #[test]
    fn rknn_active_membership_flip_batch() {
        let k = 3usize;
        let mut tree = new_tree_i64_graphs(k, k);
        let mut data = Vec::new();
        for i in 0..24 {
            let key = (i, 0);
            let id = tree.insert(key, format!("{i}")).unwrap();
            data.push((key, id));
        }
        // Initially activate first half via small diffs (incremental path).
        let first: Vec<_> = data[..12].iter().map(|(_, id)| *id).collect();
        for chunk in first.chunks(2) {
            tree.set_active_membership(chunk, &[]);
        }
        assert_eq!(
            tree.iter().filter(|&id| tree.is_active_member(id)).count(),
            12
        );
        assert_active_graph_invariants(&tree);

        let to_idle: Vec<_> = data[0..6].iter().map(|(_, id)| *id).collect();
        let to_active: Vec<_> = data[12..18].iter().map(|(_, id)| *id).collect();
        tree.set_active_membership(&to_active, &to_idle);

        let active: HashSet<_> = tree
            .iter()
            .filter(|&id| tree.is_active_member(id))
            .collect();
        assert_eq!(active.len(), 12);
        for id in &to_idle {
            assert!(!active.contains(id));
        }
        for id in &to_active {
            assert!(active.contains(id));
        }
        assert_active_graph_invariants(&tree);

        let live: Vec<_> = data.clone();
        for &(needle, id) in &live {
            if !active.contains(&id) {
                continue;
            }
            let got = tree.rknn_active_search(&needle, k).unwrap();
            let expect = naive_rknn_active(&live, &active, &needle, k);
            let got_ids: HashSet<_> = got.iter().map(|(i, _)| *i).collect();
            let exp_ids: HashSet<_> = expect.iter().map(|(i, _)| *i).collect();
            assert_eq!(got_ids, exp_ids);
        }
    }

    #[test]
    fn rknn_active_tree_erase_idle_leaves_active_intact() {
        let k = 3usize;
        let mut tree = new_tree_i64_graphs(k, k);
        let mut ids = Vec::new();
        for i in 0..16 {
            ids.push(tree.insert((i, 0), format!("{i}")).unwrap());
        }
        tree.rebuild_active_graph(&ids[..8]);
        let idle = ids[10];
        assert!(!tree.is_active_member(idle));
        let before: HashSet<_> = tree
            .iter()
            .filter(|&id| tree.is_active_member(id))
            .collect();
        assert!(tree.erase_by_id(idle));
        let after: HashSet<_> = tree
            .iter()
            .filter(|&id| tree.is_active_member(id))
            .collect();
        assert_eq!(before, after);
        assert_active_graph_invariants(&tree);
    }

    #[test]
    fn rknn_active_tree_erase_active_updates_graph() {
        let k = 3usize;
        let mut tree = new_tree_i64_graphs(k, k);
        let mut ids = Vec::new();
        for i in 0..16 {
            ids.push(tree.insert((i, 0), format!("{i}")).unwrap());
        }
        tree.rebuild_active_graph(&ids[..10]);
        let victim = ids[2];
        assert!(tree.is_active_member(victim));
        assert!(tree.erase_by_id(victim));
        assert!(!tree.is_active_member(victim));
        assert_active_graph_invariants(&tree);
        let active: HashSet<_> = tree
            .iter()
            .filter(|&id| tree.is_active_member(id))
            .collect();
        let live: Vec<_> = ids
            .iter()
            .filter(|id| tree.key(**id).is_some())
            .map(|&id| (tree.key(id).unwrap(), id))
            .collect();
        for &(needle, id) in &live {
            if !active.contains(&id) {
                continue;
            }
            let got = tree.rknn_active_search(&needle, k).unwrap();
            let expect = naive_rknn_active(&live, &active, &needle, k);
            let got_ids: HashSet<_> = got.iter().map(|(i, _)| *i).collect();
            let exp_ids: HashSet<_> = expect.iter().map(|(i, _)| *i).collect();
            assert_eq!(got_ids, exp_ids);
        }
    }

    #[test]
    fn rknn_active_rebuild_equiv_incremental_small_flips() {
        let k = 4usize;
        let mut rebuild = new_tree_i64_graphs(k, k);
        let mut incr = new_tree_i64_graphs(k, k);
        let mut data = Vec::new();
        for i in 0..20 {
            let key = (i % 5, i / 5);
            let a = rebuild.insert(key, format!("r{i}")).unwrap();
            let b = incr.insert(key, format!("i{i}")).unwrap();
            data.push((key, a, b));
        }
        let init: Vec<_> = data.iter().take(10).map(|(_, a, _)| *a).collect();
        let init_b: Vec<_> = data.iter().take(10).map(|(_, _, b)| *b).collect();
        rebuild.rebuild_active_graph(&init);
        incr.rebuild_active_graph(&init_b);

        // One small flip on each (should take incremental path on incr).
        let to_idle_a = [data[0].1];
        let to_active_a = [data[10].1];
        let to_idle_b = [data[0].2];
        let to_active_b = [data[10].2];
        rebuild.set_active_membership(&to_active_a, &to_idle_a);
        incr.set_active_membership(&to_active_b, &to_idle_b);

        for (key, a, b) in &data {
            assert_eq!(
                rebuild.is_active_member(*a),
                incr.is_active_member(*b),
                "membership mismatch at {key:?}"
            );
            if !rebuild.is_active_member(*a) {
                continue;
            }
            let na = rebuild.knn_graph_active_neighbors(*a).unwrap_or(&[]);
            let nb = incr.knn_graph_active_neighbors(*b).unwrap_or(&[]);
            let ka: Vec<_> = na
                .iter()
                .map(|(id, d)| (rebuild.key(*id).unwrap(), *d))
                .collect();
            let kb: Vec<_> = nb
                .iter()
                .map(|(id, d)| (incr.key(*id).unwrap(), *d))
                .collect();
            assert_eq!(ka.len(), kb.len(), "forward len at {key:?}");
            for (xa, xb) in ka.iter().zip(kb.iter()) {
                assert_eq!(xa.0, xb.0);
                assert!((xa.1 - xb.1).abs() < 1e-9);
            }
        }
    }
}
