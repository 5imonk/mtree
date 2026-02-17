// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hash::{Hash, Hasher};

/// Trait für Typen, die Koordinaten als f64-Slice bereitstellen
/// Ermöglicht SIMD-optimierte Distanzberechnung für beliebige Key-Typen
pub trait AsCoordinates: Send + Sync {
    /// Gibt die Koordinaten als Slice von f64 zurück
    fn coordinates(&self) -> &[f64];
}

/// Trait für Distanzfunktionen
pub trait Distance<K>: Send + Sync {
    type Output;
    fn distance(&self, a: &K, b: &K) -> Self::Output;

    /// Erstellt eine Box-Kopie der Distanzfunktion (für Query-Iteratoren).
    fn clone_box(&self) -> Box<dyn Distance<K, Output = Self::Output> + Send + Sync> {
        panic!("clone_box not supported for this distance type")
    }
}

/// Euklidische Distanzfunktion mit SIMD-Optimierungen
pub struct EuclideanDistance;

impl EuclideanDistance {
    /// Berechnet euklidische Distanz zwischen zwei f32-Vektoren mit SIMD
    #[target_feature(enable = "avx")]
    unsafe fn euclidean_f32_simd_avx(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm256_sub_ps(va, vb);
            let sq = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq);
        }
        
        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(sum_high, sum_low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        let mut result = _mm_cvtss_f32(sum32);
        
        // Handle remainder
        for i in (chunks * 8)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    /// Berechnet euklidische Distanz zwischen zwei f64-Vektoren mit SIMD
    #[target_feature(enable = "avx")]
    unsafe fn euclidean_f64_simd_avx(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_pd();
        
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(idx));
            let vb = _mm256_loadu_pd(b.as_ptr().add(idx));
            let diff = _mm256_sub_pd(va, vb);
            let sq = _mm256_mul_pd(diff, diff);
            sum = _mm256_add_pd(sum, sq);
        }
        
        // Horizontal sum
        let sum_high = _mm256_extractf128_pd(sum, 1);
        let sum_low = _mm256_castpd256_pd128(sum);
        let sum128 = _mm_add_pd(sum_high, sum_low);
        let sum64 = _mm_add_pd(sum128, _mm_shuffle_pd(sum128, sum128, 0x1));
        let mut result = _mm_cvtsd_f64(sum64);
        
        // Handle remainder
        for i in (chunks * 4)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    /// Portable/Skalar-Fallback für f32 (wird genutzt wenn kein AVX oder Nicht-x86_64)
    fn euclidean_f32_portable(a: &[f32], b: &[f32]) -> f32 {
        Self::euclidean_scalar(a, b) as f32
    }

    /// Portable/Skalar-Fallback für f64 (wird genutzt wenn kein AVX oder Nicht-x86_64)
    fn euclidean_f64_portable(a: &[f64], b: &[f64]) -> f64 {
        Self::euclidean_scalar(a, b)
    }
    
    /// Skalare Fallback-Implementierung
    fn euclidean_scalar<T>(a: &[T], b: &[T]) -> f64
    where
        T: Copy + Into<f64>,
    {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            let diff: f64 = a[i].into() - b[i].into();
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// Öffentliche Wrapper-Methode für euklidische Distanz zwischen zwei f64-Slices
    /// Nutzt SIMD-Optimierungen wenn verfügbar, sonst skalaren Fallback
    pub fn distance_slice(a: &[f64], b: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    return Self::euclidean_f64_simd_avx(a, b);
                }
            }
        }
        Self::euclidean_f64_portable(a, b)
    }
}

impl Distance<Vec<f32>> for EuclideanDistance {
    type Output = f64;

    fn clone_box(&self) -> Box<dyn Distance<Vec<f32>, Output = f64> + Send + Sync> {
        Box::new(EuclideanDistance)
    }

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    return Self::euclidean_f32_simd_avx(a, b) as f64;
                }
            }
        }
        Self::euclidean_f32_portable(a, b) as f64
    }
}

impl Distance<Vec<f64>> for EuclideanDistance {
    type Output = f64;

    fn clone_box(&self) -> Box<dyn Distance<Vec<f64>, Output = f64> + Send + Sync> {
        Box::new(EuclideanDistance)
    }

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe {
                    return Self::euclidean_f64_simd_avx(a, b);
                }
            }
        }
        Self::euclidean_f64_portable(a, b)
    }
}

/// Distance-Implementierung für alle Typen, die AsCoordinates implementieren
/// Nutzt die SIMD-optimierte Distanzberechnung
impl<P> Distance<P> for EuclideanDistance
where
    P: AsCoordinates,
{
    type Output = f64;

    fn clone_box(&self) -> Box<dyn Distance<P, Output = f64> + Send + Sync> {
        Box::new(EuclideanDistance)
    }

    fn distance(&self, a: &P, b: &P) -> f64 {
        Self::distance_slice(a.coordinates(), b.coordinates())
    }
}

/// Generische Distanzfunktion für beliebige Typen
pub struct GenericDistance<F, K, D> {
    distance_fn: F,
    _phantom: std::marker::PhantomData<(K, D)>,
}

impl<F, K, D> GenericDistance<F, K, D>
where
    F: Fn(&K, &K) -> D,
{
    pub fn new(distance_fn: F) -> Self {
        Self {
            distance_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, K, D> Distance<K> for GenericDistance<F, K, D>
where
    F: Fn(&K, &K) -> D + Send + Sync,
    K: Send + Sync,
    D: Send + Sync,
{
    type Output = D;

    fn distance(&self, a: &K, b: &K) -> D {
        (self.distance_fn)(a, b)
    }
}

/// Point-Typ als Wrapper um Vec<f64> mit Hash + Eq für erase()-Unterstützung
/// Nutzt SIMD-optimierte Distanzberechnung über AsCoordinates-Trait
#[derive(Clone, Debug, Default)]
pub struct Point(pub Vec<f64>);

impl Point {
    /// Erstellt einen neuen Point mit den gegebenen Koordinaten
    pub fn new(coords: Vec<f64>) -> Self {
        Point(coords)
    }
    
    /// Gibt die Koordinaten als Slice zurück
    pub fn coords(&self) -> &[f64] {
        &self.0
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Point {}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash über die Bits der f64-Werte für konsistente Hash-Werte
        self.0.iter().for_each(|&x| state.write_u64(x.to_bits()));
    }
}

impl AsCoordinates for Point {
    fn coordinates(&self) -> &[f64] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidean_f32() {
        let dist = EuclideanDistance;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let d = dist.distance(&a, &b);
        let expected: f64 = ((3.0_f32 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0).sqrt()).into();
        assert!((d - expected).abs() < 1e-5);
    }
    
    #[test]
    fn test_euclidean_f64() {
        let dist = EuclideanDistance;
        let a = vec![1.0f64, 2.0];
        let b = vec![4.0f64, 5.0];
        let d = dist.distance(&a, &b);
        let expected = ((3.0 * 3.0 + 3.0 * 3.0) as f64).sqrt();
        assert!((d - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_point_as_coordinates() {
        let p1 = Point::new(vec![1.0, 2.0, 3.0]);
        let p2 = Point::new(vec![4.0, 5.0, 6.0]);
        
        // Test AsCoordinates trait
        assert_eq!(p1.coordinates(), &[1.0, 2.0, 3.0]);
        assert_eq!(p2.coordinates(), &[4.0, 5.0, 6.0]);
        
        // Test Distance mit SIMD-Optimierung
        let dist = EuclideanDistance;
        let d = dist.distance(&p1, &p2);
        let expected = ((3.0 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0) as f64).sqrt();
        assert!((d - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_point_hash_eq() {
        let p1 = Point::new(vec![1.0, 2.0, 3.0]);
        let p2 = Point::new(vec![1.0, 2.0, 3.0]);
        let p3 = Point::new(vec![1.0, 2.0, 4.0]);
        
        // Test Eq
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
        
        // Test Hash (gleiche Werte sollten gleichen Hash haben)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(p1.clone());
        assert!(set.contains(&p2));
        assert!(!set.contains(&p3));
    }
}
