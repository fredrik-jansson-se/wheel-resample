//! Re-sampling functions for weighted sampling
//!
//! # Example
//!
//! ```
//! use wheel_resample::resample;
//!
//! let mut rng = rand::thread_rng();
//! let weights = [0.1, 0.2, 0.3, 0.8];
//! let population = vec![1, 2, 3, 4];
//! let samples = resample(&mut rng, &weights, &population);
//!
//! assert_eq!(samples.len(), population.len());
//!
//! // Make sure all samples are in the population
//! assert!(samples.iter().all(|s| population.contains(s)));
//! ```
use num_traits::float::Float;
use rand::{
    distributions::{uniform::SampleUniform, Distribution},
    Rng,
};

use rand_distr::Uniform;

/// Returns a vector of n indices sampled according to the weights slice.
///
/// # Example
///
/// ```
/// use wheel_resample::resample_idx;
///
/// let mut rng = rand::thread_rng();
/// let weights = [0.1, 0.2, 0.3, 0.8];
///
/// let sample_idx = resample_idx(&mut rng, &weights, weights.len());
/// assert_eq!(sample_idx.len(), weights.len());
///
/// let sample_2_idx = resample_idx(&mut rng, &weights, 2);
/// assert_eq!(sample_2_idx.len(), 2);
/// ```
pub fn resample_idx<R, W>(rng: &mut R, weights: &[W], n: usize) -> Vec<usize>
where
    R: Rng,
    W: SampleUniform + Float,
{
    let mut max_w = W::zero();
    // Can we do this more elegant given floats are not Ord?
    for &w in weights.iter() {
        if w > max_w {
            max_w = w;
        }
    }

    let uniform_n = Uniform::new(0, weights.len());
    let uniform_w = Uniform::new(W::zero(), W::from(2.0).unwrap() * max_w);

    let mut indices = Vec::with_capacity(n);

    let mut b = W::zero();
    let mut i = uniform_n.sample(rng);
    for _ in 0..n {
        b = b + uniform_w.sample(rng);
        while b > weights[i] {
            b = b - weights[i];
            i = (i + 1) % weights.len();
        }
        indices.push(i);
    }

    indices
}

/// Returns a vector of weighted samples drawn from the population vector.
///
/// # Example
///
/// ```
/// use wheel_resample::resample;
///
/// let mut rng = rand::thread_rng();
/// let weights = [0.1, 0.2, 0.3, 0.8];
/// let population = vec![1, 2, 3, 4];
/// let samples = resample(&mut rng, &weights, &population);
///
/// assert_eq!(samples.len(), population.len());
///
/// // Make sure all samples are in the population
/// assert!(samples.iter().all(|s| population.contains(s)));
/// ```
pub fn resample<R, T, W>(rng: &mut R, weights: &[W], population: &[T]) -> Vec<T>
where
    R: Rng,
    T: Clone,
    W: SampleUniform + Float,
{
    let indices = resample_idx(rng, weights, population.len());

    indices.iter().map(|&i| population[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn resample_idx() {
        let mut rng = rand::thread_rng();
        let weights = [0.1, 0.2, 0.3, 0.8];

        // Make sure we can pull fewer samples than weights
        let sample_idx_2 = super::resample_idx(&mut rng, &weights, 2);

        assert_eq!(sample_idx_2.len(), 2);
        assert!(sample_idx_2.iter().all(|&i| i < weights.len()));

        // Make sure we can pull more samples than weights
        let sample_idx_6 = super::resample_idx(&mut rng, &weights, 6);

        assert_eq!(sample_idx_6.len(), 6);
        assert!(sample_idx_6.iter().all(|&i| i < weights.len()));
    }
}
