//! Re-sampling functions for weighted sampling
//!
use num_traits::Zero;
use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform},
        Distribution,
    },
    Rng,
};

use rand_distr::Uniform;

/// Return a Vector of n indices sampled according to the weights slice.
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
    W: std::cmp::PartialOrd
        + Clone
        + SampleBorrow<W>
        + SampleUniform
        + Sized
        + Zero
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let mut max_w = W::zero();
    for w in weights.iter() {
        if *w > max_w {
            max_w = w.clone();
        }
    }

    let uniform_n = Uniform::new(0, weights.len());
    let uniform_w = Uniform::new(W::zero(), max_w);

    let mut indices = Vec::with_capacity(n);

    let mut b = W::zero();
    let mut i = uniform_n.sample(rng);
    for _ in 0..n {
        b += uniform_w.sample(rng);
        while b > weights[i] {
            b -= weights[i].clone();
            i = (i + 1) % weights.len();
        }
        indices.push(i);
    }

    indices
}

/// Return a Vector of samples drawn from the particles vector, according to the weights slice.
///
/// # Example
///
/// ```
/// use wheel_resample::resample;
///
/// let mut rng = rand::thread_rng();
/// let weights = [0.1, 0.2, 0.3, 0.8];
/// let particles = vec![1, 2, 3, 4];
/// let samples = resample(&mut rng, &weights, &particles);
///
/// assert_eq!(samples.len(), particles.len());
///
/// // Make sure all samples are particles
/// assert!(samples.iter().all(|s| particles.contains(s)));
/// ```
pub fn resample<R, T, W>(rng: &mut R, weights: &[W], particles: &[T]) -> Vec<T>
where
    R: Rng,
    T: Clone,
    W: std::cmp::PartialOrd
        + Clone
        + SampleBorrow<W>
        + SampleUniform
        + Sized
        + Zero
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let indices = resample_idx(rng, weights, particles.len());

    indices.iter().map(|&i| particles[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn resample() {
        let mut rng = rand::thread_rng();
        let weights = [0.1, 0.2, 0.3, 0.8];

        let sample_idx_2 = super::resample_idx(&mut rng, &weights, 2);

        assert_eq!(sample_idx_2.len(), 2);
        assert!(sample_idx_2.iter().all(|&i| i < weights.len()));

        let sample_idx_6 = super::resample_idx(&mut rng, &weights, 6);

        assert_eq!(sample_idx_6.len(), 6);
        assert!(sample_idx_6.iter().all(|&i| i < weights.len()));
    }
}
