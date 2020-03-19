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
    let resampler = Resampler::new(rng, weights);
    resampler.into_iter().take(n).collect()
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

/// The Resampler can be turned into an Iterator to contineously pull sample indices
///
/// # Example
///
/// ```
/// use wheel_resample::Resampler;
///
/// let mut rng = rand::thread_rng();
/// let weights = [0.1, 0.2, 0.3, 0.8];
/// let resampler = Resampler::new(&mut rng, &weights);
///
/// let population = vec![1, 2, 3, 4];
/// let samples = resampler.into_iter().take(4).map(|i| population[i].clone()).collect::<Vec<u32>>();
///
/// // Make sure we got four samples
/// assert_eq!(samples.len(), 4);
///
/// // Make sure all samples come from the population
/// assert!(samples.iter().all(|s| population.contains(s)));
/// ```
///
pub struct Resampler<'a, R: Rng, W: Float> {
    rng: &'a mut R,
    weights: &'a [W],
}

impl<'a, R: Rng, W: Float> Resampler<'a, R, W> {
    /// Create Resampler instance from random generator and weights
    pub fn new(rng: &'a mut R, weights: &'a [W]) -> Self {
        Resampler { rng, weights }
    }
}

impl<'a, R: Rng, W: SampleUniform + Float> IntoIterator for Resampler<'a, R, W> {
    type Item = usize;
    type IntoIter = ResampleIterator<'a, R, W>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut max_w = W::zero();
        // Can we do this more elegant given floats are not Ord?
        for &w in self.weights.iter() {
            if w > max_w {
                max_w = w;
            }
        }

        let uniform_n = Uniform::new(0, self.weights.len());
        let uniform_w = Uniform::new(W::zero(), W::from(2.0).unwrap() * max_w);

        ResampleIterator {
            b: W::zero(),
            uniform_w,
            index: uniform_n.sample(&mut self.rng),
            resampler: self,
        }
    }
}

pub struct ResampleIterator<'a, R: Rng, W: SampleUniform + Float> {
    b: W,
    uniform_w: Uniform<W>,
    index: usize,
    resampler: Resampler<'a, R, W>,
}

impl<'a, R: Rng, W: SampleUniform + Float> Iterator for ResampleIterator<'a, R, W> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.b = self.b + self.uniform_w.sample(self.resampler.rng);
        while self.b > self.resampler.weights[self.index] {
            self.b = self.b - self.resampler.weights[self.index];
            self.index = (self.index + 1) % self.resampler.weights.len();
        }

        Some(self.index)
    }
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

    #[test]
    fn resample_iter() {
        let mut rng = rand::thread_rng();
        let weights = [0.1, 0.2, 0.3, 0.8];

        let resampler = super::Resampler::new(&mut rng, &weights);

        let samples = resampler.into_iter().take(4).collect::<Vec<usize>>();

        dbg! { &samples};

        assert_eq!(samples.len(), 4);
        assert!(samples.iter().all(|&i| i < weights.len()));
    }
}
