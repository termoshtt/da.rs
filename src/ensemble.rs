
use std::ops::*;
use rand::distributions::*;
use ndarray::*;
use ndarray_rand::RandomExt;

use super::types::*;

/// Ensemble is saved as two-dimensional array
#[derive(Debug, Clone)]
pub struct EnsembleBase<S: DataClone<Elem = R>>(ArrayBase<S, Ix2>);
pub type Ensemble = EnsembleBase<OwnedRepr<R>>;

impl<S: DataClone<Elem = R>> EnsembleBase<S> {
    pub fn ensemble_size(&self) -> usize {
        self.rows()
    }

    pub fn dim(&self) -> usize {
        self.cols()
    }
}

impl<S: DataClone<Elem = R>> From<ArrayBase<S, Ix2>> for EnsembleBase<S> {
    fn from(a: ArrayBase<S, Ix2>) -> Self {
        EnsembleBase(a)
    }
}

impl<S: DataClone<Elem = R>> Deref for EnsembleBase<S> {
    type Target = ArrayBase<S, Ix2>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: DataClone<Elem = R>> DerefMut for EnsembleBase<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S: DataClone<Elem = R>> EnsembleBase<S> {
    /// Generate ensemble as an isotropic Gaussian distribution
    pub fn isotropic_gaussian<S1>(center: &ArrayBase<S1, Ix1>, size: usize, noise: R) -> Ensemble
        where S1: Data<Elem = R>
    {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((size, n), dist);
        (dx + center).into()
    }
}
