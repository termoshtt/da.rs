
use ndarray::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

use super::types::*;

/// Ensemble is saved as two-dimensional array
#[derive(Debug, Clone)]
pub struct EnsembleBase<S: DataClone<Elem = R>>(ArrayBase<S, Ix2>);
pub type Ensemble = EnsembleBase<OwnedRepr<R>>;

impl<S: DataClone<Elem = R>> From<ArrayBase<S, Ix2>> for EnsembleBase<S> {
    fn from(a: ArrayBase<S, Ix2>) -> Self {
        EnsembleBase(a)
    }
}

impl<S: DataClone<Elem = R>> EnsembleBase<S> {
    /// Generate ensemble from the center vector and noise intensity
    pub fn generate<S1>(center: &ArrayBase<S1, Ix1>, size: usize, noise: R) -> Ensemble
        where S1: Data<Elem = R>
    {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((n, size), dist);
        (dx + center).into()
    }
}
