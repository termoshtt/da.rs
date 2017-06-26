
use ndarray::*;
use ndarray_linalg::vector::outer;
use ndarray_rand::RandomExt;
use rand::distributions::*;
use std::ops::*;

use super::gaussian::Gaussian;
use super::types::*;

/// Ensemble is saved as two-dimensional array
#[derive(Debug, Clone)]
pub struct EnsembleBase<S: DataClone<Elem = R>>(ArrayBase<S, Ix2>);
pub type Ensemble = EnsembleBase<OwnedRepr<R>>;

impl<S: DataClone<Elem = R>> EnsembleBase<S> {
    /// size of ensemble
    pub fn size(&self) -> usize {
        self.rows()
    }

    /// size of each state vector
    pub fn dim(&self) -> usize {
        self.cols()
    }

    /// immutable ensemble iterator
    pub fn eiter(&self) -> iter::AxisIter<R, Ix1> {
        self.axis_iter(Axis(0))
    }

    /// mutable ensemble iterator
    pub fn eiter_mut(&mut self) -> iter::AxisIterMut<R, Ix1>
    where
        S: DataMut<Elem = R>,
    {
        self.axis_iter_mut(Axis(0))
    }

    /// center of ensemble
    pub fn center(&self) -> Array<R, Ix1> {
        self.mean(Axis(0))
    }

    /// regard ensemble as a Gaussian distribution
    pub fn as_gaussian(&self) -> Gaussian {
        // XXX this may be slow.
        let c = self.center();
        let n = self.dim();
        let m = self.size();
        let mut cov = Array::zeros((n, n));
        for v in self.eiter() {
            let dx = &v - &c;
            cov = cov + outer(&dx, &dx);
        }
        cov *= 1.0 / (m as f64 - 1.0);
        Gaussian {
            center: c,
            cov: cov,
        }
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
    where
        S1: Data<Elem = R>,
    {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((size, n), dist);
        (dx + center).into()
    }
}
