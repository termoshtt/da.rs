use ndarray::*;
use ndarray_rand::RandomExt;
use rand::distributions::*;

use super::gaussian::*;
use super::types::*;

/// Ensemble as two-dimensional array
#[derive(Debug, Clone)]
pub struct Ensemble(Array2<R>);

impl Ensemble {
    /// size of ensemble
    pub fn size(&self) -> usize {
        self.0.rows()
    }

    /// size of each state vector
    pub fn dim(&self) -> usize {
        self.0.cols()
    }

    /// immutable ensemble iterator
    pub fn ens_iter(&self) -> iter::AxisIter<R, Ix1> {
        self.0.axis_iter(Axis(0))
    }

    /// mutable ensemble iterator
    pub fn ens_iter_mut(&mut self) -> iter::AxisIterMut<R, Ix1> {
        self.0.axis_iter_mut(Axis(0))
    }

    /// center of ensemble
    pub fn center(&self) -> Array<R, Ix1> {
        self.0.mean(Axis(0))
    }

    /// regard ensemble as a Gaussian distribution
    pub fn as_gaussian(&self) -> Gaussian {
        // XXX this may be slow.
        let c = self.center();
        let dx = &self.0 - &c;
        let mut cov = dx.t().dot(&dx);
        let m = self.size() as f64;
        cov *= 1.0 / (m - 1.0);
        Gaussian::from_mean(c, cov)
    }

    /// Generate ensemble as an isotropic Gaussian distribution
    pub fn isotropic_gaussian<S: Data<Elem = R>>(center: &ArrayBase<S, Ix1>, size: usize, noise: R) -> Ensemble {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((size, n), dist);
        Ensemble(dx + center)
    }
}

impl ::std::ops::Deref for Ensemble {
    type Target = Array2<R>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Weight matrix
#[derive(Debug, Clone)]
pub struct Weights(Array2<R>);

impl Weights {
    pub fn trivial(n: usize) -> Self {
        Weights(Array::eye(n))
    }

    /// transform ensemble
    pub fn transform(&self, ens: &Ensemble) -> Ensemble {
        Ensemble(self.0.dot(&ens.0))
    }

    /// size of ensemble
    pub fn size(&self) -> usize {
        self.0.rows()
    }

    /// immutable ensemble iterator
    pub fn ens_iter(&self) -> iter::AxisIter<R, Ix1> {
        self.0.axis_iter(Axis(0))
    }
}

impl ::std::ops::Deref for Weights {
    type Target = Array2<R>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
