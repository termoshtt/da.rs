use ndarray::*;
use ndarray_linalg::*;
use ndarray_rand::RandomExt;
use rand::distributions::*;

use super::gaussian::*;
use super::types::*;

/// Ensemble as two-dimensional array
#[derive(Debug, Clone, NewType)]
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
    pub fn center(&self) -> Array1<R> {
        self.0.mean(Axis(0))
    }

    /// Calculate center and covariance matrix
    pub fn stat(&self) -> (Array1<R>, Array2<R>) {
        stat(self)
    }

    /// regard ensemble as a Gaussian distribution
    pub fn as_gaussian(&self) -> Gaussian {
        let (center, cov) = self.stat();
        Gaussian::from_mean(center, cov)
    }

    /// Generate ensemble as an isotropic Gaussian distribution
    pub fn isotropic_gaussian<S: Data<Elem = R>>(center: &ArrayBase<S, Ix1>, size: usize, noise: R) -> Ensemble {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((size, n), dist);
        Ensemble(dx + center)
    }
}

/// Ensemble on the weight space (ensemble-transform)
///
/// This weight is independent from ensembles due to the ETKF assumption,
/// i.e. a weight can be used with `t` and `t+1` ensembles.
#[derive(Debug, Clone, NewType)]
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

    pub fn center(&self) -> Array1<R> {
        self.0.mean(Axis(0))
    }

    /// Calculate center and covariance matrix
    pub fn stat(&self) -> (Array1<R>, Array2<R>) {
        stat(self)
    }

    /// immutable ensemble iterator
    pub fn ens_iter(&self) -> iter::AxisIter<R, Ix1> {
        self.0.axis_iter(Axis(0))
    }

    /// mutable ensemble iterator
    pub fn ens_iter_mut(&mut self) -> iter::AxisIterMut<R, Ix1> {
        self.0.axis_iter_mut(Axis(0))
    }
}

/// Transform projected Gaussian on the real space to ensemble space
pub fn ensemble_transform(ens: &Ensemble, pg: PGaussian) -> PGaussian {
    let xm = ens.center();
    let hxm = pg.projection.dot(&xm);
    let p = ens.dot(&pg.projection.t()) - &hxm;
    let m: M = pg.gaussian.into();
    let g = M {
        center: m.center - hxm,
        cov: m.cov,
    }.into();
    PGaussian {
        projection: p.reversed_axes(),
        gaussian: g,
    }
}

/// Core function for square-root filter
///
/// Sampling weights from Gaussian in weight space.
/// Gaussian must have an eigenvector `(1, ..., 1)`.
pub fn ssqrt_sampling(m: &M) -> Weights {
    let k = m.size() as f64;
    let mut ws = ((k - 1.0) * &m.cov).ssqrt_into(UPLO::Upper).unwrap();
    ws += &m.center;
    let s = ws.subview(Axis(0), 0).scalar_sum();
    ws += (1.0 - s) / k;
    Weights(ws)
}


/// Calculate center and covariance matrix (assuming 0-index denotes ensemble)
fn stat(a: &Array2<R>) -> (Array1<R>, Array2<R>) {
    let c = a.mean(Axis(0));
    let dx = a - &c;
    let mut cov = dx.t().dot(&dx);
    let m = a.rows() as f64;
    cov *= 1.0 / (m - 1.0);
    (c, cov)
}
