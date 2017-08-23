use ndarray::*;
use ndarray_rand::RandomExt;
use rand::distributions::*;

use super::gaussian::*;
use super::types::*;

/// Ensemble as two-dimensional array
#[derive(Debug, Clone)]
pub struct Ensemble(Array<R, Ix2>);

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
}

impl Ensemble {
    /// Generate ensemble as an isotropic Gaussian distribution
    pub fn isotropic_gaussian<S: Data<Elem = R>>(center: &ArrayBase<S, Ix1>, size: usize, noise: R) -> Ensemble {
        let n = center.len();
        let dist = Normal::new(0.0, noise);
        let dx = Array::random((size, n), dist);
        Ensemble(dx + center)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size() {
        let n = 2; // dimension of each state
        let m = 10; // ensemble size
        let x0 = arr1(&[1.0, 2.0]);
        let xs = Ensemble::isotropic_gaussian(&x0, m, 1.0);
        assert_eq!(xs.dim(), n);
        assert_eq!(xs.size(), m);

        let g = xs.as_gaussian().into_m();
        assert_eq!(g.center.shape(), [n]);
        assert_eq!(g.cov.shape(), [n, n]);
    }

    #[test]
    fn ensemble_iter() {
        let n = 2; // dimension of each state
        let m = 10; // ensemble size
        let x0 = arr1(&[1.0, 2.0]);
        let mut xs = Ensemble::isotropic_gaussian(&x0, m, 1.0);
        for v in xs.ens_iter() {
            assert_eq!(v.len(), n);
        }
        for v in xs.ens_iter_mut() {
            assert_eq!(v.len(), n);
        }
    }
}
