use ndarray::*;
use ndarray_linalg::*;

use super::gaussian;
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
        self.0.mean_axis(Axis(0))
    }

    /// m-Projection onto a normal distribution
    pub fn to_m(&self) -> gaussian::M {
        let c = self.center();
        let dx = &self.0 - &c;
        let mut cov = dx.t().dot(&dx);
        let k = self.size() as f64;
        cov *= 1.0 / (k - 1.0);
        gaussian::M { center: c, cov }
    }
}
