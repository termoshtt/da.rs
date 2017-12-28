use super::*;
use ndarray::*;
use ndarray_linalg::*;

/// Ensemble as two-dimensional array
#[derive(Debug, Clone, NewType)]
pub struct Ensemble(Array2<R>);

impl Ensemble {
    /// Generate random ensemble
    pub fn random(size: usize, dim: usize) -> Self {
        Ensemble(random((size, dim)))
    }

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
    pub fn center(&self) -> State {
        self.0.mean_axis(Axis(0)).into()
    }

    pub(crate) fn deviation(&self) -> (Array1<R>, Array2<R>) {
        let c = self.center().into();
        let dx = &self.0 - &c;
        (c, dx)
    }

    /// m-Projection onto a normal distribution
    pub fn to_m(&self) -> gaussian::M {
        let (c, dx) = self.deviation();
        let mut cov = dx.t().dot(&dx);
        let k = self.size() as f64;
        cov *= 1.0 / (k - 1.0);
        gaussian::M { center: c, cov }
    }
}
