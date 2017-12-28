use super::*;
use ndarray::*;
use ndarray_linalg::*;

/// Ensemble as two-dimensional array
#[derive(Debug, Clone, NewType)]
pub struct Ensemble(Array2<R>);

impl Ensemble {
    /// Generate random ensemble
    pub fn random(size: EnsembleSize, dim: StateSize) -> Self {
        Ensemble(random((size.0, dim.0)))
    }

    /// size of ensemble
    pub fn size(&self) -> EnsembleSize {
        self.0.rows().into()
    }

    /// size of each state vector
    pub fn dim(&self) -> StateSize {
        self.0.cols().into()
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

    pub fn deviation(&self) -> (Array1<R>, Array2<R>) {
        let c = self.center().into();
        let dx = &self.0 - &c;
        (c, dx)
    }

    /// m-Projection onto a normal distribution
    pub fn to_m(&self) -> gaussian::M<StateSize> {
        let (c, dx) = self.deviation();
        let mut cov = dx.t().dot(&dx);
        let k = self.size().0 as f64;
        cov *= 1.0 / (k - 1.0);
        gaussian::M::new(c, cov)
    }
}
