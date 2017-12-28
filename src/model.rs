use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

/// Specification of the state in the model
pub trait StateSpec {
    type Scalar: Scalar;
    type Dim: Dimension;
}

/// Specification of the noise in the model
pub trait ObsSpec: StateSpec {
    /// Make an observation from truth
    fn observe(&self, st: ArrayView<Self::Scalar, Self::Dim>) -> Array1<Self::Scalar>;
    /// Return the covariance matrix of the observation noise
    fn obs_cov(&self) -> Array2<Self::Scalar>;
}

/// Deterministic (Dtm) state-space model (SSM)
pub struct DtmSSM<TEO> {
    _f: TEO,
    _n_state: usize,
    n_obs: usize,
    h: Array2<R>,
    rs: Array2<R>, // square of the covariance matrix of obs
}

impl<TEO> StateSpec for DtmSSM<TEO> {
    type Scalar = R;
    type Dim = Ix1;
}

impl<TEO> ObsSpec for DtmSSM<TEO> {
    fn observe(&self, st: ArrayView1<R>) -> Array1<R> {
        let n0: Array2<R> = random((self.n_obs, self.n_obs));
        self.h.dot(&st) + self.rs.dot(&n0)
    }

    fn obs_cov(&self) -> Array2<R> {
        self.rs.t().dot(&self.rs)
    }
}
