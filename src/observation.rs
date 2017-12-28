use super::*;

use ndarray::*;
use ndarray_linalg::*;

/// Linear observation operator, and Gaussian observation noise
pub struct LinearNormal {
    /// Precision matrix of the observation noise covariance matrix
    pub rinv: Array2<R>,
    /// Observation operator
    pub h: Array2<R>,
}

impl LinearNormal {
    /// Apply observation operator without noise
    pub fn no_noise(&self, st: &State) -> Obs {
        self.h.dot(&st.0).into()
    }

    /// Size of state vector
    pub fn state_size(&self) -> StateSize {
        self.h.cols().into()
    }

    /// Size of observation
    pub fn obs_size(&self) -> ObsSize {
        assert_eq!(self.rinv.cols(), self.rinv.rows(), "Rinv is not square");
        assert_eq!(self.rinv.cols(), self.h.rows(), "R and H are inconsistent");
        self.h.rows().into()
    }

    /// Construct new observation using random numbers
    pub fn random(state_size: StateSize, obs_size: ObsSize) -> Self {
        let rinv = random_hpd(obs_size.0);
        let h = random((obs_size.0, state_size.0));
        Self { rinv, h }
    }

    /// Construct new observation with identity operator with isotropic Gaussian
    pub fn isotropic(n: ObsSize, r: f64) -> Self {
        let rinv = Array::eye(n.0) / r;
        let h = Array::eye(n.0);
        Self { rinv, h }
    }

    /// Increment of Bayes update
    pub fn increment(&self, y: &Obs) -> gaussian::E<StateSize> {
        let hr = self.h.t().dot(&self.rinv);
        let ab = hr.dot(&y.0);
        let prec = hr.dot(&self.h);
        gaussian::E::new(ab, prec)
    }

    /// Increment in ensemble space
    pub fn et_increment(&self, xs: &ensemble::Ensemble, y: &Obs) -> gaussian::E<EnsembleSize> {
        let (c, dx) = xs.deviation();
        let ys = self.h.dot(&dx.t()); // Y
        let yr = ys.t().dot(&self.rinv); // YR^{-1}
        let mdy = self.h.dot(&c) - &y.0; // -dy
        let ab = -yr.dot(&mdy);
        let prec = yr.dot(&ys);
        gaussian::E::new(ab, prec)
    }
}
