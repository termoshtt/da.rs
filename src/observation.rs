use super::*;

use ndarray::*;
use ndarray_linalg::*;
use std::ops::Deref;

/// Linear observation operator, and Gaussian observation noise
pub struct LinearNormal {
    /// Precision matrix of the observation noise covariance matrix
    pub rinv: Array2<R>,
    /// Observation operator
    pub h: Array2<R>,
}

impl LinearNormal {
    /// Size of state vector
    pub fn state_size(&self) -> usize {
        self.h.cols()
    }

    /// Size of observation
    pub fn obs_size(&self) -> usize {
        assert_eq!(self.rinv.cols(), self.rinv.rows(), "Rinv is not square");
        assert_eq!(self.rinv.cols(), self.h.rows(), "R and H are inconsistent");
        self.h.rows()
    }

    /// Construct new observation using random numbers
    pub fn random(state_size: usize, obs_size: usize) -> Self {
        let rinv = random_hpd(obs_size);
        let h = random((obs_size, state_size));
        Self { rinv, h }
    }

    /// Construct new observation with identity operator with isotropic Gaussian
    pub fn isotropic(n : usize, r: f64) -> Self {
        let rinv = Array::eye(n) / r;
        let h = Array::eye(n);
        Self { rinv, h }
    }

    /// Increment of Bayes update
    pub fn increment(&self, y: &Obs) -> gaussian::E {
        let hr = self.h.t().dot(&self.rinv);
        let ab = hr.dot(&y.0);
        let prec = hr.dot(&self.h);
        gaussian::E { ab, prec }
    }

    /// Increment in ensemble space
    pub fn et_increment(&self, xs: &ensemble::Ensemble, y: &Obs) -> gaussian::E {
        let (c, dx) = xs.deviation();
        let ys = self.h.dot(&dx); // Y
        let yr = ys.t().dot(&self.rinv); // YR^{-1}
        let mdy = self.h.dot(&c) - y.deref(); // -dy
        let ab = -yr.dot(&mdy);
        let prec = yr.dot(&ys);
        gaussian::E { ab, prec }
    }
}
