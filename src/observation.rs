use super::*;

use ndarray::*;
use std::ops::Deref;

/// Linear observation operator, and Gaussian observation noise
pub struct LinearNormal {
    /// Precision matrix of the observation noise covariance matrix
    pub rinv: Array2<R>,
    /// Observation operator
    pub h: Array2<R>,
}

impl LinearNormal {
    pub fn etkf_update(&self, xs: &ensemble::Ensemble, y: &Obs) -> gaussian::E {
        let (c, dx) = xs.deviation();
        let ys = self.h.dot(&dx); // Y
        let yr = ys.t().dot(&self.rinv); // YR^{-1}
        let mdy = self.h.dot(&c) - y.deref(); // -dy
        let ab = -yr.dot(&mdy);
        let prec = yr.dot(&ys);
        gaussian::E { ab, prec }
    }
}
