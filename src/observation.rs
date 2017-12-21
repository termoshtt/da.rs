use super::{ensemble, gaussian};
use super::types::*;

use ndarray::*;
use std::ops::Deref;

pub struct LinearNormal {
    /// Precision matrix of the observation noise covariance matrix
    rinv: Array2<R>,
    /// Observation operator
    h: Array2<R>,
}

impl LinearNormal {
    pub fn etkf_update(&self, xs: &ensemble::Ensemble, y: &Obs) -> gaussian::E {
        let (c, dx) = xs.deviation();
        let ys = self.h.dot(&dx); // Y
        let yr = ys.t().dot(&self.rinv); // YR^{-1}
        let dy = self.h.dot(&c) - y.deref();
        let ab = -yr.dot(&dy);
        let prec = yr.dot(&ys);
        gaussian::E { ab, prec }
    }
}
