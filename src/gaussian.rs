//! Normal distribution (a.k.a. Gaussian) as an exponential family distribution
//!
//! Two forms of Gaussian are implemented:
//!
//! - natural (m-) parameter, i.e. mean and covariance matrix
//! - e-parameter for calculating the multiplication of two Gaussian
//!
//! These forms are natural in the information-geometric context.
//! Each forms represents Gaussian, and can be converted each other.
//! Conversion cost is in the order of `O(N^2)` since it calculates an inverse matrix.

use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

/// m-parameter form of Gaussian
#[derive(Debug, Clone)]
pub struct M {
    /// Center of Gaussian
    pub center: Array1<R>,
    /// Covariance matrix of Gaussian
    pub cov: Array2<R>,
}

impl M {
    pub fn size(&self) -> usize {
        self.center.len()
    }

    pub fn to_e(&self) -> E {
        let prec = self.cov.invh().expect("Covariance matrix is singular");
        let ab = prec.dot(&self.center);
        E { ab, prec }
    }
}

/// e-parameter form of Gaussian
#[derive(Debug, Clone)]
pub struct E {
    pub ab: Array1<R>,
    /// Precision matrix (inverse of the covariance matrix) of Gaussian
    pub prec: Array2<R>,
}

impl E {
    pub fn size(&self) -> usize {
        self.ab.len()
    }

    pub fn to_m(&self) -> M {
        let cov = self.prec.invh().expect("Precision matrix is singular");
        let center = cov.dot(&self.ab);
        M { center, cov }
    }
}

impl<'a> ::std::ops::Mul<&'a E> for E {
    type Output = Self;
    fn mul(mut self, rhs: &'a E) -> Self {
        self *= rhs;
        self
    }
}

impl<'a, 'b> ::std::ops::Mul<&'a E> for &'b E {
    type Output = E;
    fn mul(self, rhs: &'a E) -> E {
        let ab = &self.ab + &rhs.ab;
        let prec = &self.prec + &rhs.prec;
        E { ab, prec }
    }
}

impl<'a> ::std::ops::MulAssign<&'a E> for E {
    fn mul_assign(&mut self, rhs: &'a E) {
        self.ab += &rhs.ab;
        self.prec += &rhs.prec;
    }
}

impl From<E> for M {
    fn from(e: E) -> Self {
        let cov = e.prec.invh_into().expect("Precision matrix is singular");
        let center = cov.dot(&e.ab);
        M { center, cov }
    }
}

impl From<M> for E {
    fn from(m: M) -> Self {
        let prec = m.cov.invh_into().expect("Covariance matrix is singular");
        let ab = prec.dot(&m.center);
        E { ab, prec }
    }
}
