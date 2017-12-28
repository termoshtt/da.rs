//! Normal distribution (a.k.a. Gaussian) as an exponential family distribution

use ndarray::*;
use ndarray_linalg::*;
use std::marker::PhantomData;

use super::*;

/// m-parameter form of Gaussian
#[derive(Debug, Clone, new)]
pub struct M<Si: Size> {
    /// Center of Gaussian
    pub center: Array1<R>,
    /// Covariance matrix of Gaussian
    pub cov: Array2<R>,
    ph: PhantomData<Si>,
}

impl<Si: Size> M<Si> {
    pub fn random(n: Si) -> Self {
        let n = n.into();
        let cov = random_hpd(n);
        let center = random(n);
        Self::new(center, cov)
    }

    pub fn size(&self) -> Si {
        assert_eq!(
            self.cov.cols(),
            self.cov.rows(),
            "Covariance matrix is not square"
        );
        assert_eq!(
            self.cov.cols(),
            self.center.len(),
            "Sizes of covariance matrix and center are inconsistent"
        );
        self.center.len().into()
    }

    pub fn to_e(&self) -> E<Si> {
        let prec = self.cov.invh().expect("Covariance matrix is singular");
        let ab = prec.dot(&self.center);
        E {
            ab,
            prec,
            ph: PhantomData {},
        }
    }
}

/// e-parameter form of Gaussian
#[derive(Debug, Clone, new)]
pub struct E<Si: Size> {
    pub ab: Array1<R>,
    /// Precision matrix (inverse of the covariance matrix) of Gaussian
    pub prec: Array2<R>,
    ph: PhantomData<Si>,
}

impl<Si: Size> E<Si> {
    pub fn random(n: Si) -> Self {
        let n = n.into();
        let prec = random_hpd(n);
        let ab = random(n);
        Self::new(ab, prec)
    }

    pub fn size(&self) -> Si {
        assert_eq!(
            self.prec.cols(),
            self.prec.rows(),
            "Covariance matrix is not square"
        );
        assert_eq!(
            self.prec.cols(),
            self.ab.len(),
            "Sizes of covariance matrix and center are inconsistent"
        );
        self.ab.len().into()
    }

    pub fn to_m(&self) -> M<Si> {
        let cov = self.prec.invh().expect("Precision matrix is singular");
        let center = cov.dot(&self.ab);
        M::new(center, cov)
    }
}

impl<'a, Si: Size> ::std::ops::Mul<&'a E<Si>> for E<Si> {
    type Output = Self;
    fn mul(mut self, rhs: &'a E<Si>) -> Self {
        self *= rhs;
        self
    }
}

impl<'a, 'b, Si: Size> ::std::ops::Mul<&'a E<Si>> for &'b E<Si> {
    type Output = E<Si>;
    fn mul(self, rhs: &'a E<Si>) -> E<Si> {
        let ab = &self.ab + &rhs.ab;
        let prec = &self.prec + &rhs.prec;
        E::new(ab, prec)
    }
}

impl<'a, Si: Size> ::std::ops::MulAssign<&'a E<Si>> for E<Si> {
    fn mul_assign(&mut self, rhs: &'a E<Si>) {
        self.ab += &rhs.ab;
        self.prec += &rhs.prec;
    }
}

impl<Si: Size> From<E<Si>> for M<Si> {
    fn from(e: E<Si>) -> Self {
        let cov = e.prec.invh_into().expect("Precision matrix is singular");
        let center = cov.dot(&e.ab);
        M::new(center, cov)
    }
}

impl<Si: Size> From<M<Si>> for E<Si> {
    fn from(m: M<Si>) -> Self {
        let prec = m.cov.invh_into().expect("Covariance matrix is singular");
        let ab = prec.dot(&m.center);
        E::new(ab, prec)
    }
}
