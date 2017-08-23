use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

/// Gaussian as an exponential family distribution
///
/// Two forms of Gaussian are implemented:
///
/// - natural (m-) parameter, i.e. mean and covariance matrix
/// - e-parameter for calculating the multiplication of two Gaussian
///
/// Each forms totally represents Gaussian, and can be converted each other
/// Conversion cost is in the order of `O(N^2)` since it calculates the inverse matrix.
/// You will find more knowledge in textbooks of information-geometory.
#[derive(Debug, Clone, IntoEnum)]
pub enum Gaussian {
    M(M),
    E(E),
}

impl Gaussian {
    pub fn from_mean(center: Array1<R>, cov: Array2<R>) -> Self {
        Gaussian::M(M { center, cov })
    }

    /// Get the center of Gaussian
    ///
    /// if the Gaussian is in E form, it is recalculated.
    pub fn center(&self) -> Array1<R> {
        match *self {
            Gaussian::M(ref m) => m.center.clone(),
            Gaussian::E(ref e) => e.prec.solveh(&e.ab).unwrap(),
        }
    }

    /// Get the covariance matrix of Gaussian
    ///
    /// if the Gaussian is in E form, it is recalculated.
    pub fn cov(&self) -> Array2<R> {
        match *self {
            Gaussian::M(ref m) => m.cov.clone(),
            Gaussian::E(ref e) => e.prec.invh().unwrap(),
        }
    }
}

/// natural (m-) parameter as an exponential family
#[derive(Debug, Clone)]
pub struct M {
    pub center: Array1<R>,
    pub cov: Array2<R>,
}

/// e-parameter as an exponential family
#[derive(Debug, Clone)]
pub struct E {
    pub ab: Array1<R>,
    pub prec: Array2<R>,
}

pub trait IntoM {
    fn into_m(self) -> M;
}

impl<T: Into<M>> IntoM for T {
    fn into_m(self) -> M {
        self.into()
    }
}

pub trait IntoE {
    fn into_e(self) -> E;
}

impl<T: Into<E>> IntoE for T {
    fn into_e(self) -> E {
        self.into()
    }
}

impl From<E> for M {
    fn from(e: E) -> Self {
        let cov = e.prec.invh_into().unwrap();
        let center = cov.dot(&e.ab);
        M { center, cov }
    }
}

impl From<M> for E {
    fn from(m: M) -> Self {
        let prec = m.cov.invh_into().unwrap();
        let ab = prec.dot(&m.center);
        E { ab, prec }
    }
}

impl From<Gaussian> for M {
    fn from(g: Gaussian) -> M {
        match g {
            Gaussian::M(m) => m,
            Gaussian::E(e) => e.into(),
        }
    }
}

impl From<Gaussian> for E {
    fn from(g: Gaussian) -> E {
        match g {
            Gaussian::M(m) => m.into(),
            Gaussian::E(e) => e,
        }
    }
}

impl<'a> ::std::ops::Mul<&'a E> for E {
    type Output = Self;
    fn mul(mut self, rhs: &'a E) -> Self {
        self.ab += &rhs.ab;
        self.prec += &rhs.prec;
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
