use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

/// Gaussian as an exponential family distribution
#[derive(Debug, Clone)]
pub enum Gaussian {
    M(M),
    E(E),
}

impl Gaussian {
    pub fn from_mean(center: Array1<R>, cov: Array2<R>) -> Self {
        Gaussian::M(M { center, cov })
    }

    pub fn into_m(self) -> M {
        match self {
            Gaussian::M(m) => m,
            Gaussian::E(e) => e.into(),
        }
    }

    pub fn into_e(self) -> E {
        match self {
            Gaussian::M(m) => m.into(),
            Gaussian::E(e) => e,
        }
    }
}

/// natural (m-) parameter as an exponential family
#[derive(Debug, Clone)]
pub struct M {
    center: Array1<R>,
    cov: Array2<R>,
}

/// e-parameter as an exponential family
#[derive(Debug, Clone)]
pub struct E {
    ab: Array1<R>,
    prec: Array2<R>,
}

impl From<E> for M {
    fn from(e: E) -> Self {
        let cov = e.prec.inv_into().unwrap();
        let center = cov.dot(&e.ab);
        M { center, cov }
    }
}

impl From<M> for E {
    fn from(m: M) -> Self {
        let prec = m.cov.inv_into().unwrap();
        let ab = prec.dot(&m.center);
        E { ab, prec }
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
