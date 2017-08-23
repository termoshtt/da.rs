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
    pub center: Array1<R>,
    pub cov: Array2<R>,
}

/// e-parameter as an exponential family
#[derive(Debug, Clone)]
pub struct E {
    pub ab: Array1<R>,
    pub prec: Array2<R>,
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
        g.into_m()
    }
}

impl From<Gaussian> for E {
    fn from(g: Gaussian) -> E {
        g.into_e()
    }
}

impl Into<Gaussian> for M {
    fn into(self) -> Gaussian {
        Gaussian::M(self)
    }
}

impl Into<Gaussian> for E {
    fn into(self) -> Gaussian {
        Gaussian::E(self)
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
