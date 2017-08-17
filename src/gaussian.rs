use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

pub enum Gaussian {
    M(M),
    E(E),
}

impl Gaussian {
    pub fn from_mean(center: Array1<R>, cov: Array2<R>) -> Self {
        Gaussian::M(M { center, cov })
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

fn solve_prec(p: &Array2<R>, x: Array1<R>) -> Array1<R> {
    // FIXME `solve` uses LU decomposition
    // This code should be use Cholesky decomposition,
    // but corresponding interface is absent in ndarray-linalg
    let f: Factorized<OwnedRepr<R>> = p.factorize().unwrap();
    f.solve(Transpose::No, x).unwrap()
}

impl<'a> ::std::ops::Mul<&'a E> for E {
    type Output = Self;
    fn mul(mut self, rhs: &'a E) -> Self {
        self.ab += &rhs.ab;
        self.prec += &rhs.prec;
        self
    }
}
