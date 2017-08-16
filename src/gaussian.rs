use ndarray::*;
use ndarray_linalg::*;

use super::types::{R, State};

/// Precision matrix type
pub type Precision = Array<R, Ix2>;
/// Matrix for projection
pub type Projection = Array<R, Ix2>;

#[derive(Debug, Clone)]
pub struct Gaussian {
    pub center: State,
    pub prec: Precision,
}

#[derive(Debug, Clone)]
pub struct ProjectedGaussian {
    pub center: State,
    pub prec: Precision,
    pub p: Projection,
}

/// Probability Density Function
pub trait PDF {
    /// dimension of state
    fn dim(&self) -> usize;
    /// relative probability of state (not normalized)
    fn prob(&State) -> f64;
    /// log of relative probability of state (not normalized)
    fn log_prob(&State) -> f64;
}

fn solve_prec(p: &Precision, x: State) -> State {
    // FIXME `solve` uses LU decomposition
    // This code should be use Cholesky decomposition,
    // but corresponding interface is absent in ndarray-linalg
    let f: Factorized<OwnedRepr<R>> = p.factorize().unwrap();
    f.solve(Transpose::No, x).unwrap()
}

impl<'a> ::std::ops::Mul<&'a Gaussian> for Gaussian {
    type Output = Self;
    fn mul(mut self, rhs: &'a Gaussian) -> Self {
        let c = self.prec.dot(&self.center) + rhs.prec.dot(&rhs.center);
        self.prec += &rhs.prec;
        self.center = solve_prec(&self.prec, c);
        self
    }
}
