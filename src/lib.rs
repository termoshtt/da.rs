
#[macro_use]
extern crate procedurals;

extern crate num_complex;
extern crate ndarray;
extern crate ndarray_linalg;

pub mod gaussian;
pub mod ensemble;
pub mod observation;

use ndarray::*;
use num_complex::Complex;

/// Real number
pub type R = f64;
/// Complex number
pub type C = Complex<f64>;

/// State of the simulation
#[derive(Debug, Clone, NewType)]
pub struct State(Array1<R>);

impl State {
    /// Generate random state
    pub fn random(n: usize) -> Self {
        ndarray_linalg::random(n).into()
    }
}

/// Obverstion
#[derive(Debug, Clone, NewType)]
pub struct Obs(Array1<R>);

impl Obs {
    /// Generate random observation
    pub fn random(n: usize) -> Self {
        ndarray_linalg::random(n).into()
    }
}
