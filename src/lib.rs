
#[macro_use]
extern crate procedurals;
#[macro_use]
extern crate derive_new;

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

pub trait Size: Copy + Eq + Ord + Into<usize> + From<usize> {}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, NewType, new)]
pub struct StateSize(usize);
impl Size for StateSize {}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, NewType, new)]
pub struct ObsSize(usize);
impl Size for ObsSize {}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, NewType, new)]
pub struct EnsembleSize(usize);
impl Size for EnsembleSize {}

/// State of the simulation
#[derive(Debug, Clone, NewType, new)]
pub struct State(Array1<R>);

impl State {
    /// Generate random state
    pub fn random(n: StateSize) -> Self {
        ndarray_linalg::random(n.0).into()
    }

    pub fn size(&self) -> StateSize {
        self.len().into()
    }
}

/// Obverstion
#[derive(Debug, Clone, NewType, new)]
pub struct Obs(Array1<R>);

impl Obs {
    /// Generate random observation
    pub fn random(n: ObsSize) -> Self {
        ndarray_linalg::random(n.0).into()
    }

    pub fn size(&self) -> ObsSize {
        self.len().into()
    }
}
