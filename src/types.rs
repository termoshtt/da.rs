use ndarray::*;
use num_complex::Complex;

pub type R = f64;
pub type C = Complex<f64>;

#[derive(Debug, Clone, NewType)]
pub struct State(Array1<R>);

#[derive(Debug, Clone, NewType)]
pub struct Obs(Array1<R>);
