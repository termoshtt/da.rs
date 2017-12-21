
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

pub type R = f64;
pub type C = Complex<f64>;

#[derive(Debug, Clone, NewType)]
pub struct State(Array1<R>);

#[derive(Debug, Clone, NewType)]
pub struct Obs(Array1<R>);

pub trait Analysis {
    type PDF;
    fn analysis(&self, &mut Self::PDF, &Obs);
}

pub trait Forecast {
    type PDF;
    fn forecast(&self, &mut Self::PDF);
}
