
#[macro_use]
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate dars;

use dars::*;
use dars::types::R;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_odeint::*;

fn main() {
    // lorenz63 setting
    let n = 3;
    let dt = 0.01;
    // ensemble setting
    let k = 6;
    // observation setting
    let m = 3;
    let tau = 8;
    let hm = Array2::<R>::eye(3); // observe all variables
    let r = 2.0;
    let rm = r * Array::eye(3);
    // DA setting
    let count = 10000;

    // Generate initial state and ensemble
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let x0 = time_series(array![1.0, 0.0, 0.0], &teo)
        .take(1000)
        .last()
        .unwrap();
    let xs = ensemble::Ensemble::isotropic_gaussian(&x0, k, r);

    // truth and observation
    let teo = adaptor::nstep(teo, tau);
    let truth: Vec<_> = time_series(x0, &teo).take(count).collect();
    let obs: Vec<Array1<R>> = truth.iter().map(|x| r.sqrt() * random(m) + x).collect();

    // execute DA
    for (t, o) in truth.into_iter().zip(obs.into_iter()) {
        //
    }
}
