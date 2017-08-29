
extern crate dars;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use dars::gaussian::*;
use ndarray_linalg::*;

#[test]
fn merge_e() {
    let g1: E = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into();
    let g2: E = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into();
    let g3e = &g1 * &g2;
    println!("g3E = {:?}", &g3e);
    let g3m: M = g3e.into();
    println!("g3M = {:?}", &g3m);
    assert_close_l2!(&g3m.center, &array![1.0, 0.0], 1e-7);
    assert_close_l2!(&g3m.cov, &array![[0.5, 0.0], [0.0, 0.5]], 1e-7);
}

#[test]
fn merge_gaussian() {
    let g1 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]);
    let g2 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]);
    let mut g3 = &g1 * &g2;
    println!("g3(E) = {:?}", &g3);
    g3.as_m();
    println!("g3(M) = {:?}", &g3);
    assert_close_l2!(&g3.center(), &array![1.0, 0.0], 1e-7);
    assert_close_l2!(&g3.cov(), &array![[0.5, 0.0], [0.0, 0.5]], 1e-7);
}
