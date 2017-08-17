
extern crate da;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use da::ensemble::*;
use da::gaussian::*;
use ndarray_linalg::*;

#[test]
fn merge_e() {
    let g1 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into_e();
    let g2 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into_e();
    let g3e = &g1 * &g2;
    println!("g3E = {:?}", &g3e);
    let g3m = g3e.into_m();
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

#[test]
fn ssqrt() {
    let c = random(3);
    let cov = random_hpd(3);
    let m = Gaussian::from_mean(c, cov).into_m();
    println!("m.center = {:?}", &m.center);
    println!("m.cov = \n{:?}", &m.cov);
    let xs: Ensemble = m.ssqrt().into();
    println!("xs = {:?}", &xs);
    let m2 = xs.as_gaussian().into_m();
    println!("m2.center = {:?}", &m2.center);
    println!("m2.cov = \n{:?}", &m2.cov);
    assert_close_l2!(&m2.center, &m.center, 1e-7);
    assert_close_l2!(&m2.cov, &m.cov, 1e-7);
}
