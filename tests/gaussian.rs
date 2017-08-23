
extern crate da;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use da::gaussian::*;
use ndarray_linalg::*;

#[test]
fn merge() {
    let g1 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into_e();
    let g2 = Gaussian::from_mean(array![1.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]).into_e();
    let g3e = &g1 * &g2;
    println!("g3E = {:?}", &g3e);
    let g3m = g3e.into_m();
    println!("g3M = {:?}", &g3m);
    assert_close_l2!(&g3m.center, &array![1.0, 0.0], 1e-7);
    assert_close_l2!(&g3m.cov, &array![[0.5, 0.0], [0.0, 0.5]], 1e-7);
}
