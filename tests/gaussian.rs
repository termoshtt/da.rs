
extern crate dars;
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use dars::gaussian::*;

#[test]
fn m2e2m() {
    let n = 5;
    let m = M::random(n);
    let e = m.to_e();
    let m2 = e.to_m();
    assert_close_l2!(&m.center, &m2.center, 1e-7);
    assert_close_l2!(&m.cov, &m2.cov, 1e-7);
}

#[test]
fn e2m2e() {
    let n = 5;
    let e = E::random(n);
    let m = e.to_m();
    let e2 = m.to_e();
    assert_close_l2!(&e.prec, &e2.prec, 1e-7);
    assert_close_l2!(&e.ab, &e2.ab, 1e-7);
}
