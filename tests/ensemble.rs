
extern crate dars;
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use dars::ensemble::*;
use ndarray::*;
use ndarray_linalg::*;

#[test]
fn size() {
    let n = 2; // dimension of each state
    let m = 10; // ensemble size
    let x0 = arr1(&[1.0, 2.0]);
    let xs = Ensemble::isotropic_gaussian(&x0, m, 1.0);
    assert_eq!(xs.dim(), n);
    assert_eq!(xs.size(), m);

    let g: dars::gaussian::M = xs.as_gaussian().into();
    assert_eq!(g.center.shape(), [n]);
    assert_eq!(g.cov.shape(), [n, n]);
}

#[test]
fn ensemble_iter() {
    let n = 2; // dimension of each state
    let m = 10; // ensemble size
    let x0 = arr1(&[1.0, 2.0]);
    let mut xs = Ensemble::isotropic_gaussian(&x0, m, 1.0);
    for v in xs.ens_iter() {
        assert_eq!(v.len(), n);
    }
    for v in xs.ens_iter_mut() {
        assert_eq!(v.len(), n);
    }
}

#[test]
fn transform() {
    let _n = 2; // dimension of each state
    let m = 10; // ensemble size
    let x0 = arr1(&[1.0, 2.0]);
    let xs = Ensemble::isotropic_gaussian(&x0, m, 1.0);
    let w = Weights::trivial(m);
    let xs_new = w.transform(&xs);
    assert_close_l2!(&xs_new, &xs, 1e-7);
}
