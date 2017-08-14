
extern crate ndarray;
extern crate da;

use ndarray::*;

#[test]
fn size() {
    let n = 2; // dimension of each state
    let m = 10; // ensemble size
    let x0 = arr1(&[1.0, 2.0]);
    let xs = da::ensemble::Ensemble::isotropic_gaussian(&x0, m, 1.0);
    assert_eq!(xs.dim(), n);
    assert_eq!(xs.size(), m);

    let g = xs.as_gaussian();
    assert_eq!(g.center.shape(), [n]);
    assert_eq!(g.precision.shape(), [n, n]);
}

#[test]
fn ensemble_iter() {
    let n = 2; // dimension of each state
    let m = 10; // ensemble size
    let x0 = arr1(&[1.0, 2.0]);
    let mut xs = da::ensemble::Ensemble::isotropic_gaussian(&x0, m, 1.0);
    for v in xs.ens_iter() {
        assert_eq!(v.len(), n);
    }
    for v in xs.ens_iter_mut() {
        assert_eq!(v.len(), n);
    }
}
