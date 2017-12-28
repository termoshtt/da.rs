
extern crate dars;
extern crate ndarray;
extern crate ndarray_linalg;

use dars::ensemble::*;

#[test]
fn shape() {
    let size = 10; // ensemble size
    let dim = 2; // dimension of each state
    let xs = Ensemble::random(size, dim);
    assert_eq!(xs.dim(), dim);
    assert_eq!(xs.size(), size);

    assert_eq!(xs.ens_iter().len(), size);
    for v in xs.ens_iter() {
        assert_eq!(v.len(), dim);
    }

    let g = xs.to_m();
    assert_eq!(g.center.shape(), [dim]);
    assert_eq!(g.cov.shape(), [dim, dim]);
}

#[test]
fn iter(){
    let size = 10; // ensemble size
    let dim = 2; // dimension of each state
    let xs = Ensemble::random(size, dim);
    assert_eq!(xs.ens_iter().len(), size);
    for v in xs.ens_iter() {
        assert_eq!(v.len(), dim);
    }
    let mut xs_mut = Ensemble::random(size, dim);
    assert_eq!(xs_mut.ens_iter().len(), size);
    for v in xs_mut.ens_iter_mut() {
        assert_eq!(v.len(), dim);
    }
}
