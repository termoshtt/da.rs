
extern crate dars;
extern crate ndarray;
extern crate ndarray_linalg;

use dars::*;
use dars::ensemble::*;

#[test]
fn shape() {
    let size = EnsembleSize::new(10);
    let dim = StateSize::new(3);
    let xs = Ensemble::random(size, dim);
    assert_eq!(xs.dim(), dim);
    assert_eq!(xs.size(), size);
}

#[test]
fn to_m() {
    let size = EnsembleSize::new(10);
    let dim = StateSize::new(3);
    let xs = Ensemble::random(size, dim);
    let g = xs.to_m();
    assert_eq!(g.center.shape(), [dim.into()]);
    assert_eq!(g.cov.shape(), [dim.into(), dim.into()]);
}

#[test]
fn shape_dx() {
    let size = EnsembleSize::new(10);
    let dim = StateSize::new(3);
    let xs = Ensemble::random(size, dim);
    let (c, dx) = xs.deviation();
    assert_eq!(c.len(), dim.into(), "size of state is invalid");
    assert_eq!(
        dx.shape(),
        [size.into(), dim.into()],
        "size of state is invalid"
    );
}

#[test]
fn iter() {
    let size = EnsembleSize::new(10);
    let dim = StateSize::new(3);
    let xs = Ensemble::random(size, dim);
    assert_eq!(
        xs.ens_iter().len(),
        size.into(),
        "Length of iterator mismatch"
    );
    for v in xs.ens_iter() {
        assert_eq!(v.len(), dim.into(), "Dim of ensemble mismatch");
    }
    let mut xs_mut = Ensemble::random(size, dim);
    assert_eq!(
        xs_mut.ens_iter().len(),
        size.into(),
        "Length of iterator mismatch"
    );
    for v in xs_mut.ens_iter_mut() {
        assert_eq!(v.len(), dim.into(), "Dim of ensemble mismatch");
    }
}
