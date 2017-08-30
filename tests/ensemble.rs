
extern crate dars;
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use dars::ensemble::*;
use dars::gaussian::*;
use dars::types::*;
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

#[test]
fn ensemble_transform_() {
    let k = 6;
    let c: Array1<R> = random(3);
    let h: Array2<R> = random((2, 3));
    let g = Gaussian::from_mean(random(2), Array::eye(2));
    let ens = Ensemble::isotropic_gaussian(&c, k, 0.1);
    let pg = PGaussian::new(h, g);
    assert_eq!(pg.size(), 3);
    let pg_t = ensemble_transform(&ens, pg);
    assert_eq!(pg_t.size(), 6);
    let g = pg_t.reduction();
    assert_eq!(g.size(), 6);
}

#[test]
fn ssqrt_sampling_() {
    let u = array![[1.0, -1.0, 0.0], [1.0, 1.0, -2.0], [1.0, 1.0, 1.0]];
    let a = u.t().dot(&u) / 3.0;
    let v = array![1.0, 1.0, 1.0];
    println!("av = {:?}", a.dot(&v));
    let c = random(3);
    println!("c = {:?}", &c);
    let m: M = Gaussian::from_mean(c.clone(), a).into();
    let w = ssqrt_sampling(&m);
    println!("w = {:?}", &w);
    let d = w.center() - &c;
    println!("w.center - c = {:?}", d);
    assert_close_l2!(&d, &(d[0] * v), 1e-7);
}
