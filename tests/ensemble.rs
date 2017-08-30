
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
    let v = array![1.0, 1.0, 1.0];
    let v1 = array![1.0, -1.0, 0.0];
    let v2 = array![1.0, 1.0, -2.0];
    let u = array![[1.0, -1.0, 0.0], [1.0, 1.0, -2.0], [1.0, 1.0, 1.0]];

    let a = u.t().dot(&u) / 3.0;
    println!("a = {:?}", &a);
    println!("av = {:?}", a.dot(&v));
    assert_close_l2!(&a.dot(&v), &v, 1e-7; "v is 1-eigenvector of a");

    let c = random(3);
    let m: M = Gaussian::from_mean(c.clone(), a.clone()).into();
    println!("Gaussian = \n{:?}", &m);

    let w = ssqrt_sampling(&m);
    println!("w = {:?}", &w);
    // check sum of weight is one
    println!("sum of weights = {:?}", w.sum(Axis(1)));
    assert_close_l2!(&w.sum(Axis(1)), &v, 1e-7; "weight condition");
    // check center
    let d = w.center() - &c;
    println!("w.center - c = {:?}", d);
    assert_close_l2!(&d, &(d[0] * &v), 1e-7);
    // check covariance matrix
    let wc = w.stat().1;
    println!("w.cov = \n{:?}", &wc);
    assert_close_max!(&wc.dot(&v), &Array::zeros(3), 1e-7; "v direction is dropped");
    assert_close_l2!(&wc.dot(&v1), &a.dot(&v1), 1e-7; "check v1 direction");
    assert_close_l2!(&wc.dot(&v2), &a.dot(&v2), 1e-7; "check v2 direction");
}
