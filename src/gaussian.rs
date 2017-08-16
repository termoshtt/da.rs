use super::types::*;
use ndarray::*;
use ndarray_linalg::*;

#[derive(Debug, Clone)]
pub struct Gaussian {
    pub center: Array<R, Ix1>,
    pub precision: Array<R, Ix2>,
}

#[derive(Debug, Clone)]
pub struct ProjectedGaussian {
    pub center: Array<R, Ix1>,
    pub precision: Array<R, Ix2>,
    pub p: Array<R, Ix2>,
}

pub fn merge(a: &Gaussian, b: &Gaussian) -> Gaussian {
    let p = &a.precision + &b.precision;
    let c = a.precision.dot(&a.center) + b.precision.dot(&b.center);
    let f: Factorized<OwnedRepr<R>> = p.factorize().unwrap();
    let c = f.solve(Transpose::No, c).unwrap(); // FIXME `solve` uses LU decomposition
    Gaussian {
        center: c,
        precision: p,
    }
}
