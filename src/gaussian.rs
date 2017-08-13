use super::types::*;
use ndarray::*;

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
