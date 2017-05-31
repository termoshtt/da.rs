
use ndarray::*;
use super::types::*;

#[derive(Debug, Clone, new)]
pub struct GaussianBase<S1, S2>
    where S1: DataClone<Elem = R>,
          S2: DataClone<Elem = R>
{
    pub mean: ArrayBase<S1, Ix1>,
    pub cov: ArrayBase<S2, Ix2>,
}

pub type Gaussian = GaussianBase<OwnedRepr<R>, OwnedRepr<R>>;

#[derive(Debug, Clone, new)]
pub struct ProjectedGaussian<S1, S2, S3>
    where S1: DataClone<Elem = R>,
          S2: DataClone<Elem = R>,
          S3: DataClone<Elem = R>
{
    pub mean: ArrayBase<S1, Ix1>,
    pub cov: ArrayBase<S2, Ix2>,
    pub projection: ArrayBase<S3, Ix2>,
}
