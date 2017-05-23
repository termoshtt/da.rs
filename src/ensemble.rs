
use ndarray::*;
use super::types::*;

#[derive(Debug, Clone)]
pub struct Ensemble<S>(ArrayBase<S, Ix2>) where S: DataClone<Elem = R>;
