
use rand::*;
use ndarray::*;
use super::types::*;

#[derive(Debug, Clone)]
pub struct WeightBase<S: DataClone<Elem = R>>(ArrayBase<S, Ix1>);

pub type Weight = WeightBase<OwnedRepr<R>>;

impl<S: DataClone<Elem = R>> From<ArrayBase<S, Ix1>> for WeightBase<S> {
    fn from(a: ArrayBase<S, Ix1>) -> Self {
        WeightBase(a)
    }
}

impl<S: DataClone<Elem = R>> ::std::ops::Deref for WeightBase<S> {
    type Target = ArrayBase<S, Ix1>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: DataClone<Elem = R>> ::std::ops::DerefMut for WeightBase<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S: DataClone<Elem = R>> WeightBase<S> {
    pub fn normalize(&mut self)
        where S: DataMut
    {
        let n = 1.0 / self.iter().sum::<f64>();
        for x in self.iter_mut() {
            *x *= n;
        }
    }

    pub fn normalized(mut self) -> Self
        where S: DataMut
    {
        self.normalize();
        self
    }

    pub fn uniform(n: usize) -> Self
        where S: DataOwned
    {
        WeightBase(ArrayBase::from_vec(vec![1.0/n as f64; n]))
    }

    pub fn random(n: usize) -> Self
        where S: DataOwned + DataMut
    {
        let mut rng = thread_rng();
        let w = WeightBase((0..n).map(|_| rng.gen_range(0.0, 1.0)).collect());
        w.normalized()
    }
}
