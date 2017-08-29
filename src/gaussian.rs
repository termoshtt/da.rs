use ndarray::*;
use ndarray_linalg::*;

use super::types::R;

/// Gaussian as an exponential family distribution
///
/// Two forms of Gaussian are implemented:
///
/// - natural (m-) parameter, i.e. mean and covariance matrix
/// - e-parameter for calculating the multiplication of two Gaussian
///
/// Each forms totally represents Gaussian, and can be converted each other
/// Conversion cost is in the order of `O(N^2)` since it calculates the inverse matrix.
/// You will find more knowledge in textbooks of information-geometory.
#[derive(Debug, Clone, IntoEnum)]
pub enum Gaussian {
    M(M),
    E(E),
}

impl Gaussian {
    pub fn from_mean(center: Array1<R>, cov: Array2<R>) -> Self {
        Gaussian::M(M { center, cov })
    }

    pub fn size(&self) -> usize {
        match *self {
            Gaussian::M(ref m) => m.center.len(),
            Gaussian::E(ref e) => e.ab.len(),
        }
    }

    /// Get the center of Gaussian
    ///
    /// if the Gaussian is in E form, it is recalculated.
    pub fn center(&self) -> Array1<R> {
        match *self {
            Gaussian::M(ref m) => m.center.clone(),
            Gaussian::E(ref e) => e.prec.solveh(&e.ab).expect("Precision matrix is singular"),
        }
    }

    /// Get the covariance matrix of Gaussian
    ///
    /// if the Gaussian is in E form, it is recalculated.
    pub fn cov(&self) -> Array2<R> {
        match *self {
            Gaussian::M(ref m) => m.cov.clone(),
            Gaussian::E(ref e) => e.prec.invh().expect("Precision matrix is singular"),
        }
    }

    /// Force to m-parameter
    pub fn as_m<'a>(&'a mut self) -> &'a mut Self {
        match *self {
            Gaussian::M(_) => return self,
            Gaussian::E(_) => {}
        }
        let tmp = E {
            ab: arr1(&[]),
            prec: arr2(&[[]]),
        }.into();
        let m: M = ::std::mem::replace(self, tmp).into();
        ::std::mem::replace(self, m.into());
        self
    }

    /// Force to e-parameter
    pub fn as_e<'a>(&'a mut self) -> &'a mut Self {
        match *self {
            Gaussian::E(_) => return self,
            Gaussian::M(_) => {}
        }
        let tmp = M {
            center: arr1(&[]),
            cov: arr2(&[[]]),
        }.into();
        let e: E = ::std::mem::replace(self, tmp).into();
        ::std::mem::replace(self, e.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct PGaussian {
    pub projection: Array2<R>,
    pub gaussian: Gaussian,
}

impl PGaussian {
    pub fn size(&self) -> usize {
        self.projection.cols()
    }

    pub fn reduction(self) -> Gaussian {
        assert!(
            self.size() <= self.gaussian.size(),
            "Upward reduction is prohibited"
        );
        let e: E = self.gaussian.into();
        let ab = self.projection.t().dot(&e.ab);
        let prec = self.projection.t().dot(&e.prec).dot(&self.projection);
        E { ab, prec }.into()
    }
}

/// natural (m-) parameter as an exponential family
#[derive(Debug, Clone)]
pub struct M {
    pub center: Array1<R>,
    pub cov: Array2<R>,
}

/// e-parameter as an exponential family
#[derive(Debug, Clone)]
pub struct E {
    pub ab: Array1<R>,
    pub prec: Array2<R>,
}

impl<'a> ::std::ops::Mul<&'a E> for E {
    type Output = Self;
    fn mul(mut self, rhs: &'a E) -> Self {
        self *= rhs;
        self
    }
}

impl<'a, 'b> ::std::ops::Mul<&'a E> for &'b E {
    type Output = E;
    fn mul(self, rhs: &'a E) -> E {
        let ab = &self.ab + &rhs.ab;
        let prec = &self.prec + &rhs.prec;
        E { ab, prec }
    }
}

impl<'a> ::std::ops::MulAssign<&'a E> for E {
    fn mul_assign(&mut self, rhs: &'a E) {
        self.ab += &rhs.ab;
        self.prec += &rhs.prec;
    }
}

impl<'a> ::std::ops::Mul<&'a Gaussian> for Gaussian {
    type Output = Self;
    fn mul(self, rhs: &'a Gaussian) -> Self {
        let self_e: E = self.into();
        match *rhs {
            Gaussian::M(ref m) => (self_e * &m.clone().into()).into(),
            Gaussian::E(ref e) => (self_e * &e).into(),
        }
    }
}

impl<'a, 'b> ::std::ops::Mul<&'a Gaussian> for &'b Gaussian {
    type Output = Gaussian;
    fn mul(self, rhs: &'a Gaussian) -> Gaussian {
        self.clone() * rhs
    }
}

impl<'a> ::std::ops::MulAssign<&'a Gaussian> for Gaussian {
    fn mul_assign(&mut self, rhs: &'a Gaussian) {
        self.as_e();
        match *self {
            Gaussian::M(_) => unreachable!(),
            Gaussian::E(ref mut e) => {
                match *rhs {
                    Gaussian::M(ref m_) => *e *= &m_.clone().into(),
                    Gaussian::E(ref e_) => *e *= e_,
                };
            }
        }
    }
}

impl From<E> for M {
    fn from(e: E) -> Self {
        let cov = e.prec.invh_into().expect("Precision matrix is singular");
        let center = cov.dot(&e.ab);
        M { center, cov }
    }
}

impl From<M> for E {
    fn from(m: M) -> Self {
        let prec = m.cov.invh_into().expect("Covariance matrix is singular");
        let ab = prec.dot(&m.center);
        E { ab, prec }
    }
}

impl From<Gaussian> for M {
    fn from(g: Gaussian) -> M {
        match g {
            Gaussian::M(m) => m,
            Gaussian::E(e) => e.into(),
        }
    }
}

impl From<Gaussian> for E {
    fn from(g: Gaussian) -> E {
        match g {
            Gaussian::M(m) => m.into(),
            Gaussian::E(e) => e,
        }
    }
}
