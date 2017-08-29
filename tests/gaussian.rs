
extern crate dars;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use dars::gaussian::*;
use dars::types::*;

use ndarray::*;
use ndarray_linalg::*;

fn center() -> Array1<R> {
    array![1.0, 0.0]
}

fn cov() -> Array2<R> {
    Array::eye(2)
}

mod e {
    use super::*;

    fn g2e() -> E {
        Gaussian::from_mean(center(), cov()).into()
    }

    #[test]
    fn merge() {
        let g1 = g2e();
        let g2 = g2e();
        let g3e = &g1 * &g2;
        println!("g3E = {:?}", &g3e);
        let g3m: M = g3e.into();
        println!("g3M = {:?}", &g3m);
        assert_close_l2!(&g3m.center, &center(), 1e-7);
        assert_close_l2!(&g3m.cov, &(0.5 * cov()), 1e-7);
    }
}

mod gaussian {
    use super::*;

    pub fn g() -> Gaussian {
        Gaussian::from_mean(center(), cov())
    }

    #[test]
    fn merge() {
        let g1 = g();
        let g2 = g();
        let mut g3 = &g1 * &g2;
        println!("g3(E) = {:?}", &g3);
        g3.as_m();
        println!("g3(M) = {:?}", &g3);
        assert_close_l2!(&g3.center(), &center(), 1e-7);
        assert_close_l2!(&g3.cov(), &(0.5 * cov()), 1e-7);
    }
}

mod pgaussian {
    use super::*;

    fn pg_3to2() -> PGaussian {
        let h = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let g = gaussian::g();
        PGaussian {
            projection: h,
            gaussian: g,
        }
    }

    fn pg_2to3() -> PGaussian {
        let h = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let g = Gaussian::from_mean(random(3), random_hpd(3));
        PGaussian {
            projection: h,
            gaussian: g,
        }
    }

    #[test]
    fn size() {
        assert_eq!(pg_3to2().size(), 3);
        assert_eq!(pg_2to3().size(), 2);
    }

    #[should_panic]
    #[test]
    fn upward_reduction() {
        let _m: M = pg_3to2().reduction().into();
    }

    #[test]
    fn reduction() {
        let pg = pg_2to3();
        let m: M = pg.reduction().into(); // should not panic
        assert_eq!(m.size(), 2);
    }
}
