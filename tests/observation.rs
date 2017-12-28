
extern crate dars;
#[macro_use]
extern crate ndarray_linalg;

use dars::observation::*;

#[test]
fn linear_normal() {
    let n = 10;
    let obs = LinearNormal::random(n);
}
