
extern crate dars;
extern crate ndarray_linalg;

use dars::observation::*;
use ndarray_linalg::*;

#[test]
fn size_random() {
    let state_size = 4;
    let obs_size = 3;
    let op = LinearNormal::random(state_size, obs_size);
    assert_eq!(op.state_size(), state_size, "State size mismatch");
    assert_eq!(op.obs_size(), obs_size, "State size mismatch");
}

#[test]
fn size_increment() {
    let state_size = 4;
    let obs_size = 3;
    let op = LinearNormal::random(state_size, obs_size);
    let obs = random(obs_size).into();
    let e = op.increment(&obs);
    assert_eq!(e.size(), state_size, "Size of increment is invalid");
}
