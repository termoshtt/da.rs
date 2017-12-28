
extern crate dars;

use dars::*;
use dars::observation::*;
use dars::ensemble::*;

#[test]
fn size_random() {
    let state_size = 4;
    let obs_size = 3;
    let op = LinearNormal::random(state_size, obs_size);
    assert_eq!(op.state_size(), state_size, "State size mismatch");
    assert_eq!(op.obs_size(), obs_size, "State size mismatch");

    let state = State::random(state_size);
    let obs = op.no_noise(&state);
    assert_eq!(obs.len(), obs_size, "Size of observation is mismatch");
}

#[test]
fn size_increment() {
    let state_size = 4;
    let obs_size = 3;
    let op = LinearNormal::random(state_size, obs_size);
    let obs = Obs::random(obs_size);
    let e = op.increment(&obs);
    assert_eq!(e.size(), state_size, "Size of increment is invalid");
}

#[test]
fn size_et_increment() {
    let state_size = 20;
    let obs_size = 3;
    let ens_size = 10;
    let ens = Ensemble::random(ens_size, state_size);
    let op = LinearNormal::random(state_size, obs_size);
    let obs = Obs::random(obs_size);
    let e = op.et_increment(&ens, &obs);
    assert_eq!(e.size(), ens_size, "Size of et_increment is invalid");
}
