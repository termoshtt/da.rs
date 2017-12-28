
extern crate dars;

use dars::*;
use dars::ensemble::*;
use dars::observation::*;

#[test]
fn size_random() {
    let state_size: StateSize = 4.into();
    let obs_size: ObsSize = 3.into();
    let op = LinearNormal::random(state_size, obs_size);
    assert_eq!(op.state_size(), state_size, "State size mismatch");
    assert_eq!(op.obs_size(), obs_size, "State size mismatch");

    let state = State::random(state_size);
    let obs = op.no_noise(&state);
    assert_eq!(obs.size(), obs_size, "Size of observation is mismatch");
}

#[test]
fn size_increment() {
    let state_size: StateSize = 4.into();
    let obs_size: ObsSize = 3.into();
    let op = LinearNormal::random(state_size, obs_size);
    let obs = Obs::random(obs_size);
    let e = op.increment(&obs);
    assert_eq!(e.size(), state_size, "Size of increment is invalid");
}

#[test]
fn size_et_increment() {
    let state_size: StateSize = 20.into();
    let obs_size: ObsSize = 3.into();
    let ens_size: EnsembleSize = 10.into();
    let ens = Ensemble::random(ens_size, state_size);
    let op = LinearNormal::random(state_size, obs_size);
    let obs = Obs::random(obs_size);
    let e = op.et_increment(&ens, &obs);
    assert_eq!(e.size(), ens_size, "Size of et_increment is invalid");
}
