from rung_registration import RungRegistrationV1


def _reg(**kw):
    base = dict(
        rung_id="vfull",
        primary_estimator="ResidualAffectProbeV1",
        feature_blocks=("hdc64", "base_pre_nudge"),
        baselines=("persistence", "steelman_ridge", "steelman_gbm"),
    )
    base.update(kw)
    return RungRegistrationV1(**base)


def test_digest_is_deterministic():
    assert _reg().digest() == _reg().digest()


def test_post_hoc_change_detected():
    a = _reg()
    b = _reg(rel_floor=0.10)  # a moved goalpost
    assert a.digest() != b.digest()
    # feature-set change also changes the digest
    c = _reg(feature_blocks=("hdc64",))
    assert a.digest() != c.digest()


def test_seed_binds_to_digest():
    a = _reg()
    b = _reg(abs_floor=0.05)
    assert a.episode_seed() != b.episode_seed()  # split changes if the registration changes
    assert a.episode_seed() == a.episode_seed()  # but is stable for a fixed registration


def test_single_primary_estimator():
    assert _reg().primary_estimator == "ResidualAffectProbeV1"  # one estimator, not a shopping list
