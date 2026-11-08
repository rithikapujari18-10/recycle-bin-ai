from recyclebin.stats import wilson_ci, two_proportion_z


def test_wilson_matches_paper():
    lo, hi = wilson_ci(317, 380)
    assert abs(lo - 0.7935) < 1e-3 and abs(hi - 0.8682) < 1e-3


def test_two_proportion_z():
    z, p = two_proportion_z(317, 380, 201, 380)
    assert 8.9 < z < 9.2 and p < 1e-3
