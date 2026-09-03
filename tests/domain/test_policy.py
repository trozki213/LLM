"""FitPolicy is merchant-tunable. Its invariants stop an incoherent policy shipping."""
import pytest

from fitkit.domain.policy import EaseWindow, FitPolicy, FitPreference, Tone
from fitkit.domain.regions import BodyRegion

WEIGHTS = {BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8, BodyRegion.THIGH: 0.4}


def policy(**overrides) -> FitPolicy:
    fields = dict(
        policy_id="policy/merchant-a",
        version=3,
        tau_single=0.65,
        tau_pair=0.85,
        max_critical_sigma_cm=2.5,
        region_weights=WEIGHTS,
        critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
        tightness_penalty_ratio=1.8,
    )
    fields.update(overrides)
    return FitPolicy(**fields)


class TestFitPolicy:
    def test_version_key_identifies_the_exact_policy(self):
        assert policy().version_key == "policy/merchant-a/3"

    @pytest.mark.parametrize("bad", [0.0, 1.01, -0.2])
    def test_tau_single_is_a_probability(self, bad):
        with pytest.raises(ValueError, match="tau_single"):
            policy(tau_single=bad)

    def test_pair_threshold_cannot_be_easier_than_the_single_threshold(self):
        with pytest.raises(ValueError, match="tau_pair"):
            policy(tau_pair=0.5)

    def test_critical_regions_must_be_weighted(self):
        with pytest.raises(ValueError, match="critical"):
            policy(critical_regions=frozenset({BodyRegion.BUST}))

    def test_rejects_an_empty_weight_set(self):
        with pytest.raises(ValueError, match="region_weights"):
            policy(region_weights={})

    def test_rejects_non_positive_weights(self):
        with pytest.raises(ValueError, match="region_weights"):
            policy(region_weights={BodyRegion.WAIST: 0.0})

    def test_tightness_penalty_must_not_favour_tightness(self):
        """Too tight is worse than too loose. A ratio below 1 inverts that."""
        with pytest.raises(ValueError, match="tightness_penalty_ratio"):
            policy(tightness_penalty_ratio=0.9)

    def test_rejects_a_non_positive_sigma_ceiling(self):
        with pytest.raises(ValueError, match="max_critical_sigma_cm"):
            policy(max_critical_sigma_cm=0.0)

    def test_weights_are_immutable(self):
        with pytest.raises(TypeError):
            policy().region_weights[BodyRegion.BUST] = 1.0  # type: ignore[index]


class TestEaseWindow:
    def test_bounds_must_be_ordered(self):
        with pytest.raises(ValueError, match="ordered"):
            EaseWindow(min_cm=3.0, preferred_cm=2.0, max_cm=5.0)

    def test_max_must_not_precede_preferred(self):
        with pytest.raises(ValueError, match="ordered"):
            EaseWindow(min_cm=1.0, preferred_cm=4.0, max_cm=3.0)

    def test_negative_ease_is_allowed_for_stretch_garments(self):
        assert EaseWindow(min_cm=-2.0, preferred_cm=0.0, max_cm=2.0).min_cm == -2.0

    def test_contains_reports_membership(self):
        window = EaseWindow(min_cm=1.0, preferred_cm=2.0, max_cm=5.0)
        assert window.contains(2.0)
        assert not window.contains(0.5)
        assert not window.contains(5.5)


class TestPreferenceVocabulary:
    def test_preference_is_a_closed_three_way_choice(self):
        assert set(FitPreference) == {
            FitPreference.TIGHTER,
            FitPreference.AS_DESIGNED,
            FitPreference.LOOSER,
        }

    def test_tone_is_a_render_hint_not_a_decision(self):
        assert Tone.NEUTRAL in set(Tone)
