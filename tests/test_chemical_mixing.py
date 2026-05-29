"""Tests for mixing and separation exergy."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eie.chemical.mixing import (
    SeparationTask,
    co2_capture_min_work_j_per_kg_co2,
    desalination_min_work_j_per_kg_product,
    mixing_exergy_j_per_mol,
    separation_work_j_per_mol_feed,
    separation_work_to_target_j_per_mol_product,
)
from eie.core.errors import DomainError


def test_mixing_exergy_is_non_positive():
    # Mixing always releases exergy (or zero for pure components)
    x = mixing_exergy_j_per_mol([0.5, 0.5])
    assert x <= 0


def test_mixing_exergy_pure_component_is_zero():
    x = mixing_exergy_j_per_mol([1.0])
    assert x == pytest.approx(0.0, abs=1e-12)


def test_mixing_exergy_two_components_equimolar():
    # R*T0*2*(0.5*ln(0.5)) = R*T0*ln(0.5)
    x = mixing_exergy_j_per_mol([0.5, 0.5])
    from eie.chemical.reference_environment import R_J_MOL_K, T0_K
    from math import log
    expected = R_J_MOL_K * T0_K * log(0.5)
    assert x == pytest.approx(expected, rel=1e-9)


def test_separation_work_is_negative_of_mixing_exergy():
    fracs = [0.3, 0.4, 0.3]
    mix = mixing_exergy_j_per_mol(fracs)
    sep = separation_work_j_per_mol_feed(fracs)
    assert sep == pytest.approx(-mix, rel=1e-9)
    assert sep >= 0


def test_separation_work_increases_with_purity():
    # Separating to 90% is harder than separating to 50%
    feed = [0.1, 0.9]
    sep_90 = separation_work_to_target_j_per_mol_product(feed, [0.9, 0.1])
    sep_50 = separation_work_to_target_j_per_mol_product(feed, [0.5, 0.5])
    assert sep_90 > sep_50


def test_separation_to_same_as_feed_requires_no_work():
    feed = [0.3, 0.7]
    sep = separation_work_to_target_j_per_mol_product(feed, feed)
    assert sep == pytest.approx(0.0, abs=1e-9)


def test_separation_with_mismatched_lengths_raises():
    with pytest.raises(DomainError, match="equal length"):
        separation_work_to_target_j_per_mol_product([0.5, 0.5], [0.5, 0.3, 0.2])


def test_separation_when_feed_has_zero_fraction_raises():
    # Cannot separate a component that is absent from feed
    with pytest.raises(DomainError, match="infinite work"):
        separation_work_to_target_j_per_mol_product([0.0, 1.0], [0.5, 0.5])


def test_desalination_min_work_positive():
    w = desalination_min_work_j_per_kg_product()
    assert w > 0


def test_desalination_min_work_in_physical_range():
    # RO minimum work for seawater: ~1–2 kJ/kg (real is 3–5 kJ/kg)
    w = desalination_min_work_j_per_kg_product(
        feed_salinity_g_kg=35.0, recovery_fraction=0.45
    )
    # Allow broad range since our model is simplified
    assert 500 < w < 10_000  # J/kg


def test_desalination_higher_recovery_requires_more_work():
    w_low = desalination_min_work_j_per_kg_product(recovery_fraction=0.3)
    w_high = desalination_min_work_j_per_kg_product(recovery_fraction=0.7)
    assert w_high > w_low


def test_desalination_recovery_at_or_above_one_raises():
    with pytest.raises(DomainError):
        desalination_min_work_j_per_kg_product(recovery_fraction=1.0)


def test_co2_capture_min_work_positive():
    w = co2_capture_min_work_j_per_kg_co2()
    assert w > 0


def test_co2_capture_min_work_less_than_real_plants():
    # Real MEA plants: ~3–4 MJ/kg_CO2; minimum should be well below
    w = co2_capture_min_work_j_per_kg_co2(
        flue_gas_co2_mole_fraction=0.15,
        capture_fraction=0.90,
    )
    assert w < 2_000_000  # J/kg


def test_co2_capture_more_concentrated_feed_requires_less_work():
    w_dilute = co2_capture_min_work_j_per_kg_co2(flue_gas_co2_mole_fraction=0.05)
    w_rich = co2_capture_min_work_j_per_kg_co2(flue_gas_co2_mole_fraction=0.30)
    assert w_dilute > w_rich


def test_co2_capture_invalid_concentration_raises():
    with pytest.raises(DomainError):
        co2_capture_min_work_j_per_kg_co2(flue_gas_co2_mole_fraction=0.0)
    with pytest.raises(DomainError):
        co2_capture_min_work_j_per_kg_co2(flue_gas_co2_mole_fraction=1.0)


def test_separation_task_validates():
    task = SeparationTask(
        name="air-separation",
        feed_mole_fractions=[0.21, 0.79],
        product_mole_fractions=[0.95, 0.05],
        component_names=["O2", "N2"],
    )
    work_product = task.minimum_work_j_per_mol_product()
    work_feed = task.minimum_work_j_per_mol_feed()
    assert work_product > 0
    assert work_feed > 0


def test_separation_task_invalid_names_length():
    with pytest.raises(DomainError, match="component_names"):
        SeparationTask(
            name="bad",
            feed_mole_fractions=[0.5, 0.5],
            product_mole_fractions=[0.9, 0.1],
            component_names=["only_one"],
        )


def test_mixing_exergy_rejects_fractions_not_summing_to_one():
    with pytest.raises(DomainError, match="sum to"):
        mixing_exergy_j_per_mol([0.3, 0.3])


# ── Property-based ───────────────────────────────────────────────────────────

@settings(deadline=None, max_examples=100)
@given(
    x1=st.floats(min_value=0.01, max_value=0.99),
)
def test_mixing_exergy_two_component_is_always_non_positive(x1):
    x = mixing_exergy_j_per_mol([x1, 1.0 - x1])
    assert x <= 1.0e-12  # allow tiny positive due to float arithmetic


@settings(deadline=None, max_examples=100)
@given(
    x1=st.floats(min_value=0.01, max_value=0.99),
)
def test_separation_work_is_non_negative(x1):
    sep = separation_work_j_per_mol_feed([x1, 1.0 - x1])
    assert sep >= -1.0e-12
