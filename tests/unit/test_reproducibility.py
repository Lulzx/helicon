"""Tests for helicon._reproducibility (spec §14)."""

from __future__ import annotations

from helicon._reproducibility import collect_metadata
from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    SimConfig,
)


def _make_config(**plasma_kwargs) -> SimConfig:
    return SimConfig(
        nozzle=NozzleConfig(
            coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
            domain=DomainConfig(z_min=0.0, z_max=1.0, r_max=0.5),
        ),
        plasma=PlasmaSourceConfig(
            n0=1e19,
            T_i_eV=100.0,
            T_e_eV=50.0,
            v_injection_ms=50000.0,
            **plasma_kwargs,
        ),
    )


class TestCollectMetadataBasics:
    def test_returns_dict(self) -> None:
        meta = collect_metadata()
        assert isinstance(meta, dict)

    def test_has_helicon_version(self) -> None:
        meta = collect_metadata()
        assert "helicon_version" in meta
        assert meta["helicon_version"] is not None

    def test_has_python_version(self) -> None:
        meta = collect_metadata()
        assert "python_version" in meta

    def test_has_timestamp(self) -> None:
        meta = collect_metadata()
        assert "timestamp" in meta

    def test_has_hostname(self) -> None:
        meta = collect_metadata()
        assert "hostname" in meta


class TestCollectMetadataMassRatioReduced:
    """Tests for mass_ratio_reduced flag (spec §14)."""

    def test_no_config_no_flag(self) -> None:
        meta = collect_metadata()
        assert "mass_ratio_reduced" not in meta

    def test_full_ratio_not_flagged(self) -> None:
        config = _make_config(mass_ratio=None)
        meta = collect_metadata(config)
        assert meta["mass_ratio_reduced"] is False

    def test_reduced_ratio_flagged(self) -> None:
        config = _make_config(mass_ratio=100.0)
        meta = collect_metadata(config)
        assert meta["mass_ratio_reduced"] is True

    def test_physical_ratio_exact_boundary(self) -> None:
        """A mass ratio of exactly 1836 is not 'reduced'."""
        config = _make_config(mass_ratio=1836.0)
        meta = collect_metadata(config)
        assert meta["mass_ratio_reduced"] is False

    def test_near_physical_ratio_below_boundary(self) -> None:
        """1835.9 < 1836 → considered reduced."""
        config = _make_config(mass_ratio=1835.9)
        meta = collect_metadata(config)
        assert meta["mass_ratio_reduced"] is True

    def test_above_physical_ratio_not_flagged(self) -> None:
        """mass_ratio > 1836 (e.g. heavy ions) is not flagged."""
        config = _make_config(mass_ratio=2000.0)
        meta = collect_metadata(config)
        assert meta["mass_ratio_reduced"] is False

    def test_config_hash_present(self) -> None:
        config = _make_config()
        meta = collect_metadata(config)
        assert "config_hash" in meta
        assert len(meta["config_hash"]) == 64  # SHA-256 hex

    def test_config_contents_present(self) -> None:
        config = _make_config()
        meta = collect_metadata(config)
        assert "config_contents" in meta
        assert isinstance(meta["config_contents"], str)


class TestCollectMetadataRandomSeed:
    """Tests for random_seed in metadata (spec §14)."""

    def test_no_seed_when_no_config(self) -> None:
        meta = collect_metadata()
        assert "random_seed" not in meta

    def test_seed_present_with_config(self) -> None:
        config = _make_config()
        meta = collect_metadata(config)
        assert "random_seed" in meta
        assert isinstance(meta["random_seed"], int)

    def test_user_seed_preserved(self) -> None:
        config = SimConfig(
            nozzle=_make_config().nozzle,
            plasma=_make_config().plasma,
            random_seed=42,
        )
        meta = collect_metadata(config)
        assert meta["random_seed"] == 42

    def test_deterministic_from_config_hash(self) -> None:
        """Same config → same seed, different config → different seed."""
        c1 = _make_config()
        c2 = _make_config(mass_ratio=100.0)
        s1 = collect_metadata(c1)["random_seed"]
        s2 = collect_metadata(c2)["random_seed"]
        # Same config twice → same seed
        assert collect_metadata(c1)["random_seed"] == s1
        # Different config → different seed
        assert s1 != s2

    def test_seed_in_valid_range(self) -> None:
        """Derived seed must fit in 32-bit signed int."""
        config = _make_config()
        meta = collect_metadata(config)
        assert 0 <= meta["random_seed"] < 2**31
