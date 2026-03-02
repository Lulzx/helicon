"""Tests for helicon.runner.insitu module."""

from __future__ import annotations

from helicon.runner.insitu import InsituTimeSeries, make_thrust_callback


class TestInsituTimeSeries:
    def test_append_and_length(self) -> None:
        ts = InsituTimeSeries()
        ts.append(0.0, 1.0, 0.1)
        ts.append(1e-4, 1.1, 0.11)
        assert len(ts) == 2

    def test_mean_thrust_empty(self) -> None:
        ts = InsituTimeSeries()
        assert ts.mean_thrust() == 0.0

    def test_mean_thrust(self) -> None:
        ts = InsituTimeSeries()
        ts.append(0.0, 2.0, 0.2)
        ts.append(1.0, 4.0, 0.4)
        assert abs(ts.mean_thrust() - 3.0) < 1e-10

    def test_lists_populated(self) -> None:
        ts = InsituTimeSeries()
        ts.append(0.5, 1.5, 0.15)
        assert ts.times == [0.5]
        assert ts.thrust_N == [1.5]
        assert ts.mass_flow_kgs == [0.15]


class TestCallbackFactory:
    def test_returns_callable(self) -> None:
        acc = InsituTimeSeries()
        cb = make_thrust_callback(acc)
        assert callable(cb)

    def test_callback_raises_without_pywarpx(self) -> None:
        """Calling the callback outside WarpX should raise ImportError."""
        import pytest

        acc = InsituTimeSeries()
        # Use backend="numpy" to avoid MLX ImportError before pywarpx check
        cb = make_thrust_callback(acc, backend="numpy")
        with pytest.raises(ImportError, match="pywarpx"):
            cb()

    def test_factory_accepts_mlx_backend(self) -> None:
        """Factory should not raise when backend='mlx' (only runtime error)."""
        acc = InsituTimeSeries()
        cb = make_thrust_callback(acc, backend="mlx")
        assert callable(cb)

    def test_factory_accepts_numpy_backend(self) -> None:
        acc = InsituTimeSeries()
        cb = make_thrust_callback(acc, backend="numpy")
        assert callable(cb)
