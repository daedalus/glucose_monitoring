import argparse

import pytest

from agp.config import build_config


def _make_args(**overrides):
    """Return a minimal Namespace matching the defaults from build_parser."""
    defaults = dict(
        very_low_threshold=54,
        low_threshold=70,
        high_threshold=180,
        very_high_threshold=250,
        tight_low=70,
        tight_high=140,
        bin_minutes=5,
        min_samples=5,
        sensor_interval=5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_returns_all_expected_keys():
    cfg = build_config(_make_args())
    expected = {
        "VERY_LOW",
        "LOW",
        "HIGH",
        "VERY_HIGH",
        "TIGHT_LOW",
        "TIGHT_HIGH",
        "BIN_MINUTES",
        "MIN_SAMPLES_PER_BIN",
        "SENSOR_INTERVAL",
        "ROC_CLIP",
    }
    assert expected == set(cfg.keys())


def test_build_config_default_values():
    cfg = build_config(_make_args())
    assert cfg["VERY_LOW"] == 54
    assert cfg["LOW"] == 70
    assert cfg["HIGH"] == 180
    assert cfg["VERY_HIGH"] == 250
    assert cfg["TIGHT_LOW"] == 70
    assert cfg["TIGHT_HIGH"] == 140
    assert cfg["BIN_MINUTES"] == 5
    assert cfg["MIN_SAMPLES_PER_BIN"] == 5
    assert cfg["SENSOR_INTERVAL"] == 5
    assert cfg["ROC_CLIP"] == 10


def test_build_config_bin_minutes_raises_when_zero():
    with pytest.raises(ValueError, match="bin_minutes must be positive"):
        build_config(_make_args(bin_minutes=0))


def test_build_config_bin_minutes_raises_when_negative():
    with pytest.raises(ValueError, match="bin_minutes must be positive"):
        build_config(_make_args(bin_minutes=-3))


def test_build_config_sensor_interval_raises_when_zero():
    with pytest.raises(ValueError, match="sensor_interval must be positive"):
        build_config(_make_args(sensor_interval=0))


def test_build_config_sensor_interval_raises_when_negative():
    with pytest.raises(ValueError, match="sensor_interval must be positive"):
        build_config(_make_args(sensor_interval=-1))


def test_build_config_roc_clip_always_10():
    cfg = build_config(_make_args())
    assert cfg["ROC_CLIP"] == 10


def test_build_config_custom_thresholds():
    cfg = build_config(
        _make_args(
            very_low_threshold=60,
            low_threshold=80,
            high_threshold=160,
            very_high_threshold=200,
        )
    )
    assert cfg["VERY_LOW"] == 60
    assert cfg["LOW"] == 80
    assert cfg["HIGH"] == 160
    assert cfg["VERY_HIGH"] == 200


def test_build_config_raises_when_threshold_order_violated():
    with pytest.raises(ValueError, match="very_low <= low <= high <= very_high"):
        build_config(_make_args(low_threshold=200, high_threshold=100))


def test_build_config_raises_when_very_low_exceeds_low():
    with pytest.raises(ValueError, match="very_low <= low <= high <= very_high"):
        build_config(_make_args(very_low_threshold=80, low_threshold=70))


def test_build_config_raises_when_high_exceeds_very_high():
    with pytest.raises(ValueError, match="very_low <= low <= high <= very_high"):
        build_config(_make_args(high_threshold=300, very_high_threshold=250))


def test_build_config_raises_when_tight_low_equals_tight_high():
    with pytest.raises(ValueError, match="tight_low must be less than tight_high"):
        build_config(_make_args(tight_low=140, tight_high=140))


def test_build_config_raises_when_tight_low_exceeds_tight_high():
    with pytest.raises(ValueError, match="tight_low must be less than tight_high"):
        build_config(_make_args(tight_low=150, tight_high=140))


def test_build_config_raises_when_min_samples_zero():
    with pytest.raises(ValueError, match="min_samples must be positive"):
        build_config(_make_args(min_samples=0))


def test_build_config_raises_when_min_samples_negative():
    with pytest.raises(ValueError, match="min_samples must be positive"):
        build_config(_make_args(min_samples=-1))
