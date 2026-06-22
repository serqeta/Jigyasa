from voiceshield import config


def test_config_constants_exist():
    """TEST-S0.3: Ensure global constants are defined."""
    assert hasattr(config, "SAMPLE_RATE")
    assert config.SAMPLE_RATE == 16000
    assert hasattr(config, "CHUNK_MS")
    assert config.CHUNK_MS == 500
    assert config.CHUNK_SAMPLES == 8000
    assert config.BUFFER_SECONDS == 10
    assert config.BUFFER_SAMPLES == 160000
    assert config.SNR_NORMAL_DB == 12.0
    assert config.SNR_GREY_DB == 8.0
    assert config.SCORE_AMBER == 0.30
    assert config.SCORE_RED == 0.70
    assert hasattr(config, "USE_FALLBACK_CLASSIFIER")
    assert getattr(config, "USE_FALLBACK_CLASSIFIER") is False
