"""
Behavioral tests: quant rung selection pure function (T3.4-rung2).

_choose_quant(total_vram_gb, required_vram) -> '4bit' / '8bit' / None.
Boundaries mirror the shipped CUDA-branch thresholds exactly.
"""
from synapse.inference.pytorch_hf_engine import PyTorchHFInferenceEngine as E


def test_4bit_when_tight():
    assert E._choose_quant(8.0, 14.0) == "4bit"    # 8 < 14*0.7=9.8
    assert E._choose_quant(5.0, 0.0) == "4bit"      # unknown params, <=6GB


def test_8bit_when_snug():
    assert E._choose_quant(12.0, 14.0) == "8bit"   # 9.8 <= 12 < 16.8
    assert E._choose_quant(10.0, 0.0) == "8bit"    # unknown params, <=12GB


def test_none_when_roomy():
    assert E._choose_quant(24.0, 14.0) is None     # 24 >= 16.8
    assert E._choose_quant(16.0, 0.0) is None      # unknown params, >12GB
    assert E._choose_quant(0.0, 0.0) == "4bit"     # degenerate: 0 <= 6.0


def test_exact_boundaries():
    assert E._choose_quant(9.8, 14.0) == "8bit"    # == 0.7x -> not strict <
    assert E._choose_quant(16.8, 14.0) is None     # == 1.2x -> not strict <
    assert E._choose_quant(6.0, 0.0) == "4bit"
    assert E._choose_quant(12.0, 0.0) == "8bit"
