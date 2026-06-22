"""Tests for microscope.saes.train (Phase 2, ADR-0004): the SAE-vs-skip-transcoder wrapper.

The whole point of this unit is a *methodologically fair* SAE-vs-skip-transcoder head-to-head:
both coders are trained from the SAME ``sparsify`` ``SaeConfig`` / ``Trainer`` and differ ONLY by
two flags (``transcode`` / ``skip_connection``) while sharing ``num_latents`` (width) and ``k`` (the
TopK L0). :func:`coder_config_dict` is the CPU-testable core that encodes exactly that invariant; it
is pure (no ``sparsify`` / ``torch`` import) so the correctness can be pinned without a GPU.

What is verified here:

* THE invariant (ADR-0004 §"E4 verification results"): SAE => ``transcode=False,
  skip_connection=False``; skip-transcoder => ``transcode=True, skip_connection=True``.
* Fairness: an SAE and a transcoder derived from the SAME :class:`RunConfig` share an identical
  ``num_latents`` and ``k`` (otherwise the Phase-3 comparison is confounded).
* Field propagation: width / k / layer / seed / hookpoint / dataset / run_name flow through from the
  config; optional fields fall back to the documented module constants.
* Input validation (fail fast, RULES.md E2): bad ``kind``; missing/non-positive/non-integral
  ``width`` / ``k``; bad ``activation``; ``layer is None``; non-positive ``lr`` / ``batch_size``.
* :func:`train_coder` on this CPU box raises :class:`GpuStackUnavailable` (sparsify is GPU-only,
  ADR-0004) and NOT :class:`GpuImplementationPending` (it is implemented, not a stub) — and it
  validates the config FIRST, so a bad config raises ``ValueError`` before the GPU gate.
* The two committed Phase-2 YAMLs load via :func:`microscope.config.load_config` and flow cleanly
  through :func:`coder_config_dict` for both kinds, carrying the expected width / k / layer.

No GPU and no ``sparsify`` are needed: every assertion runs on the CPU base box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from microscope._pending import GpuImplementationPending, GpuStackUnavailable
from microscope.config import RunConfig, load_config
from microscope.saes.train import (
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_SAVE_DIR,
    coder_config_dict,
    train_coder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"

# The two NEW Phase-2 training configs (ADR-0004). Both carry width + k (the OLD
# pythia70m_smoke.yaml / gemma2_2b_reproduce.yaml are Phase-1 configs and intentionally lack k).
TRAIN_CONFIGS = ("train_pythia70m_smoke.yaml", "train_gemma2_2b_l12.yaml")


def _run_config(**extras: object) -> RunConfig:
    """Build a minimal valid RunConfig, overriding/adding fields via kwargs.

    Defaults to a layer-bearing Pythia config with the two REQUIRED extras (width, k) present, so a
    test that wants the *valid* path gets one and a test that wants an *invalid* path can drop or
    corrupt a single field explicitly.
    """
    base: dict[str, object] = {
        "name": "unit",
        "model": "EleutherAI/pythia-70m",
        "layer": 3,
        "width": 4096,
        "k": 32,
    }
    base.update(extras)
    return RunConfig(**base)  # type: ignore[arg-type]


# --- THE core invariant: SAE vs skip-transcoder flags (ADR-0004) ---------------------------------


def test_sae_kind_sets_both_flags_false() -> None:
    # Arrange
    config = _run_config()

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert: an SAE is transcode=False AND skip_connection=False (ADR-0004 fair-comparison flags).
    assert settings["transcode"] is False
    assert settings["skip_connection"] is False


def test_transcoder_kind_sets_both_flags_true() -> None:
    # Arrange
    config = _run_config()

    # Act
    settings = coder_config_dict(config, "transcoder")

    # Assert: a skip-transcoder is transcode=True AND skip_connection=True (ADR-0004).
    assert settings["transcode"] is True
    assert settings["skip_connection"] is True


def test_kind_is_recorded_in_settings() -> None:
    # Arrange / Act / Assert: the flat dict echoes the kind it was built for.
    assert coder_config_dict(_run_config(), "sae")["kind"] == "sae"
    assert coder_config_dict(_run_config(), "transcoder")["kind"] == "transcoder"


# --- fairness: SAE and transcoder from the SAME config share width + k ---------------------------


def test_sae_and_transcoder_share_identical_width_and_k() -> None:
    # Arrange: ONE config drives BOTH coders -> the head-to-head must hold width/k fixed, or the
    # Phase-3 SAE-vs-transcoder comparison is confounded (ADR-0004; the whole point of this unit).
    config = _run_config(width=8192, k=48)

    # Act
    sae = coder_config_dict(config, "sae")
    transcoder = coder_config_dict(config, "transcoder")

    # Assert: width (=> num_latents) and k are identical across the two kinds.
    assert sae["num_latents"] == transcoder["num_latents"] == 8192
    assert sae["k"] == transcoder["k"] == 48


def test_only_the_two_flags_differ_between_sae_and_transcoder() -> None:
    # Arrange: stronger fairness check — the SAE and transcoder settings must be IDENTICAL except
    # for transcode / skip_connection / kind / run_name (which is kind-suffixed by default).
    config = _run_config(width=8192, k=48)

    # Act
    sae = coder_config_dict(config, "sae")
    transcoder = coder_config_dict(config, "transcoder")

    # Assert: every key outside the expected-different set matches between the two coders.
    expected_different = {"transcode", "skip_connection", "kind", "run_name"}
    for key in sae:
        if key in expected_different:
            continue
        assert sae[key] == transcoder[key], f"field {key!r} drifted between SAE and transcoder"


# --- field propagation from the RunConfig --------------------------------------------------------


def test_known_fields_propagate_from_run_config() -> None:
    # Arrange: a config with KNOWN typed fields + extras, so we can assert each lands in the dict.
    config = _run_config(
        model="EleutherAI/pythia-70m-deduped",
        layer=7,
        hookpoint="blocks.7.hook_resid_post",
        dataset="NeelNanda/pile-10k",
        n_tokens=123_456,
        seed=99,
        width=2048,
        k=64,
    )

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert: typed fields + required extras propagate verbatim.
    assert settings["model"] == "EleutherAI/pythia-70m-deduped"
    assert settings["num_latents"] == 2048
    assert settings["k"] == 64
    assert settings["layers"] == [7]
    assert settings["seed"] == 99
    assert settings["dataset"] == "NeelNanda/pile-10k"
    assert settings["n_tokens"] == 123_456


def test_layers_is_a_single_element_list_of_config_layer() -> None:
    # Arrange / Act: sparsify takes layers=[N]; the wrapper wraps the scalar config.layer.
    settings = coder_config_dict(_run_config(layer=11), "sae")

    # Assert
    assert settings["layers"] == [11]


def test_hookpoints_is_list_when_hookpoint_set() -> None:
    # Arrange: an explicit hookpoint string => hookpoints=[that string] (overrides layers in train).
    config = _run_config(hookpoint="blocks.3.hook_resid_post")

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert
    assert settings["hookpoints"] == ["blocks.3.hook_resid_post"]


def test_hookpoints_is_none_when_hookpoint_unset() -> None:
    # Arrange: no hookpoint => None, so sparsify derives the hookpoint from layers=[N] (ADR-0004).
    config = _run_config(hookpoint=None)

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert
    assert settings["hookpoints"] is None


def test_run_name_defaults_to_name_suffixed_by_kind() -> None:
    # Arrange: with no explicit run_name, the default is f"{name}-{kind}" so the SAE and transcoder
    # save to distinct subfolders even from one config.
    config = _run_config(name="myrun")

    # Act / Assert
    assert coder_config_dict(config, "sae")["run_name"] == "myrun-sae"
    assert coder_config_dict(config, "transcoder")["run_name"] == "myrun-transcoder"


def test_run_name_honors_explicit_override() -> None:
    # Arrange / Act: an explicit run_name extra wins over the kind-suffixed default.
    settings = coder_config_dict(_run_config(run_name="hand-named"), "sae")

    # Assert
    assert settings["run_name"] == "hand-named"


def test_optional_fields_fall_back_to_module_defaults() -> None:
    # Arrange: omit every optional extra -> documented module constants apply (only width/k given).
    config = _run_config()

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert
    assert settings["activation"] == DEFAULT_ACTIVATION
    assert settings["batch_size"] == DEFAULT_BATCH_SIZE
    assert settings["lr"] == DEFAULT_LR
    assert settings["save_dir"] == DEFAULT_SAVE_DIR


def test_optional_fields_honor_explicit_values() -> None:
    # Arrange / Act: explicit optional extras override the defaults.
    settings = coder_config_dict(
        _run_config(activation="groupmax", batch_size=8, lr=2e-4, save_dir="outputs/custom"),
        "sae",
    )

    # Assert
    assert settings["activation"] == "groupmax"
    assert settings["batch_size"] == 8
    assert settings["lr"] == pytest.approx(2e-4)
    assert settings["save_dir"] == "outputs/custom"


def test_activation_is_lowercased() -> None:
    # Arrange / Act: "TopK" must normalise to the sparsify-valid "topk" (case-insensitive input).
    settings = coder_config_dict(_run_config(activation="TopK"), "sae")

    # Assert
    assert settings["activation"] == "topk"


def test_width_and_k_accept_string_and_float_and_coerce_to_int() -> None:
    # Arrange: YAML may yield "64" or 64.0; the wrapper coerces to a positive int (E2 validation).
    config = _run_config(width="2048", k=32.0)

    # Act
    settings = coder_config_dict(config, "sae")

    # Assert: coerced to genuine ints, not left as str/float.
    assert settings["num_latents"] == 2048
    assert isinstance(settings["num_latents"], int)
    assert settings["k"] == 32
    assert isinstance(settings["k"], int)


# --- input validation: every bad input must raise ValueError (fail fast) -------------------------


def test_invalid_kind_raises_value_error() -> None:
    # Arrange / Act / Assert: kind must be 'sae' or 'transcoder'.
    with pytest.raises(ValueError, match="kind"):
        coder_config_dict(_run_config(), "diffusion")  # type: ignore[arg-type]


def test_missing_width_raises_value_error() -> None:
    # Arrange: width has NO default — a fair comparison demands an explicit shared width (ADR-0004).
    config = RunConfig(name="u", model="m", layer=3, k=32)

    # Act / Assert
    with pytest.raises(ValueError, match="width"):
        coder_config_dict(config, "sae")


def test_missing_k_raises_value_error() -> None:
    # Arrange: k has NO default — the SAE and transcoder MUST share an explicit k (ADR-0004).
    config = RunConfig(name="u", model="m", layer=3, width=4096)

    # Act / Assert
    with pytest.raises(ValueError, match="k"):
        coder_config_dict(config, "sae")


def test_width_none_raises_value_error() -> None:
    # Arrange: an explicit None width is treated as missing (not a silent default).
    config = _run_config(width=None)

    # Act / Assert
    with pytest.raises(ValueError, match="width"):
        coder_config_dict(config, "sae")


def test_k_none_raises_value_error() -> None:
    # Arrange: an explicit None k is treated as missing.
    config = _run_config(k=None)

    # Act / Assert
    with pytest.raises(ValueError, match="k"):
        coder_config_dict(config, "sae")


@pytest.mark.parametrize("bad_width", [0, -1, -4096])
def test_non_positive_width_raises_value_error(bad_width: int) -> None:
    # Arrange / Act / Assert: width must be strictly positive.
    with pytest.raises(ValueError, match="width"):
        coder_config_dict(_run_config(width=bad_width), "sae")


@pytest.mark.parametrize("bad_k", [0, -1, -32])
def test_non_positive_k_raises_value_error(bad_k: int) -> None:
    # Arrange / Act / Assert: k must be strictly positive.
    with pytest.raises(ValueError, match="k"):
        coder_config_dict(_run_config(k=bad_k), "sae")


def test_non_integral_width_raises_value_error() -> None:
    # Arrange / Act / Assert: 4096.5 is not an integer dictionary size.
    with pytest.raises(ValueError, match="width"):
        coder_config_dict(_run_config(width=4096.5), "sae")


def test_non_integral_k_raises_value_error() -> None:
    # Arrange / Act / Assert: a fractional L0 is meaningless.
    with pytest.raises(ValueError, match="k"):
        coder_config_dict(_run_config(k=32.5), "sae")


def test_non_numeric_width_raises_value_error() -> None:
    # Arrange / Act / Assert: a non-numeric width string cannot be coerced.
    with pytest.raises(ValueError, match="width"):
        coder_config_dict(_run_config(width="wide"), "sae")


def test_non_numeric_k_raises_value_error() -> None:
    # Arrange / Act / Assert: a non-numeric k string cannot be coerced.
    with pytest.raises(ValueError, match="k"):
        coder_config_dict(_run_config(k="lots"), "sae")


def test_invalid_activation_raises_value_error() -> None:
    # Arrange: only sparsify's {topk, groupmax} are valid (ADR-0004 SaeConfig.activation).
    config = _run_config(activation="relu")

    # Act / Assert
    with pytest.raises(ValueError, match="activation"):
        coder_config_dict(config, "sae")


def test_layer_none_raises_value_error() -> None:
    # Arrange: sparsify needs a concrete layer index for the hookpoint; None is rejected.
    config = RunConfig(name="u", model="m", layer=None, width=4096, k=32)

    # Act / Assert
    with pytest.raises(ValueError, match="layer"):
        coder_config_dict(config, "sae")


@pytest.mark.parametrize("bad_lr", [0, -1e-4])
def test_non_positive_lr_raises_value_error(bad_lr: float) -> None:
    # Arrange / Act / Assert: a non-positive learning rate is invalid.
    with pytest.raises(ValueError, match="lr"):
        coder_config_dict(_run_config(lr=bad_lr), "sae")


def test_non_numeric_lr_raises_value_error() -> None:
    # Arrange / Act / Assert: a non-numeric lr cannot be parsed to a float.
    with pytest.raises(ValueError, match="lr"):
        coder_config_dict(_run_config(lr="fast"), "sae")


@pytest.mark.parametrize("bad_batch", [0, -16])
def test_non_positive_batch_size_raises_value_error(bad_batch: int) -> None:
    # Arrange / Act / Assert: batch_size must be a strictly-positive int.
    with pytest.raises(ValueError, match="batch_size"):
        coder_config_dict(_run_config(batch_size=bad_batch), "sae")


# --- train_coder: GPU gate on the CPU box (sparsify absent) --------------------------------------


def test_train_coder_raises_gpu_stack_unavailable_on_cpu() -> None:
    # Arrange: sparsify is NOT installed on this box (it lives on the Modal [gpu] image, ADR-0004),
    # so train_coder must surface GpuStackUnavailable — mirroring harvest_resid_activations /
    # load_pretrained_sae, the other implemented-but-GPU-only stages.
    config = _run_config()

    # Act / Assert: message names the absent library and the Modal [gpu] image.
    with pytest.raises(GpuStackUnavailable, match="sparsify"):
        train_coder(config, "sae")


def test_train_coder_error_names_gpu_image() -> None:
    # Arrange
    config = _run_config()

    # Act
    with pytest.raises(GpuStackUnavailable) as exc_info:
        train_coder(config, "sae")

    # Assert: actionable message points at the [gpu] image (not a bare RuntimeError).
    assert "[gpu]" in str(exc_info.value).lower()


def test_train_coder_error_is_runtime_not_pending() -> None:
    # Arrange: pin the behaviour change documented in the handoff — train_coder moved from a
    # GpuImplementationPending STUB to a real implementation that raises GpuStackUnavailable. It IS
    # a RuntimeError (callers catching RuntimeError still work) but is NOT GpuImplementationPending
    # (that would mean it regressed back to an unimplemented stub).
    config = _run_config()

    # Act
    with pytest.raises(GpuStackUnavailable) as exc_info:
        train_coder(config, "sae")

    # Assert
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, GpuImplementationPending)


def test_train_coder_transcoder_also_hits_gpu_gate() -> None:
    # Arrange / Act / Assert: the transcoder path reaches the same GPU gate as the SAE path.
    with pytest.raises(GpuStackUnavailable):
        train_coder(_run_config(), "transcoder")


def test_train_coder_validates_before_gpu_gate() -> None:
    # Arrange: a config missing 'k' is invalid. train_coder calls coder_config_dict FIRST, so the
    # ValueError must fire BEFORE the GPU import gate — a bad config fails fast on CPU and never
    # pretends it needs a GPU (RULES.md input validation; documented fail-fast ordering).
    config = RunConfig(name="u", model="m", layer=3, width=4096)  # no k

    # Act / Assert: ValueError (validation), NOT GpuStackUnavailable (the gate).
    with pytest.raises(ValueError, match="k"):
        train_coder(config, "sae")


def test_train_coder_validates_missing_width_before_gpu_gate() -> None:
    # Arrange: symmetry with the missing-k case — missing width is also caught before the gate.
    config = RunConfig(name="u", model="m", layer=3, k=32)  # no width

    # Act / Assert
    with pytest.raises(ValueError, match="width"):
        train_coder(config, "sae")


def test_train_coder_invalid_kind_raises_value_error_before_gpu_gate() -> None:
    # Arrange / Act / Assert: an invalid kind is a validation error, surfaced before the GPU gate.
    with pytest.raises(ValueError, match="kind"):
        train_coder(_run_config(), "nonsense")  # type: ignore[arg-type]


# --- the two committed Phase-2 YAMLs load + flow through the pure core ----------------------------


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_train_yaml_loads_via_load_config(config_name: str) -> None:
    # Arrange / Act: both committed Phase-2 training YAMLs must parse into a RunConfig cleanly.
    cfg = load_config(CONFIGS_DIR / config_name)

    # Assert: each carries an explicit, positive width + k (required for a fair comparison).
    extras = dict(cfg.model_extra or {})
    assert "width" in extras and int(extras["width"]) > 0
    assert "k" in extras and int(extras["k"]) > 0
    assert cfg.layer is not None


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
@pytest.mark.parametrize("kind", ["sae", "transcoder"])
def test_train_yaml_flows_through_coder_config_dict(config_name: str, kind: str) -> None:
    # Arrange
    cfg = load_config(CONFIGS_DIR / config_name)
    extras = dict(cfg.model_extra or {})

    # Act
    settings = coder_config_dict(cfg, kind)  # type: ignore[arg-type]

    # Assert: width/k/layer carry through, and the kind flags are correct for a real config.
    assert settings["num_latents"] == int(extras["width"])
    assert settings["k"] == int(extras["k"])
    assert settings["layers"] == [cfg.layer]
    is_transcoder = kind == "transcoder"
    assert settings["transcode"] is is_transcoder
    assert settings["skip_connection"] is is_transcoder


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_train_yaml_sae_and_transcoder_are_a_fair_pair(config_name: str) -> None:
    # Arrange: the real configs must yield a fair SAE-vs-transcoder pair (shared width + k), since
    # ONE YAML serves BOTH coders via the --kind flag (ADR-0004 / config header comments).
    cfg = load_config(CONFIGS_DIR / config_name)

    # Act
    sae = coder_config_dict(cfg, "sae")
    transcoder = coder_config_dict(cfg, "transcoder")

    # Assert
    assert sae["num_latents"] == transcoder["num_latents"]
    assert sae["k"] == transcoder["k"]


def test_pythia_smoke_yaml_specific_values() -> None:
    # Arrange / Act: pin the documented smoke values (layer 3, width 4096, k 32, hookpoint unset).
    settings = coder_config_dict(load_config(CONFIGS_DIR / "train_pythia70m_smoke.yaml"), "sae")

    # Assert
    assert settings["layers"] == [3]
    assert settings["num_latents"] == 4096
    assert settings["k"] == 32
    assert settings["hookpoints"] is None  # train_pythia70m_smoke.yaml has hookpoint: null


def test_gemma_yaml_specific_values() -> None:
    # Arrange / Act: pin the documented Gemma values (layer 12, width 16384, k 64, hookpoint set).
    settings = coder_config_dict(load_config(CONFIGS_DIR / "train_gemma2_2b_l12.yaml"), "sae")

    # Assert
    assert settings["layers"] == [12]
    assert settings["num_latents"] == 16384
    assert settings["k"] == 64
    assert settings["hookpoints"] == ["blocks.12.hook_resid_post"]
