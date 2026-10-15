"""Check comparison formulas and inference isolation without training models.

Examples::

    python tools/check_comparison.py --device cpu
    python tools/check_comparison.py --device cuda --output outputs/checks-cuda.json

CPU mode hides CUDA devices before any CUDA calls.  Existing ``models/vae.pt``
and ``models/gan.pt`` are loaded read-only on CPU when present; their absence is
reported explicitly and does not prevent the independent mathematical checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from gan.checkpoint import load_checkpoint as load_gan
from gan.model import ConditionalGenerator
import generative.comparison_metrics as metrics
from generative.comparison_study import generate
from vae.checkpoint import load_checkpoint as load_vae
from vae.model import ConvVAE


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_distribution_formulas() -> dict:
    """Compare blocked tensor calculations with separate NumPy references."""
    previous_block_size = metrics._BLOCK_SIZE
    metrics._BLOCK_SIZE = 2  # Exercise several blocks with tiny, cheap examples.
    try:
        real = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64)
        collapsed = torch.zeros((4, 1), dtype=torch.float64)
        scores = metrics.distribution_metrics(real, collapsed, k=3)
        np.testing.assert_allclose(scores["mmd2"], 19 / 6, rtol=0, atol=1e-12)
        _require(scores["precision"] == 1 and scores["recall"] == 0,
                 "A collapsed center must be covered by the real manifold, but cover no real points")
        reversed_scores = metrics.distribution_metrics(collapsed, real, k=3)
        _require(reversed_scores["precision"] == 0 and reversed_scores["recall"] == 1,
                 "Swapping real/fake must swap precision and recall")
        identical = metrics.distribution_metrics(real, real, k=3)
        np.testing.assert_allclose(identical["mmd2"], -187 / 6, rtol=0, atol=1e-12)
        _require(identical["precision"] == 1 and identical["recall"] == 1,
                 "An identical set must cover itself")

        rng = np.random.default_rng(123)
        x, y = rng.normal(size=(9, 5)), rng.normal(size=(7, 5))
        kxx = (x @ x.T / 5 + 1) ** 3
        kyy = (y @ y.T / 5 + 1) ** 3
        kxy = (x @ y.T / 5 + 1) ** 3
        expected_mmd = ((kxx.sum() - np.trace(kxx)) / (9 * 8)
                        + (kyy.sum() - np.trace(kyy)) / (7 * 6) - 2 * kxy.mean())
        # Direct broadcasted differences are independent of the production
        # blocked squared-distance and incremental-neighbour implementation.
        dxx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
        dyy = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        dxy = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        np.fill_diagonal(dxx, np.inf)
        np.fill_diagonal(dyy, np.inf)
        real_radii = np.sort(dxx, axis=1)[:, 2]
        fake_radii = np.sort(dyy, axis=1)[:, 2]
        expected_precision = (dxy <= real_radii[:, None]).any(axis=0).mean()
        expected_recall = (dxy <= fake_radii[None, :]).any(axis=1).mean()
        actual = metrics.distribution_metrics(torch.from_numpy(x), torch.from_numpy(y))
        np.testing.assert_allclose(actual["mmd2"], expected_mmd, rtol=1e-12, atol=1e-12)
        _require(actual["precision"] == expected_precision and actual["recall"] == expected_recall,
                 "Blocked precision/recall disagree with the brute-force NumPy reference")
        return {"asymmetric_precision": scores["precision"], "asymmetric_recall": scores["recall"],
                "retained_negative_mmd2": identical["mmd2"], "numpy_reference": "passed"}
    finally:
        metrics._BLOCK_SIZE = previous_block_size


def check_macro_and_confusion() -> dict:
    """An imbalanced example distinguishes class-macro from sample-micro means."""
    real_labels = torch.arange(10).repeat_interleave(4)
    requested = torch.cat((torch.zeros(8, dtype=torch.long), torch.arange(1, 10).repeat_interleave(4)))
    predictions = requested.clone()
    predictions[:4] = 1
    real_features = torch.arange(40, dtype=torch.float64).reshape(-1, 1)
    fake_features = torch.arange(44, dtype=torch.float64).reshape(-1, 1)
    result = metrics.classwise_metrics(real_features, real_labels, fake_features, requested, predictions)
    np.testing.assert_allclose(result["macro"]["accuracy"], 0.95, rtol=0, atol=1e-12)
    _require(result["confusion"][0][0] == 4 and result["confusion"][0][1] == 4,
             "Confusion matrix must use requested rows and predicted columns")
    _require(sum(sum(row) for row in result["confusion"]) == 44,
             "Confusion counts must include every generated sample")
    _require(len(result["per_class"]) == 10, "All ten digits must be reported")
    return {"expected_macro_accuracy": 0.95, "micro_accuracy_for_contrast": 40 / 44,
            "confusion_count": 44}


def _rng_state(device: torch.device) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
                    if device.type == "cuda" else []}


def _require_same_rng(before: dict, device: torch.device) -> None:
    after = _rng_state(device)
    _require(before["python"] == after["python"], "Python RNG changed")
    _require(before["numpy"][0] == after["numpy"][0]
             and np.array_equal(before["numpy"][1], after["numpy"][1])
             and before["numpy"][2:] == after["numpy"][2:], "NumPy RNG changed")
    _require(torch.equal(before["cpu"], after["cpu"]), "PyTorch CPU RNG changed")
    _require(len(before["cuda"]) == len(after["cuda"])
             and all(torch.equal(left, right) for left, right in zip(before["cuda"], after["cuda"])),
             "A PyTorch CUDA default RNG changed")


def check_rng_restoration(device: torch.device) -> dict:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    before = _rng_state(device)

    class DeliberateError(Exception):
        pass

    for exceptional_exit in (False, True):
        try:
            with metrics.preserve_rng():
                random.random()
                np.random.rand(10)
                torch.rand(10)
                if device.type == "cuda":
                    for index in range(torch.cuda.device_count()):
                        torch.rand(10, device=f"cuda:{index}")
                if exceptional_exit:
                    raise DeliberateError
        except DeliberateError:
            pass
        _require_same_rng(before, device)
    return {"normal_exit": "passed", "exceptional_exit": "passed",
            "cuda_generators_checked": len(before["cuda"])}


def check_evaluator(device: torch.device) -> dict:
    model = metrics.MNISTEvaluator().to(device)
    model.train()
    images = torch.rand(5, 1, 28, 28)
    with torch.inference_mode():
        logits = model(images.to(device)).cpu()
    before = _rng_state(device)
    features, predicted = metrics.extract_features(model, images, device, batch_size=2)
    _require_same_rng(before, device)
    _require(model.training, "Feature extraction did not restore the evaluator's training mode")
    _require(features.shape == (5, 128) and predicted.shape == (5,), "Incorrect evaluator output shapes")
    _require(features.device.type == predicted.device.type == "cpu", "Features and labels must return to CPU")
    _require(torch.equal(predicted, logits.argmax(1)), "Batched extraction changes predicted labels")
    with torch.inference_mode():
        reconstructed_logits = model.classifier(features.to(device)).cpu()
    torch.testing.assert_close(reconstructed_logits, logits, rtol=1e-4, atol=1e-5)
    return {"feature_shape": list(features.shape), "parameters": sum(p.numel() for p in model.parameters())}


def check_generation_isolation(device: torch.device) -> dict:
    checked = []
    for family, model in (("vae", ConvVAE(4)), ("gan", ConditionalGenerator(4))):
        model.to(device)
        for training in (True, False):
            model.train(training)
            buffer_before = {name: value.clone() for name, value in model.named_buffers()}
            rng_before = _rng_state(device)
            images, labels = generate(model, family, 2, 1729, device)
            _require_same_rng(rng_before, device)
            _require(model.training == training, f"{family} generation changed the model mode")
            _require(all(module.training == training for module in model.modules()),
                     f"{family} generation failed to restore a submodule mode")
            for name, value in model.named_buffers():
                _require(torch.equal(value, buffer_before[name]),
                         f"{family} generation changed a buffer: {name}")
            _require(images.shape == (20, 1, 28, 28), "Incorrect generated image shape")
            _require(torch.equal(labels, torch.arange(10).repeat_interleave(2)), "Generated labels lost their order")
            _require(bool(torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1),
                     "Generated images must be finite and within [0, 1]")
            repeated, repeated_labels = generate(model, family, 2, 1729, device)
            torch.testing.assert_close(images, repeated, rtol=0, atol=0)
            _require(torch.equal(labels, repeated_labels), "Fixed-seed labels changed")
            _require_same_rng(rng_before, device)
            checked.append(f"{family}:{'train' if training else 'eval'}")
    return {"modes": checked, "global_rng": "unchanged", "gan_batchnorm_buffers": "unchanged"}


def check_existing_checkpoints() -> dict:
    checked, absent = [], []
    cpu = torch.device("cpu")
    for family, loader in (("vae", load_vae), ("gan", load_gan)):
        path = ROOT / "models" / f"{family}.pt"
        if not path.exists():
            absent.append(str(path.relative_to(ROOT)))
            continue
        loaded = loader(path, cpu)
        model, metadata = loaded[0], loaded[-1]
        _require(all(parameter.device.type == "cpu" for parameter in model.parameters()),
                 f"{family} existing checkpoint was not loaded on CPU")
        images, labels = generate(model, family, 1, 1729, cpu)
        _require(images.shape == (10, 1, 28, 28) and torch.equal(labels, torch.arange(10)),
                 f"{family} existing checkpoint is incompatible with the comparison generation path")
        checked.append({"model": family, "epoch": metadata["epoch"], "latent_dim": model.latent_dim,
                        "path": str(path.relative_to(ROOT))})
    return {"loaded_and_generated_on_cpu": checked, "skipped_missing_checkpoints": absent}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="optional JSON result file")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or run with GPU access")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    checks = [
        ("distribution_formulas_and_pr_direction", check_distribution_formulas),
        ("macro_and_confusion", check_macro_and_confusion),
        ("rng_restoration", lambda: check_rng_restoration(device)),
        ("evaluator_features", lambda: check_evaluator(device)),
        ("generation_isolation", lambda: check_generation_isolation(device)),
        ("existing_checkpoint_compatibility_cpu", check_existing_checkpoints),
    ]
    result = {"device": str(device), "torch_version": str(torch.__version__), "checks": [], "passed": True}
    for name, check in checks:
        start = time.perf_counter()
        try:
            details = check()
            result["checks"].append({"name": name, "passed": True,
                                     "seconds": time.perf_counter() - start, "details": details})
            print(f"PASS {name}", flush=True)
            for missing in details.get("skipped_missing_checkpoints", []):
                print(f"  SKIP absent checkpoint: {missing}", flush=True)
        except Exception as error:
            result["passed"] = False
            result["checks"].append({"name": name, "passed": False,
                                     "seconds": time.perf_counter() - start,
                                     "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Results: {args.output}", flush=True)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {sum(row['passed'] for row in result['checks'])}/{len(checks)} checks ({device})")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
