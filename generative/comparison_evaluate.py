"""Final held-out evaluation and fixed-rule visual evidence for the study."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from gan.checkpoint import load_checkpoint as load_gan
from vae.checkpoint import load_checkpoint as load_vae
from generative.comparison_metrics import extract_features, preserve_rng, classwise_metrics
from generative.comparison_study import (
    ROOT, evaluate_model, evaluator_accuracy, generate, read_json, sha256, sync, write_json,
)


def load_model(root, family, seed, epoch, device):
    path = root / family / f"seed_{seed}" / f"epoch_{epoch:03d}.pt"
    loaded = load_vae(path, device) if family == "vae" else load_gan(path, device)
    model = loaded[0]
    model.eval()
    return model, path


@torch.inference_mode()
def benchmark_generation(model, family, device, seed, smoke=False):
    """Time only the decoder/generator forward pass on resident GPU inputs."""
    count, batch = (256 if smoke else 10000), 128
    with preserve_rng():
        rng = torch.Generator().manual_seed(seed)
        latent = torch.randn(count, model.latent_dim, generator=rng).to(device)
        labels = (torch.arange(count) % 10).to(device)
        model.eval()
        def forward(z, y):
            return model.decode(z, y) if family == "vae" else model(z, y)
        for _ in range(10):
            forward(latent[:batch], labels[:batch])
        seconds = []
        for _ in range(5):
            sync(device)
            start = time.perf_counter()
            for i in range(0, count, batch):
                forward(latent[i:i + batch], labels[i:i + batch])
            sync(device)
            seconds.append(time.perf_counter() - start)
    median = float(np.median(seconds))
    return {"seconds_median": median, "images_per_second": count / median,
            "repeats_seconds": seconds, "images": count, "batch_size": batch,
            "scope": "forward only; latent and labels already on GPU; excludes transfer, RNG, postprocessing, evaluator and PNG writing"}


@torch.inference_mode()
def nearest_training(features, labels, training_features, training_labels):
    indices = torch.empty(len(features), dtype=torch.long)
    distances = torch.empty(len(features))
    for digit in range(10):
        queries = torch.nonzero(labels == digit).flatten()
        candidates = torch.nonzero(training_labels == digit).flatten()
        reference = training_features[candidates].double()
        for offset in range(0, len(queries), 256):
            ids = queries[offset:offset + 256]
            squared = torch.cdist(features[ids].double(), reference).square()
            minimum, match = squared.min(1)
            indices[ids] = candidates[match]
            distances[ids] = minimum.sqrt().float()
    return indices, distances


def evaluate_study(cfg, data, references, evaluator, device):
    study = cfg["study"]
    root = Path(study["run_dir"])
    start = time.perf_counter()
    count = study["final_samples_per_class"]
    sample_seed = study["sampling_seed"]
    reference_images, reference_labels = references["test"]
    reference_features, _ = extract_features(evaluator, reference_images, device)
    test_accuracy = evaluator_accuracy(evaluator, data["test"], device)
    evaluator_meta = read_json(root / "evaluator/complete.json")
    evaluator_meta["test_accuracy"] = test_accuracy
    write_json(root / "evaluator/complete.json", evaluator_meta)
    print(f"Frozen evaluator real test accuracy: {test_accuracy:.4%}", flush=True)

    train_features, _ = extract_features(evaluator, data["train"][0], device)
    center = train_features.double().mean(0)
    centered = train_features.double() - center
    covariance = centered.T @ centered / (len(centered) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    axes = vectors[:, -2:].flip(1)
    # Deterministic signs make re-rendered figures stable.
    for j in range(2):
        index = axes[:, j].abs().argmax()
        axes[:, j] *= torch.sign(axes[index, j])
    def project(features):
        return ((features.double() - center) @ axes).float().numpy()
    write_json(root / "pca.json", {"fit_split": "train", "samples": len(train_features),
        "explained_variance_ratio": (values[-2:].flip(0) / values.sum()).tolist()})

    visuals = {"real_images": reference_images.reshape(10, count, 1, 28, 28)[:, :8].numpy(),
               "pca_real": project(reference_features), "pca_labels": reference_labels.numpy()}
    final_rows, scale_rows, diagnostics = [], [], {}
    packet_dir = root / "evaluation"
    packet_dir.mkdir(exist_ok=True)
    variants = [("VAE20", "vae", 20), ("GAN20", "gan", 20), ("GAN50", "gan", 50)]
    if cfg["smoke"]:
        variants = [(name, family, 1) for name, family, _ in variants]
    for variant, family, epoch in variants:
        for seed in study["seeds"]:
            model, checkpoint = load_model(root, family, seed, epoch, device)
            metadata = read_json(root / family / f"seed_{seed}" / "complete.json")
            history = read_json(root / family / f"seed_{seed}" / "history.json")
            train_seconds = next(row["train_seconds"] for row in history if row["epoch"] == epoch)
            row, images, features, labels, predictions = evaluate_model(
                model, family, evaluator, (reference_features, reference_labels), count,
                sample_seed, device)
            row.update(variant=variant, model=family, epoch=epoch, seed=seed,
                       train_seconds=train_seconds, parameters_total=metadata["parameters_total"],
                       parameters_generate=metadata["parameters_generate"], checkpoint_sha256=sha256(checkpoint),
                       generation_benchmark=benchmark_generation(model, family, device, sample_seed, cfg["smoke"]),
                       samples_per_class=count, reference_split="test")
            final_rows.append(row)
            write_json(root / "final_metrics.json", final_rows)
            torch.save({"images": images, "features": features, "labels": labels,
                        "predictions": predictions, "checkpoint_sha256": row["checkpoint_sha256"]},
                       packet_dir / f"{variant}_seed_{seed}.pt")
            visuals[f"samples_{variant}_s{seed}"] = images.reshape(10, count, 1, 28, 28)[:, :8].numpy()
            print(f"Final {variant} seed={seed}: {row['macro']}", flush=True)

            for tau in study["scales"]:
                if tau == 1.0:
                    scores, scale_images = {key: row[key] for key in ("macro", "per_class", "confusion")}, images
                else:
                    scores, scale_images, _, _, _ = evaluate_model(model, family, evaluator,
                        (reference_features, reference_labels), count, sample_seed, device, tau=tau)
                scale_rows.append({"variant": variant, "model": family, "epoch": epoch,
                                   "seed": seed, "tau": tau, "samples_per_class": count, **scores})
                if seed == study["seeds"][0]:
                    visuals[f"scale_{variant}_{tau}"] = scale_images.reshape(10, count, 1, 28, 28)[:, 0].numpy()
            write_json(root / "scale_metrics.json", scale_rows)

            if seed == study["seeds"][0]:
                visuals[f"pca_{variant}"] = project(features)
                rng = torch.Generator().manual_seed(sample_seed + 1)
                endpoints = torch.randn(10, 2, model.latent_dim, generator=rng)
                t = torch.linspace(0, 1, 11)
                interpolated = ((1 - t[None, :, None]) * endpoints[:, :1] + t[None, :, None] * endpoints[:, 1:]).reshape(-1, model.latent_dim)
                path_images, _ = generate(model, family, 0, 0, device,
                    latent=interpolated, labels=torch.arange(10).repeat_interleave(11))
                visuals[f"interpolation_{variant}"] = path_images.reshape(10, 11, 1, 28, 28).numpy()
                _, path_predictions = extract_features(evaluator, path_images, device)
                z = torch.randn(8, model.latent_dim, generator=torch.Generator().manual_seed(sample_seed + 2))
                controlled, _ = generate(model, family, 0, 0, device, latent=z.repeat(10, 1),
                    labels=torch.arange(10).repeat_interleave(8))
                visuals[f"conditional_{variant}"] = controlled.reshape(10, 8, 1, 28, 28).numpy()
                _, controlled_predictions = extract_features(evaluator, controlled, device)
                nearest, distances = nearest_training(features, labels, train_features, data["train"][1])
                selected = torch.arange(10) * count
                visuals[f"nearest_generated_{variant}"] = images[selected].numpy()
                visuals[f"nearest_real_{variant}"] = data["train"][0][nearest[selected]].numpy()
                visuals[f"nearest_distances_{variant}"] = distances.numpy()
                failures = torch.nonzero(predictions != labels).flatten()[:20]
                visuals[f"failure_images_{variant}"] = images[failures].numpy()
                visuals[f"failure_requested_{variant}"] = labels[failures].numpy()
                visuals[f"failure_predicted_{variant}"] = predictions[failures].numpy()
                diagnostics[variant] = {
                    "seed": seed,
                    "label_switch_accuracy": (controlled_predictions == torch.arange(10).repeat_interleave(8)).float().mean().item(),
                    "interpolation_accuracy": (path_predictions == torch.arange(10).repeat_interleave(11)).float().mean().item(),
                    "interpolation_adjacent_pixel_l1": path_images.reshape(10, 11, 1, 28, 28).diff(dim=1).abs().mean().item(),
                    "nearest_distance_mean": distances.mean().item(),
                    "nearest_distance_median": distances.median().item(),
                    "displayed_failure_count": len(failures),
                    "sampling_rule": "predeclared first indices; interpolation and label switching use separate fixed seeds",
                }
                write_json(packet_dir / f"{variant}-visual-selection.json", {
                    "sample_indices": [d * count + j for d in range(10) for j in range(8)],
                    "nearest_generated_indices": selected.tolist(), "nearest_training_indices_in_split": nearest[selected].tolist(),
                    "failure_indices": failures.tolist(), "rule": "first indices, no quality-based selection",
                    "sampling_seed": sample_seed, "interpolation_seed": sample_seed + 1, "condition_seed": sample_seed + 2})
            del model

    for variant, _, _ in variants:
        visuals[f"scale_{variant}"] = np.stack([visuals.pop(f"scale_{variant}_{tau}") for tau in study["scales"]])
    for family in ("vae", "gan"):
        progress = []
        for epoch in cfg[family]["evaluation_epochs"]:
            model, _ = load_model(root, family, study["seeds"][0], epoch, device)
            z = torch.randn(3, model.latent_dim, generator=torch.Generator().manual_seed(sample_seed + 3))
            images, _ = generate(model, family, 0, 0, device,
                                 latent=z.repeat(3, 1), labels=torch.tensor([1, 5, 8]).repeat_interleave(3))
            progress.append(images.reshape(3, 3, 1, 28, 28).numpy())
        visuals[f"progress_{family}"] = np.stack(progress)
        visuals[f"progress_{family}_epochs"] = np.array(cfg[family]["evaluation_epochs"])
    np.savez_compressed(root / "visuals.npz", **visuals)
    # A real-train vs held-out reference diagnoses the metric scale without using
    # generated data or selecting any model hyperparameters.
    chosen = []
    rng = np.random.default_rng(sample_seed)
    for digit in range(10):
        ids = torch.nonzero(data["train"][1] == digit).flatten().numpy()
        chosen.extend(rng.permutation(ids)[:count].tolist())
    chosen = torch.tensor(chosen)
    with torch.inference_mode():
        baseline_predictions = evaluator.classifier(train_features[chosen].to(device)).argmax(1).cpu()
    baseline = classwise_metrics(reference_features, reference_labels, train_features[chosen],
                                data["train"][1][chosen], baseline_predictions)
    write_json(root / "real_baseline.json", baseline)
    write_json(root / "visual_diagnostics.json", diagnostics)
    write_json(root / "evaluation-manifest.json", {
        "elapsed_seconds": time.perf_counter() - start, "samples_per_class": count,
        "sampling_seed": sample_seed, "feature_space": "raw frozen MNIST classifier 128-dimensional ReLU features",
        "kernel": "(x dot y / 128 + 1)^3; unbiased MMD squared; negative values retained",
        "precision_recall": "k=3, Euclidean balls, self excluded, macro over requested class",
        "evaluator_checkpoint_sha256": sha256(root / "evaluator/best.pt"),
        "visuals_sha256": sha256(root / "visuals.npz"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "generative").glob("comparison_*.py")},
    })
    write_json(root / "evaluation_summary.json", {"wall_seconds": time.perf_counter() - start,
        "test_accuracy": test_accuracy, "pca": read_json(root / "pca.json"), "visual_diagnostics": diagnostics})
    print(f"Final evaluation completed in {time.perf_counter() - start:.1f}s", flush=True)
