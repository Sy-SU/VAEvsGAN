"""Reproducible paired experiments, using the repository's unchanged models.

Training uses a lossless in-memory cache of the existing MNIST PNG data. All
random draws used for evaluation are isolated from the training RNG streams.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gan.checkpoint import load_checkpoint as load_gan, save_checkpoint as save_gan
from gan.model import ConditionalDiscriminator, ConditionalGenerator
from generative.comparison_metrics import (
    MNISTEvaluator, classwise_metrics, extract_features, preserve_rng,
)
from generative.data import MNISTPNGDataset
from generative.runtime import resolve_device, set_seed
from vae.checkpoint import load_checkpoint as load_vae, save_checkpoint as save_vae
from vae.model import ConvVAE, vae_loss

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_config(path: Path, smoke=False) -> dict:
    cfg = tomllib.loads(path.read_text())
    for name in ("data_dir", "run_dir", "report_dir"):
        p = Path(cfg["study"][name])
        cfg["study"][name] = str(p if p.is_absolute() else ROOT / p)
    cfg["smoke"] = smoke
    if smoke:
        cfg["study"]["run_dir"] += "-smoke"
        cfg["study"]["report_dir"] = str(Path(cfg["study"]["run_dir"]) / "report")
        cfg["study"].update(seeds=[42], process_samples_per_class=8, final_samples_per_class=8)
        cfg["evaluator"]["epochs"] = 1
        for model in ("vae", "gan"):
            cfg[model].update(epochs=1, evaluation_epochs=[1])
    return cfg


def load_images(data_dir: Path, cache_dir: Path):
    cache = cache_dir / "mnist-tensors.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)
    result = {}
    for split in ("train", "test"):
        dataset = MNISTPNGDataset(data_dir, split)
        images = torch.empty((len(dataset), 1, 28, 28), dtype=torch.uint8)
        labels = torch.tensor([label for _, label in dataset.samples])
        for i, (path, _) in enumerate(dataset.samples):
            with Image.open(path) as image:
                images[i, 0] = torch.from_numpy(np.array(image.convert("L"), copy=True))
        result[split + "_images"] = images
        result[split + "_labels"] = labels
        print(f"Cached {split}: {len(labels)} images", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache)
    write_json(cache_dir / "data-provenance.json", {
        "tensor_cache_sha256": sha256(cache),
        "labels_sha256": {s: sha256(data_dir / s / "labels.csv") for s in ("train", "test")},
        "source": "Existing MNIST PNG files, grayscale uint8, no augmentation; divide by 255 at load",
    })
    return result


def stratified_split(labels: torch.Tensor, validation_size: int, seed: int):
    rng = np.random.default_rng(seed)
    lab = labels.numpy()
    counts = np.bincount(lab, minlength=10)
    exact = counts * validation_size / len(lab)
    quotas = np.floor(exact).astype(int)
    for digit in np.argsort(-(exact - quotas), kind="stable")[:validation_size - quotas.sum()]:
        quotas[digit] += 1
    train, validation = [], []
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(lab == digit))
        validation.extend(ids[:quotas[digit]].tolist())
        train.extend(ids[quotas[digit]:].tolist())
    return {"train": sorted(train), "validation": sorted(validation), "seed": seed}


def balanced_indices(labels, pool, count, seed):
    rng = np.random.default_rng(seed)
    pool = np.array(pool, dtype=np.int64)
    labels = labels.numpy()
    selected = []
    for digit in range(10):
        candidates = pool[labels[pool] == digit]
        if len(candidates) < count:
            raise ValueError(f"Digit {digit} has {len(candidates)} samples, need {count}")
        selected.extend(rng.permutation(candidates)[:count].tolist())
    return torch.tensor(selected, dtype=torch.long)


def get_data(cfg):
    study = cfg["study"]
    run_dir = Path(study["run_dir"])
    # Smoke and formal studies share only the raw, lossless source cache.
    raw = load_images(Path(study["data_dir"]), ROOT / "outputs/comparison-data")
    split = stratified_split(raw["train_labels"], study["validation_size"], study["split_seed"])
    if cfg["smoke"]:
        split["train"] = balanced_indices(raw["train_labels"], split["train"], 64, 100).tolist()
        split["validation"] = balanced_indices(raw["train_labels"], split["validation"], 16, 101).tolist()
    if (run_dir / "split.json").exists() and read_json(run_dir / "split.json") != split:
        raise ValueError("Existing split differs from requested protocol")
    write_json(run_dir / "split.json", split)
    data = {}
    for part in ("train", "validation"):
        idx = torch.tensor(split[part])
        data[part] = (raw["train_images"][idx].float().div_(255), raw["train_labels"][idx])
    data["test"] = (raw["test_images"].float().div_(255), raw["test_labels"])
    reference = {}
    for part, count in (("validation", study["process_samples_per_class"]), ("test", study["final_samples_per_class"])):
        x, y = data[part]
        ids = balanced_indices(y, range(len(y)), count, study["sampling_seed"])
        reference[part] = (x[ids], y[ids])
        write_json(run_dir / f"{part}-reference-indices.json", {
            "indices_in_split": ids.tolist(),
            "source_indices": [split[part][i] for i in ids.tolist()] if part == "validation" else ids.tolist(),
        })
    return data, reference


def loader_for(pair, batch_size, seed, device, shuffle=True):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*pair), batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, pin_memory=device.type == "cuda", generator=generator)
    return loader, generator


@torch.inference_mode()
def evaluator_accuracy(evaluator, pair, device):
    _, predictions = extract_features(evaluator, pair[0], device)
    return (predictions == pair[1]).float().mean().item()


def prepare_evaluator(cfg, data, device, allow_train):
    directory = Path(cfg["study"]["run_dir"]) / "evaluator"
    path = directory / "best.pt"
    with preserve_rng():
        set_seed(cfg["evaluator"]["seed"])
        evaluator = MNISTEvaluator().to(device)
        if path.exists() and (directory / "complete.json").exists():
            evaluator.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
            evaluator.eval()
            return evaluator
        if not allow_train:
            raise FileNotFoundError("Train the evaluator first (--stage train)")
        directory.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(evaluator.parameters(), lr=cfg["evaluator"]["learning_rate"])
        loader, _ = loader_for(data["train"], cfg["study"]["batch_size"], cfg["evaluator"]["seed"], device)
        best = -1.0
        history = []
        start = time.perf_counter()
        for epoch in range(1, cfg["evaluator"]["epochs"] + 1):
            evaluator.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(evaluator(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            evaluator.eval()
            accuracy = evaluator_accuracy(evaluator, data["validation"], device)
            history.append({"epoch": epoch, "loss": total / len(data["train"][1]), "validation_accuracy": accuracy})
            if accuracy > best:
                best = accuracy
                torch.save({"state_dict": evaluator.state_dict(), "epoch": epoch, "validation_accuracy": best}, path)
            print(f"Evaluator {epoch}: validation accuracy={accuracy:.4%}", flush=True)
            write_json(directory / "history.json", history)
        evaluator.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
        evaluator.eval()
        write_json(directory / "complete.json", {"validation_accuracy": best, "wall_seconds": time.perf_counter() - start,
                                                  "checkpoint_sha256": sha256(path), "config": cfg["evaluator"]})
        return evaluator


@torch.inference_mode()
def generate(model, family, per_class, seed, device, tau=1.0, latent=None, labels=None):
    was_training = model.training
    model.eval()
    try:
        if latent is None:
            rng = torch.Generator().manual_seed(seed)
            latent = torch.randn(10 * per_class, model.latent_dim, generator=rng) * tau
            labels = torch.arange(10).repeat_interleave(per_class)
        result = []
        for i in range(0, len(latent), 512):
            z = latent[i:i + 512].to(device)
            y = labels[i:i + 512].to(device)
            images = model.decode(z, y) if family == "vae" else model(z, y).add(1).div(2)
            result.append(images.clamp(0, 1).cpu())
        return torch.cat(result), labels.cpu()
    finally:
        model.train(was_training)


def evaluate_model(model, family, evaluator, reference_features, count, sample_seed, device, tau=1.0):
    with preserve_rng():
        images, labels = generate(model, family, count, sample_seed, device, tau=tau)
        features, predictions = extract_features(evaluator, images, device)
        metrics = classwise_metrics(reference_features[0], reference_features[1], features, labels, predictions)
    return metrics, images, features, labels, predictions


def history_csv(path, rows):
    flat = []
    for row in rows:
        item = {k: v for k, v in row.items() if not isinstance(v, dict)}
        item.update({"validation_" + k: v for k, v in row.get("evaluation", {}).get("macro", {}).items()})
        flat.append(item)
    keys = sorted(set().union(*(row.keys() for row in flat)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def train_model(cfg, family, seed, data, evaluator, reference_features, device):
    study, settings = cfg["study"], cfg[family]
    directory = Path(study["run_dir"]) / family / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config_fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if (directory / "complete.json").exists():
        if read_json(directory / "complete.json")["config_fingerprint"] != config_fingerprint:
            raise ValueError("Run configuration changed; use a new run_dir")
        print(f"Reusing completed {family} seed {seed}", flush=True)
        return
    set_seed(seed)
    model = (ConvVAE(study["latent_dim"]) if family == "vae" else ConditionalGenerator(
        study["latent_dim"], label_embedding_dim=settings["label_embedding_dim"])).to(device)
    discriminator = ConditionalDiscriminator().to(device) if family == "gan" else None
    betas = (0.9, 0.999) if family == "vae" else (settings["adam_beta1"], settings["adam_beta2"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], betas=betas)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=settings["learning_rate"], betas=betas) if discriminator else None
    loader, loader_rng = loader_for(data["train"], study["batch_size"], seed, device)
    train_cfg = dict(settings, latent_dim=study["latent_dim"], seed=seed, batch_size=study["batch_size"],
                     split_seed=study["split_seed"], training_samples=len(data["train"][1]), conditional=True)
    write_json(directory / "config.json", train_cfg)
    history, start_epoch, cumulative = [], 1, 0.0
    resume = directory / "resume.pt"
    prior_wall = 0.0
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)  # Locally written trusted run state.
        if state["config_fingerprint"] != config_fingerprint:
            raise ValueError("Interrupted run configuration changed; use a new run_dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if discriminator:
            discriminator.load_state_dict(state["discriminator"])
            d_optimizer.load_state_dict(state["d_optimizer"])
        history = state["history"]
        start_epoch, cumulative = state["epoch"] + 1, history[-1]["train_seconds"]
        loader_rng.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        prior_wall = state["wall_seconds"]
        print(f"Resuming {family} seed {seed}, epoch {start_epoch}", flush=True)
    wall_start = time.perf_counter()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch, settings["epochs"] + 1):
        model.train()
        if discriminator:
            discriminator.train()
        sync(device)
        start = time.perf_counter()
        totals = {}
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch = len(labels)
            if family == "vae":
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mu, logvar = model(images, labels)
                losses = vae_loss(reconstruction, images, mu, logvar)
                losses.total.backward()
                optimizer.step()
                values = {"loss": losses.total, "reconstruction": losses.reconstruction, "kl": losses.kl}
            else:
                images = images.mul(2).sub(1)
                real = torch.ones(batch, device=device)
                fake = torch.zeros(batch, device=device)
                wrong = (labels + torch.randint(1, 10, labels.shape, device=device)) % 10
                latent = torch.randn(batch, model.latent_dim, device=device)
                generated = model(latent, labels)
                d_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(images, labels)
                wrong_logits = discriminator(images, wrong)
                fake_logits = discriminator(generated.detach(), labels)
                d_loss = criterion(real_logits, real) + settings["conditional_negative_weight"] * (
                    criterion(fake_logits, fake) + criterion(wrong_logits, fake))
                d_loss.backward()
                d_optimizer.step()
                discriminator.requires_grad_(False)
                optimizer.zero_grad(set_to_none=True)
                g_loss = criterion(discriminator(generated, labels), real)
                g_loss.backward()
                optimizer.step()
                discriminator.requires_grad_(True)
                values = {"g_loss": g_loss, "d_loss": d_loss,
                          "real_score": real_logits.detach().sigmoid().mean(),
                          "fake_score": fake_logits.detach().sigmoid().mean(),
                          "wrong_label_score": wrong_logits.detach().sigmoid().mean()}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * batch
        sync(device)
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        row = {"epoch": epoch, "epoch_seconds": elapsed, "train_seconds": cumulative,
               **{key: value / len(data["train"][1]) for key, value in totals.items()}}
        if epoch in settings["evaluation_epochs"]:
            eval_start = time.perf_counter()
            metrics, _, _, _, _ = evaluate_model(model, family, evaluator, reference_features,
                study["process_samples_per_class"], study["sampling_seed"], device)
            row["evaluation"] = metrics
            row["evaluation_seconds"] = time.perf_counter() - eval_start
            path = directory / f"epoch_{epoch:03d}.pt"
            if family == "vae":
                save_vae(path, model, epoch, train_cfg)
            else:
                save_gan(path, model, discriminator, epoch, train_cfg)
            print(f"{family} seed={seed} epoch={epoch} train={cumulative:.1f}s val={metrics['macro']}", flush=True)
        else:
            print(f"{family} seed={seed} epoch={epoch} {elapsed:.2f}s", flush=True)
        history.append(row)
        write_json(directory / "history.json", history)
        history_csv(directory / "history.csv", history)
        state = {"config_fingerprint": config_fingerprint, "epoch": epoch, "history": history,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "loader_rng": loader_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "wall_seconds": prior_wall + time.perf_counter() - wall_start}
        if discriminator:
            state.update(discriminator=discriminator.state_dict(), d_optimizer=d_optimizer.state_dict())
        torch.save(state, resume.with_suffix(".tmp"))
        resume.with_suffix(".tmp").replace(resume)
    write_json(directory / "complete.json", {"config_fingerprint": config_fingerprint,
        "epochs": settings["epochs"], "train_seconds": cumulative,
        "wall_seconds": prior_wall + time.perf_counter() - wall_start,
        "parameters_total": sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in discriminator.parameters()) if discriminator else 0),
        "parameters_generate": sum(p.numel() for p in (list(model.decoder_input.parameters()) + list(model.decoder.parameters()) if family == "vae" else model.parameters())),
        "checkpoints": {p.name: sha256(p) for p in directory.glob("epoch_*.pt")}})


def run_study(config_path: Path, stage: str, smoke=False):
    cfg = load_config(config_path, smoke)
    study = cfg["study"]
    run_dir = Path(study["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(study["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(study["device"])
    if (run_dir / "config.json").exists() and read_json(run_dir / "config.json") != cfg:
        raise ValueError("Study configuration changed; choose a new run_dir")
    write_json(run_dir / "config.json", cfg)
    if not (run_dir / "environment.json").exists():
        write_json(run_dir / "environment.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "tf32": False, "cpu_threads": study["cpu_threads"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for folder in ("vae", "gan", "generative", "tools") for p in (ROOT / folder).glob("*.py")},
        })
    if stage in ("train", "evaluate", "all"):
        data, references = get_data(cfg)
        evaluator = prepare_evaluator(cfg, data, device, stage in ("train", "all"))
        if stage in ("train", "all"):
            validation_features, _ = extract_features(evaluator, references["validation"][0], device)
            for seed in study["seeds"]:
                for family in ("vae", "gan"):
                    train_model(cfg, family, seed, data, evaluator,
                                (validation_features, references["validation"][1]), device)
        if stage in ("evaluate", "all"):
            from generative.comparison_evaluate import evaluate_study
            evaluate_study(cfg, data, references, evaluator, device)
    if stage in ("report", "all"):
        from generative.comparison_report import create_report
        create_report(cfg)
    print(f"Completed stage={stage}; results: {run_dir}", flush=True)
