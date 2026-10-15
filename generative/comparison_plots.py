"""Publication figures for the reproducible MNIST comparison experiment.

This module only reads experiment artifacts. It does not train or evaluate models.
All error bars summarize independent training seeds, using sample standard
deviation (ddof=1); digits and generated images are not treated as extra runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


VARIANTS = ("VAE20", "GAN20", "GAN50")
VARIANT_LABELS = {
    "VAE20": "VAE · epoch 20",
    "GAN20": "GAN · epoch 20",
    "GAN50": "GAN · epoch 50",
}
COLORS = {"VAE20": "#247BA0", "GAN20": "#E99335", "GAN50": "#C44E62"}
DIGIT_COLORS = plt.get_cmap("tab10")(np.arange(10))
METRICS = ("accuracy", "mmd2", "precision", "recall")
METRIC_LABELS = {
    "accuracy": "Condition accuracy (%) ↑",
    "mmd2": "Feature MMD² estimate ↓",
    "precision": "Feature precision (%) ↑",
    "recall": "Feature recall (%) ↑",
}
TAUS = (0.5, 1.0, 1.5, 2.0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _value(row: dict, metric: str) -> float:
    value = float(row["macro"][metric])
    return value * 100 if metric != "mmd2" else value


def _style_axis(axis, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
    axis.set_axisbelow(True)


def _metric_ticks(axis, metric: str) -> None:
    if metric == "mmd2":
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


def _strip(images: np.ndarray, gap: int = 2) -> np.ndarray:
    """Join NCHW/NHW images with an explicit black separator."""
    images = np.asarray(images, dtype=float)
    if images.ndim == 4:
        images = images[:, 0]
    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3 or not len(images):
        raise ValueError(f"Expected nonempty NCHW/NHW images, got {images.shape}")
    separator = np.zeros((images.shape[1], gap))
    parts = [part for im in images for part in (im, separator)]
    return np.concatenate(parts[:-1], axis=1)


def _show_strip(axis, images: np.ndarray, *, label: str | None = None) -> None:
    axis.imshow(_strip(images), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if label is not None:
        axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=11)


class _FigureWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.figures: list[dict[str, str]] = []

    def save(self, figure, name: str, description: str, *, statistical: bool = False) -> None:
        target = self.directory / f"{name}.png"
        try:
            figure.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
            if statistical:
                figure.savefig(
                    self.directory / f"{name}.svg", bbox_inches="tight", facecolor="white"
                )
        finally:
            plt.close(figure)
        self.figures.append({"name": name, "path": str(target.resolve()), "description": description})


def _sample_figures(writer: _FigureWriter, arrays: Any, seeds: list[int]) -> None:
    display_seed = 42 if 42 in seeds else seeds[0]
    variants = [v for v in VARIANTS if f"samples_{v}_s{display_seed}" in arrays]
    if "real_images" in arrays and variants:
        for start in (0, 5):
            figure, axes = plt.subplots(5, len(variants) + 1, figsize=(16.5, 5.0), squeeze=False)
            for row, digit in enumerate(range(start, start + 5)):
                _show_strip(axes[row, 0], arrays["real_images"][digit], label=f"Digit {digit}")
                for col, variant in enumerate(variants, 1):
                    _show_strip(axes[row, col], arrays[f"samples_{variant}_s{display_seed}"][digit])
            for axis, name in zip(axes[0], ["Real MNIST", *[VARIANT_LABELS[v] for v in variants]]):
                axis.set_title(name, pad=9)
            figure.suptitle(f"Conditioned samples · digits {start}–{start + 4} · training seed {display_seed}", y=0.99)
            figure.text(0.5, 0.015, "Eight fixed samples per digit. Shared noise inputs do not align semantics across models.", ha="center", fontsize=9)
            figure.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.08, hspace=0.2, wspace=0.12)
            writer.save(figure, f"samples_digits_{start}_{start + 4}", "真实 MNIST 与三个模型检查点的全类别样本网格；固定采样种子，不按生成效果挑选样本。")

    for variant in VARIANTS:
        found_seeds = [s for s in seeds if f"samples_{variant}_s{s}" in arrays]
        if not found_seeds:
            continue
        figure, axes = plt.subplots(10, len(found_seeds), figsize=(12.0, 7.8), squeeze=False)
        for row in range(10):
            for col, seed in enumerate(found_seeds):
                _show_strip(axes[row, col], arrays[f"samples_{variant}_s{seed}"][row], label=str(row) if col == 0 else None)
        for axis, seed in zip(axes[0], found_seeds):
            axis.set_title(f"Training seed {seed}", pad=8)
        figure.suptitle(f"{VARIANT_LABELS[variant]} · independent training seeds", y=0.985)
        figure.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.02, hspace=0.14, wspace=0.12)
        writer.save(figure, f"seed_samples_{variant}", f"{VARIANT_LABELS[variant]} 在独立训练种子上的逐类样本，作为主样本网格的附录。")


def _metric_figures(writer: _FigureWriter, rows: list[dict]) -> None:
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    figure, axes = plt.subplots(1, 4, figsize=(15.6, 4.0), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        for index, variant in enumerate(variants):
            selected = sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["seed"])
            values = np.array([_value(r, metric) for r in selected])
            mean, sd = _mean_sd(values)
            jitter = np.linspace(-0.1, 0.1, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(index + jitter, values, color=COLORS[variant], s=30, alpha=0.75, zorder=3)
            axis.errorbar(index, mean, yerr=sd, fmt="D", color="black", ms=5, capsize=5, elinewidth=1.3, zorder=4)
        axis.set_xticks(range(len(variants)), variants)
        axis.set_title(METRIC_LABELS[metric])
        _style_axis(axis)
        _metric_ticks(axis, metric)
    figure.suptitle("Final generation metrics · colored dots = training seeds · black diamonds = mean ± 1 SD", fontsize=12)
    writer.save(figure, "generation_metrics", "生成指标的各训练种子原始值与均值 ± 样本标准差；标准差来自独立训练种子。", statistical=True)

    figure, axes = plt.subplots(1, len(variants), figsize=(14.7, 4.6), squeeze=False, constrained_layout=True)
    for axis, variant in zip(axes[0], variants):
        matrices = np.array([r["confusion"] for r in rows if r["variant"] == variant], dtype=float)
        totals = matrices.sum(axis=2, keepdims=True)
        normalized = np.divide(matrices, totals, out=np.zeros_like(matrices), where=totals > 0).mean(axis=0)
        im = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for digit in range(10):
            for predicted in range(10):
                value = normalized[digit, predicted]
                if value >= 0.015 or predicted == digit:
                    axis.text(predicted, digit, f"{value * 100:.0f}", fontsize=7, ha="center", va="center", color="white" if value > 0.55 else "#333333")
        axis.set(xticks=range(10), yticks=range(10), xlabel="Classifier prediction", ylabel="Requested digit", title=VARIANT_LABELS[variant])
    figure.colorbar(im, ax=list(axes[0]), shrink=0.83, label="Mean row probability")
    figure.suptitle("Condition confusion · each row normalized within each training seed", fontsize=12)
    writer.save(figure, "condition_confusion", "各训练种子先按请求数字行归一化，再跨种子平均的条件混淆矩阵；格内为百分比。", statistical=True)

    figure, axis = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    for offset, variant in zip((-0.15, 0, 0.15), variants):
        selected = [r for r in rows if r["variant"] == variant]
        values = np.array([[next(c["accuracy"] for c in r["per_class"] if int(c["digit"]) == d) * 100 for d in range(10)] for r in selected])
        means = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(10)
        for seed_values in values:
            axis.scatter(np.arange(10) + offset, seed_values, color=COLORS[variant], s=15, alpha=0.32)
        axis.errorbar(np.arange(10) + offset, means, yerr=sd, color=COLORS[variant], marker="o", markersize=4, linewidth=1.4, capsize=3, label=VARIANT_LABELS[variant])
    axis.set(xticks=range(10), xlabel="Requested digit", ylabel="Condition accuracy (%)", title="Per-digit condition accuracy · mean ± 1 SD across training seeds")
    axis.legend(loc="best", fontsize=9)
    _style_axis(axis)
    writer.save(figure, "per_digit_accuracy", "逐数字的条件准确率及训练种子间波动，避免总体均值掩盖薄弱类别。", statistical=True)

    figure, axis = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    markers = ("o", "s", "^")
    all_seeds = sorted({int(r["seed"]) for r in rows})
    for variant in variants:
        selected = [r for r in rows if r["variant"] == variant]
        for row in selected:
            axis.scatter(_value(row, "recall"), _value(row, "precision"), color=COLORS[variant], marker=markers[all_seeds.index(int(row["seed"])) % len(markers)], s=60, alpha=0.85)
        precision = [_value(r, "precision") for r in selected]
        recall = [_value(r, "recall") for r in selected]
        x, x_sd = _mean_sd(recall)
        y, y_sd = _mean_sd(precision)
        axis.errorbar(x, y, xerr=x_sd, yerr=y_sd, fmt="D", color=COLORS[variant], markeredgecolor="black", capsize=4, ms=6, label=VARIANT_LABELS[variant])
    axis.set(xlabel="Feature recall / coverage (%) ↑", ylabel="Feature precision (%) ↑", title="Generation quality and coverage · independent seeds")
    variant_legend = axis.legend(loc="best", fontsize=9)
    axis.add_artist(variant_legend)
    axis.legend(handles=[Line2D([], [], marker=markers[i % len(markers)], linestyle="", color="#555555", label=f"Seed {seed}") for i, seed in enumerate(all_seeds)], loc="lower left", fontsize=8)
    _style_axis(axis)
    writer.save(figure, "precision_recall", "共同特征空间中的生成精确率与覆盖率；各点是训练种子，菱形和误差线为跨种子均值与标准差。", statistical=True)

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), constrained_layout=True)
    for axis, metric in zip(axes, ("mmd2", "precision", "recall")):
        table = []
        for variant in variants:
            selected = [r for r in rows if r["variant"] == variant]
            table.append([np.mean([next(float(c[metric]) for c in r["per_class"] if int(c["digit"]) == digit) for r in selected]) for digit in range(10)])
        values = np.array(table) * (1 if metric == "mmd2" else 100)
        im = axis.imshow(values, aspect="auto", cmap="magma_r" if metric == "mmd2" else "YlGnBu")
        axis.set(xticks=range(10), yticks=range(len(variants)), yticklabels=variants, title=METRIC_LABELS[metric])
        for row in range(len(variants)):
            for col in range(10):
                val = values[row, col]
                red, green, blue, _ = im.cmap(im.norm(val))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.48 else "#222222"
                axis.text(col, row, f"{val:.2g}" if metric == "mmd2" else f"{val:.1f}", ha="center", va="center", fontsize=8, color=text_color)
        figure.colorbar(im, ax=axis, shrink=0.9, pad=0.02)
    axes[-1].set_xlabel("Requested digit")
    figure.suptitle("Per-digit generation metrics · mean across training seeds", fontsize=12)
    writer.save(figure, "per_digit_metrics", "逐类 MMD²、特征精确率和覆盖率热图；每一列对应请求生成的数字。", statistical=True)


def _efficiency_figure(writer: _FigureWriter, rows: list[dict]) -> None:
    fields = (("train_seconds", "Training time (minutes)", 1 / 60), ("images_per_second", "Generation throughput (images/s)", 1), ("parameters_generate", "Generation parameters (millions)", 1e-6))
    if not all(all(k in row for k in ("train_seconds", "parameters_generate", "generation_benchmark")) for row in rows):
        return
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    for axis, (field, label, scale) in zip(axes, fields):
        for index, variant in enumerate(variants):
            selected = [r for r in rows if r["variant"] == variant]
            vals = [(r["generation_benchmark"][field] if field == "images_per_second" else r[field]) * scale for r in selected]
            mean, sd = _mean_sd(vals)
            axis.bar(index, mean, color=COLORS[variant], width=0.62, alpha=0.78)
            axis.errorbar(index, mean, yerr=sd, color="#333333", capsize=4, fmt="none")
            axis.scatter(index + np.linspace(-0.07, 0.07, len(vals)), vals, s=18, color="#333333", zorder=3)
            axis.annotate(f"{mean:,.2f}" if field == "parameters_generate" else f"{mean:,.0f}", (index, mean + sd), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        axis.set_xticks(range(len(variants)), variants)
        axis.set_title(label)
        axis.margins(y=0.2)
        _style_axis(axis)
    figure.suptitle("Measured compute cost · same hardware and generation protocol", fontsize=12)
    writer.save(figure, "compute_cost", "实测训练耗时、生成吞吐量与生成端参数量；吞吐量使用每次训练所得模型的重复计时中位数。", statistical=True)


def _pca_figure(writer: _FigureWriter, arrays: Any) -> None:
    keys = ["pca_real", *[f"pca_{v}" for v in VARIANTS]]
    if not all(key in arrays for key in [*keys, "pca_labels"]):
        return
    labels = np.asarray(arrays["pca_labels"])
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.7), sharex=True, sharey=True)
    all_points = np.concatenate([arrays[key] for key in keys])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((upper - lower) * 0.035, 1e-6)
    for axis, key, title in zip(axes, keys, ["Real MNIST", *[VARIANT_LABELS[v] for v in VARIANTS]]):
        points = np.asarray(arrays[key])
        if len(points) != len(labels):
            raise ValueError(f"{key}: points and pca_labels have different lengths")
        for digit in range(10):
            selected = labels == digit
            axis.scatter(points[selected, 0], points[selected, 1], s=4, alpha=0.35, color=DIGIT_COLORS[digit], linewidths=0, rasterized=True)
        axis.set(title=title, xlabel="Shared PC 1", xlim=(lower[0] - pad[0], upper[0] + pad[0]), ylim=(lower[1] - pad[1], upper[1] + pad[1]))
        _style_axis(axis, grid=False)
    axes[0].set_ylabel("Shared PC 2")
    figure.legend(handles=[Line2D([], [], linestyle="", marker="o", markersize=6, color=DIGIT_COLORS[d], label=str(d)) for d in range(10)], loc="lower center", ncol=10, title="Digit", bbox_to_anchor=(0.5, 0.02), frameon=False)
    figure.suptitle("Real and generated images in the same MNIST feature projection", y=0.98)
    figure.subplots_adjust(top=0.84, bottom=0.26, left=0.05, right=0.99, wspace=0.12)
    writer.save(figure, "shared_feature_pca", "共同 MNIST 分类器特征及同一 PCA 投影中的真实与生成分布；坐标和范围一致，二维重叠仅作定性解释。", statistical=True)


def _history_figures(writer: _FigureWriter, run_root: Path, seeds: list[int]) -> None:
    histories: dict[str, dict[int, list[dict]]] = {}
    for model in ("vae", "gan"):
        histories[model] = {}
        for seed in seeds:
            path = run_root / model / f"seed_{seed}" / "history.json"
            if path.exists():
                histories[model][seed] = _read_json(path)
    colors = {"vae": COLORS["VAE20"], "gan": COLORS["GAN50"]}
    seed_styles = ("-", "--", ":", "-.")

    has_evaluation = any("evaluation" in row for runs in histories.values() for history in runs.values() for row in history)
    if has_evaluation:
        for against_time in (False, True):
            figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
            for axis, metric in zip(axes.flat, METRICS):
                for model, runs in histories.items():
                    by_epoch: dict[int, list[float]] = {}
                    for index, (seed, history) in enumerate(runs.items()):
                        evaluated = [r for r in history if r.get("evaluation") and metric in r["evaluation"].get("macro", {})]
                        if not evaluated:
                            continue
                        x = [float(r["train_seconds"]) / 60 if against_time else r["epoch"] for r in evaluated]
                        y = [_value(r["evaluation"], metric) for r in evaluated]
                        axis.plot(x, y, color=colors[model], alpha=0.7 if against_time else 0.26, linewidth=1.3, linestyle=seed_styles[index % len(seed_styles)], marker=".", label=f"{model.upper()} s{seed}" if against_time else None)
                        for row, value in zip(evaluated, y):
                            by_epoch.setdefault(int(row["epoch"]), []).append(value)
                    if by_epoch and not against_time:
                        epochs = sorted(by_epoch)
                        means, sds = zip(*[_mean_sd(by_epoch[e]) for e in epochs])
                        means, sds = np.array(means), np.array(sds)
                        axis.plot(epochs, means, marker="o", markersize=3, linewidth=2, color=colors[model], label=model.upper())
                        axis.fill_between(epochs, means - sds, means + sds, color=colors[model], alpha=0.14)
                axis.set(xlabel="Cumulative training time (minutes)" if against_time else "Epoch", ylabel=METRIC_LABELS[metric])
                _style_axis(axis)
                _metric_ticks(axis, metric)
            axes.flat[0].legend(fontsize=8, ncols=2)
            figure.suptitle("Validation metrics during training · individual seed trajectories" if against_time else "Validation metrics during training · seed mean ± 1 SD", fontsize=12)
            writer.save(figure, "training_metrics_time" if against_time else "training_metrics_epoch", "随累计训练耗时变化的各训练种子指标，时间轴不进行插值平均。" if against_time else "随训练轮数变化的生成指标；细线是训练种子，粗线和阴影为同一轮的均值与标准差。", statistical=True)

    for model, keys in (("vae", ("loss", "reconstruction", "kl")), ("gan", ("g_loss", "d_loss"))):
        runs = histories[model]
        if not runs:
            continue
        figure, axes = plt.subplots(1, len(keys), figsize=(12, 3.8), squeeze=False, constrained_layout=True)
        for axis, key in zip(axes[0], keys):
            epoch_values: dict[int, list[float]] = {}
            for index, (seed, history) in enumerate(runs.items()):
                history = [r for r in history if key in r]
                axis.plot([r["epoch"] for r in history], [r[key] for r in history], color=colors[model], alpha=0.35, linewidth=1, linestyle=seed_styles[index % len(seed_styles)], label=f"Seed {seed}")
                for row in history:
                    epoch_values.setdefault(int(row["epoch"]), []).append(float(row[key]))
            if epoch_values:
                epochs = sorted(epoch_values)
                means, sds = zip(*[_mean_sd(epoch_values[e]) for e in epochs])
                means, sds = np.array(means), np.array(sds)
                axis.plot(epochs, means, color=colors[model], linewidth=2, label="Mean ± SD")
                axis.fill_between(epochs, means - sds, means + sds, color=colors[model], alpha=0.12)
            axis.set(xlabel="Epoch", ylabel=key.replace("_", " "), title={"loss": "Total objective", "reconstruction": "Reconstruction BCE", "kl": "KL divergence", "g_loss": "Generator loss", "d_loss": "Discriminator loss"}[key])
            _style_axis(axis)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"{model.upper()} native training objectives · interpreted within this model only", fontsize=12)
        writer.save(figure, f"native_losses_{model}", f"{model.upper()} 自身训练目标的跨种子变化；损失值不用于 VAE 与 GAN 间的性能排序。", statistical=True)


def _progress_figures(writer: _FigureWriter, arrays: Any) -> None:
    for model, default_epochs in (("vae", (1, 5, 10, 15, 20)), ("gan", (1, 5, 10, 15, 20, 30, 40, 50))):
        key = f"progress_{model}"
        if key not in arrays:
            continue
        values = np.asarray(arrays[key])
        epochs = arrays[f"{key}_epochs"].tolist() if f"{key}_epochs" in arrays else default_epochs
        if len(values) != len(epochs):
            raise ValueError(f"{key}: expected {len(epochs)} snapshot epochs, got {len(values)}")
        figure, axes = plt.subplots(3, len(epochs), figsize=(15, 4.0), squeeze=False)
        for row, digit in enumerate((1, 5, 8)):
            for col, epoch in enumerate(epochs):
                _show_strip(axes[row, col], values[col, row], label=f"Digit {digit}" if col == 0 else None)
                if row == 0:
                    axes[row, col].set_title(f"Epoch {epoch}", fontsize=9, pad=8)
        figure.suptitle(f"{model.upper()} training progress · fixed labels and three fixed noise vectors", y=0.98)
        figure.subplots_adjust(left=0.07, right=0.99, top=0.79, bottom=0.035, hspace=0.18, wspace=0.1)
        writer.save(figure, f"training_progress_{model}", f"{model.upper()} 的训练过程：数字 1、5、8 各跟踪三个固定潜变量，避免仅展示一个样本。")


def _intervention_figures(writer: _FigureWriter, arrays: Any) -> None:
    for kind, title, caption in (
        ("interpolation", "Within-digit linear interpolation · t = 0, 0.1, …, 1 between fixed endpoints", "同类别的随机潜变量端点之间进行线性插值；各行固定数字，三种检查点按同一采样协议展示，不解释为跨模型语义对齐。"),
        ("conditional", "Change the requested digit while holding each noise column fixed", "每一列保持潜变量固定，从数字 0 到 9 改变条件标签，观察类别控制与笔画风格变化。"),
    ):
        variants = [v for v in VARIANTS if f"{kind}_{v}" in arrays]
        if not variants:
            continue
        figure, axes = plt.subplots(10, len(variants), figsize=(15.5, 7.2), squeeze=False)
        for col, variant in enumerate(variants):
            for digit in range(10):
                _show_strip(axes[digit, col], arrays[f"{kind}_{variant}"][digit], label=str(digit) if col == 0 else None)
            axes[0, col].set_title(VARIANT_LABELS[variant], fontsize=10, pad=10)
        figure.suptitle(title, fontsize=13, y=0.99)
        figure.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.025, hspace=0.16, wspace=0.12)
        writer.save(figure, f"latent_{kind}", caption)


def _scale_figures(writer: _FigureWriter, arrays: Any, rows: list[dict]) -> None:
    if rows:
        figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), constrained_layout=True)
        for axis, metric in zip(axes, METRICS):
            for variant in VARIANTS:
                selected = [r for r in rows if r["variant"] == variant]
                if not selected:
                    continue
                taus = sorted({float(r["tau"]) for r in selected})
                means, sds = zip(*[_mean_sd([_value(r, metric) for r in selected if float(r["tau"]) == tau]) for tau in taus])
                axis.errorbar(taus, means, yerr=sds, color=COLORS[variant], marker="o", markersize=4, capsize=3, linewidth=1.5, label=variant)
            axis.set(xticks=TAUS, xlabel="Noise scale τ", ylabel=METRIC_LABELS[metric])
            axis.axvline(1, color="#666666", linestyle=":", linewidth=0.8)
            _style_axis(axis)
            _metric_ticks(axis, metric)
        axes[0].legend(fontsize=8)
        figure.suptitle("Sampling sensitivity · z → τz · mean ± 1 SD across training seeds", fontsize=12)
        writer.save(figure, "noise_scale_metrics", "改变高斯噪声尺度后的质量、覆盖与条件准确率；τ=1 为训练先验，误差线来自训练种子。", statistical=True)

    variants = [v for v in VARIANTS if f"scale_{v}" in arrays]
    if not variants:
        return
    figure, axes = plt.subplots(len(variants), 4, figsize=(16, 4.2), squeeze=False)
    for row, variant in enumerate(variants):
        for col, tau in enumerate(TAUS):
            _show_strip(axes[row, col], arrays[f"scale_{variant}"][col], label=variant if col == 0 else None)
            if row == 0:
                axes[row, col].set_title(f"τ = {tau:g}", fontsize=10, pad=10)
    figure.suptitle("Noise-scale sample strips · each strip contains digits 0–9", fontsize=13, y=0.99)
    figure.text(0.5, 0.03, "For each digit, the underlying noise direction is held fixed across scales.", ha="center", fontsize=9)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.13, hspace=0.25, wspace=0.12)
    writer.save(figure, "noise_scale_samples", "每个数字使用同一潜变量方向、只改变尺度的样本网格；每条由数字 0–9 组成。")


def _nearest_figures(writer: _FigureWriter, arrays: Any) -> None:
    variants = [v for v in VARIANTS if f"nearest_generated_{v}" in arrays and f"nearest_real_{v}" in arrays]
    if variants:
        figure, axes = plt.subplots(len(variants) * 2, 1, figsize=(13, 7), squeeze=False)
        for index, variant in enumerate(variants):
            _show_strip(axes[index * 2, 0], arrays[f"nearest_generated_{variant}"], label=f"{variant}\nGenerated")
            _show_strip(axes[index * 2 + 1, 0], arrays[f"nearest_real_{variant}"], label="Train nearest")
        axes[0, 0].set_title("Generated samples and same-digit training nearest neighbors", fontsize=13, pad=15)
        figure.text(0.55, 0.018, "Nearest neighbors use the shared MNIST feature space. Visual similarity alone is not evidence of memorization.", ha="center", fontsize=9)
        figure.subplots_adjust(left=0.14, right=0.985, top=0.91, bottom=0.07, hspace=0.16)
        writer.save(figure, "nearest_neighbor_pairs", "生成样本及同类别训练集特征最近邻；相似性用于检查，不能单独证明模型记忆训练图像。")
    available = [v for v in VARIANTS if f"nearest_distances_{v}" in arrays]
    if available:
        figure, axis = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
        for variant in available:
            distances = np.asarray(arrays[f"nearest_distances_{variant}"], dtype=float).ravel()
            distances = np.sort(distances[np.isfinite(distances)])
            if len(distances):
                axis.plot(distances, np.arange(1, len(distances) + 1) / len(distances), color=COLORS[variant], linewidth=1.7, label=f"{variant} (n={len(distances):,})")
        axis.set(xlabel="Distance to nearest same-digit training feature", ylabel="Fraction of generated samples", title="Nearest-training-neighbor distance · empirical CDF", ylim=(0, 1.015))
        axis.legend(fontsize=9)
        _style_axis(axis)
        writer.save(figure, "nearest_neighbor_distances", "生成图像到同类训练特征最近邻距离的经验累积分布；距离大小不是独立的生成质量或记忆程度指标。", statistical=True)


def _failure_figures(writer: _FigureWriter, arrays: Any) -> None:
    for variant in VARIANTS:
        key = f"failure_images_{variant}"
        if key not in arrays:
            continue
        images = np.asarray(arrays[key])
        if len(images) == 0:
            continue
        count = min(len(images), 20)
        columns = min(10, count)
        row_count = (count + columns - 1) // columns
        figure, axes = plt.subplots(row_count, columns, figsize=(13.4, 1.7 * row_count + 0.6), squeeze=False)
        requested = arrays[f"failure_requested_{variant}"]
        predicted = arrays[f"failure_predicted_{variant}"]
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.axis("off")
                continue
            _show_strip(axis, images[index:index + 1])
            axis.set_title(f"{int(requested[index])} → {int(predicted[index])}", fontsize=10, pad=5)
        figure.suptitle(f"{VARIANT_LABELS[variant]} · classifier disagreements · requested → predicted", y=0.99, fontsize=12)
        figure.subplots_adjust(left=0.015, right=0.99, top=0.83, bottom=0.035, wspace=0.1, hspace=0.32)
        writer.save(figure, f"condition_failures_{variant}", f"{VARIANT_LABELS[variant]} 的分类器条件不一致案例；需结合图像判断，分类器预测不等于人工真值。")


def render_figures(run_root: Path, report_dir: Path) -> list[dict[str, str]]:
    """Render completed experiment artifacts into ``report_dir/figures``.

    Required inputs are final_metrics.json and visuals.npz. Noise-scale metrics
    and training histories are optional. An absent optional visual is skipped;
    incompatible array shapes fail explicitly instead of creating a wrong plot.
    Returned paths are absolute PNG paths; statistical plots also have sibling
    SVG files. The caller is responsible for linking figures into its report.
    """
    run_root, report_dir = Path(run_root), Path(report_dir)
    rows = _read_json(run_root / "final_metrics.json")
    if not rows:
        raise ValueError("final_metrics.json contains no completed runs")
    scale_path = run_root / "scale_metrics.json"
    scale_rows = _read_json(scale_path) if scale_path.exists() else []
    seeds = sorted({int(row["seed"]) for row in rows})
    writer = _FigureWriter(report_dir / "figures")
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.titlesize": 14,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        with np.load(run_root / "visuals.npz", allow_pickle=False) as arrays:
            _sample_figures(writer, arrays, seeds)
            _metric_figures(writer, rows)
            _efficiency_figure(writer, rows)
            _pca_figure(writer, arrays)
            _history_figures(writer, run_root, seeds)
            _progress_figures(writer, arrays)
            _intervention_figures(writer, arrays)
            _scale_figures(writer, arrays, scale_rows)
            _nearest_figures(writer, arrays)
            _failure_figures(writer, arrays)
    return writer.figures
