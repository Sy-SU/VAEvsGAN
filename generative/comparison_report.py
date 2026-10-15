"""Write the Chinese comparison report directly from verified study artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

METRICS = ("accuracy", "mmd2", "precision", "recall")
NAMES = {"accuracy": "条件准确率", "mmd2": "MMD²", "precision": "分布 precision", "recall": "分布 recall"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values):
    values = list(values)
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _format(values, metric, uncertainty=True):
    average, deviation = _stats(values)
    if metric in ("accuracy", "precision", "recall"):
        return f"{average * 100:.2f} ± {deviation * 100:.2f}%" if uncertainty else f"{average * 100:.2f}%"
    if metric == "seconds_median":
        return f"{average:.5f} ± {deviation:.5f}" if uncertainty else f"{average:.5f}"
    if metric in ("train_seconds", "wall_seconds", "evaluation_seconds"):
        return f"{average:.2f} ± {deviation:.2f}" if uncertainty else f"{average:.2f}"
    return f"{average:.5g} ± {deviation:.3g}" if uncertainty else f"{average:.5g}"


def _value(row, metric):
    if metric in METRICS:
        return row["macro"][metric]
    if metric in ("seconds_median", "images_per_second"):
        return row["generation_benchmark"][metric]
    return row[metric]


def _group(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], int(row["epoch"])), []).append(row)
    return {key: sorted(value, key=lambda row: row["seed"]) for key, value in sorted(groups.items(), key=lambda item: (item[0][0] != "vae", item[0][1]))}


def _label(key):
    return f"{'VAE' if key[0] == 'vae' else 'GAN'}{key[1]}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _comparison(groups, left, right):
    if left not in groups or right not in groups:
        return "本次小规模检查未包含此组正式对照。"
    a, b = groups[left], groups[right]
    chunks = []
    for metric in METRICS:
        av, bv = mean(_value(row, metric) for row in a), mean(_value(row, metric) for row in b)
        delta = bv - av
        if metric == "mmd2":
            expression = f"{'降低' if delta < 0 else '升高'} {abs(delta):.5g}"
        else:
            expression = f"{'降低' if delta < 0 else '提高'} {abs(delta) * 100:.2f} 个百分点"
        paired = [(next(row for row in a if row["seed"] == seed), next(row for row in b if row["seed"] == seed)) for seed in sorted({row["seed"] for row in a} & {row["seed"] for row in b})]
        favorable = sum((_value(second, metric) < _value(first, metric)) if metric == "mmd2" else (_value(second, metric) > _value(first, metric)) for first, second in paired)
        chunks.append(f"{NAMES[metric]}{expression}（{favorable}/{len(paired)} 个相同编号种子的结果沿有利方向变化）")
    return f"**{_label(right)} 相比 {_label(left)}**：" + "；".join(chunks) + "。这是预设比较的描述性结果，三个种子不足以据此宣称统计显著性。"


def _figure_path(figure, report_dir):
    path = Path(figure["path"])
    if path.is_absolute():
        return path.relative_to(report_dir).as_posix()
    if (report_dir / path).is_file():
        return path.as_posix()
    if path.is_file():
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    raise FileNotFoundError(f"Missing figure: {path}")


def _validate_report_links(path):
    text = path.read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in links:
        if "://" in target or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        if not (path.parent / target).is_file():
            raise FileNotFoundError(f"Broken report link: {target}")


def create_report(cfg) -> Path:
    """Render all figures and write a report; incomplete formal studies fail closed."""
    run_root = Path(cfg["study"]["run_dir"])
    report_dir = Path(cfg["study"]["report_dir"])
    data_dir = report_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    final = _read(run_root / "final_metrics.json")
    scales = _read(run_root / "scale_metrics.json")
    groups = _group(final)
    seeds = list(cfg["study"]["seeds"])
    if not cfg.get("smoke"):
        expected = {("vae", 20), ("gan", 20), ("gan", 50)}
        if set(groups) != expected or any([row["seed"] for row in rows] != seeds for rows in groups.values()):
            raise ValueError("Formal report requires VAE20/GAN20/GAN50 for every configured seed")
    if not cfg.get("smoke"):
        for key in groups:
            for tau in cfg["study"]["scales"]:
                selected = [row for row in scales if (row["model"], int(row["epoch"])) == key and float(row["tau"]) == float(tau)]
                if sorted(row["seed"] for row in selected) != seeds:
                    raise ValueError(f"Incomplete scale evaluation: {key} tau={tau}")
    for row in final:
        checkpoint = run_root / row["model"] / f"seed_{row['seed']}" / f"epoch_{row['epoch']:03d}.pt"
        if _hash(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Evaluated checkpoint changed: {checkpoint}")
    evaluator = _read(run_root / "evaluator/complete.json")
    if "test_accuracy" not in evaluator:
        raise ValueError("Evaluator test accuracy must be measured before report generation")
    if _hash(run_root / "evaluator/best.pt") != evaluator["checkpoint_sha256"]:
        raise ValueError("Evaluator checkpoint differs from recorded fingerprint")
    environment = _read(run_root / "environment.json")
    split = _read(run_root / "split.json")
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Training and validation indices overlap")
    if not cfg.get("smoke") and (len(split["train"]) != 50000 or len(split["validation"]) != 10000 or set(split["train"]) | set(split["validation"]) != set(range(60000))):
        raise ValueError("Formal split must partition all 60000 training samples")
    training, histories = {}, {}
    for family in ("vae", "gan"):
        for seed in seeds:
            directory = run_root / family / f"seed_{seed}"
            training[(family, seed)] = _read(directory / "complete.json")
            histories[(family, seed)] = _read(directory / "history.json")
            history = histories[(family, seed)]
            if [row["epoch"] for row in history] != list(range(1, cfg[family]["epochs"] + 1)):
                raise ValueError(f"Incomplete epoch history: {family}, {seed}")
            if [row["epoch"] for row in history if "evaluation" in row] != cfg[family]["evaluation_epochs"]:
                raise ValueError(f"Missing scheduled evaluation: {family}, {seed}")
    evaluation_summary = _read(run_root / "evaluation_summary.json") if (run_root / "evaluation_summary.json").exists() else {}
    for name in ("final_metrics.json", "scale_metrics.json", "config.json", "environment.json", "split.json", "validation-reference-indices.json", "test-reference-indices.json", "evaluation_summary.json", "visual_diagnostics.json", "pca.json", "real_baseline.json", "evaluation-manifest.json", "visual_observations.json", "code-provenance.json"):
        source = run_root / name
        if source.exists():
            shutil.copyfile(source, data_dir / name)
    observations_path = run_root / "visual_observations.json"
    observations = _read(observations_path) if observations_path.exists() else {}
    if observations.get("final_metrics_sha256") != _hash(run_root / "final_metrics.json"):
        observations = {}
    for name in ("training_comparison_study.py", "training-to-final.patch"):
        source = run_root / "provenance" / name
        if source.exists():
            shutil.copyfile(source, data_dir / name)
    shutil.copyfile(run_root / "evaluator/complete.json", data_dir / "evaluator.json")
    _json(data_dir / "training_histories.json", {f"{family}_seed_{seed}": history for (family, seed), history in histories.items()})
    _json(data_dir / "training_runs.json", {f"{family}_seed_{seed}": value for (family, seed), value in training.items()})
    repository = Path(__file__).resolve().parents[1]
    validation_records = []
    for source in sorted((repository / "outputs").glob("comparison-checks-*.json")):
        record = _read(source)
        if not record.get("passed"):
            raise ValueError(f"Failed verification record: {source}")
        shutil.copyfile(source, data_dir / source.name)
        validation_records.append((source.name, record))
    summary_rows = []
    for key, rows in groups.items():
        item = {"variant": _label(key), "seeds": len(rows)}
        for metric in (*METRICS, "train_seconds", "seconds_median", "images_per_second"):
            average, deviation = _stats(_value(row, metric) for row in rows)
            item[metric + "_mean"], item[metric + "_sample_sd"] = average, deviation
        summary_rows.append(item)
    _csv(data_dir / "summary.csv", summary_rows)
    _csv(data_dir / "per_seed.csv", [{"variant": row["variant"], "seed": row["seed"], "epoch": row["epoch"], **row["macro"], "train_seconds": row["train_seconds"], "generation_seconds": row["generation_benchmark"]["seconds_median"], "images_per_second": row["generation_benchmark"]["images_per_second"], "checkpoint_sha256": row["checkpoint_sha256"]} for row in final])
    _csv(data_dir / "per_class.csv", [{"variant": row["variant"], "seed": row["seed"], **digit} for row in final for digit in row["per_class"]])
    _csv(data_dir / "scale_per_seed.csv", [{"variant": row["variant"], "seed": row["seed"], "tau": row["tau"], **row["macro"]} for row in scales])
    from generative.comparison_plots import render_figures
    figures = render_figures(run_root, report_dir)
    used_figures = set()
    def figures_for(*keywords):
        chunks = []
        for index, figure in enumerate(figures):
            name = str(figure["name"]).lower()
            if index not in used_figures and any(keyword in name for keyword in keywords):
                relative = _figure_path(figure, report_dir)
                chunks.extend([f"![{figure['name']}]({relative})", str(figure.get("description", ""))])
                used_figures.add(index)
        return "\n\n".join(chunks)
    text = []
    def add(value):
        text.append(str(value).strip())
    stamp = datetime.fromisoformat(environment["created_at"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    add("# MNIST 条件 VAE 与 GAN：质量、覆盖、控制与效率对比")
    if cfg.get("smoke"):
        add("> 本文件是小规模真实数据冒烟检查产物，不是正式研究结论。")
    add(f"实验启动：{stamp}（Asia/Shanghai）。本报告由保存的实际权重与评估结果生成；数据为 MNIST，生成模型是本仓库既有条件卷积 VAE 和条件 DCGAN。{"冒烟实验" if cfg.get("smoke") else "正式实验"}共 {len(training)} 次训练，随机种子为 {', '.join(map(str, seeds))}。")
    add("## 1. 主要结果：分别回答两种预算下的比较")
    add("下表为独立训练种子的均值 ± **样本标准差**（ddof=1），百分数后的标准差同样以百分点表示。条件准确率、precision、recall 越高越好，MMD² 越低越好；训练时间以秒计。误差条描述训练种子的差异，不是置信区间。")
    add(_table(["检查点", "条件准确率 ↑", "MMD² ↓", "分布 precision ↑", "分布 recall ↑", "累计训练秒数"], [[_label(key), *[_format((_value(row, metric) for row in rows), metric) for metric in METRICS], _format((row["train_seconds"] for row in rows), "train_seconds")] for key, rows in groups.items()]))
    if ("vae", 20) in groups and ("gan", 20) in groups:
        compared = {key: {metric: mean(_value(row, metric) for row in groups[key]) for metric in (*METRICS, "train_seconds", "images_per_second")} for key in (("vae", 20), ("gan", 20))}
        def preferred(metric, lower=False):
            function = min if lower else max
            return function(compared, key=lambda key: compared[key][metric])
        quality = preferred("mmd2", lower=True)
        coverage = preferred("recall")
        correct = preferred("accuracy")
        faster = preferred("train_seconds", lower=True)
        generator = preferred("images_per_second")
        add(f"本次同为 20 epoch 的直接答案是：**{_label(correct)} 的条件正确率较高，{_label(quality)} 的特征分布 MMD² 较低，{_label(coverage)} 的覆盖 recall 较高；{_label(faster)} 训练更快，{_label(generator)} 的生成吞吐量更高。** 这些结论分别来自不同测量轴，下面保留数值和逐种子证据，避免把单项优势概括为所有方面都更好。")
        precision_delta = compared[("gan", 20)]["precision"] - compared[("vae", 20)]["precision"]
        recall_delta = compared[("gan", 20)]["recall"] - compared[("vae", 20)]["recall"]
        if precision_delta * recall_delta < 0:
            add("同 20 epoch 对照中，precision 与 recall 的优势分属不同模型，出现了质量线索与覆盖的实际取舍：应根据需求分别判断生成图靠近真实分布的程度，以及真实字形覆盖的程度。")
        if quality != coverage:
            add("MMD² 与覆盖 recall 也没有给出同一排序。这种分歧保留在结论中：前者检验核特征统计的接近程度，后者检验近邻球对真实样本的覆盖，二者不能相互替代。")
    add("### 相同数据遍数：VAE20 与 GAN20")
    add(_comparison(groups, ("vae", 20), ("gan", 20)))
    add("两者在 20 epoch 时看过相同数量的训练样本，但 GAN 每批交替更新生成器与判别器、网络规模也不同，因此这项对照不等价于相同计算预算。")
    add("### 默认训练设置：VAE20 与 GAN50")
    add(_comparison(groups, ("vae", 20), ("gan", 50)))
    add("### 同一个 GAN 从 20 到 50 epoch")
    add(_comparison(groups, ("gan", 20), ("gan", 50)))
    if ("gan", 20) in groups and ("gan", 50) in groups:
        earlier = {row["seed"]: row for row in groups[("gan", 20)]}
        later = {row["seed"]: row for row in groups[("gan", 50)]}
        regressions = [seed for seed in seeds if later[seed]["macro"]["mmd2"] > earlier[seed]["macro"]["mmd2"]]
        if regressions:
            details = "；".join(f"种子 {seed} 的 MMD² 从 {earlier[seed]['macro']['mmd2']:.3f} 升至 {later[seed]['macro']['mmd2']:.3f}" for seed in regressions)
            add(f"更多训练并非每个种子都一致改善：{details}。虽然跨种子均值可能更好，单次轨迹仍存在退化或波动；不能把 50 epoch 概括成在所有运行上都优于 20 epoch。")
    add("GAN20 是这三次 GAN50 训练的中途检查点，不是另外三次训练。这里报告预先约定的两个时点，没有根据测试集结果选择最佳 epoch。所有逐种子数值见 [per_seed.csv](data/per_seed.csv)，逐类数值见 [per_class.csv](data/per_class.csv)。")
    add(figures_for("generation_metrics"))
    add("## 2. 统一协议与评价含义")
    add(f"MNIST 原训练集被固定分层划为 {len(split['train']):,} 张训练图、{len(split['validation']):,} 张验证图，划分种子 {cfg['study']['split_seed']}。官方测试集仅用于最终评估。过程评估从验证集按每类 {cfg['study']['process_samples_per_class']} 张固定抽样，最终从测试集按每类 {cfg['study']['final_samples_per_class']} 张固定抽样，并生成相同数量的每类图片。参考索引和完整划分已随报告保存，训练阶段没有使用测试指标选模型。")
    add(_table(["项目", "VAE", "GAN"], [["Epoch", cfg["vae"]["epochs"], cfg["gan"]["epochs"]], ["潜变量维度 / batch size", f"{cfg['study']['latent_dim']} / {cfg['study']['batch_size']}", f"{cfg['study']['latent_dim']} / {cfg['study']['batch_size']}"], ["Adam 学习率", cfg["vae"]["learning_rate"], cfg["gan"]["learning_rate"]], ["Adam betas", "(0.9, 0.999)", f"({cfg['gan']['adam_beta1']}, {cfg['gan']['adam_beta2']})"], ["标签表示", "10 维 one-hot", f"{cfg['gan']['label_embedding_dim']} 维 embedding"], ["原生训练目标", "按样本平均 BCE 重建项 + KL", "D: real + 0.5×(fake+wrong-label); G: real-target BCE"]]))
    add("所有图片为原始 28×28 灰度图，不进行数据增强。已有 PNG 被无损缓存为 uint8，加载后除以 255；GAN 在训练时进一步映射到 [−1,1]，评估输出统一还原到 [0,1]。两模型保留原始结构、初始化及损失函数；每个 epoch 不丢弃最后不足一批的样本。")
    add(f"统一分类器为两层 3×3 卷积（32/64 通道，padding=1；各接 ReLU 和 2×2 最大池化）、128 维 ReLU 特征层和十分类输出。只使用同一训练划分，Adam 学习率 {cfg['evaluator']['learning_rate']}，batch size {cfg['study']['batch_size']}，训练 {cfg['evaluator']['epochs']} epoch；按验证准确率保存最佳权重后冻结。最佳验证准确率为 **{evaluator['validation_accuracy']:.3%}**，官方完整测试集准确率为 **{evaluator['test_accuracy']:.3%}**。分类器误差和特征偏好仍会影响生成评价，分类器判断与人类视觉质量不完全相同。")
    add("### 指标定义")
    add("**条件准确率**：分类器预测是否等于请求标签 y。它检测条件遵循程度，不单独判断自然程度或类内多样性。下述分布指标按**请求标签**与真实标签分别分组计算，再对十个数字等权平均；没有先筛掉分类错误的生成图。")
    add(r"""**MNIST 特征 MMD²**：在冻结分类器的 128 维、未经 PCA 投影的特征上，使用 $k(x,y)=(x^\top y/128+1)^3$。无偏估计为

$$\widehat{\mathrm{MMD}}_u^2=\frac{\sum_{i\ne j}k(x_i,x_j)}{m(m-1)}+\frac{\sum_{i\ne j}k(z_i,z_j)}{n(n-1)}-\frac{2\sum_{i,j}k(x_i,z_j)}{mn}.$$

核距离借鉴 [Bińkowski 等，Demystifying MMD GANs，ICLR 2018](https://arxiv.org/abs/1801.01401) 的 KID 评价思路。本实验替换了特征提取器，称为 MNIST 特征 MMD²，不能直接当作标准 Inception KID 与论文分数比较。有限样本无偏估计可以为负，因此保留原值；绝对数值也依赖特征的尺度。""")
    add("**分布 precision / recall**：采用 [Kynkäänniemi 等，Improved Precision and Recall Metric，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html) 的近邻球覆盖思想。每个特征点以第 3 个其他近邻的距离为半径，球的并集近似其分布支持域；precision 是生成点落入真实球并集的比例，recall 是真实点落入生成球并集的比例。二者均在 128 维空间计算。高 precision 与高 recall 分别提供质量、覆盖的线索，二者可能出现取舍；它们不是分类 precision/recall，也不能完整表示视觉质量和所有写法多样性。样本量、离群点和特征空间都会影响这些估计。")
    add("## 3. 全类别生成与条件控制")
    add(figures_for("samples_digits_"))
    add("主图固定使用训练种子 42，真实样本和每种检查点每类展示前 8 个固定索引。图片不按分类置信度或视觉好坏筛选。每行对应请求标签，列展示同类的不同采样；除类别是否正确，还应对照真实图检查笔画连贯性、字形重复和灰度边缘。另两个训练种子的同规则网格放在附录。")
    if "samples" in observations:
        add("**固定样本的视觉观察：**" + observations["samples"])
    add(figures_for("condition_confusion", "per_digit_accuracy"))
    for key, rows in groups.items():
        class_acc = {digit: mean(next(item for item in row["per_class"] if int(item["digit"]) == digit)["accuracy"] for row in rows) for digit in range(10)}
        hardest = min(class_acc, key=class_acc.get)
        easiest = max(class_acc, key=class_acc.get)
        confusion = [[sum(row["confusion"][i][j] for row in rows) for j in range(10)] for i in range(10)]
        i, j = max(((i, j) for i in range(10) for j in range(10) if i != j), key=lambda pair: confusion[pair[0]][pair[1]])
        add(f"**{_label(key)}**：条件准确率最低的数字为 {hardest}（跨种子均值 {class_acc[hardest]:.2%}），最高为 {easiest}（{class_acc[easiest]:.2%}）。合并种子的最大非对角混淆是请求 {i}、预测 {j}，共 {confusion[i][j]} 张，占请求 {i} 的 {confusion[i][j] / sum(confusion[i]):.2%}。这个现象由冻结分类器判断，应结合失败案例看其是否属于字形模糊或分类器误识别。")
    add("## 4. 生成质量、类内覆盖与共同特征空间")
    add(figures_for("precision_recall", "per_digit_metrics"))
    for key, rows in groups.items():
        av = {metric: mean(row["macro"][metric] for row in rows) for metric in METRICS}
        deviation = {metric: _stats(row["macro"][metric] for row in rows)[1] for metric in METRICS}
        lowest = min(range(10), key=lambda digit: mean(next(item for item in row["per_class"] if int(item["digit"]) == digit)["recall"] for row in rows))
        add(f"{_label(key)} 的分布 precision 为 {av['precision']:.2%}，recall 为 {av['recall']:.2%}，MMD² 为 {av['mmd2']:.5g}；recall 最低的是数字 {lowest}。三种子的条件准确率标准差为 {deviation['accuracy'] * 100:.2f} 个百分点，recall 标准差为 {deviation['recall'] * 100:.2f} 个百分点。这些数值描述本次实现的质量、覆盖与训练波动，不能单凭一个指标把模型排成全面优劣。")
    if any(mean(row["macro"]["precision"] for row in rows) >= .97 or mean(row["macro"]["recall"] for row in rows) >= .97 for rows in groups.values()):
        add("部分 precision/recall 已接近上限。高覆盖比例存在饱和效应，此时模型之间的细小差异更需要结合逐类结果、MMD² 和固定样本网格解释，不能把接近 1 视为分布完全一致。")
    baseline_path = run_root / "real_baseline.json"
    if baseline_path.exists():
        baseline = _read(baseline_path)["macro"]
        add(f"作为量尺检查，用真实训练图片作为待评估样本、真实测试参考不变时，MMD² 为 {baseline['mmd2']:.5g}，precision 为 {baseline['precision']:.2%}，recall 为 {baseline['recall']:.2%}。这一实对实基线也受有限样本影响，并非必须恰好得到 MMD²=0 或 precision/recall=100%。它用于理解量级，不用于训练选择。完整记录见 [real_baseline.json](data/real_baseline.json)。")
    add(figures_for("pca", "feature_space"))
    if "pca" in evaluation_summary:
        ratio = evaluation_summary["pca"]["explained_variance_ratio"]
        add(f"第一、第二主成分分别解释真实训练特征方差的 {ratio[0]:.2%} 与 {ratio[1]:.2%}，合计 {sum(ratio):.2%}。因此二维图保留的信息有限，重叠区域不能被直接解释为两组数据分布相同。")
    add("PCA 仅在真实训练集特征上拟合一次，所有面板共享中心、投影轴与坐标范围。数字颜色表示真实标签或请求标签；相同位置只说明在这两个投影方向上接近，不保证高维分布一致。上述统计仍在原始 128 维特征上计算，二维图不用于替代 MMD² 或覆盖指标。")
    add("## 5. 训练效率、稳定性与生成速度")
    add(figures_for("compute_cost", "training_metrics_", "training_progress_", "native_losses_"))
    add("曲线在预先约定的 epoch 节点评估：验证参考和生成图均每类固定抽取同样数量，阴影或误差条为训练种子标准差；实际种子点保留用于识别波动。累计训练时间包含内存数据加载、主机到 GPU 传输、前反向传播与优化器更新，GPU 同步后记时；排除首次 PNG 解码和缓存构建、辅助分类器训练、在线评估、权重保存和绘图。由于统一采用无损内存缓存，这里的加载耗时不代表逐张 PNG 磁盘读取的旧入口性能。")
    add("训练进展网格固定多个数字和多份潜变量，观察同一输入随训练的变化。VAE 的 BCE/KL 与 GAN 的对抗损失分别展示，它们含义和尺度不同，不能直接跨方法比较大小；GAN 的某项 loss 下降也不自动表示生成质量提高。过程曲线使用验证集且每类样本量更小，因此不能直接把最终测试 P/R 接在曲线末端。")
    timing_rows = []
    for family in ("vae", "gan"):
        runs = [training[(family, seed)] for seed in seeds]
        eval_times = [sum(row.get("evaluation_seconds", 0.0) for row in histories[(family, seed)]) for seed in seeds]
        walls = [row["wall_seconds"] for row in runs]
        trains = [row["train_seconds"] for row in runs]
        others = [wall - train - ev for wall, train, ev in zip(walls, trains, eval_times)]
        timing_rows.append([family.upper(), _format(trains, "train_seconds"), _format(eval_times, "evaluation_seconds"), _format(others, "wall_seconds"), _format(walls, "wall_seconds")])
    add(_table(["完整训练任务", "训练计算秒数", "在线验证秒数", "保存等其他秒数", "任务墙钟秒数"], timing_rows))
    add(f"辅助分类器训练及验证墙钟耗时：{evaluator['wall_seconds']:.2f} 秒。{len(training)} 次生成模型训练任务的墙钟总和为 {sum(row['wall_seconds'] for row in training.values()):.2f} 秒。以上每行按训练种子汇总，任务墙钟从模型和 loader 构建完成后开始，不是整个实验端到端耗时；GPU 顺序运行各任务，没有把 GAN20 额外计算成独立任务。")
    if "wall_seconds" in evaluation_summary:
        add(f"最终评估、尺度扫描及诊断数据准备共耗时 {evaluation_summary['wall_seconds']:.2f} 秒，单独记录，不计入训练秒数。")
    add(_table(["检查点", "训练参数总数", "生成使用参数", "生成 10,000 张耗时 / 秒", "吞吐量 / 张每秒"], [[_label(key), f"{rows[0]['parameters_total']:,}", f"{rows[0]['parameters_generate']:,}", _format((_value(row, "seconds_median") for row in rows), "seconds_median"), _format((_value(row, "images_per_second") for row in rows), "images_per_second")] for key, rows in groups.items()]))
    add("生成基准统一 batch size 128，每次生成 10,000 张，预热后重复 5 次并在 GPU 同步条件下计时；每个种子先取 5 次耗时中位数，再跨种子汇总。该指标仅衡量本次硬件和批量下的生成计算，只计已驻留 GPU 的输入通过 decoder/generator 前向传播；不计输入随机采样、传输、输出后处理、分类器或 PNG 写盘。VAE 训练参数包括编码器与解码器，生成参数仅计 decoder_input 和 decoder；GAN 训练参数是 G+D，生成只使用 G。")
    if ("vae", 20) in groups and ("gan", 20) in groups:
        vae_time = mean(row["train_seconds"] for row in groups[("vae", 20)])
        gan_time = mean(row["train_seconds"] for row in groups[("gan", 20)])
        add(f"本轮同为 20 epoch，GAN/VAE 的累计训练耗时比为 **{gan_time / vae_time:.2f} 倍**。因此评价效率时应同时看生成质量变化与实际时间，而不是仅看 epoch 数。")
    add("## 6. 潜变量插值与标签切换")
    add(figures_for("interpolation", "label_switch", "conditional", "control"))
    add("每个方法在相同请求数字下，对两份预先固定的随机潜变量进行 11 步线性插值；另外固定潜变量，把条件标签依次设为 0–9。前者检查连续改变噪声时字形是否平滑变化，后者检查相同噪声是否能随标签改变类别。各方法使用相同输入协议，但潜变量坐标的语义没有对齐，不能要求两种模型在某个坐标产生同样的笔画变化。")
    add("线性插值中点的潜变量方差比端点小：独立标准高斯端点在中点的逐维方差为 1/2。因此中间字形的变化同时受采样半径变化影响，不能全部解释为语义控制；下一节的噪声尺度扫描提供相邻证据。这里只做随机先验潜变量的共同对照，未把 VAE 专有的真实图片编码重建混入双方生成比较。")
    diagnostics = evaluation_summary.get("visual_diagnostics", {})
    if diagnostics:
        add(_table(["检查点 / 种子", "固定噪声标签切换准确率", "插值点条件准确率", "相邻插值图像素 L1 均值"], [[f"{variant} / {values['seed']}", f"{values['label_switch_accuracy']:.2%}", f"{values['interpolation_accuracy']:.2%}", f"{values['interpolation_adjacent_pixel_l1']:.5f}"] for variant, values in diagnostics.items()]))
        add("标签切换表基于 8 份固定噪声×10 个标签，共 80 张图片；插值表基于 10 个数字×11 个位置，共 110 张图片，只对展示种子计算。相邻像素 L1 测量图像变化幅度，值小可能表示平滑，也可能表示变化不足，不能独立用来评判语义可控性或多样性。")
    if "control" in observations:
        add("**固定样本的视觉观察：**" + observations["control"])
    add("## 7. 采样尺度：正确性与覆盖如何变化")
    add(figures_for("scale", "tau"))
    add("对同一批基础噪声 ε，分别使用 z=τε，τ∈{0.5,1,1.5,2}。各尺度使用同一批请求标签、相同最终参考和采样数；τ=1 是主实验标准先验。该扫描是预先设定的诊断，不使用测试集结果选择新的正式最优配置。")
    scale_rows = []
    for key, rows in groups.items():
        subset = [row for row in scales if (row["model"], int(row["epoch"])) == key]
        for tau in cfg["study"]["scales"]:
            selected = [row for row in subset if float(row["tau"]) == float(tau)]
            if selected:
                scale_rows.append([_label(key), tau, *[_format((row["macro"][metric] for row in selected), metric) for metric in METRICS]])
        endpoints = [[row for row in subset if float(row["tau"]) == tau] for tau in (.5, 2.)]
        if all(endpoints):
            low = {metric: mean(row["macro"][metric] for row in endpoints[0]) for metric in METRICS}
            high = {metric: mean(row["macro"][metric] for row in endpoints[1]) for metric in METRICS}
            add(f"**{_label(key)}**：τ=0.5 时条件准确率 {low['accuracy']:.2%}、precision {low['precision']:.2%}、recall {low['recall']:.2%}；τ=2 时分别为 {high['accuracy']:.2%}、{high['precision']:.2%}、{high['recall']:.2%}。")
            deltas = {metric: mean(row["macro"][metric] for row in endpoints[1]) - mean(row["macro"][metric] for row in endpoints[0]) for metric in METRICS}
            add(f"{_label(key)} 从 τ=0.5 增至 τ=2 时，条件准确率变化 {deltas['accuracy'] * 100:+.2f} 个百分点，precision 变化 {deltas['precision'] * 100:+.2f} 个百分点，recall 变化 {deltas['recall'] * 100:+.2f} 个百分点，MMD² 变化 {deltas['mmd2']:+.5g}。增加潜变量幅度是否带来更好的覆盖，需要与错误类别和分布偏离同时判断，不能仅按图像变化幅度定性。")
    add("小尺度下即使条件准确率接近满分，recall 也可能很低：正确生成指定数字并不意味着覆盖了该数字的多种写法。另一端，若大尺度同时降低准确率与 precision、却提高 recall，不能把这种提高直接称为更多有效写法；离群点及其较大的近邻球可能扩大估计支持域。这正是需要把正确性、质量与覆盖并列观察的原因。主比较始终保留 τ=1。")
    add(_table(["检查点", "τ", "条件准确率", "MMD²", "precision", "recall"], scale_rows))
    if "scales" in observations:
        add("**固定样本的视觉观察：**" + observations["scales"])
    add("## 8. 最近邻与失败案例")
    add(figures_for("nearest", "failure", "failed", "diagnostic"))
    add("最近邻只在同类训练图片中搜索，依据同一个冻结分类器的高维特征距离；图中给出固定选取的生成图片及匹配到的训练图。接近的笔画既可能来自 MNIST 类内本来就相似，也可能反映有限覆盖，因此最近邻相似本身不证明记忆训练集。失败案例按预设顺序选取条件预测错误的样本，附请求与预测数字；这些是诊断展示，不能用少量案例替代总体错误率。")
    for key, rows in groups.items():
        row = next((item for item in rows if item["seed"] == 42), rows[0])
        total = sum(sum(line) for line in row["confusion"])
        wrong = total - sum(row["confusion"][digit][digit] for digit in range(10))
        add(f"{_label(key)} 在展示用种子 {row['seed']} 的 {total:,} 张最终生成图中，有 {wrong:,} 张不被分类器识别为请求类别（{wrong / total:.2%}）；错误图库只能代表这一固定检测规则下的例子。")
    if diagnostics:
        add(_table(["检查点", "同类训练最近邻距离均值", "距离中位数", "实际展示错误数"], [[variant, f"{values['nearest_distance_mean']:.4f}", f"{values['nearest_distance_median']:.4f}", values["displayed_failure_count"]] for variant, values in diagnostics.items()]))
        add("最近邻距离统计基于展示种子的全部最终生成图；配对主图只显示每个请求数字的第一个样本。失败图按整体生成索引选最早 20 个条件错误，因此可能集中在前几个数字，这是固定排序的结果，不表示未展示的数字不会出错。更多原始诊断量见 [visual_diagnostics.json](data/visual_diagnostics.json)。")
    if "diagnostics" in observations:
        add("**固定样本的视觉观察：**" + observations["diagnostics"])
    add("## 9. 结论边界与复现")
    add("本实验比较的是本仓库两种具体条件模型与各自固定优化器设置，没有控制架构参数量，也没有对两种方法等量调参。VAE 使用 one-hot 条件，GAN 使用可学习 embedding；两者条件注入方式和生成网络容量也不同，不能把观察到的差异全部归因于 VAE/GAN 目标函数本身。因此不能将结果推广为所有 VAE 与所有 GAN 的普遍结论。只有三个训练种子；固定采样噪声和固定参考集减少对照噪声，但误差条不覆盖新参考数据、分类器重训或新采样种子的所有不确定性。MMD² 和 P/R 共同依赖同一分类器特征，不是完全独立的证据。MNIST 的结论也不直接代表彩色自然图像。")
    add("### 实际运行环境")
    add(_table(["项目", "本次记录"], [["GPU", environment.get("gpu", "CPU")], ["PyTorch / torchvision", f"{environment['torch']} / {environment['torchvision']}"], ["Python", environment["python"].split("|")[0].strip()], ["CUDA / cuDNN", f"{environment.get('cuda')} / {environment.get('cudnn')}"], ["NumPy", environment.get("numpy")], ["系统", environment["platform"]], ["TF32 / CPU threads", f"{environment.get('tf32')} / {environment.get('cpu_threads')}"], ["实验起始 Git commit", environment["git_commit"]]]))
    add("种子同时控制模型初始化、训练随机数和各自独立 DataLoader 随机流。评估使用独立采样器并保存/恢复 Python、NumPy、PyTorch CPU/CUDA 随机数，生成时切换 eval 后恢复状态。每个完整 epoch 保存可继续运行的模型、优化器和随机数状态；只复用相同配置的完成任务。cuDNN 采用确定性模式、关闭 benchmark 与 TF32；这不保证跨硬件或跨框架版本逐位一致。")
    add("从仓库根目录，在已有 `vae` 环境运行：\n\n```bash\npython tools/compare.py --config configs/comparison.toml --stage all\n# 已训练后重做最终评估与绘图\npython tools/compare.py --config configs/comparison.toml --stage evaluate\npython tools/compare.py --config configs/comparison.toml --stage report\n# 独立目录的小规模真实数据检查\npython tools/compare.py --config configs/comparison.toml --stage all --smoke\n```\n\n重新训练一份独立实验时，复制配置并为 `run_dir`、`report_dir` 指定新目录；不要覆盖已有正式结果。原始缓存保持固定数据源，变更数据源时需单独管理缓存。完整训练权重和可恢复状态留在实验运行目录，不复制到报告目录。")
    add("### 验证与产物完整性")
    add("正式报告生成前逐项核验：训练/验证索引无交集且覆盖原始 60,000 张训练图；两模型各种子的 epoch 历史完整；规定检查点都有在线评估；正式三组检查点各有三个种子，四个尺度的所有结果完整；最终评估所用权重与冻结分类器的实际 SHA-256 匹配。汇总 CSV 与正文从同一份逐种子 JSON 计算，报告生成后检查所有本地图片和数据链接。")
    for filename, record in validation_records:
        names = ", ".join(check["name"] for check in record.get("checks", []))
        add(f"{record.get('device', '未知设备')} 独立检查通过：{names}。详细断言与执行时间见 [{filename}](data/{filename})。")
    add("### 原始记录与完整性")
    add("[汇总表](data/summary.csv) · [逐种子表](data/per_seed.csv) · [逐类表](data/per_class.csv) · [尺度扫描](data/scale_per_seed.csv) · [完整最终指标](data/final_metrics.json) · [配置](data/config.json) · [训练历史](data/training_histories.json) · [完整划分索引](data/split.json) · [最终参考索引](data/test-reference-indices.json) · [环境](data/environment.json) · [文件哈希](data/manifest.json)。起始 Git commit 与逐文件 SHA-256 共同记录未提交代码状态；每个最终结果同时记录实际检查点哈希。")
    if (data_dir / "code-provenance.json").exists():
        add("训练起始的源码哈希保留在原始 environment.json 中。训练后，comparison_study.py 仅删除一个未使用的 import，并允许 report 阶段在无 CUDA 的环境运行，训练逻辑不变；已反向还原并验证训练版本哈希。保存了 [训练时源码](data/training_comparison_study.py)、[训练至最终版本的补丁](data/training-to-final.patch) 和 [源码来源说明](data/code-provenance.json)，最终报告 manifest 另记录当前源码哈希。")
    if observations:
        add("少量固定样本的视觉观察单独保存在 [visual_observations.json](data/visual_observations.json)，绑定这份最终指标文件的 SHA-256；它们是人工可读的图像解释，不是额外的盲测评分。")
    add("全部图表提供高分辨率 PNG；统计曲线、散点图和热图在相同 figures 目录另存同名 SVG，便于放大和排版。")
    add("## 附录：其他训练种子与补充图")
    add(figures_for("seed_samples_", "appendix"))
    for index, figure in enumerate(figures):
        if index not in used_figures:
            add(f"![{figure['name']}]({_figure_path(figure, report_dir)})\n\n{figure.get('description', '')}")
    output = report_dir / "comparison.md"
    output.write_text("\n\n".join(text) + "\n", encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    manifest = {
        "report_generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "split_sha256": _hash(run_root / "split.json"),
        "evaluator_checkpoint_sha256": evaluator["checkpoint_sha256"],
        "data_files": {path.name: _hash(path) for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "manifest.json"},
        "figures": {path.relative_to(report_dir).as_posix(): _hash(path) for path in sorted((report_dir / "figures").iterdir()) if path.suffix in (".png", ".svg")},
        "report_source_sha256": {str(path.relative_to(repository)): _hash(path) for path in sorted((repository / "generative").glob("comparison*.py"))},
        "report_sha256": _hash(output),
    }
    _json(data_dir / "manifest.json", manifest)
    _validate_report_links(output)
    print(f"Report saved: {output} ({len(figures)} figures, links verified)", flush=True)
    return output
