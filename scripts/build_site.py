#!/usr/bin/env python3
"""Convert the modular LaTeX report into Markdown and a standalone HTML page."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


CITATION_ORDER = [
    "bishop2006prml",
    "blei2017vi",
    "dempster1977em",
    "sohldickstein2015diffusion",
    "kingma2014aevb",
    "goodfellow2014gan",
    "ho2020ddpm",
    "song2021ddim",
    "lee2020variational-learning",
    "larsen2016autoencoding",
    "alemi2018broken-elbo",
    "peebles2023dit",
    "ma2024sit",
    "tian2024var",
    "yu2025repa",
    "geng2025meanflow",
]

REFERENCES = [
    "BISHOP C M. *Pattern Recognition and Machine Learning* [M]. New York, NY, USA: Springer, 2006. [出版社页面](https://link.springer.com/book/9780387310732)",
    "BLEI D M, KUCUKELBIR A, MCAULIFFE J D. Variational Inference: A Review for Statisticians [J/OL]. *Journal of the American Statistical Association*, 2017, 112(518): 859–877. [DOI](https://doi.org/10.1080/01621459.2017.1285773)",
    "DEMPSTER A P, LAIRD N M, RUBIN D B. Maximum Likelihood from Incomplete Data Via the EM Algorithm [J/OL]. *Journal of the Royal Statistical Society: Series B (Methodological)*, 1977, 39(1): 1–22. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)",
    "SOHL-DICKSTEIN J, WEISS E, MAHESWARANATHAN N, et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics [C]//*Proceedings of the 32nd International Conference on Machine Learning*. PMLR, 2015, 37: 2256–2265. [论文页面](https://proceedings.mlr.press/v37/sohl-dickstein15.html)",
    "KINGMA D P, WELLING M. Auto-Encoding Variational Bayes [C]//*International Conference on Learning Representations*. 2014. [论文页面](https://arxiv.org/abs/1312.6114)",
    "GOODFELLOW I J, POUGET-ABADIE J, MIRZA M, et al. Generative Adversarial Nets [C]//*Advances in Neural Information Processing Systems*. 2014, 27: 2672–2680. [论文页面](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)",
    "HO J, JAIN A N, ABBEEL P. Denoising Diffusion Probabilistic Models [C]//*Advances in Neural Information Processing Systems*. 2020, 33: 6840–6851. [论文页面](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)",
    "SONG J, MENG C, ERMON S. Denoising Diffusion Implicit Models [C]//*International Conference on Learning Representations*. 2021. [论文页面](https://iclr.cc/virtual/2021/poster/2804)",
    "LEE J Y, CHOI S I. Improvement of Learning Stability of Generative Adversarial Network Using Variational Learning [J/OL]. *Applied Sciences*, 2020, 10(13): 4528. [DOI](https://doi.org/10.3390/app10134528)",
    "LARSEN A B L, SØNDERBY S K, LAROCHELLE H, et al. Autoencoding beyond pixels using a learned similarity metric [C]//*Proceedings of the 33rd International Conference on Machine Learning*. PMLR, 2016, 48: 1558–1566. [论文页面](https://proceedings.mlr.press/v48/larsen16.html)",
    "ALEMI A, POOLE B, FISCHER I, et al. Fixing a Broken ELBO [C]//*Proceedings of the 35th International Conference on Machine Learning*. PMLR, 2018, 80: 159–168. [论文页面](https://proceedings.mlr.press/v80/alemi18a.html)",
    "PEEBLES W, XIE S. Scalable Diffusion Models with Transformers [C]//*Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023: 4195–4205. [论文页面](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)",
    "MA N, GOLDSTEIN M, ALBERGO M S, et al. SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers [C/OL]//*Computer Vision – ECCV 2024*. Cham: Springer, 2024, 15135: 23–40. [论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9828_ECCV_2024_paper.php)",
    "TIAN K, JIANG Y, YUAN Z, et al. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction [C/OL]//*Advances in Neural Information Processing Systems*. 2024, 37: 84839–84865. [论文页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a24e284b187f662681440ba15c416fb-Abstract-Conference.html)",
    "YU S, KWAK S, JANG H, et al. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think [C]//*International Conference on Learning Representations*. 2025: 87400–87442. [论文页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html)",
    "GENG Z, DENG M, BAI X, et al. Mean Flows for One-step Generative Modeling [C/OL]//*Advances in Neural Information Processing Systems*. 2025, 38: 75460–75482. [论文页面](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)",
]


def link_entire_reference(reference: str) -> str:
    match = re.fullmatch(r"(.*) \[[^\]]+\]\(([^)]+)\)", reference)
    if not match:
        return reference
    label, url = match.groups()
    label = label.replace("[M]", "（M）")
    label = label.replace("[C/OL]", "（C/OL）")
    label = label.replace("[J/OL]", "（J/OL）")
    label = label.replace("[C]", "（C）")
    return f"[{label}]({url})"


def format_references() -> str:
    return "\n".join(
        f"{i}. {link_entire_reference(reference)}"
        for i, reference in enumerate(REFERENCES, 1)
    )


def braced_argument(text: str, command: str) -> str:
    """Return a command's balanced braced argument, allowing an optional [] arg."""
    match = re.search(r"\\" + re.escape(command) + r"(?:\[[^\]]*\])?\s*\{", text)
    if not match:
        return ""
    start = match.end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:index]
    return ""


def collect_numbered_labels(tex: str, environment: str) -> dict[str, int]:
    labels: list[str] = []
    pattern = re.compile(
        rf"\\begin\{{{environment}\}}(?:\[[^\]]*\])?(.*?)\\end\{{{environment}\}}",
        re.S,
    )
    for block in pattern.findall(tex):
        label = re.search(r"\\label\{([^}]+)\}", block)
        if label:
            labels.append(label.group(1))
    return {label: number for number, label in enumerate(labels, 1)}


def convert_inline(
    text: str,
    citations: dict[str, int],
    equations: dict[str, int],
    figures: dict[str, int],
    tables: dict[str, int],
) -> str:
    text = text.replace(r"\protect", "")
    text = re.sub(
        r"\\href\{([^{}]+)\}\{([^{}]+)\}",
        lambda m: f"[{m.group(2)}]({m.group(1)})",
        text,
    )
    text = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda m: "".join(
            f"[[{citations[key.strip()]}]](#ref-{citations[key.strip()]})"
            for key in m.group(1).split(",")
            if key.strip() in citations
        ),
        text,
    )
    text = re.sub(
        r"\\eqref\{([^}]+)\}",
        lambda m: f"[({equations.get(m.group(1), '?')})](#{m.group(1)})",
        text,
    )

    def replace_ref(match: re.Match[str]) -> str:
        label = match.group(1)
        if label in figures:
            return f"[{figures[label]}](#{label})"
        if label in tables:
            return f"[{tables[label]}](#{label})"
        if label in equations:
            return f"[{equations[label]}](#{label})"
        return "?"

    text = re.sub(r"\\ref\{([^}]+)\}", replace_ref, text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = text.replace("``", "“").replace("''", "”")
    text = text.replace("~", " ")
    return text.strip()


def table_to_markdown(
    block: str,
    citations: dict[str, int],
    equations: dict[str, int],
    figures: dict[str, int],
    tables: dict[str, int],
) -> str:
    caption = braced_argument(block, "caption")
    label_match = re.search(r"\\label\{([^}]+)\}", block)
    label = label_match.group(1) if label_match else "table"
    number = tables[label]
    tabular = re.search(r"\\begin\{tabularx\}.*?\{[^\n]*\}(.*?)\\end\{tabularx\}", block, re.S)
    if not tabular:
        return ""
    body = tabular.group(1)
    body = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", body)
    body = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "", body)
    rows: list[list[str]] = []
    for raw_row in re.split(r"\\\\", body):
        row = raw_row.strip()
        if not row or "&" not in row:
            continue
        cells = [
            convert_inline(cell.strip(), citations, equations, figures, tables)
            for cell in row.split("&")
        ]
        rows.append(cells)
    if not rows:
        return ""
    width = len(rows[0])
    output = [
        f'<a id="{label}"></a>',
        "",
        f"**表 {number}　{convert_inline(caption, citations, equations, figures, tables)}**",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(output)


def latex_to_markdown(source: Path) -> str:
    section_paths = sorted((source / "sections").glob("*.tex"))
    section_texts = [path.read_text(encoding="utf-8") for path in section_paths]
    all_tex = "\n".join(section_texts)
    citations = {key: number for number, key in enumerate(CITATION_ORDER, 1)}
    equations = collect_numbered_labels(all_tex, "equation")
    figures = collect_numbered_labels(all_tex, "figure")
    tables = collect_numbered_labels(all_tex, "table")

    converted_sections: list[str] = []
    for text in section_texts:
        text = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("%")
        )

        def equation_replacement(match: re.Match[str]) -> str:
            block = match.group(1)
            label_match = re.search(r"\\label\{([^}]+)\}", block)
            label = label_match.group(1) if label_match else "equation"
            number = equations[label]
            body = re.sub(r"\\label\{[^}]+\}", "", block).strip()
            return f'\n<a id="{label}"></a>\n\n$$\n{body}\n\\tag{{{number}}}\n$$\n'

        text = re.sub(
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
            equation_replacement,
            text,
            flags=re.S,
        )

        def figure_replacement(match: re.Match[str]) -> str:
            block = match.group(1)
            label_match = re.search(r"\\label\{([^}]+)\}", block)
            image_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
            if not label_match or not image_match:
                return ""
            label = label_match.group(1)
            number = figures[label]
            caption = convert_inline(
                braced_argument(block, "caption"), citations, equations, figures, tables
            )
            image_name = Path(image_match.group(1)).name
            alt = re.sub(r"\[.*?\]\(.*?\)|[*_`]", "", caption)
            return (
                f'\n<figure id="{label}">\n'
                f'  <img src="assets/images/{image_name}" alt="图 {number}：{html.escape(alt)}" loading="lazy">\n'
                f'  <figcaption>图 {number}　{caption}</figcaption>\n'
                f"</figure>\n"
            )

        text = re.sub(
            r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}",
            figure_replacement,
            text,
            flags=re.S,
        )
        text = re.sub(
            r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}",
            lambda m: "\n"
            + table_to_markdown(m.group(1), citations, equations, figures, tables)
            + "\n",
            text,
            flags=re.S,
        )
        text = re.sub(
            r"\\section\{([^{}]+)\}\\label\{([^}]+)\}",
            lambda m: f'<a id="{m.group(2)}"></a>\n\n## {m.group(1)}',
            text,
        )
        text = re.sub(
            r"\\subsection\{([^{}]+)\}\\label\{([^}]+)\}",
            lambda m: f'<a id="{m.group(2)}"></a>\n\n### {m.group(1)}',
            text,
        )
        text = re.sub(r"\\(?:FloatBarrier|clearpage)\b", "", text)
        text = convert_inline(text, citations, equations, figures, tables)
        text = re.sub(r"\n{3,}", "\n\n", text)
        converted_sections.append(text)

    header = "# 传统生成模型到 VAE、GAN 和 Diffusion Model 的发展"

    references = format_references()
    return (
        header.strip()
        + "\n\n"
        + "\n\n".join(converted_sections)
        + '\n\n<a id="references"></a>\n\n## 参考文献\n\n'
        + references
        + "\n"
    )


def render_inline(text: str) -> str:
    placeholders: dict[str, str] = {}

    def protect(pattern: str, value: str) -> str:
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = value
        return key

    text = re.sub(
        r"\$(.+?)\$",
        lambda m: protect("math", r"\(" + m.group(1) + r"\)"),
        text,
    )
    text = html.escape(text, quote=False)
    text = re.sub(r"\[\[([0-9]+)\]\]\((#[^)]+)\)", lambda m: f'<sup class="citation"><a href="{m.group(2)}">[{m.group(1)}]</a></sup>', text)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: f'<img src="{html.escape(m.group(2), quote=True)}" alt="{html.escape(m.group(1), quote=True)}" loading="lazy">',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: (
            f'<a href="{html.escape(m.group(2), quote=True)}"'
            + (" target=\"_blank\" rel=\"noreferrer\"" if not m.group(2).startswith("#") else "")
            + f">{m.group(1)}</a>"
        ),
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    for key, value in placeholders.items():
        text = text.replace(html.escape(key), value)
    return text


def render_markdown(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    output: list[str] = []
    toc: list[tuple[int, str, str]] = []
    pending_anchor = ""
    in_references = False
    reference_number = 0
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue

        anchor = re.fullmatch(r'<a id="([^"]+)"></a>', line.strip())
        if anchor:
            pending_anchor = anchor.group(1)
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            heading_id = pending_anchor or ("report-title" if level == 1 else "")
            id_attr = f' id="{heading_id}"' if heading_id else ""
            output.append(f"<h{level}{id_attr}>{render_inline(title)}</h{level}>")
            if level in (2, 3) and heading_id:
                toc.append((level, heading_id, title))
            in_references = title in {"参考文献", "References"}
            pending_anchor = ""
            index += 1
            continue

        if pending_anchor:
            output.append(f'<a id="{pending_anchor}"></a>')
            pending_anchor = ""

        if line.startswith("```"):
            language = line[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed Markdown code fence")
            index += 1
            css_class = f' class="language-{html.escape(language, quote=True)}"' if language else ""
            code = html.escape("\n".join(code_lines))
            output.append(f"<pre><code{css_class}>{code}</code></pre>")
            continue

        if line.startswith("> "):
            output.append(f"<p class=\"eyebrow\">{render_inline(line[2:])}</p>")
            index += 1
            continue

        if line.startswith("$$"):
            if len(line) > 4 and line.endswith("$$"):
                math_lines = [line[2:-2]]
                index += 1
            else:
                math_lines = [line[2:]] if line[2:] else []
                index += 1
                while index < len(lines) and not lines[index].startswith("$$"):
                    math_lines.append(lines[index])
                    index += 1
                if index == len(lines):
                    raise ValueError("Unclosed Markdown display math")
                index += 1
            output.append('<div class="math-display">\\[' + "\n".join(math_lines) + "\\]</div>")
            continue

        if line.startswith("<figure"):
            raw = [line]
            index += 1
            while index < len(lines):
                raw.append(lines[index])
                if lines[index].strip() == "</figure>":
                    index += 1
                    break
                index += 1
            figure_html = "\n".join(raw)
            figure_html = re.sub(
                r"(<figcaption>)(.*?)(</figcaption>)",
                lambda m: m.group(1) + render_inline(m.group(2)) + m.group(3),
                figure_html,
                flags=re.S,
            )
            output.append(figure_html)
            continue

        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) >= 2:
                output.append('<div class="table-wrap"><table><thead><tr>')
                output.extend(f"<th>{render_inline(cell)}</th>" for cell in rows[0])
                output.append("</tr></thead><tbody>")
                for row in rows[2:]:
                    output.append("<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>")
                output.append("</tbody></table></div>")
            continue

        if re.match(r"^\d+\.\s+", line):
            items: list[str] = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index]))
                index += 1
            list_class = ' class="references"' if in_references else ""
            output.append(f"<ol{list_class}>")
            for item in items:
                reference_number += 1
                item_id = f' id="ref-{reference_number}"' if in_references else ""
                output.append(f"<li{item_id}>{render_inline(item)}</li>")
            output.append("</ol>")
            continue

        if line.startswith("- "):
            items: list[str] = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(lines[index][2:])
                index += 1
            output.append("<ul>" + "".join(f"<li>{render_inline(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            candidate = lines[index].rstrip()
            if (
                re.match(r"^(#{1,3})\s+", candidate)
                or candidate.startswith(("$$", "```", "<figure", "|", "> ", "- "))
                or re.match(r"^\d+\.\s+", candidate)
                or re.fullmatch(r'<a id="([^"]+)"></a>', candidate.strip())
            ):
                break
            paragraph.append(candidate)
            index += 1
        output.append(f"<p>{render_inline(' '.join(paragraph))}</p>")

    toc_html = []
    for level, anchor_id, title in toc:
        css_class = "toc-sub" if level == 3 else "toc-section"
        toc_html.append(
            f'<a class="{css_class}" href="#{anchor_id}">{html.escape(title)}</a>'
        )
    return "\n".join(output), "\n".join(toc_html)


def build_html(english: str, chinese: str, template: str) -> str:
    article_en, toc_en = render_markdown(english)
    article_zh, toc_zh = render_markdown(chinese)
    return (
        template.replace("{{ARTICLE_EN}}", article_en)
        .replace("{{TOC_EN}}", toc_en)
        .replace("{{ARTICLE_ZH}}", article_zh)
        .replace("{{TOC_ZH}}", toc_zh)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="LaTeX source directory")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="Site output directory")
    args = parser.parse_args()

    chinese = latex_to_markdown(args.source)
    english_source = args.output / "translations" / "report.en.md"
    english = english_source.read_text(encoding="utf-8").strip()
    english_references = (
        format_references()
        .replace("（M）", "(M)")
        .replace("（J/OL）", "(J/OL)")
        .replace("（C/OL）", "(C/OL)")
        .replace("（C）", "(C)")
    )
    english += (
        '\n\n<a id="references"></a>\n\n## References\n\n'
        + english_references
        + "\n"
    )
    template = (args.output / "template.html").read_text(encoding="utf-8")
    (args.output / "report.en.md").write_text(english, encoding="utf-8")
    (args.output / "report.zh.md").write_text(chinese, encoding="utf-8")
    (args.output / "index.html").write_text(
        build_html(english, chinese, template), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
