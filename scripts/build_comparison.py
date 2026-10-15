#!/usr/bin/env python3
"""Publish the repository's verified comparison report without rerunning experiments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

from build_site import render_markdown


def prepare_markdown(markdown: str) -> str:
    """Add stable section anchors and responsive figures to the web rendition."""
    section = 0
    lines = []
    for line in markdown.splitlines():
        if re.match(r"^#{2,3} ", line):
            section += 1
            lines.extend([f'<a id="section-{section}"></a>', ""])
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            alt, src = (html.escape(value, quote=True) for value in image.groups())
            lines.extend([
                "<figure>",
                f'<a href="{src}"><img src="{src}" alt="{alt}" loading="lazy" decoding="async"></a>',
                "</figure>",
            ])
        else:
            lines.append(line)
    return "\n".join(lines)


def build(source: Path, output: Path) -> None:
    report = source / "reports"
    manifest = json.loads((report / "data/manifest.json").read_text(encoding="utf-8"))
    hashes = {"comparison.md": manifest["report_sha256"]}
    hashes.update({f"data/{name}": value for name, value in manifest["data_files"].items()})
    hashes.update(manifest["figures"])
    for name, expected in hashes.items():
        actual = hashlib.sha256((report / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Report artifact checksum mismatch: {name}")
    markdown = (report / "comparison.md").read_text(encoding="utf-8")
    article, toc = render_markdown(prepare_markdown(markdown))
    template = (output / "comparison-template.html").read_text(encoding="utf-8")
    rendered = template.replace("{{ARTICLE}}", article).replace("{{TOC}}", toc)
    target = output / "comparison"
    for name in [*hashes, "data/manifest.json"]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report / name, destination)
    (target / "index.html").write_text(rendered, encoding="utf-8")
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    (target / "source.json").write_text(json.dumps({
        "repository": "https://github.com/Sy-SU/VAEvsGAN",
        "source_commit": source_commit,
        "report_sha256": manifest["report_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Published {len(hashes) + 1} verified report files to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="VAEvsGAN main checkout")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="gh-pages checkout")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
