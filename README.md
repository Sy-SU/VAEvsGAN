# VAEvsGAN 读书报告站点

本分支保存《传统生成模型到 VAE、GAN 和 Diffusion Model 的发展》的中英文 GitHub Pages 静态站点。首页默认显示英文，可在页面右上角切换中文。

首页的 Experiments / 对比实验入口打开 `comparison/`，展示完整中英文三种子实验报告、27 张 PNG 图、13 张 SVG 图以及原始数据下载。实验页默认英文，支持 EN / 中文切换；两个报告之间的导航保留所选语言，也可使用 `?lang=en` 或 `?lang=zh` 分享指定语言。

- `report.en.md`：英文 Markdown 正文。
- `report.zh.md`：由最终 LaTeX 章节转换得到的中文 Markdown 正文。
- `translations/report.en.md`：英文翻译源稿。
- `translations/comparison.en.md`：实验报告完整英文译稿。
- `translations/comparison.source.json`：英文译稿所对应中文报告的 SHA-256。
- `comparison/comparison.en.md`、`comparison/comparison.md`：分别用于下载的英文和原始中文实验报告。
- `index.html`：包含无刷新中英文切换的静态页面。
- `assets/`：页面样式、交互脚本和实验图片。
- `scripts/build_site.py`：将模块化 LaTeX 中文源稿与英文翻译构建为 Markdown 和 HTML。
- `scripts/build_comparison.py`：核验主分支报告哈希，复制完整报告和数据，构建实验报告页面。
- `comparison-template.html`：实验报告页面模板，沿用站点样式。
- `comparison/source.json`：报告来源提交及两种语言 Markdown 的 SHA-256。

在本分支目录构建（将路径替换为本地源码位置）：

```bash
python3 -B scripts/build_site.py --source /path/to/LaTeX --output .
python3 -B scripts/build_comparison.py --source /path/to/VAEvsGAN --output .
```

两个构建命令仅使用 Python 标准库。实验页直接使用已保存的报告及图表，不重新训练或评估；若报告文件与 manifest 不符，构建会失败。中文报告更新后需同步完整英文译稿与 `comparison.source.json` 中的源哈希；构建还会检查两种语言的章节层级、证据链接、公式和表格数值一致。

本地预览：

```bash
python3 -m http.server 4173
```

然后访问 <http://127.0.0.1:4173/>。
