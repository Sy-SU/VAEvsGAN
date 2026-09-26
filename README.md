# VAEvsGAN 读书报告站点

本分支保存《传统生成模型到 VAE、GAN 和 Diffusion Model 的发展》的中英文 GitHub Pages 静态站点。首页默认显示英文，可在页面右上角切换中文。

- `report.en.md`：英文 Markdown 正文。
- `report.zh.md`：由最终 LaTeX 章节转换得到的中文 Markdown 正文。
- `translations/report.en.md`：英文翻译源稿。
- `index.html`：包含无刷新中英文切换的静态页面。
- `assets/`：页面样式、交互脚本和实验图片。
- `scripts/build_site.py`：将模块化 LaTeX 中文源稿与英文翻译构建为 Markdown 和 HTML。

本地预览：

```bash
python3 -m http.server 4173
```

然后访问 <http://127.0.0.1:4173/>。
