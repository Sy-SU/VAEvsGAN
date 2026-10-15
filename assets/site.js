const progress = document.querySelector('.reading-progress span');
const backToTop = document.querySelector('.back-to-top');
const report = document.querySelector('.report-content');
const toc = document.querySelector('.toc-content');
const tocContainer = document.querySelector('.toc');
const tocLabel = document.querySelector('.toc-label');
const markdownLink = document.querySelector('.markdown-link');
const comparisonLink = document.querySelector('.comparison-link');
const readingLink = document.querySelector('.reading-link');
const comparisonPage = document.body.dataset.report === 'comparison';
const footerLabel = document.querySelector('.footer-label');
const footerProject = document.querySelector('.footer-project');
const brand = document.querySelector('.brand');
const languageSwitch = document.querySelector('.language-switch');
const languageButtons = [...document.querySelectorAll('[data-language]')];

const content = {
  en: {
    article: report.innerHTML,
    toc: toc.innerHTML,
    documentLanguage: 'en',
    title: 'From Traditional Generative Models to VAEs, GANs, and Diffusion Models',
    description: 'A comparative reading report on Gaussian mixture models, VAEs, GANs, and diffusion models.',
    contents: 'Contents',
    tocLabel: 'Report contents',
    markdown: 'report.en.md',
    comparison: 'Experiments',
    reading: 'Reading Report',
    pageLinks: 'Page links',
    footer: 'Pattern Recognition Reading Report',
    project: 'VAEvsGAN Experimental Project',
    back: 'Back to top',
    brand: 'Back to the report title',
    language: 'Language',
    ...(comparisonPage ? {
      title: 'Conditional VAE vs GAN on MNIST: Comparison Study',
      description: 'A three-seed comparison of conditional VAE and GAN on MNIST: generation quality, coverage, conditional control, and efficiency.',
      tocLabel: 'Comparison study contents',
      markdown: 'comparison.en.md',
      footer: 'Three-seed Comparison Study',
    } : {}),
  },
  zh: {
    article: document.querySelector('#article-zh').innerHTML,
    toc: document.querySelector('#toc-zh').innerHTML,
    documentLanguage: 'zh-CN',
    title: '传统生成模型到 VAE、GAN 和 Diffusion Model 的发展',
    description: '关于高斯混合模型、VAE、GAN 与扩散模型的模式识别技术读书报告。',
    contents: '目录',
    tocLabel: '报告目录',
    markdown: 'report.zh.md',
    comparison: '对比实验',
    reading: '读书报告',
    pageLinks: '页面链接',
    footer: '模式识别技术读书报告',
    project: 'VAEvsGAN 实验项目',
    back: '返回顶部',
    brand: '返回报告顶部',
    language: '语言',
    ...(comparisonPage ? {
      title: 'MNIST 条件 VAE 与 GAN：对比实验',
      description: 'MNIST 条件 VAE 与 GAN 的三种子实验：生成质量、覆盖、条件控制与效率的完整对比报告。',
      tocLabel: '实验报告目录',
      markdown: 'comparison.md',
      footer: '三种子对比实验',
    } : {}),
  },
};

let activeLanguage = 'en';
let sectionObserver;

function updateScrollUI() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
  progress.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
  backToTop.classList.toggle('visible', window.scrollY > 700);
}

function observeSections() {
  if (sectionObserver) sectionObserver.disconnect();
  const tocLinks = [...toc.querySelectorAll('a')];
  const sections = tocLinks
    .map((link) => document.getElementById(link.getAttribute('href').slice(1)))
    .filter(Boolean);

  sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!visible) return;
    tocLinks.forEach((link) => {
      link.classList.toggle('active', link.getAttribute('href') === `#${visible.target.id}`);
    });
  }, { rootMargin: '-15% 0px -72% 0px' });

  sections.forEach((section) => sectionObserver.observe(section));
}

function setLanguage(language, updateUrl = true) {
  if (!content[language] || language === activeLanguage) return;
  const selected = content[language];
  if (window.MathJax?.typesetClear) window.MathJax.typesetClear([report]);
  report.innerHTML = selected.article;
  toc.innerHTML = selected.toc;
  document.documentElement.lang = selected.documentLanguage;
  document.title = selected.title;
  document.querySelector('meta[name="description"]').content = selected.description;
  tocLabel.textContent = selected.contents;
  tocContainer.setAttribute('aria-label', selected.tocLabel);
  markdownLink.href = selected.markdown;
  if (comparisonLink) {
    comparisonLink.textContent = selected.comparison;
    comparisonLink.href = `comparison/?lang=${language}`;
  }
  if (readingLink) {
    readingLink.textContent = selected.reading;
    readingLink.href = `../?lang=${language}`;
  }
  document.querySelector('.header-links').setAttribute('aria-label', selected.pageLinks);
  footerLabel.textContent = selected.footer;
  footerProject.textContent = selected.project;
  backToTop.setAttribute('aria-label', selected.back);
  brand.setAttribute('aria-label', selected.brand);
  languageSwitch.setAttribute('aria-label', selected.language);
  languageButtons.forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.language === language));
  });
  activeLanguage = language;
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set('lang', language);
    window.history.replaceState(null, '', url);
  }
  observeSections();
  window.scrollTo({ top: 0, behavior: 'smooth' });
  updateScrollUI();
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([report]);
}

languageButtons.forEach((button) => {
  button.addEventListener('click', () => setLanguage(button.dataset.language));
});
window.addEventListener('scroll', updateScrollUI, { passive: true });
backToTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
const initialLanguage = new URLSearchParams(window.location.search).get('lang');
if (initialLanguage === 'zh') {
  setLanguage('zh', false);
  if (window.location.hash) {
    document.getElementById(decodeURIComponent(window.location.hash.slice(1)))?.scrollIntoView();
  }
} else {
  observeSections();
  updateScrollUI();
}
