window.MathJax = {
  // Nạp sẵn extension boldsymbol: để autoload lười thì \boldsymbol hiện thành chữ đỏ
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Material dùng instant navigation: render lại toán mỗi khi đổi trang
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
