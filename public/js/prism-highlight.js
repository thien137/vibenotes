(function () {
  function init() {
    if (typeof Prism === 'undefined') return;

    document.querySelectorAll('pre code:not([class*="language-"])').forEach(function (block) {
      block.classList.add('language-python');
    });

    Prism.highlightAll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
