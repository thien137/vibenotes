(function () {
  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();
  }

  function initToc() {
    const tocList = document.getElementById('toc-list');
    const tocContainer = document.getElementById('toc-container');
    const sidebarToc = document.getElementById('sidebar-note-toc');
    const sidebarTocList = document.getElementById('sidebar-toc-list');
    const noteBody = document.querySelector('.note-body');
    if (!tocList || !noteBody) return;

    const headings = noteBody.querySelectorAll('h2, h3');
    if (headings.length === 0) {
      tocContainer.style.display = 'none';
      return;
    }

    function toRoman(n) {
      const vals = [1000,900,500,400,100,90,50,40,10,9,5,4,1];
      const syms = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I'];
      let s = '';
      for (let i = 0; i < vals.length; i++) {
        while (n >= vals[i]) { s += syms[i]; n -= vals[i]; }
      }
      return s;
    }

    function makeTocLink(h, id, mlaLabel) {
      const a = document.createElement('a');
      a.href = '#' + id;
      a.className = 'toc-link';
      if (mlaLabel) {
        const span = document.createElement('span');
        span.className = 'toc-link-label';
        span.textContent = mlaLabel + ' ';
        a.appendChild(span);
        a.appendChild(document.createTextNode(h.textContent));
      } else {
        a.textContent = h.textContent;
      }
      a.addEventListener('click', function (e) {
        e.preventDefault();
        document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
      });
      return a;
    }

    const fragment = document.createDocumentFragment();
    const sidebarFragment = document.createDocumentFragment();
    let romanCount = 0;
    let letterCount = 0;
    let numCount = 0;

    headings.forEach(function (h) {
      const level = parseInt(h.tagName.charAt(1), 10);
      let id = h.getAttribute('id');
      if (!id) {
        id = slugify(h.textContent || 'section');
        h.setAttribute('id', id);
      }

      let mlaLabel = '';
      if (level === 2) {
        romanCount++;
        letterCount = 0;
        numCount = 0;
        mlaLabel = toRoman(romanCount) + '.';
      } else if (level === 3) {
        letterCount++;
        numCount = 0;
        mlaLabel = String.fromCharCode(64 + letterCount) + '.';
      } else if (level === 4) {
        numCount++;
        mlaLabel = numCount + '.';
      }

      const li = document.createElement('li');
      li.className = 'toc-item toc-level-' + level;
      li.appendChild(makeTocLink(h, id, mlaLabel));
      fragment.appendChild(li);

      if (sidebarTocList) {
        const sideLi = document.createElement('li');
        sideLi.className = 'sidebar-toc-item sidebar-toc-level-' + level;
        sideLi.appendChild(makeTocLink(h, id, mlaLabel));
        sidebarFragment.appendChild(sideLi);
      }
    });

    tocList.appendChild(fragment);

    if (sidebarToc && sidebarTocList) {
      const noteTitleEl = document.querySelector('.note-title');
      const sidebarTitleEl = document.getElementById('sidebar-note-current-title');
      const dividerEl = document.getElementById('sidebar-toc-divider');
      if (noteTitleEl && sidebarTitleEl) {
        sidebarTitleEl.textContent = noteTitleEl.textContent;
        sidebarTitleEl.classList.add('sidebar-note-title-link');
        sidebarTitleEl.setAttribute('role', 'button');
        sidebarTitleEl.setAttribute('tabindex', '0');
        sidebarTitleEl.setAttribute('title', 'Scroll to top');
        sidebarTitleEl.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        sidebarTitleEl.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }
        });
      }
      sidebarTocList.appendChild(sidebarFragment);
      sidebarToc.style.display = 'block';
      if (dividerEl) dividerEl.style.display = 'block';
      const topDivider = document.getElementById('sidebar-note-toc-divider');
      if (topDivider) topDivider.style.display = 'block';
    }
  }

  function initReadingProgress() {
    const bar = document.getElementById('reading-progress');
    if (!bar) return;

    function updateProgress() {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (scrollHeight <= 0) {
        bar.style.setProperty('--progress', '0%');
        bar.setAttribute('aria-valuenow', '0');
        return;
      }
      const percent = Math.min(100, (scrollTop / scrollHeight) * 100);
      bar.style.setProperty('--progress', percent + '%');
      bar.setAttribute('aria-valuenow', Math.round(percent));
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  function initTocHighlight() {
    const sidebarTocLinks = document.querySelectorAll('.sidebar-toc-list .toc-link');
    const headings = document.querySelectorAll('.note-body h2[id], .note-body h3[id]');
    const noteBody = document.querySelector('.note-body');
    if (sidebarTocLinks.length === 0 || headings.length === 0 || !noteBody) return;

    function setActive(id) {
      sidebarTocLinks.forEach(function (link) {
        const href = link.getAttribute('href');
        const parent = link.closest('li');
        if (href === '#' + id) {
          link.classList.add('toc-active');
          if (parent) parent.classList.add('toc-active');
        } else {
          link.classList.remove('toc-active');
          if (parent) parent.classList.remove('toc-active');
        }
      });
    }

    function getActiveHeadingFromMouse(clientY) {
      for (let i = headings.length - 1; i >= 0; i--) {
        const rect = headings[i].getBoundingClientRect();
        if (clientY >= rect.top) return headings[i].getAttribute('id');
      }
      return headings[0] ? headings[0].getAttribute('id') : null;
    }

    sidebarTocLinks.forEach(function (link) {
      link.addEventListener('mouseenter', function () {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
          setActive(href.slice(1));
        }
      });
    });

    noteBody.addEventListener('mousemove', function (e) {
      const id = getActiveHeadingFromMouse(e.clientY);
      if (id) setActive(id);
    });
  }

  function initCodeCopyButtons() {
    const pres = document.querySelectorAll('.note-body pre');
    pres.forEach(function (pre) {
      if (pre.querySelector('.code-copy-btn')) return;

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'code-copy-btn';
      btn.setAttribute('aria-label', 'Copy code');
      btn.textContent = 'Copy';

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      wrapper.appendChild(btn);

      btn.addEventListener('click', function () {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        navigator.clipboard.writeText(text).then(
          function () {
            btn.textContent = 'Copied!';
            setTimeout(function () {
              btn.textContent = 'Copy';
            }, 1500);
          },
          function () {
            btn.textContent = 'Failed';
            setTimeout(function () {
              btn.textContent = 'Copy';
            }, 1500);
          }
        );
      });
    });
  }

  function init() {
    initToc();
    initReadingProgress();
    initTocHighlight();
    initCodeCopyButtons();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
