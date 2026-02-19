document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;

  document.querySelectorAll('.topic-dropdown').forEach((dropdown) => {
    const summary = dropdown.querySelector('.topic-summary');
    const expandBtn = dropdown.querySelector('.topic-expand-btn');

    function toggle() {
      const isOpen = dropdown.classList.toggle('is-open');
      summary.setAttribute('aria-expanded', isOpen);
    }

    summary.addEventListener('click', (e) => {
      if (!e.target.closest('.topic-link') && !e.target.closest('.topic-expand-btn')) {
        e.preventDefault();
        toggle();
      }
    });
    if (expandBtn) expandBtn.addEventListener('click', (e) => { e.preventDefault(); toggle(); });
    summary.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!e.target.closest('.topic-link') && !e.target.closest('.topic-expand-btn')) toggle();
      }
    });
  });

  if (path.includes('/topic/')) {
    const match = path.match(/\/topic\/([^/]+)/);
    if (match) {
      const topicId = match[1];
      const dropdown = document.querySelector('.topic-dropdown[data-topic="' + topicId + '"]');
      if (dropdown) {
        dropdown.classList.add('is-open');
        const s = dropdown.querySelector('.topic-summary');
        if (s) s.setAttribute('aria-expanded', 'true');
      }
    }
  }
  if (path.includes('/notes/')) {
    const match = path.match(/\/notes\/([^/]+)/);
    if (match) {
      const noteId = match[1];
      document.querySelectorAll('.topic-dropdown').forEach((d) => {
        const link = d.querySelector('.note-link[href*="' + noteId + '"]');
        if (link) {
          d.classList.add('is-open');
          const s = d.querySelector('.topic-summary');
          if (s) s.setAttribute('aria-expanded', 'true');
        }
      });
    }
  }

  const sidebarNoteToc = document.getElementById('sidebar-note-toc');
  const sidebarTocExpandBtn = document.getElementById('sidebar-note-toc-expand-btn');
  const sidebarTocPanel = document.getElementById('sidebar-note-toc-panel');
  if (sidebarNoteToc && sidebarTocExpandBtn && sidebarTocPanel) {
    const toggle = function () {
      const isOpen = sidebarTocPanel.classList.toggle('is-open');
      sidebarTocExpandBtn.classList.toggle('is-open', isOpen);
      sidebarTocExpandBtn.setAttribute('aria-expanded', isOpen);
    };
    sidebarTocExpandBtn.addEventListener('click', function (e) {
      e.preventDefault();
      toggle();
    });
    sidebarTocPanel.classList.add('is-open');
    sidebarTocExpandBtn.classList.add('is-open');
  }

  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const sidebarShowBtn = document.getElementById('sidebar-show-btn');
  const SIDEBAR_KEY = 'vibenotes-sidebar-hidden';

  function setSidebarHidden(hidden) {
    document.body.classList.toggle('sidebar-hidden', hidden);
    if (sidebarShowBtn) sidebarShowBtn.style.display = hidden ? 'flex' : 'none';
    try { localStorage.setItem(SIDEBAR_KEY, hidden ? 'true' : 'false'); } catch (_e) {}
  }

  function initSidebarToggle() {
    try {
      const stored = localStorage.getItem(SIDEBAR_KEY);
      if (stored === 'true') setSidebarHidden(true);
    } catch (_e) {}
    if (sidebarToggleBtn) {
      sidebarToggleBtn.addEventListener('click', function () {
        setSidebarHidden(true);
      });
    }
    if (sidebarShowBtn) {
      sidebarShowBtn.addEventListener('click', function () {
        setSidebarHidden(false);
      });
    }
  }
  initSidebarToggle();
});
