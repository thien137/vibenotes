(function () {
  const THEME_KEY = 'vibenotes-theme';
  const PARTICLES_KEY = 'vibenotes-particles';

  const THEMES = [
    { id: 'light', name: 'Light', color: '#faf8f5' },
    { id: 'dark', name: 'Dark', color: '#1e1c1a' },
    { id: 'nord', name: 'Nord', color: '#2e3440' },
    { id: 'dracula', name: 'Dracula', color: '#282a36' },
    { id: 'gruvbox', name: 'Gruvbox', color: '#282828' },
    { id: 'mint', name: 'Mint', color: '#1a1d1c' },
    { id: 'ocean', name: 'Ocean', color: '#0d1117' },
    { id: 'rose', name: 'Rose', color: '#1c1917' },
  ];

  function getTheme() {
    try {
      const v = localStorage.getItem(THEME_KEY);
      return v && THEMES.some(function (t) { return t.id === v; }) ? v : 'light';
    } catch (_e) {
      return 'light';
    }
  }

  function setTheme(themeId) {
    document.documentElement.setAttribute('data-theme', themeId);
    try {
      localStorage.setItem(THEME_KEY, themeId);
    } catch (_e) {}
    updateThemeUI(themeId);
  }

  function particlesEnabled() {
    try {
      const v = localStorage.getItem(PARTICLES_KEY);
      return v === null || v === 'true';
    } catch (_e) {
      return true;
    }
  }

  function setParticles(enabled) {
    try {
      localStorage.setItem(PARTICLES_KEY, enabled ? 'true' : 'false');
    } catch (_e) {}
    document.dispatchEvent(new CustomEvent('particlesChange'));
  }

  function openModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) {
      modal.hidden = false;
      modal.setAttribute('aria-hidden', 'false');
    }
  }

  function closeModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) {
      modal.hidden = true;
      modal.setAttribute('aria-hidden', 'true');
    }
  }

  function updateThemeUI(themeId) {
    document.querySelectorAll('.settings-theme-option').forEach(function (btn) {
      btn.setAttribute('aria-pressed', btn.dataset.theme === themeId ? 'true' : 'false');
    });
  }

  function updateParticlesUI() {
    const cb = document.getElementById('settings-particles-toggle');
    if (cb) cb.checked = particlesEnabled();
  }

  function buildModal() {
    if (document.getElementById('settings-modal')) return;

    const modal = document.createElement('div');
    modal.id = 'settings-modal';
    modal.className = 'settings-modal';
    modal.hidden = true;
    modal.setAttribute('aria-hidden', 'true');
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-labelledby', 'settings-modal-title');

    const overlay = document.createElement('div');
    overlay.className = 'settings-modal-overlay';
    overlay.addEventListener('click', closeModal);

    const panel = document.createElement('div');
    panel.className = 'settings-modal-panel';
    panel.addEventListener('click', function (e) { e.stopPropagation(); });

    const title = document.createElement('h2');
    title.id = 'settings-modal-title';
    title.className = 'settings-modal-title';
    title.textContent = 'Settings';

    const themeSection = document.createElement('div');
    themeSection.className = 'settings-section';
    themeSection.innerHTML = '<div class="settings-label">Color theme</div>';
    const themeGrid = document.createElement('div');
    themeGrid.className = 'settings-theme-grid';

    THEMES.forEach(function (t) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'settings-theme-option';
      btn.setAttribute('aria-label', t.name);
      btn.setAttribute('aria-pressed', t.id === getTheme() ? 'true' : 'false');
      btn.dataset.theme = t.id;
      btn.title = t.name;
      btn.style.backgroundColor = t.color;
      btn.addEventListener('click', function () {
        setTheme(t.id);
      });
      themeGrid.appendChild(btn);
    });
    themeSection.appendChild(themeGrid);

    const particlesSection = document.createElement('div');
    particlesSection.className = 'settings-section';
    const particlesLabel = document.createElement('label');
    particlesLabel.className = 'settings-toggle-label';
    const particlesInput = document.createElement('input');
    particlesInput.id = 'settings-particles-toggle';
    particlesInput.type = 'checkbox';
    particlesInput.checked = particlesEnabled();
    particlesInput.addEventListener('change', function () {
      setParticles(particlesInput.checked);
    });
    particlesLabel.appendChild(particlesInput);
    particlesLabel.appendChild(document.createTextNode(' Particles'));
    particlesSection.appendChild(particlesLabel);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'settings-modal-close';
    closeBtn.textContent = 'Done';
    closeBtn.addEventListener('click', closeModal);

    panel.appendChild(title);
    panel.appendChild(themeSection);
    panel.appendChild(particlesSection);
    panel.appendChild(closeBtn);
    modal.appendChild(overlay);
    modal.appendChild(panel);
    document.body.appendChild(modal);
  }

  function init() {
    setTheme(getTheme());

    buildModal();
    updateParticlesUI();

    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
      settingsBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        openModal();
        updateParticlesUI();
      });
    }

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        const modal = document.getElementById('settings-modal');
        if (modal && !modal.hidden) closeModal();
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
