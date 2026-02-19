(function () {
  const PARTICLES_KEY = 'vibenotes-particles';
  const NUM_FIREFLIES = 12;

  function particlesEnabled() {
    try {
      const v = localStorage.getItem(PARTICLES_KEY);
      return v === null || v === 'true';
    } catch (_e) {
      return true;
    }
  }

  let container = null;

  function createFirefly() {
    const f = document.createElement('div');
    f.className = 'firefly';
    f.style.left = Math.random() * 100 + 'vw';
    f.style.setProperty('--ff-mid-x', (Math.random() * 80 - 40) + 'px');
    f.style.setProperty('--ff-end-x', (Math.random() * 60 - 30) + 'px');
    f.style.setProperty('--ff-pause-y', '-' + (15 + Math.random() * 70) + 'vh');
    f.style.animationDuration = (10 + Math.random() * 12) + 's';
    f.style.animationDelay = '-' + (Math.random() * 20) + 's';
    return f;
  }

  function init() {
    if (!particlesEnabled()) return;

    if (!container) {
      container = document.createElement('div');
      container.className = 'fireflies';
      container.setAttribute('aria-hidden', 'true');
      for (let i = 0; i < NUM_FIREFLIES; i++) {
        container.appendChild(createFirefly());
      }
      document.body.appendChild(container);
    }
  }

  function remove() {
    if (container) {
      container.remove();
      container = null;
    }
  }

  function check() {
    if (particlesEnabled()) {
      init();
    } else {
      remove();
    }
  }

  window.addEventListener('storage', function (e) {
    if (e.key === PARTICLES_KEY) check();
  });

  const observer = new MutationObserver(check);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  document.addEventListener('particlesChange', check);

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', check);
  } else {
    check();
  }
})();
