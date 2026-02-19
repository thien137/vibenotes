(function () {
  const PARTICLES_KEY = 'vibenotes-particles';
  let container = null;
  let listenersAttached = false;
  let lastSpawn = 0;
  const MIN_INTERVAL = 85;
  const MAX_GLITTER = 35;
  const SHAPES = ['star', 'diamond'];

  function spawnGlitter(x, y) {
    if (!container) return;

    const all = container.querySelectorAll('.mouse-fairy');
    if (all.length >= MAX_GLITTER) {
      all[0].remove();
    }

    const offsetX = (Math.random() - 0.5) * 12;
    const offsetY = (Math.random() - 0.5) * 12;
    const size = 3 + Math.random() * 4;
    const shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];

    const fairy = document.createElement('div');
    fairy.className = 'mouse-fairy mouse-fairy--' + shape;
    fairy.style.left = (x + offsetX) + 'px';
    fairy.style.top = (y + offsetY) + 'px';
    fairy.style.width = size + 'px';
    fairy.style.height = size + 'px';
    container.appendChild(fairy);

    fairy.addEventListener('animationend', function () {
      fairy.remove();
    });
  }

  function attachListeners() {
    if (!container || listenersAttached) return;
    listenersAttached = true;

    function onMove(e) {
      const now = Date.now();
      if (now - lastSpawn >= MIN_INTERVAL) {
        lastSpawn = now;
        spawnGlitter(e.clientX, e.clientY);
      }
    }

    document.addEventListener('mousemove', onMove, { passive: true });
  }

  function removeGlow() {
    if (container) {
      container.remove();
      container = null;
      listenersAttached = false;
    }
  }

  function particlesEnabled() {
    try {
      const v = localStorage.getItem(PARTICLES_KEY);
      return v === null || v === 'true';
    } catch (_e) {
      return true;
    }
  }

  function init() {
    if (!particlesEnabled()) return;

    if (!container) {
      container = document.createElement('div');
      container.className = 'mouse-glow';
      container.setAttribute('aria-hidden', 'true');
      document.body.appendChild(container);
      attachListeners();
    }
  }

  function onThemeChange() {
    if (particlesEnabled()) {
      init();
    } else {
      removeGlow();
    }
  }

  document.addEventListener('particlesChange', onThemeChange);

  const observer = new MutationObserver(onThemeChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
