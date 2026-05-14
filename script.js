/* ============================================================
   ACE — NeurIPS supplementary site
   Lightweight progressive enhancement: lightbox, in-view bar reveal.
   ============================================================ */

(() => {
  const $ = (sel) => document.querySelector(sel);

  // ---------- Lightbox ----------
  const lightbox = $('#lightbox');
  if (lightbox) {
    const lbImg = lightbox.querySelector('img');
    const lbCap = lightbox.querySelector('.lb-cap');
    const lbClose = lightbox.querySelector('.lb-close');

    document.querySelectorAll('.lb').forEach(fig => {
      fig.addEventListener('click', () => {
        const full = fig.getAttribute('data-full') || fig.querySelector('img')?.src;
        const cap = fig.querySelector('figcaption')?.textContent?.trim() || '';
        if (!full) return;
        lbImg.src = full;
        lbCap.textContent = cap;
        lightbox.classList.add('open');
        lightbox.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
      });
    });

    function close() {
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      lbImg.removeAttribute('src');
      document.body.style.overflow = '';
    }
    lbClose.addEventListener('click', close);
    lightbox.addEventListener('click', e => { if (e.target === lightbox) close(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  }

  // ---------- In-view bar reveal ----------
  const bars = document.querySelectorAll('.bar');
  if ('IntersectionObserver' in window && bars.length) {
    const widths = new WeakMap();
    bars.forEach(b => {
      widths.set(b, b.style.getPropertyValue('--w'));
      b.style.setProperty('--w', '0%');
    });
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          requestAnimationFrame(() => {
            e.target.style.setProperty('--w', widths.get(e.target));
          });
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.3 });
    bars.forEach(b => io.observe(b));
  }

  // ---------- Demos: tabbed scenario player ----------
  const demoVideo = document.getElementById('demo-video');
  const demoCap   = document.getElementById('demo-caption');
  const demoTabs  = document.querySelectorAll('.demo-tab');

  const DEMO_DATA = {
    friction: {
      src: 'assets/animations/hero_friction.mp4',
      html: 'Friction &times; 2.5 &mdash; <strong>&epsilon; says LEARN</strong>. ACE adapts; verdict panel reads +91&nbsp;% better than Frozen.',
    },
    noise: {
      src: 'assets/animations/hero_sensor_noise.mp4',
      html: '&sigma; = 12&nbsp;cm sensor noise &mdash; <strong>a says HOLD</strong>. ACE matches Frozen (no-adapt baseline); ER drifts to &minus;88&nbsp;%.',
    },
    platforms: {
      src: 'assets/animations/platform_grid.mp4',
      html: '<strong>Same gate, three robots.</strong> Top row: disturbance &mdash; ACE adapts. Bottom row: sensor noise &mdash; ACE holds; ER drifts on the drone (3.3&times; worse).',
    },
  };

  if (demoVideo && demoTabs.length) {
    demoTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const key = tab.dataset.scenario;
        const data = DEMO_DATA[key];
        if (!data) return;
        demoVideo.pause();
        demoVideo.src = data.src;
        demoVideo.load();
        const playPromise = demoVideo.play();
        if (playPromise && typeof playPromise.catch === 'function') {
          playPromise.catch(() => { /* autoplay blocked: user can tap to play */ });
        }
        demoCap.innerHTML = data.html;
        demoTabs.forEach(t => {
          const active = (t === tab);
          t.classList.toggle('is-active', active);
          t.setAttribute('aria-selected', active ? 'true' : 'false');
        });
      });
    });
  }
})();
