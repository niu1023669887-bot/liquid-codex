// Shared UI utilities

// ── Status bar clock ──
function startClock() {
  const el = document.getElementById('status-time');
  if (!el) return;
  function tick() {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    el.textContent = `${hh}:${mm}:${ss}`;
  }
  tick();
  setInterval(tick, 1000);
}

// ── Scroll reveal (with stagger for .vol-card groups) ──
function initReveal() {
  // Pre-assign stagger delays to vol-cards so they cascade in on scroll
  document.querySelectorAll('.vol-card.reveal').forEach((el, i) => {
    el.style.transitionDelay = `${i * 0.06}s`;
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      e.target.classList.add('visible');
      observer.unobserve(e.target);
      // Clear delay after reveal so hover is always instant
      const delay = parseFloat(e.target.style.transitionDelay) || 0;
      setTimeout(() => { e.target.style.transitionDelay = ''; }, (delay + 0.8) * 1000);
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

// ── Tabs ──
function initTabs(containerSelector) {
  const containers = document.querySelectorAll(containerSelector || '[data-tabs]');
  containers.forEach(container => {
    const btns = container.querySelectorAll('.tab-btn');
    const panels = container.querySelectorAll('.tab-panel');

    btns.forEach((btn, i) => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        panels[i]?.classList.add('active');
      });
    });

    // Activate first by default
    if (btns.length) { btns[0].classList.add('active'); panels[0]?.classList.add('active'); }
  });
}

// ── Expanders ──
function initExpanders() {
  document.querySelectorAll('.expander-header').forEach(header => {
    header.addEventListener('click', () => {
      header.closest('.expander').classList.toggle('open');
    });
  });
}

// ── Range display ──
function initRanges() {
  document.querySelectorAll('input[type="range"]').forEach(input => {
    const display = input.nextElementSibling;
    if (display && display.classList.contains('range-val')) {
      display.textContent = input.value;
      input.addEventListener('input', () => { display.textContent = input.value; });
    }
  });
}

// ── Split-hero cinematic reveal (all .split-hero on page) ──
function initSplitHero() {
  const heroes = document.querySelectorAll('.split-hero');
  if (!heroes.length) return;
  // Double rAF ensures paint is flushed before class is added
  requestAnimationFrame(() => requestAnimationFrame(() => {
    heroes.forEach(h => h.classList.add('revealed'));
  }));
}

// ── Odometer count-up (integer targets only) ──
function initOdometers() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      observer.unobserve(e.target);
      const target = parseInt(e.target.dataset.odometer, 10);
      if (isNaN(target)) return;
      const duration = 1400;
      const start = performance.now();
      (function step(now) {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        e.target.textContent = Math.round(eased * target);
        if (p < 1) requestAnimationFrame(step);
        else {
          e.target.textContent = target;
          e.target.classList.add('odometer-land');
        }
      })(start);
    });
  }, { threshold: 0.6 });

  document.querySelectorAll('[data-odometer]').forEach(el => observer.observe(el));
}

// ── Ticker seamless loop (duplicates DOM content) ──
function initTicker() {
  const track = document.getElementById('ticker-track');
  if (!track) return;
  track.innerHTML += track.innerHTML; // [A][A] → animate -50% for seamless loop
}

// ── Flash a numeric result on change ──
function flashResult(el) {
  if (!el) return;
  el.classList.remove('result-flash');
  void el.offsetWidth; // force reflow to restart animation
  el.classList.add('result-flash');
  el.addEventListener('animationend', () => el.classList.remove('result-flash'), { once: true });
}

// ── Live material count ({n} in i18n strings) ──
function refreshMaterialCounts() {
  const base = (typeof API_BASE !== 'undefined') ? API_BASE : '';
  fetch(`${base}/api/materials?t=${Date.now()}`)
    .then(r => r.ok ? r.json() : [])
    .then(arr => {
      if (!Array.isArray(arr) || !arr.length) return;
      const n = arr.length;
      document.querySelectorAll('[data-i18n="vol1.desc"], [data-i18n="login.footer"]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const tpl = (typeof t === 'function' ? t(key) : null) || el.textContent;
        if (tpl && tpl.includes('{n}')) el.textContent = tpl.replace('{n}', n);
      });
    })
    .catch(() => {});
}

// ── Init everything on load ──
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  initReveal();
  initTabs();
  initExpanders();
  initRanges();
  initSplitHero();
  initOdometers();
  initTicker();
  refreshMaterialCounts();
});
