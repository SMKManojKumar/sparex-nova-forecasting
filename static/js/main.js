'use strict';

/* ══════════════════════════════════════════
   SPREX NOVA v3 — Animations + Interactions
══════════════════════════════════════════ */

/* ── 1. THEME ─────────────────────────── */
const TKEY = 'sprex_theme';
function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem(TKEY, t);
  document.querySelectorAll('.th-btn').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  if (window._bgAnim) window._bgAnim.updateTheme(t);
}
function initTheme() {
  setTheme(localStorage.getItem(TKEY) || 'light');
  document.querySelectorAll('.th-btn').forEach(b => b.addEventListener('click', () => setTheme(b.dataset.t)));
}

/* ── 2. BACKGROUND CANVAS ANIMATION ──── */
function initBgCanvas() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [], theme = document.documentElement.getAttribute('data-theme') || 'light';

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Floating particles (dots + connecting lines)
  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x  = Math.random() * W;
      this.y  = Math.random() * H;
      this.vx = (Math.random() - .5) * .4;
      this.vy = (Math.random() - .5) * .4;
      this.r  = Math.random() * 2.5 + 1;
      this.alpha = Math.random() * .5 + .15;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > W) this.vx *= -1;
      if (this.y < 0 || this.y > H) this.vy *= -1;
    }
  }

  // init 60 particles
  for (let i = 0; i < 60; i++) particles.push(new Particle());

  function getColors() {
    if (theme === 'dark')  return { dot: '100,140,255', line: '80,110,220' };
    if (theme === 'neon')  return { dot: '96,165,250',  line: '59,130,246' };
    return                        { dot: '37,99,235',   line: '26,58,143'  };
  }

  // ripple rings from clicks
  let rings = [];
  document.addEventListener('click', e => {
    rings.push({ x: e.clientX, y: e.clientY, r: 0, max: 120, alpha: .6 });
  });

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const { dot, line } = getColors();
    const baseAlpha = theme === 'dark' ? .55 : theme === 'neon' ? .7 : .3;

    // draw connection lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(${line},${baseAlpha * (1 - dist/130) * .5})`;
          ctx.lineWidth = .8;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // draw dots
    particles.forEach(p => {
      p.update();
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${dot},${baseAlpha * p.alpha})`;
      ctx.fill();
    });

    // click ripple rings
    rings = rings.filter(rg => rg.r < rg.max);
    rings.forEach(rg => {
      rg.r  += 3;
      rg.alpha -= rg.alpha / 40;
      ctx.beginPath();
      ctx.arc(rg.x, rg.y, rg.r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${dot},${rg.alpha})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    requestAnimationFrame(draw);
  }
  draw();

  window._bgAnim = { updateTheme: t => { theme = t; } };
}

/* ── 3. CURSOR ───────────────────────── */
function initCursor() {
  const dot  = document.getElementById('cdot');
  const ring = document.getElementById('cring');
  if (!dot) return;
  let mx = 0, my = 0, rx = 0, ry = 0;

  document.addEventListener('mousemove', e => {
    mx = e.clientX; my = e.clientY;
    dot.style.left = mx + 'px'; dot.style.top = my + 'px';
  });
  (function tick() {
    rx += (mx - rx) * .14; ry += (my - ry) * .14;
    ring.style.left = rx + 'px'; ring.style.top = ry + 'px';
    requestAnimationFrame(tick);
  })();

  const growEls = 'a,button,.btn,.pill,.sb-a,.tb-btn,.sc,.feat,.dz,.card';
  document.querySelectorAll(growEls).forEach(el => {
    el.addEventListener('mouseenter', () => document.body.classList.add('cur-grow'));
    el.addEventListener('mouseleave', () => document.body.classList.remove('cur-grow'));
  });
}

/* ── 4. CLICK BURST ──────────────────── */
function initClickBurst() {
  document.addEventListener('click', e => {
    if (e.target.closest('.fm-x')) return; // don't burst on close buttons
    const b = document.createElement('div');
    b.className = 'cburst';
    b.style.cssText = `left:${e.clientX}px;top:${e.clientY}px;width:40px;height:40px`;
    document.body.appendChild(b);
    b.addEventListener('animationend', () => b.remove());
  });
}

/* ── 5. RIPPLE ON BUTTONS ────────────── */
function initRipple() {
  document.querySelectorAll('.btn,.pill,.sb-a').forEach(el => {
    el.addEventListener('click', e => {
      const r = document.createElement('span');
      const rect = el.getBoundingClientRect();
      const sz = Math.max(rect.width, rect.height);
      r.className = 'rpl';
      r.style.cssText = `width:${sz}px;height:${sz}px;left:${e.clientX-rect.left-sz/2}px;top:${e.clientY-rect.top-sz/2}px`;
      el.appendChild(r);
      r.addEventListener('animationend', () => r.remove());
    });
  });
}

/* ── 6. FLASH AUTO-DISMISS ───────────── */
function initFlash() {
  document.querySelectorAll('.fm').forEach((m, i) => {
    m.querySelector('.fm-x')?.addEventListener('click', () => dismiss(m));
    setTimeout(() => dismiss(m), 5000 + i * 300);
  });
}
function dismiss(el) {
  if (el._gone) return; el._gone = true;
  el.style.animation = 'fmOut .28s var(--ease) forwards';
  setTimeout(() => el.remove(), 280);
}

/* ── 7. SIDEBAR MOBILE ───────────────── */
function initSidebar() {
  const ham = document.querySelector('.ham');
  const sb  = document.querySelector('.sidebar');
  const ov  = document.querySelector('.sb-overlay');
  if (!ham) return;
  const open  = () => { sb.classList.add('open'); ov?.classList.add('show'); };
  const close = () => { sb.classList.remove('open'); ov?.classList.remove('show'); };
  ham.addEventListener('click', open);
  ov?.addEventListener('click', close);
}

/* ── 8. PASSWORD EYE ─────────────────── */
function initPwEye() {
  document.querySelectorAll('.peye').forEach(btn => {
    btn.addEventListener('click', () => {
      const inp = btn.previousElementSibling || btn.closest('.finp')?.querySelector('input');
      if (!inp) return;
      inp.type = inp.type === 'password' ? 'text' : 'password';
      btn.textContent = inp.type === 'password' ? '◎' : '◌';
    });
  });
}

/* ── 9. DROP ZONE ────────────────────── */
function initDrop() {
  const dz  = document.querySelector('.dz');
  const inp = dz?.querySelector('input[type=file]');
  if (!dz) return;
  ['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('over'); }));
  ['dragleave','drop'].forEach(e => dz.addEventListener(e, () => dz.classList.remove('over')));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && inp) { const dt = new DataTransfer(); dt.items.add(f); inp.files = dt.files; setFile(f.name); }
  });
  inp?.addEventListener('change', () => { if (inp.files[0]) setFile(inp.files[0].name); });
}
function setFile(name) {
  const t = document.querySelector('.dz-title'), h = document.querySelector('.dz-hint');
  if (t) t.textContent = '✅ ' + name;
  if (h) h.textContent = 'File ready — click Run Forecast';
}

/* ── 10. COUNT-UP NUMBERS ────────────── */
function animNums() {
  document.querySelectorAll('[data-num]').forEach(el => {
    const target = parseFloat(el.dataset.num) || 0;
    const t0 = performance.now(), dur = 900;
    function step(now) {
      const p = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(e * target).toLocaleString();
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

/* ── 11. NOTIF BADGE POLL ────────────── */
function initNotifPoll() {
  const badge = document.getElementById('nbadge');
  if (!badge) return;
  async function poll() {
    try {
      const r = await fetch('/api/nc');
      const d = await r.json();
      badge.textContent = d.count || '';
      badge.style.display = d.count ? 'flex' : 'none';
    } catch (_) {}
  }
  poll(); setInterval(poll, 30000);
}

/* ── 12. FORM LOADING STATE ──────────── */
function initFormLoad() {
  document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', () => {
      const btn = form.querySelector('[type=submit]');
      if (!btn || btn._loading) return;
      btn._loading = true; btn.disabled = true;
      btn._orig = btn.innerHTML;
      btn.innerHTML = `<span class="spin"></span> Processing…`;
      setTimeout(() => { btn.disabled = false; btn.innerHTML = btn._orig; btn._loading = false; }, 40000);
    });
  });
}

/* ── 13. PLOTLY CHART ────────────────── */
window.renderChart = function(id, pd) {
  const el = document.getElementById(id);
  if (!el || !pd) return;
  const theme = document.documentElement.getAttribute('data-theme');
  const dark = theme === 'dark', neon = theme === 'neon';
  const grid  = neon ? '#1a2040' : dark ? '#202840' : '#e8ecf8';
  const label = neon ? '#5060a0' : dark ? '#6070a0' : '#8090c0';
  const acc   = neon ? '#60a5fa' : '#2563eb';
  const fore  = neon ? '#f472b6' : '#d97706';

  const traces = [
    { x: pd.hist_dates, y: pd.hist_actual, name: 'Actual', type: 'scatter', mode: 'lines+markers',
      line: { color: acc, width: 2.5 }, marker: { size: 5, color: acc },
      hovertemplate: '<b>%{y}</b><extra>Actual</extra>' },
    { x: pd.hist_dates, y: pd.hist_pred, name: 'Model Fit', type: 'scatter', mode: 'lines',
      line: { color: acc, width: 1.5, dash: 'dot' }, opacity: .45,
      hovertemplate: '%{y}<extra>Fit</extra>' },
    { x: pd.fut_dates, y: pd.fut_vals, name: 'Forecast', type: 'scatter', mode: 'lines+markers',
      line: { color: fore, width: 2.5 },
      marker: { size: 7, color: fore, symbol: 'diamond' },
      fill: 'tozeroy',
      fillcolor: neon ? 'rgba(244,114,182,.06)' : dark ? 'rgba(217,119,6,.07)' : 'rgba(217,119,6,.06)',
      hovertemplate: '<b>%{y}</b><extra>Forecast</extra>' },
  ];

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Inter,sans-serif', color: label, size: 12 },
    margin: { l: 48, r: 16, t: 12, b: 50 },
    legend: { orientation: 'h', y: -0.24, font: { size: 11 } },
    xaxis: { gridcolor: grid, linecolor: grid, tickfont: { size: 11 }, showgrid: true, zeroline: false },
    yaxis: { gridcolor: grid, linecolor: grid, tickfont: { size: 11 }, showgrid: true, zeroline: false },
    hovermode: 'x unified',
    shapes: pd.hist_dates?.length ? [{
      type: 'line',
      x0: pd.hist_dates.at(-1), x1: pd.hist_dates.at(-1),
      y0: 0, y1: 1, yref: 'paper',
      line: { color: label, width: 1.2, dash: 'dot' }
    }] : [],
  };
  Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
};

/* ── BOOT ────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initBgCanvas();
  initCursor();
  initClickBurst();
  initRipple();
  initFlash();
  initSidebar();
  initPwEye();
  initDrop();
  animNums();
  initNotifPoll();
  initFormLoad();
});
