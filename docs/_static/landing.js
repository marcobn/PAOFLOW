document.addEventListener('DOMContentLoaded', function () {
  function applyMode(mode) {
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.theme = mode;
    localStorage.setItem('mode', mode);
    localStorage.setItem('theme', mode);
    document.querySelectorAll('.dropdown-menu').forEach(function (el) {
      if (mode === 'dark') {
        el.classList.add('dropdown-menu-dark');
      } else {
        el.classList.remove('dropdown-menu-dark');
      }
    });
  }

  function currentMode() {
    var mode = document.documentElement.dataset.theme || localStorage.getItem('theme') || localStorage.getItem('mode') || 'dark';
    return mode === 'light' ? 'light' : 'dark';
  }

  function addLandingThemeToggle() {
    if (document.querySelector('.pf-landing-theme-toggle')) {
      return;
    }

    var stored = localStorage.getItem('mode') || localStorage.getItem('theme');
    if (!stored || stored === 'auto') {
      applyMode(currentMode());
    }

    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'pf-landing-theme-toggle';

    function syncButton() {
      var nextMode = currentMode() === 'light' ? 'dark' : 'light';
      button.textContent = nextMode === 'light' ? 'Light' : 'Dark';
      button.setAttribute('aria-label', 'Switch to ' + nextMode + ' mode');
    }

    button.addEventListener('click', function () {
      applyMode(currentMode() === 'light' ? 'dark' : 'light');
      syncButton();
    });

    syncButton();
    document.body.appendChild(button);
  }

  if (document.querySelector('.paoflow-landing')) {
    document.body.classList.add('pf-landing');
    addLandingThemeToggle();
  }

  var title = document.querySelector('.paoflow-typewriter');
  if (!title) {
    return;
  }

  var fullText = title.dataset.text || title.textContent.trim();
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reducedMotion) {
    title.textContent = fullText;
    title.classList.add('is-complete');
    return;
  }

  title.textContent = '';
  title.classList.add('is-typing');

  Array.from(fullText).forEach(function (letter, index) {
    window.setTimeout(function () {
      title.textContent += letter;

      if (index === fullText.length - 1) {
        title.classList.remove('is-typing');
        title.classList.add('is-complete');
      }
    }, 190 * (index + 1));
  });
});
