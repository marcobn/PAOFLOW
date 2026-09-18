(function () {
  // Restrict the PST theme toggle to light ↔ dark only (no auto/system mode).
  function applyMode(mode) {
    document.documentElement.dataset.mode  = mode;
    document.documentElement.dataset.theme = mode;
    localStorage.setItem('mode',  mode);
    localStorage.setItem('theme', mode);
    document.querySelectorAll('.dropdown-menu').forEach(function (el) {
      if (mode === 'dark') el.classList.add('dropdown-menu-dark');
      else el.classList.remove('dropdown-menu-dark');
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    // Resolve any stored 'auto' preference to the actual resolved theme
    var stored = localStorage.getItem('mode');
    if (!stored || stored === 'auto') {
      var visual = document.documentElement.dataset.theme || 'light';
      applyMode(visual === 'dark' ? 'dark' : 'light');
    }

    // Intercept PST's three-way cycle in the capture phase (fires before PST's bubble handler)
    document.querySelectorAll('.theme-switch-button').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopImmediatePropagation();
        var current = document.documentElement.dataset.theme || 'light';
        applyMode(current === 'light' ? 'dark' : 'light');
      }, true);
    });
  });
})();
