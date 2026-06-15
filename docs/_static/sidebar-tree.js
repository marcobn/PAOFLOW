window.addEventListener('load', () => {
  const navs = document.querySelectorAll(
    '.bd-docs-nav.bd-links[aria-label="Site Navigation"]'
  );

  const bindNav = (nav) => {
    nav.querySelectorAll('li').forEach((item) => {
      if (item.dataset.sidebarTreeBound === 'true') {
        return;
      }

      const link = Array.from(item.children).find(
        (child) => child.tagName === 'A'
      );
      const details = Array.from(item.children).find(
        (child) => child.tagName === 'DETAILS'
      );
      const childList = details
        ? Array.from(details.children).find((child) => child.tagName === 'UL')
        : null;

      if (!details || !childList || !link) {
        return;
      }

      const isCurrentBranch =
        item.classList.contains('current') ||
        item.classList.contains('active') ||
        item.querySelector('li.current, li.active') !== null;

      item.dataset.sidebarTreeBound = 'true';
      item.classList.add('sidebar-tree-parent');

      // Track open state independently of details.open so CSS max-height
      // transitions can animate the child ul smoothly.
      let expanded = details.hasAttribute('open') || isCurrentBranch;

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'sidebar-tree-toggle';
      toggle.setAttribute('aria-label', `Toggle ${link.textContent.trim()}`);

      const applyOpenState = (isOpen) => {
        expanded = isOpen;
        item.classList.toggle('is-open', isOpen);
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      };

      // Always keep <details> structurally open so the <ul> remains in the
      // DOM and CSS can animate its max-height / opacity.
      details.open = true;
      applyOpenState(expanded);

      toggle.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        applyOpenState(!expanded);
      });

      // If something re-closes <details> natively (e.g. browser restore),
      // re-open it immediately so our CSS stays in control.
      details.addEventListener('toggle', () => {
        if (!details.open) {
          details.open = true;
        }
      });

      link.insertAdjacentElement('afterend', toggle);
    });

    // Add the ready class only after all items are processed so the initial
    // is-open states are already set when CSS transitions become active.
    // This prevents a flash of all-expanded content on first render.
    nav.classList.add('sidebar-tree-ready');
  };

  navs.forEach((nav) => {
    bindNav(nav);
  });
});
