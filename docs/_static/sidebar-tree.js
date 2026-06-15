window.addEventListener('load', () => {
  const navs = document.querySelectorAll(
    '.bd-docs-nav.bd-links[aria-label="Site Navigation"]'
  );

  const bindNav = (nav) => {
    nav.classList.add('sidebar-tree-ready');

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

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'sidebar-tree-toggle';
      toggle.setAttribute('aria-label', `Toggle ${link.textContent.trim()}`);

      const applyOpenState = (isOpen) => {
        details.open = isOpen;
        item.classList.toggle('is-open', isOpen);
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      };

      applyOpenState(details.hasAttribute('open') || isCurrentBranch);

      toggle.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();

        applyOpenState(!details.open);
      });

      details.addEventListener('toggle', () => {
        applyOpenState(details.open);
      });

      link.insertAdjacentElement('afterend', toggle);
    });
  };

  navs.forEach((nav) => {
    bindNav(nav);
  });
});
