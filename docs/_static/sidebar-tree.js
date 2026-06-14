document.addEventListener('DOMContentLoaded', () => {
  const navs = document.querySelectorAll(
    '.bd-docs-nav.bd-links[aria-label="Site Navigation"]'
  );

  navs.forEach((nav) => {
    nav.classList.add('sidebar-tree-ready');

    nav.querySelectorAll('li').forEach((item) => {
      if (item.dataset.sidebarTreeBound === 'true') {
        return;
      }

      const childList = Array.from(item.children).find(
        (child) => child.tagName === 'UL'
      );
      const link = Array.from(item.children).find(
        (child) => child.tagName === 'A'
      );

      if (!childList || !link) {
        return;
      }

      item.dataset.sidebarTreeBound = 'true';
      item.classList.add('sidebar-tree-parent');

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'sidebar-tree-toggle';
      toggle.setAttribute('aria-label', `Toggle ${link.textContent.trim()}`);

      const isCurrentBranch =
        item.classList.contains('current') ||
        item.classList.contains('active') ||
        item.querySelector('li.current, li.active') !== null;

      item.classList.toggle('is-open', isCurrentBranch);
      toggle.setAttribute(
        'aria-expanded',
        item.classList.contains('is-open') ? 'true' : 'false'
      );

      toggle.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();

        const isOpen = item.classList.toggle('is-open');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      });

      link.insertAdjacentElement('afterend', toggle);
    });
  });
});
