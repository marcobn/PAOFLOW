document.addEventListener('DOMContentLoaded', function () {
  const copyButtons = Array.from(document.querySelectorAll('.pf-citation-copy'));
  if (!copyButtons.length) {
    return;
  }

  const copyText = async function (text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand('copy');
    ta.remove();
  };

  copyButtons.forEach(function (button) {
    button.addEventListener('click', async function () {
      const entry = button.closest('.pf-citation-entry');
      const textNode = entry ? entry.querySelector('.pf-citation-text') : null;
      const citationText = textNode ? (textNode.textContent || '').trim() : '';

      if (!citationText) {
        return;
      }

      const originalLabel = button.textContent;
      button.textContent = 'Copying...';

      try {
        await copyText(citationText);
        button.textContent = 'Copied';
      } catch (_err) {
        button.textContent = 'Copy failed';
      }

      window.setTimeout(function () {
        button.textContent = originalLabel || 'Copy';
      }, 1200);
    });
  });

  document.addEventListener('keydown', function (event) {
    if ((event.key === 'Enter' || event.key === ' ') && event.target instanceof Element && event.target.classList.contains('pf-citation-copy')) {
      event.preventDefault();
      event.target.click();
    }
  });
});
