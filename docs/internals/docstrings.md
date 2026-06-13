# Docstrings

!!! note This section is for developers and contributors to the project. It provides guidelines and best practices for writing docstrings in the codebase.

## Writing Equations in Docstrings

To ensure equations render correctly in the generated documentation, always write mathematical expressions using standard LaTeX syntax and Sphinx-compatible directives. Use inline math for short expressions (e.g., ``:math:`E = mc^2```), and use a dedicated math block for displayed equations:

.. math::

G(E) = \left[E S - H - \Sigma(E)\right]^{-1}

Avoid plain-text approximations such as sqrt(x), Gamma^2, or A^-1; instead use proper LaTeX commands (\sqrt{x}, \Gamma^2, A^{-1}). Use braces around superscripts and subscripts containing more than one character (e.g., \Sigma\_{\mathrm{corr}}, G^{<}). Mathematical symbols, matrices, operators, and Greek letters should always be expressed in LaTeX rather than Unicode characters.
