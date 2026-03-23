"""Intentional lint fixture for CI checks.

Contains:
- one autofixable issue (double quotes -> single quotes via formatter)
- one unfixable issue (undefined name, F821)
"""

import numpy as np
import matplotlib.pyplot as plt

def build_message() -> str:
    greeting = 'hello from ci lint fixture'
    return greeting + missing_symbol
