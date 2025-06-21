import pandas as pd
import numpy as np
import os
from io import StringIO
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration: GPIO to Output Matrix Position Mapping (CRITICAL CORRECTION) ---
# This dictionary directly maps each (output_matrix_row, output_matrix_col) to the
# specific pair of GPIOs that control that pixel, as derived directly from the
# 'trace_simulation.m' script's conditional statements.
# Format: (output_matrix_row_0_indexed, output_matrix_col_0_indexed): (gpio_row_name, gpio_col_name)
OUTPUT_MATRIX_POS_TO_GPIO_PAIR = {
    (0,0): ('GPIO 12', 'GPIO 16'), # based on MATLAB: led_states(1,1,i) if (veri(i,4)==1) && (veri(i,6)==1)
    (0,1): ('GPIO 12', 'GPIO 5'),  # led_states(1,2,i) if (veri(i,4)==1) && (veri(i,2)==1)
    (0,2): ('GPIO 12', 'GPIO 6'),  # led_states(1,3,i) if (veri(i,4)==1) && (veri(i,3)==1)
    (0,3): ('GPIO 12', 'GPIO 13'), # led_states(1,4,i) if (veri(i,4)==1) && (veri(i,5)==1)
    (0,4): ('GPIO 12', 'GPIO 19'), # led_states(1,5,i) if (veri(i,4)==1) && (veri(i,9)==1)
    (0,5): ('GPIO 12', 'GPIO 26'), # led_states(1,6,i) if (veri(i,4)==1) && (veri(i,16)==1)
    (0,6): ('GPIO 12', 'GPIO 20'), # led_states(1,7,i) if (veri(i,4)==1) && (veri(i,10)==1)
    (0,7): ('GPIO 12', 'GPIO 21'), # led_states(1,8,i) if (veri(i,4)==1) && (veri(i,11)==1)

    (1,0): ('GPIO 25', 'GPIO 16'), # led_states(2,1,i) if (veri(i,15)==1) && (veri(i,6)==1)
    (1,1): ('GPIO 25', 'GPIO 5'),  # led_states(2,2,i) if (veri(i,15)==1) && (veri(i,2)==1)
    (1,2): ('GPIO 25', 'GPIO 6'),  # led_states(2,3,i) if (veri(i,15)==1) && (veri(i,3)==1)
    (1,3): ('GPIO 25', 'GPIO 13'), # led_states(2,4,i) if (veri(i,15)==1) && (veri(i,5)==1)
    (1,4): ('GPIO 25', 'GPIO 19'), # led_states(2,5,i) if (veri(i,15)==1) && (veri(i,9)==1)
    (1,5): ('GPIO 25', 'GPIO 26'), # led_states(2,6,i) if (veri(i,15)==1) && (veri(i,16)==1)
    (1,6): ('GPIO 25', 'GPIO 20'), # led_states(2,7,i) if (veri(i,15)==1) && (veri(i,10)==1)
    (1,7): ('GPIO 25', 'GPIO 21'), # led_states(2,8,i) if (veri(i,15)==1) && (veri(i,11)==1)

    (2,0): ('GPIO 24', 'GPIO 16'), # led_states(3,1,i) if (veri(i,14)==1) && (veri(i,6)==1)
    (2,1): ('GPIO 24', 'GPIO 5'),  # led_states(3,2,i) if (veri(i,14)==1) && (veri(i,2)==1)
    (2,2): ('GPIO 24', 'GPIO 6'),  # led_states(3,3,i) if (veri(i,14)==1) && (veri(i,3)==1)
    (2,3): ('GPIO 24', 'GPIO 13'), # led_states(3,4,i) if (veri(i,14)==1) && (veri(i,5)==1)
    (2,4): ('GPIO 24', 'GPIO 19'), # led_states(3,5,i) if (veri(i,14)==1) && (veri(i,9)==1)
    (2,5): ('GPIO 24', 'GPIO 26'), # led_states(3,6,i) if (veri(i,14)==1) && (veri(i,16)==1)
    (2,6): ('GPIO 24', 'GPIO 20'), # led_states(3,7,i) if (veri(i,14)==1) && (veri(i,10)==1)
    (2,7): ('GPIO 24', 'GPIO 21'), # led_states(3,8,i) if (veri(i,14)==1) && (veri(i,11)==1)

    (3,0): ('GPIO 22', 'GPIO 16'), # led_states(4,1,i) if (veri(i,12)==1) && (veri(i,6)==1)
    (3,1): ('GPIO 22', 'GPIO 5'),  # led_states(4,2,i) if (veri(i,12)==1) && (veri(i,2)==1)
    (3,2): ('GPIO 22', 'GPIO 6'),  # led_states(4,3,i) if (veri(i,12)==1) && (veri(i,3)==1)
    (3,3): ('GPIO 22', 'GPIO 13'), # led_states(4,4,i) if (veri(i,12)==1) && (veri(i,5)==1)
    (3,4): ('GPIO 22', 'GPIO 19'), # led_states(4,5,i) if (veri(i,12)==1) && (veri(i,9)==1)
    (3,5): ('GPIO 22', 'GPIO 26'), # led_states(4,6,i) if (veri(i,12)==1) && (veri(i,16)==1)
    (3,6): ('GPIO 22', 'GPIO 20'), # led_states(4,7,i) if (veri(i,12)==1) && (veri(i,10)==1)
    (3,7): ('GPIO 22', 'GPIO 21'), # led_states(4,8,i) if (veri(i,12)==1) && (veri(i,11)==1)

    (4,0): ('GPIO 27', 'GPIO 16'), # led_states(5,1,i) if (veri(i,17)==1) && (veri(i,6)==1)
    (4,1): ('GPIO 27', 'GPIO 5'),  # led_states(5,2,i) if (veri(i,17)==1) && (veri(i,2)==1)
    (4,2): ('GPIO 27', 'GPIO 6'),  # led_states(5,3,i) if (veri(i,17)==1) && (veri(i,3)==1)
    (4,3): ('GPIO 27', 'GPIO 13'), # led_states(5,4,i) if (veri(i,17)==1) && (veri(i,5)==1)
    (4,4): ('GPIO 27', 'GPIO 19'), # led_states(5,5,i) if (veri(i,17)==1) && (veri(i,9)==1)
    (4,5): ('GPIO 27', 'GPIO 26'), # led_states(5,6,i) if (veri(i,17)==1) && (veri(i,16)==1)
    (4,6): ('GPIO 27', 'GPIO 20'), # led_states(5,7,i) if (veri(i,17)==1) && (veri(i,10)==1)
    (4,7): ('GPIO 27', 'GPIO 21'), # led_states(5,8,i) if (veri(i,17)==1) && (veri(i,11)==1)

    (5,0): ('GPIO 17', 'GPIO 16'), # led_states(6,1,i) if (veri(i,7)==1) && (veri(i,6)==1)
    (5,1): ('GPIO 17', 'GPIO 5'),  # led_states(6,2,i) if (veri(i,7)==1) && (veri(i,2)==1)
    (5,2): ('GPIO 17', 'GPIO 6'),  # led_states(6,3,i) if (veri(i,7)==1) && (veri(i,3)==1)
    (5,3): ('GPIO 17', 'GPIO 13'), # led_states(6,4,i) if (veri(i,7)==1) && (veri(i,5)==1)
    (5,4): ('GPIO 17', 'GPIO 19'), # led_states(6,5,i) if (veri(i,7)==1) && (veri(i,9)==1)
    (5,5): ('GPIO 17', 'GPIO 26'), # led_states(6,6,i) if (veri(i,7)==1) && (veri(i,16)==1)
    (5,6): ('GPIO 17', 'GPIO 20'), # led_states(6,7,i) if (veri(i,7)==1) && (veri(i,10)==1)
    (5,7): ('GPIO 17', 'GPIO 21'), # led_states(6,8,i) if (veri(i,7)==1) && (veri(i,11)==1)

    (6,0): ('GPIO 18', 'GPIO 16'), # led_states(7,1,i) if (veri(i,8)==1) && (veri(i,6)==1)
    (6,1): ('GPIO 18', 'GPIO 5'),  # led_states(7,2,i) if (veri(i,8)==1) && (veri(i,2)==1)
    (6,2): ('GPIO 18', 'GPIO 6'),  # led_states(7,3,i) if (veri(i,8)==1) && (veri(i,3)==1)
    (6,3): ('GPIO 18', 'GPIO 13'), # led_states(7,4,i) if (veri(i,8)==1) && (veri(i,5)==1)
    (6,4): ('GPIO 18', 'GPIO 19'), # led_states(7,5,i) if (veri(i,8)==1) && (veri(i,9)==1)
    (6,5): ('GPIO 18', 'GPIO 26'), # led_states(7,6,i) if (veri(i,8)==1) && (veri(i,16)==1)
    (6,6): ('GPIO 18', 'GPIO 20'), # led_states(7,7,i) if (veri(i,8)==1) && (veri(i,10)==1)
    (6,7): ('GPIO 18', 'GPIO 21'), # led_states(7,8,i) if (veri(i,8)==1) && (veri(i,11)==1)

    (7,0): ('GPIO 23', 'GPIO 16'), # led_states(8,1,i) if (veri(i,13)==1) && (veri(i,6)==1)
    (7,1): ('GPIO 23', 'GPIO 5'),  # led_states(8,2,i) if (veri(i,13)==1) && (veri(i,2)==1)
    (7,2): ('GPIO 23', 'GPIO 6'),  # led_states(8,3,i) if (veri(i,13)==1) && (veri(i,3)==1)
    (7,3): ('GPIO 23', 'GPIO 13'), # led_states(8,4,i) if (veri(i,13)==1) && (veri(i,5)==1)
    (7,4): ('GPIO 23', 'GPIO 19'), # led_states(8,5,i) if (veri(i,13)==1) && (veri(i,9)==1)
    (7,5): ('GPIO 23', 'GPIO 26'), # led_states(8,6,i) if (veri(i,13)==1) && (veri(i,16)==1)
    (7,6): ('GPIO 23', 'GPIO 20'), # led_states(8,7,i) if (veri(i,13)==1) && (veri(i,10)==1)
    (7,7): ('GPIO 23', 'GPIO 21')  # led_states(8,8,i) if (veri(i,13)==1) && (veri(i,11)==1)
}

# Consolidate all GPIO names involved for efficient initial DataFrame filtering and numpy conversion
ALL_REQUIRED_GPIOS = sorted(list(set([gpio for pair in OUTPUT_MATRIX_POS_TO_GPIO_PAIR.values() for gpio in pair])))

# --- Character Templates (HIGHLY REFINED AND EXPANDED, ORDERED ALPHABETICALLY/NUMERICALLY) ---
# These templates are designed to precisely match the pixel patterns observed
# in the provided trace data for the flag characters, and include a comprehensive
# set for the rest of the alphabet and numbers, ordered for clarity.
# Each template is an 8x8 NumPy array of integers (0 for OFF, 1 for ON).
CHARACTER_TEMPLATES = {
    # --- Letters (A-Z) ---
    'A': np.array([
        [0,0,1,1,1,1,0,0], [0,1,0,0,0,0,1,0], [0,1,0,0,0,0,1,0], [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1]
    ], dtype=int),
    'B': np.array([
        [1,1,0,0,0,1,1,1], [1,1,0,0,0,1,1,1], [1,1,0,0,0,1,0,0], [1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,0,0], [1,1,0,0,0,1,0,0], [1,1,0,0,0,1,0,0], [1,1,0,0,0,1,0,0]
    ], dtype=int),
    'C': np.array([
        [0,1,1,1,1,1,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0]
    ], dtype=int),
    'D': np.array([
        [1,1,1,1,1,1,0,0], [1,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,1,0], [1,1,1,1,1,1,0,0]
    ], dtype=int),
    'E': np.array([[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]], dtype=int),
    'F': np.array([
        [1,1,1,1,1,1,1,1], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,1,1,1,1,1,0],
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0]
    ], dtype=int),
    'G': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,0,0,0,0,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,1,1,1,1,1,0]], dtype=int),
    'H': np.array([
        [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1], [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1]
    ], dtype=int),
    'I': np.array([
        [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0],
        [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0]
    ], dtype=int),
    'J': np.array([[0,0,0,0,1,1,1,1],[0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,1,0,0],[0,1,1,1,1,1,0,0]], dtype=int),
    'K': np.array([[1,0,0,0,0,0,1,0],[1,0,0,0,1,1,0,0],[1,0,0,1,0,0,0,0],[1,1,1,0,0,0,0,0],[1,0,0,1,0,0,0,0],[1,0,0,0,1,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1]], dtype=int),
    'L': np.array([
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]
    ], dtype=int),
    'M': np.array([
        [1,1,1,0,0,0,1,1], [1,1,1,1,0,0,1,1], [1,1,1,1,1,0,1,1], [1,0,1,1,1,1,0,1],
        [1,0,1,1,1,1,0,1], [1,0,1,1,1,1,0,1], [1,0,1,1,1,1,0,1], [1,0,1,1,1,1,0,1]
    ], dtype=int),
    'N': np.array([
        [1,1,0,0,0,0,1,1], [1,1,1,0,0,0,1,1], [1,1,0,1,0,0,1,1], [1,1,0,0,1,0,1,1],
        [1,1,0,0,0,1,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1]
    ], dtype=int),
    'O': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,1,1,1,1,1,0]], dtype=int),
    'P': np.array([
        [1,1,1,1,1,1,1,0], [1,1,0,0,0,0,0,1], [1,1,0,0,0,0,0,1], [1,1,1,1,1,1,1,0],
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0]
    ], dtype=int),
    'Q': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,1],[1,0,0,0,0,0,1,1],[0,1,1,1,1,1,1,1]], dtype=int),
    'R': np.array([[1,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1]], dtype=int),
    'S': np.array([
        [0,1,1,1,1,1,1,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,1,1], [0,0,0,0,0,0,1,1], [1,1,0,0,0,0,1,1], [0,1,1,1,1,1,1,0]
    ], dtype=int),
    'T': np.array([
        [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0],
        [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0]
    ], dtype=int),
    'U': np.array([
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]
    ], dtype=int),
    'V': np.array([[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,0,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,0,1,1,0,0,0]], dtype=int),
    'W': np.array([[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,1,0,1,0,0,1],[0,1,0,1,0,1,0,0],[0,0,1,0,1,0,0,0]], dtype=int),
    'X': np.array([
        [1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0], [0,0,1,0,0,1,0,0], [0,0,0,1,1,0,0,0],
        [0,0,0,1,1,0,0,0], [0,0,1,0,0,1,0,0], [0,1,0,0,0,0,1,0], [1,0,0,0,0,0,0,1]
    ], dtype=int), # Modified for more typical X
    'Y': np.array([[1,0,0,0,0,0,0,1],[0,1,0,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]], dtype=int),
    'Z': np.array([
        [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]
    ], dtype=int), # Modified for more typical Z

    # --- Numbers (0-9) ---
    '0': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,1,1,1,1,1,0]], dtype=int),
    '1': np.array([
        [0,0,1,0,0,0,0,0], [0,1,1,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0], [1,1,1,1,1,1,1,1]
    ], dtype=int), # Modified for a common '1'
    '2': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,0,0,0,0,1,1,0],[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]], dtype=int),
    '3': np.array([[0,1,1,1,1,1,1,0],[1,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,1,1,1,1,1,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,1,1,1,1,1,0]], dtype=int),
    '4': np.array([
        [0,0,0,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,1,0,1,0,0,0,0], [1,0,0,1,0,0,0,0],
        [1,1,1,1,1,1,1,1], [0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0]
    ], dtype=int),
    '5': np.array([
        [1,1,1,1,1,1,1,1], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [1,1,1,1,1,1,1,0]
    ], dtype=int),
    '6': np.array([
        [0,1,1,1,1,1,1,0], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,0],
        [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0]
    ], dtype=int),
    '7': np.array([
        [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0]
    ], dtype=int),
    '8': np.array([
        [0,1,1,1,1,1,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0],
        [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0]
    ], dtype=int),
    '9': np.array([
        [0,1,1,1,1,1,1,0], [1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,1], [0,1,1,1,1,1,1,0]
    ], dtype=int),

    # --- Special Characters ---
    '_': np.array([
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]
    ], dtype=int), # Underscore - usually at the bottom
    '{': np.array([
        [0,0,1,1,1,0,0,0], [0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,1,1,0,0,0]
    ], dtype=int), # Left curly brace
    '}': np.array([
        [0,0,0,1,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,1,1,1,0,0]
    ], dtype=int), # Right curly brace
    '.': np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0]], dtype=int), # Period
    ':': np.array([[0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0]], dtype=int), # Colon
    '-': np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]], dtype=int), # Hyphen
    ' ': np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]], dtype=int), # Space
    '!': np.array([
        [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0],
        [0,0,1,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,1,1,0,0,0,0], [0,0,1,1,0,0,0,0]
    ], dtype=int), # Exclamation Mark
    '@': np.array([
        [0,1,1,1,1,1,0,0], [1,0,0,0,0,0,1,0], [1,0,1,1,1,1,1,0], [1,0,1,0,0,1,0,0],
        [1,0,1,1,1,1,0,0], [1,0,0,0,0,0,1,0], [0,1,1,1,1,1,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # At symbol
    '#': np.array([
        [0,1,1,0,0,1,1,0], [0,1,1,0,0,1,1,0], [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1], [0,1,1,0,0,1,1,0], [0,1,1,0,0,1,1,0]
    ], dtype=int), # Hash/Number sign
    '$': np.array([
        [0,0,1,1,1,0,0,0], [0,1,1,1,1,0,0,0], [1,1,0,0,0,0,0,0], [0,0,1,1,1,0,0,0],
        [0,0,0,1,1,1,0,0], [0,0,0,0,0,1,1,0], [0,0,0,0,1,1,0,0], [0,1,1,1,0,0,0,0]
    ], dtype=int), # Dollar sign
    '%': np.array([
        [1,1,0,0,0,0,1,1], [1,1,0,0,0,1,1,1], [0,0,0,0,1,1,0,0], [0,0,0,1,1,0,0,0],
        [0,0,1,1,0,0,0,0], [0,1,1,1,0,0,1,1], [1,1,0,0,0,0,1,1], [1,1,0,0,0,0,1,1]
    ], dtype=int), # Percent sign
    '^': np.array([
        [0,0,1,1,1,0,0,0], [0,1,0,0,0,1,0,0], [1,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Caret/Circumflex
    '&': np.array([
        [0,1,1,1,1,1,0,0], [1,0,0,0,0,0,1,0], [1,0,1,1,1,1,0,0], [1,0,1,0,0,0,0,0],
        [1,0,1,0,0,0,0,0], [1,0,1,1,1,1,0,0], [0,1,0,0,0,0,1,0], [0,0,1,1,1,1,0,0]
    ], dtype=int), # Ampersand
    '*': np.array([
        [0,0,0,0,0,0,0,0], [0,1,0,1,0,1,0,0], [0,0,1,1,1,0,0,0], [0,1,0,1,0,1,0,0],
        [0,0,1,1,1,0,0,0], [0,1,0,1,0,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Asterisk
    '+': np.array([
        [0,0,0,0,0,0,0,0], [0,0,0,1,1,0,0,0], [0,0,0,1,1,0,0,0], [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1], [0,0,0,1,1,0,0,0], [0,0,0,1,1,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Plus sign
    '=': np.array([
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1]
    ], dtype=int), # Equals sign
    '(': np.array([
        [0,0,0,1,1,0,0,0], [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,1,0,0,0]
    ], dtype=int), # Left Parenthesis
    ')': np.array([
        [0,0,0,0,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,1,0,0]
    ], dtype=int), # Right Parenthesis
    '[': np.array([
        [1,1,1,1,1,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,1,1,1,0,0,0]
    ], dtype=int), # Left Square Bracket
    ']': np.array([
        [0,0,0,1,1,1,1,1], [0,0,0,0,0,1,1,1], [0,0,0,0,0,1,1,1], [0,0,0,0,0,1,1,1],
        [0,0,0,0,0,1,1,1], [0,0,0,0,0,1,1,1], [0,0,0,0,0,1,1,1], [0,0,0,1,1,1,1,1]
    ], dtype=int), # Right Square Bracket
    '<': np.array([
        [0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]
    ], dtype=int), # Less than sign
    '>': np.array([
        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0]
    ], dtype=int), # Greater than sign
    '/': np.array([
        [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0]
    ], dtype=int), # Forward slash
    '\\': np.array([
        [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]
    ], dtype=int), # Backslash
    '?': np.array([
        [0,1,1,1,1,1,0,0], [1,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0]
    ], dtype=int), # Question Mark
    '~': np.array([
        [0,0,0,0,0,0,0,0], [0,1,0,1,0,1,0,0], [1,0,1,0,1,0,1,0], [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Tilde
    "'": np.array([
        [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Single Quote
    '"': np.array([
        [0,1,0,1,0,0,0,0], [0,1,0,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    ], dtype=int), # Double Quote
}


# --- Utility Functions ---
def print_matrix_as_chars(matrix):
    """
    Prints an 8x8 matrix as a string of '#' (ON) and '.' (OFF) characters.
    Pixels with a value > 0 are considered ON.
    """
    char_representation = []
    binarized_matrix = (matrix > 0).astype(int)
    for row in binarized_matrix:
        char_representation.append(''.join(['#' if pixel == 1 else '.' for pixel in row]))
    return "\n".join(char_representation)

def get_best_character_match(matrix, templates=CHARACTER_TEMPLATES):
    """
    Compares a matrix against known character templates and returns the best match.
    The input matrix is first binarized (pixels > 0 are ON).
    """
    binarized_matrix = (matrix > 0).astype(int)
    best_match_char = '?'
    best_match_score = -1
    
    for char, template in templates.items():
        score = np.sum(binarized_matrix == template) # Count matching pixels
        if score > best_match_score:
            best_match_score = score
            best_match_char = char
    
    confidence = best_match_score / 64.0 if best_match_score != -1 else 0.0 # 64 pixels total
    return best_match_char, confidence, best_match_score

# --- Core Logic for Frame Reconstruction (CRITICAL CORRECTION) ---
def reconstruct_pixel_states_optimized(df_gpio_data_np, gpio_name_to_col_idx):
    """
    Reconstructs the LED states for each pixel across all timestamps based on the
    explicit GPIO mapping and pixel activation logic from 'trace_simulation.m'.
    """
    num_samples = df_gpio_data_np.shape[0]
    # Initialize the 8x8xN_samples array where N_samples is the number of rows in trace.csv
    led_states_per_timestamp = np.zeros((8, 8, num_samples), dtype=int)

    # Iterate through each pixel position in the 8x8 output matrix
    for r_out in range(8):
        for c_out in range(8):
            # Get the specific GPIO names that control this output pixel from our derived mapping
            gpio_row_name, gpio_col_name = OUTPUT_MATRIX_POS_TO_GPIO_PAIR[(r_out, c_out)]
            
            # Get the column indices for these GPIOs in the df_gpio_data_np array
            row_gpio_col_idx = gpio_name_to_col_idx[gpio_row_name]
            col_gpio_col_idx = gpio_name_to_col_idx[gpio_col_name]
            
            # Extract the state of these two GPIOs for all timestamps
            row_gpio_states_all_samples = df_gpio_data_np[:, row_gpio_col_idx]
            col_gpio_states_all_samples = df_gpio_data_np[:, col_gpio_col_idx]

            # CORRECTED PIXEL ACTIVATION LOGIC: Pixel is ON if BOTH GPIOs are HIGH (1)
            pixel_on_states_for_all_samples = (row_gpio_states_all_samples == 1) & \
                                              (col_gpio_states_all_samples == 1)
            
            # Assign the resulting pixel states to the corresponding position in the 3D array
            led_states_per_timestamp[r_out, c_out, :] = pixel_on_states_for_all_samples.astype(int)
            
    return led_states_per_timestamp


def aggregate_and_rotate_frames_optimized(led_states_per_timestamp):
    """
    Aggregates LED states over a sliding window of 8 timestamps and applies rotation,
    mimicking the display behavior. Each element in the output list is a fully
    aggregated and rotated 8x8 matrix representing a 'displayed frame'.
    """
    num_samples = led_states_per_timestamp.shape[2]
    displayed_frames_with_indices = []

    # The 8-sample window for aggregation simulates a full matrix refresh cycle,
    # where each of the 8 rows is scanned once within these 8 samples.
    # We start from the 7th index to ensure a full window of 8 samples (0-7, 1-8, etc.)
    for current_sample_idx in range(7, num_samples):
        # Sum the pixel states over the last 8 samples (from current_sample_idx - 7 to current_sample_idx)
        # to get the intensity for each pixel in the 'full frame'.
        # Pixels that were ON in more samples within this window will have higher values.
        aggregated_matrix = np.sum(led_states_per_timestamp[:, :, current_sample_idx - 7 : current_sample_idx + 1], axis=2)
        
        # Rotate the matrix to match the expected visual orientation (90 degrees counter-clockwise)
        # This rotation is crucial for mapping the internally represented matrix to the
        # human-readable character templates.
        rotated_matrix_for_display = np.rot90(aggregated_matrix, k=1)
        
        # Store the processed matrix along with the index of the last original sample
        # that contributed to it. This index helps align with the 'Time [s]' data later for segmentation.
        displayed_frames_with_indices.append((rotated_matrix_for_display, current_sample_idx))
    
    return displayed_frames_with_indices

# --- Animation Function ---
def animate_frames(frames_to_animate, save_path=None):
    """
    Generates and displays an animation of the given frames.
    Optionally saves the animation to a GIF file.
    :param frames_to_animate: A list of 8x8 NumPy arrays, where each array is a single frame.
    :param save_path: Optional file path to save the animation as a GIF (e.g., 'animation.gif').
    """
    if not frames_to_animate:
        print("No frames to animate.")
        return

    fig, ax = plt.subplots(figsize=(6, 6)) # Adjust figure size as needed for better visibility
    
    # Display the first frame
    # vmin=0 and vmax=8 because each pixel can be 'on' for up to 8 scan lines in the aggregation window.
    im = ax.imshow(frames_to_animate[0], cmap='Greys', vmin=0, vmax=8, interpolation='nearest')
    
    # Remove ticks and labels for a cleaner display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("LED Matrix Animation")

    # Function to update the frame data for the animation
    def update(frame):
        im.set_data(frames_to_animate[frame])
        return [im]

    # Create the animation
    # interval: Delay between frames in milliseconds (e.g., 50ms for 20 frames/sec)
    # blit: Optimize drawing by only redrawing what has changed (can improve performance)
    ani = animation.FuncAnimation(fig, update, frames=len(frames_to_animate), interval=50, blit=True)

    if save_path:
        print(f"\n--- Saving Animation to {save_path} ---")
        try:
            # Use Pillow writer as it's generally reliable and included with matplotlib installations
            # Ensure 'pillow' is installed (pip install pillow)
            ani.save(save_path, writer='pillow', fps=20) # fps corresponds to 1000/interval
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Error saving animation: {e}. Please ensure 'pillow' library is installed for GIF saving.")

    print("\n--- Displaying Animation ---")
    plt.show() # This will open a new window to show the animation

# --- Main Deciphering Function ---
def decipher_message_optimized(csv_file_path='traces.csv'):
    print(f"--- Starting Matrix Message Deciphering from {csv_file_path} ---")

    # 1. Load Trace Data
    df_traces = None # Initialize df_traces to None
    try:
        if not os.path.exists(csv_file_path):
            print(f"Error: {csv_file_path} not found. Using internal fallback content.")
            # Use the fallback content if the file is not found
            fallback_content = """Time [s],GPIO 5,GPIO 6,GPIO 12,GPIO 13,GPIO 16,GPIO 17,GPIO 18,GPIO 19,GPIO 20,GPIO 21,GPIO 22,GPIO 23,GPIO 24,GPIO 25,GPIO 26,GPIO 27
0.000569105,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
0.001182079,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
0.001703977,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
0.002223014,1,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1
0.0.003775119,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
0.004293918,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
0.104838132,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0
0.105206966,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,0
0.105572938,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0
0.105940103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
0.106305122,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
0.106670141,1,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0
0.107037067,1,1,0,1,1,0,0,1,0,0,0,0,0,1,1,0
0.107401132,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0
0.207886934,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
0.208255052,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0
0.208621978,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0
0.208987951,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1
0.209352970,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0
0.209720134,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0
0.210086107,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
0.210451126,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
0.310941934,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
0.311310052,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0
0.311676025,1,1,0,1,0,1,0,1,1,0,0,0,0,0,1,0
3.857671976,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0
3.958368062,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
3.958891153,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0
3.959417104,1,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0
3.959938049,1,1,0,1,0,0,0,1,1,1,0,0,0,0,1,1
3.960461139,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0
3.960983037,1,1,0,1,0,0,0,1,1,1,0,0,1,0,1,0
3.961506128,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0
3.962027072,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0
4.062721967,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
4.063246965,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0
4.063768148,1,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0
4.064293146,0,1,0,1,1,0,0,0,1,0,0,0,0,0,1,1
4.064814090,0,1,0,1,1,0,0,0,1,0,1,0,0,0,1,0
4.065336942,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0
4.065856933,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0
4.066379070,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0
4.167073011,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
4.167597055,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0
4.168162107,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0
4.168687105,1,1,0,1,1,0,0,0,1,0,0,0,0,0,1,1
4.169209957,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0
4.169733047,1,1,0,1,1,0,0,1,1,0,0,0,1,0,1,0
4.170256137,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0
4.170778036,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0
4.271485090,1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0
4.272011995,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0
4.272535085,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0
4.273055076,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1
4.273576021,1,1,0,1,0,0,0,1,1,0,1,0,0,0,1,0
4.274096965,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0
4.274616003,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0
0.002740144,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0
0.003258943,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,0
4.275135993,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0
"""
            df_traces = pd.read_csv(StringIO(fallback_content))
            print("Loaded content from internal fallback.")
        else:
            df_traces = pd.read_csv(csv_file_path)
            print(f"Successfully read data from {csv_file_path}.")

    except Exception as e:
        print(f"An error occurred during CSV loading: {e}")
        print("Deciphering aborted due to loading error.")
        return # Ensure exit if loading fails

    # Critical check: Ensure df_traces is not None before proceeding
    if df_traces is None:
        print("Fatal Error: df_traces is None after loading attempt. Deciphering cannot proceed.")
        return # Explicitly exit if df_traces is still None

    time_series = df_traces['Time [s]'].to_numpy()
    
    try:
        # Filter df_traces to include only the GPIO columns before converting to numpy
        df_gpio_data = df_traces[ALL_REQUIRED_GPIOS]
        df_gpio_data_np = df_gpio_data.to_numpy(dtype=int)
    except KeyError as e:
        print(f"Error: Missing expected GPIO columns in traces.csv: {e}")
        print(f"Expected: {ALL_REQUIRED_GPIOS}")
        print(f"Available: {df_traces.columns.tolist()}")
        print("Deciphering aborted due to missing columns.")
        return

    # Create a mapping from GPIO name to its column index in the df_gpio_data_np array.
    # This is crucial because df_gpio_data_np might not have columns in the original order.
    gpio_name_to_col_idx = {name: df_gpio_data.columns.get_loc(name) for name in ALL_REQUIRED_GPIOS}
    
    # 2. Reconstruct raw pixel states per timestamp (with corrected logic and mapping)
    led_states_per_timestamp = reconstruct_pixel_states_optimized(df_gpio_data_np, gpio_name_to_col_idx)
    print("\n--- Raw pixel states reconstructed per timestamp (using corrected logic and mapping). ---")

    # 3. Aggregate and rotate frames for display
    displayed_frames_with_indices = aggregate_and_rotate_frames_optimized(led_states_per_timestamp)
    print("--- Frames aggregated and rotated for display. ---")

    # 4. Decipher and display the message, using precise time-based grouping
    final_deciphered_string = ""
    deciphered_blocks_info = [] 
    
    print("\n--- Deciphering Message Sequence (Precise Time-Based Grouping Applied) ---")
    
    # Time threshold to define a new character display.
    TIME_JUMP_THRESHOLD = 0.01 # seconds. This value is from your original script.

    current_block_frames_data = [] # Stores (matrix, original_frame_idx) tuples for the current character block
    
    # Iterate through all displayed frames (which are already aggregated and rotated)
    for i in range(len(displayed_frames_with_indices)):
        matrix, original_gpio_sample_idx = displayed_frames_with_indices[i]
        
        # The time associated with this `matrix` (aggregated frame) is the time of the *last* GPIO sample
        # that contributed to its creation, which is `time_series[original_gpio_sample_idx]`.
        current_frame_display_time = time_series[original_gpio_sample_idx]

        if not current_block_frames_data:
            # If it's the very first frame or starting a new block, add it directly
            current_block_frames_data.append((matrix, original_gpio_sample_idx))
        else:
            # Get the time of the last frame added to the current block
            _, last_gpio_sample_idx_in_block = current_block_frames_data[-1]
            last_frame_display_time_in_block = time_series[last_gpio_sample_idx_in_block]

            # Check for a significant time jump. If the gap exceeds the threshold, it marks
            # the end of the current character's display and the beginning of a new one.
            if (current_frame_display_time - last_frame_display_time_in_block) > TIME_JUMP_THRESHOLD:
                # A new character has started. Process the previous `current_block_frames_data`
                # to determine the character it represents.
                chosen_char, chosen_conf, chosen_matrix, chosen_frame_num, chosen_transformation = \
                    process_character_block(current_block_frames_data)
                
                final_deciphered_string += chosen_char # Append the deciphered character to the result string
                deciphered_blocks_info.append({ # Store detailed information for later debugging/summary
                    'char': chosen_char, 'confidence': chosen_conf, 'transformation': chosen_transformation,
                    'matrix': chosen_matrix, 'frame_num': chosen_frame_num
                })
                
                # Start a new block with the current frame
                current_block_frames_data = [(matrix, original_gpio_sample_idx)]
            else:
                # No significant time jump, this frame belongs to the current character block.
                current_block_frames_data.append((matrix, original_gpio_sample_idx))

    # After the loop, process any remaining frames in the last character block (if any exist)
    if current_block_frames_data:
        chosen_char, chosen_conf, chosen_matrix, chosen_frame_num, chosen_transformation = \
            process_character_block(current_block_frames_data)
        
        final_deciphered_string += chosen_char
        deciphered_blocks_info.append({
            'char': chosen_char, 'confidence': chosen_conf, 'transformation': chosen_transformation,
            'matrix': chosen_matrix, 'frame_num': chosen_frame_num
        })

    # --- Call the animation function (moved before print statements) ---
    # Extract just the matrices for animation.
    animation_frames = [frame_data[0] for frame_data in displayed_frames_with_indices]
    
    # Save animation to a GIF file
    animation_output_path = 'matrix_animation.gif'
    animate_frames(animation_frames, save_path=animation_output_path)

    print("\n--- Deciphered Message Summary ---")
    if final_deciphered_string:
        print(f"Deciphered String: {final_deciphered_string}")

        # Targeted search for the flag format "HTB{...}" for easy extraction
        flag_start_index = final_deciphered_string.find("HTB{")
        if flag_start_index != -1:
            flag_end_index = final_deciphered_string.find("}", flag_start_index)
            if flag_end_index != -1:
                potential_flag = final_deciphered_string[flag_start_index : flag_end_index + 1]
                print(f"\n>>> POTENTIAL FLAG DETECTED: {potential_flag} <<<")
                
                # Provide detailed frames specifically for the detected flag section for verification
                print("\n--- Detailed Frames for Potential Flag (Best Match in Block) ---")
                
                # Iterate only over the character blocks corresponding to the potential flag for detailed output
                for i in range(flag_start_index, flag_end_index + 1):
                    if i < len(deciphered_blocks_info): # Ensure index is valid
                        info = deciphered_blocks_info[i]
                        # Display character, original frame index, confidence, and the best-matching pattern
                        print(f"\nCharacter: '{info['char']}' (Derived from Frame {info['frame_num']}, Confidence: {info['confidence']:.2f})")
                        print(f"  Best Pixel Pattern (Transformation: {info['transformation']}):")
                        print(print_matrix_as_chars(info['matrix']))
                        print("-" * 20) # Separator for readability
            else:
                print("\nDetected 'HTB{' prefix but no closing '}' found in the sequence. The flag might be incomplete or malformed.")
        else:
            print("\n'HTB{' prefix not detected in the deciphered string. The flag may not be present, or its initial characters are not recognized with high confidence.")
        
    else:
        print("No patterns recognized in the trace data. The display might have been consistently blank or too noisy.")

    print("\n--- Deciphering Process Complete ---")


# --- Helper function to process a block of frames for a single character (ENHANCED) ---
def process_character_block(block_frames_data):
    """
    Analyzes a block of frames (representing a single character display)
    by summing them into a composite matrix for more robust recognition.
    Then determines the most likely character, its confidence, and the
    best representative pixel pattern from that block.
    `block_frames_data` is a list of (matrix, original_frame_idx) tuples,
    where each matrix is an aggregated and rotated 8x8 frame.
    """
    if not block_frames_data:
        # If the block is empty, return a space character with no confidence.
        return ' ', 0.0, np.zeros((8,8), dtype=int), 0, 'Empty Block'

    # 1. Create a composite matrix by summing all matrices in the block.
    # This helps "average" out noise and reinforces consistently lit pixels.
    composite_matrix = np.sum([frame_data[0] for frame_data in block_frames_data], axis=0)
    
    # Determine the original frame index of the first frame in this block.
    # This helps in debugging and understanding the temporal context.
    first_original_frame_idx_in_block = block_frames_data[0][1]

    # Initialize best match for this block
    best_match_for_block = {
        'char': '?',
        'confidence': 0.0,
        'matrix': composite_matrix, # Default to the composite matrix (binarized later)
        'frame_num': first_original_frame_idx_in_block,
        'transformation': 'Composite (Initial)'
    }
    
    # Define transformations to check against templates for the composite matrix
    transformations_to_check = {
        "Composite (Current Orientation)": lambda m: m,
        "Composite (Flipped Horizontal)": np.fliplr,
        "Composite (Flipped Vertical)": np.flipud,
        "Composite (Rotated 90 deg CW)": lambda m: np.rot90(m, k=-1),
        "Composite (Rotated 180 deg)": lambda m: np.rot90(m, k=-2),
        "Composite (Rotated 270 deg CW)": lambda m: np.rot90(m, k=-3)
    }
    
    # Iterate through all possible transformations of the composite matrix
    for name, func in transformations_to_check.items():
        transformed_composite_matrix = func(composite_matrix)
        
        # Get the best character match for this transformed composite matrix
        char_match, confidence, _ = get_best_character_match(transformed_composite_matrix)
        
        # If this match has higher confidence, update the best match for the block
        if confidence > best_match_for_block['confidence']:
            best_match_for_block['confidence'] = confidence
            best_match_for_block['char'] = char_match
            # Store the *transformed* matrix that yielded the best match for display
            best_match_for_block['matrix'] = transformed_composite_matrix
            best_match_for_block['transformation'] = name

    # Heuristic for determining if it's a space or genuinely unrecognizable character:
    # If confidence is low AND the number of lit pixels in the *best-matched* matrix is also very low,
    # it's likely just noise or a blank period, so force it to be a space.
    # The threshold for lit pixels (e.g., < 5) can be fine-tuned.
    num_lit_pixels_in_best_match = np.sum(best_match_for_block['matrix'] > 0)
    
    if best_match_for_block['confidence'] < 0.60 and num_lit_pixels_in_best_match < 8: # A few pixels can be noise
        if best_match_for_block['char'] == '?' or best_match_for_block['char'] == ' ':
            best_match_for_block['char'] = ' '
            best_match_for_block['transformation'] = 'Heuristic: Very Low Confidence / Dim Frame -> Space'
    # Adding a debug print for each processed character block
    print(f"\n--- Processed Block (Start Sample: {first_original_frame_idx_in_block}) ---")
    print(f"  Deciphered Character: '{best_match_for_block['char']}' (Confidence: {best_match_for_block['confidence']:.2f}, Transformed as: {best_match_for_block['transformation']})")
    print("  Composite Pixel Pattern:")
    print(print_matrix_as_chars(best_match_for_block['matrix']))
    print("----------------------------")


    return (best_match_for_block['char'], best_match_for_block['confidence'], 
            best_match_for_block['matrix'], best_match_for_block['frame_num'], 
            best_match_for_block['transformation'])

# --- Execute the solution script ---
# The function will attempt to read 'traces.csv' from the current directory.
# If 'traces.csv' is not available, it will use an in-memory fallback.
if __name__ == "__main__":
    decipher_message_optimized('traces.csv')

