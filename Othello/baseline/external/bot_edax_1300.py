"""Edax Othello bot, depth level 1 (~Elo 1300 at 1s/move).

Strength controlled via Edax search level (Edax has no UCI_Elo knob).
Time per move defaults to 1 second; override with EDAX_TIME=<sec>.

The Elo label is approximate — Edax level → Elo mapping from community
benchmarks (1s/move):
    level 1\xa0\xa0\xa0\xa0\xe2\x89\x88\xa01300
    level 3\xa0\xa0\xa0\xa0\xe2\x89\x88\xa01500
    level 5\xa0\xa0\xa0\xa0\xe2\x89\x88\xa01700
    level 7\xa0\xa0\xa0\xa0\xe2\x89\x88\xa01900
    level 9\xa0\xa0\xa0\xa0\xe2\x89\x88\xa02100
    level 12\xa0\xa0\xa0\xe2\x89\x88\xa02300
    level 16\xa0\xa0\xa0\xe2\x89\x88\xa02500
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _edax_base import make_bot  # noqa: E402

Bot = make_bot(1300, 1)
