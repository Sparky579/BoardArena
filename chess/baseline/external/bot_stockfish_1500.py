"""Stockfish UCI bot, pinned to Elo 1500 for calibration.

Strength is controlled via UCI_LimitStrength + UCI_Elo; time per move
defaults to 1 second (matches Stockfish's published Elo calibration).
Override with STOCKFISH_TIME=<sec> env var if you need a different
budget.
"""

import os
import sys

# load_bot() imports this file via spec_from_file_location which does
# NOT add the file's directory to sys.path, so we have to so the
# sibling _stockfish_base module is resolvable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _stockfish_base import make_bot  # noqa: E402

Bot = make_bot(1500)
