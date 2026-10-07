"""Settings for the beyond-spec extras: beam search and n-gram repetition blocking.

Not used by the assignment pipeline (configs/base.py, src/). Kept here so the
extras stay runnable without touching the spec path.
"""

BEAM_WIDTH = 4
BEAM_LENGTH_PENALTY = 0.6
# Blocks a repeat of any n-gram of this size already generated in the same
# sequence. Off (0) in the headline run; set to 3 to enable.
NO_REPEAT_NGRAM_SIZE = 0
