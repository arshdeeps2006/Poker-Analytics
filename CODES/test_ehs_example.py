# Validates EHS computation against the report's worked example (Table 2.1):
# A(d) Q(c) on flop 3(h) 4(c) J(h)
# Report's hand-calculated values: HS=0.59, PPOT=0.20, NPOT=0.27, EHS=0.5127

import sys
sys.path.insert(0, "/home/poker_analytics")

from treys import Card
from ehs import hand_strength, hand_potential, effective_hand_strength

hole = [Card.new("Ad"), Card.new("Qc")]
board = [Card.new("3h"), Card.new("4c"), Card.new("Jh")]

# 1. HS should match exactly (it's an exact enumeration, not sampled)
hs = hand_strength(hole, board)
print(f"Computed HS = {hs:.4f}   (hand-computed HS = 0.59)")

# 2. PPOT/NPOT are Monte Carlo estimates -> run with more samples for stability,
#    and check they're in the right ballpark vs the report's exact enumeration values.
ppot, npot = hand_potential(hole, board, samples=3000, seed=42)
print(f"Computed PPOT = {ppot:.4f}   (exact PPOT = 0.20)")
print(f"Computed NPOT = {npot:.4f}   (exact NPOT = 0.27)")

ehs, hs2, ppot2, npot2 = effective_hand_strength(hole, board, samples=3000, seed=42)
print(f"\nComputed EHS = {ehs:.4f}   (hand-computed EHS = 0.5127)")
