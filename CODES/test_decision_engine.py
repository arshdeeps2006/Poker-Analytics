import sys
sys.path.insert(0, "/home/poker_analytics")

from treys import Card
from decision_engine import betting_decision

print("Validating against Table 9.1 (using report's own EHS values)")

# Row 1: A(d)Q(c) on 3(h)4(c)J(h), EHS=0.5127, expected Raise/Bet
# (pot/bet not given for this row in the report, so we just confirm the
# decision direction is sane for a high EHS with no bet info -> treat as a 
# check-or-bet spot, pot odds = 0)

r1 = betting_decision(EHS=0.5127, pot=10, bet_to_call=0, opponent_type="balanced")
print(f"\nFor (EHS=0.5127): decision={r1['decision']}  (expects Raise/Bet)")
print(r1)

# Row 2: 5(h)2(h) flush draw, EHS=0.38, expected Call
r2 = betting_decision(EHS=0.38, pot=10, bet_to_call=3, opponent_type="balanced")
print(f"\nFor (EHS=0.38, pot=10,bet=3): decision={r2['decision']}  (expects Call)")
print(r2)

# Row 3: K(h)K(d) trips on river, EHS=0.97, expected Raise/Bet
r3 = betting_decision(EHS=0.97, pot=10, bet_to_call=3, opponent_type="balanced")
print(f"\nFor (EHS=0.97, pot=10,bet=3): decision={r3['decision']}  (expects Raise/Bet)")
print(r3)


print("\n" + "=" * 60)
print("Validating against Table 9.2 (opponent sensitivity, pot=10 bet=3)")
print("=" * 60)
# Table 9.2 doesn't give EHS directly, but gives d=0.282 for all rows.
# Since d = EHS - pot_odds, and pot_odds(3,10) = 0.231 -> EHS = 0.282+0.231 = 0.513
ehs_for_table92 = 0.282 + (3 / (3 + 10))
print(f"Back-solved EHS for table 9.2 = {ehs_for_table92:.4f} (should be ~0.5127, matches row1 hand)")

for opp_type, expected in [("tight", (0.583,0.220,0.197)),
                             ("balanced", (0.621,0.211,0.168)),
                             ("loose", (0.512,0.301,0.187)),
                             ("bluffer", (0.487,0.342,0.171))]:
    res = betting_decision(EHS=ehs_for_table92, pot=10, bet_to_call=3, opponent_type=opp_type)
    print(f"{opp_type:10s} -> d={res['d']:.3f} P(Raise)={res['prob_raise']:.3f} "
          f"P(Call)={res['prob_call']:.3f} P(Fold)={res['prob_fold']:.3f}  "
          f"(report: P(R)={expected[0]} P(C)={expected[1]} P(F)={expected[2]})")
