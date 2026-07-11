import sys
sys.path.insert(0, "/home/poker_analytics")

from treys import Card
from pipeline import run_pipeline, print_report

print("#" * 60)
print("# SCENARIO: 5(h)2(h) flush draw on A(h)9(h)K(c), CO position")
print("# Facing a $3 bet into a $10 pot, vs an unprofiled opponent")
print("#" * 60)

hole = [Card.new("5h"), Card.new("2h")]
board = [Card.new("Ah"), Card.new("9h"), Card.new("Kc")]

result = run_pipeline(
    hole=hole,
    board=board,
    pot=10,
    bet_to_call=3,
    stack_hero=400,
    stack_opp=500,
    position="CO",
    opponent_type_override="balanced",  # no hand history yet -> assume balanced
    ehs_samples=2000,
    seed=7,
)
print_report(result)


print("\n\n" + "#" * 60)
print("# SAME SCENARIO, but now with real opponent hand history")
print("# (loose-aggressive opponent, e.g. someone who 3-bets light)")
print("#" * 60)

opponent_history = [
    {'entered': True, 'raised_pf': True, 'bets_raises': 4, 'calls': 1},
    {'entered': True, 'raised_pf': True, 'bets_raises': 3, 'calls': 0},
    {'entered': True, 'raised_pf': False, 'bets_raises': 2, 'calls': 2},
    {'entered': False, 'raised_pf': False, 'bets_raises': 0, 'calls': 0},
    {'entered': True, 'raised_pf': True, 'bets_raises': 5, 'calls': 1},
]

result2 = run_pipeline(
    hole=hole,
    board=board,
    pot=10,
    bet_to_call=3,
    stack_hero=400,
    stack_opp=500,
    position="CO",
    opponent_hands=opponent_history,
    ehs_samples=2000,
    seed=7,
)
print_report(result2)
