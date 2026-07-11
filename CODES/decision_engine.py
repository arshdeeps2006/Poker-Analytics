
# Combining Everything: The Betting Decision Engine
# d = EHS - Pot Odds

# P(Bet)  = 1 / (1 + e^(-a(d-f1)))
# P(Fold) = 1 / (1 + e^( a(d+f2)))
# P(Call) = e^(-20(d+fc)^2)

# The three raw probabilities are renormalized to sum to 1.

import math
from pot_odds import pot_odds

# Per-archetype sigmoid constants, exactly as specified in the report.
PARAMS = {
    "tight":    {"a": 8, "f1": 0.10,  "f2": 0.15, "fc": -0.05},
    "loose":    {"a": 5, "f1": 0.00,  "f2": 0.05, "fc": 0.00},
    "balanced": {"a": 6, "f1": 0.05,  "f2": 0.10, "fc": -0.02},
    "bluffer":  {"a": 4, "f1": -0.10, "f2": 0.05, "fc": 0.05},
}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def betting_decision(EHS, pot, bet_to_call, opponent_type="balanced"):
    
    # EHS should already be position-adjusted if you want positional
    # advantage reflected in the decision (Chapter 6).
    
    odds = pot_odds(bet_to_call, pot)
    d = EHS - odds

    p = PARAMS.get(opponent_type, PARAMS["balanced"])

    prob_bet = sigmoid(p["a"] * (d - p["f1"]))
    prob_fold = sigmoid(-p["a"] * (d + p["f2"]))
    prob_call = math.exp(-20 * (d + p["fc"]) ** 2)

    total = prob_bet + prob_fold + prob_call
    prob_bet, prob_fold, prob_call = (
        prob_bet / total,
        prob_fold / total,
        prob_call / total,
    )

    actions = {"Raise/Bet": prob_bet, "Fold": prob_fold, "Call": prob_call}
    decision = max(actions, key=actions.get)

    return {
        "d": round(d, 4),
        "pot_odds": round(odds, 4),
        "decision": decision,
        "prob_raise": round(prob_bet, 3),
        "prob_call": round(prob_call, 3),
        "prob_fold": round(prob_fold, 3),
    }
