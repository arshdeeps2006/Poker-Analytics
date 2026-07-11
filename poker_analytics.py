"""
Poker Analytics Project
=======================
Implements the concepts from the project report:
  - Hand Strength (HS)
  - Positive/Negative Potential (PPOT, NPOT)
  - Effective Hand Strength (EHS = HS*(1-NPOT) + (1-HS)*PPOT)
  - Sigmoid-based betting decision (Fold / Call / Raise)

Uses the `treys` library as the Hand Evaluator (lookup table),
exactly as described in the Poker Analytics slides.

Run:
    python poker_analytics.py
"""

import math
import random
from itertools import combinations
from treys import Card, Evaluator, Deck

evaluator = Evaluator()

# ─────────────────────────────────────────────
# 1. HAND STRENGTH  (HS)
# ─────────────────────────────────────────────

def hand_strength(hole, board):
    """
    HS = (W + T/2) / (W + T + L)

    Enumerate every possible 2-card opponent hand from the
    remaining deck and count wins, ties, losses.
    """
    deck = get_remaining_deck(hole, board)
    opponent_combos = list(combinations(deck, 2))

    W = T = L = 0

    for opp in opponent_combos:
        opp = list(opp)

        # Need 5 board cards for evaluator — if board < 5, skip potential (handled in EHS)
        # For HS we evaluate on the current board only (ignoring future cards)
        if len(board) < 3:
            continue

        hero_score = evaluator.evaluate(board, hole)
        opp_score  = evaluator.evaluate(board, opp)

        # Lower score = better hand in treys
        if hero_score < opp_score:
            W += 1
        elif hero_score == opp_score:
            T += 1
        else:
            L += 1

    total = W + T + L
    if total == 0:
        return 0.0
    return (W + T / 2) / total


# ─────────────────────────────────────────────
# 2. HAND POTENTIAL  (PPOT, NPOT)
# ─────────────────────────────────────────────

def hand_potential(hole, board, samples=200):
    """
    PPOT = probability we improve from behind to ahead
    NPOT = probability we fall from ahead to behind

    We sample opponent combos and future community cards
    to keep runtime reasonable (same idea as Gibbs sampling
    mentioned in the report).
    """
    deck = get_remaining_deck(hole, board)
    cards_needed = 5 - len(board)     # how many community cards still to come

    if cards_needed == 0:
        return 0.0, 0.0               # river — no potential left

    # We'll count:
    #   ahead[i]  = cases where we were ahead and opponent ends up ahead after runout
    #   behind[i] = cases where we were behind and we end up ahead after runout
    ahead_total  = behind_total  = 0
    ahead_caught = behind_improve = 0

    for i in range(samples):
        available = deck[:]
        random.shuffle(available)

        opp = available[:2]
        rest = available[2:]

        # Current standing (before future cards)
        if len(board) >= 3:
            hero_now = evaluator.evaluate(board, hole)
            opp_now  = evaluator.evaluate(board, opp)
            currently_ahead = hero_now < opp_now
            currently_behind = hero_now > opp_now
        else:
            currently_ahead = currently_behind = False

        # Run out the remaining community cards
        future = rest[:cards_needed]
        full_board = board + future

        hero_end = evaluator.evaluate(full_board, hole)
        opp_end  = evaluator.evaluate(full_board, opp)

        if currently_ahead:
            ahead_total += 1
            if opp_end < hero_end:       # opponent overtook us
                ahead_caught += 1

        if currently_behind:
            behind_total += 1
            if hero_end < opp_end:       # we improved past them
                behind_improve += 1

    PPOT = behind_improve / behind_total if behind_total > 0 else 0.0
    NPOT = ahead_caught  / ahead_total   if ahead_total  > 0 else 0.0

    return round(PPOT, 4), round(NPOT, 4)


# ─────────────────────────────────────────────
# 3. EFFECTIVE HAND STRENGTH  (EHS)
# ─────────────────────────────────────────────

def effective_hand_strength(hole, board, samples=200):
    """
    EHS = HS*(1 - NPOT) + (1 - HS)*PPOT
    (Formula from both project documents)
    """
    HS   = hand_strength(hole, board)
    PPOT, NPOT = hand_potential(hole, board, samples)
    EHS  = HS * (1 - NPOT) + (1 - HS) * PPOT
    return round(EHS, 4), round(HS, 4), round(PPOT, 4), round(NPOT, 4)


# ─────────────────────────────────────────────
# 4. BETTING DECISION  (sigmoid curves)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def betting_decision(EHS, pot, bet_to_call, opponent_type="balanced"):
    """
    Implements the probabilistic betting strategy from the report:

        d = EHS - pot_odds          (where pot_odds = b / (b + p))

        Bet  prob = 1 / (1 + exp(-a*(d - f1)))
        Fold prob = 1 / (1 + exp( a*(d + f2)))
        Call prob = exp(-20*(d + fc)^2)

    Parameters a, f1, f2, fc vary by opponent stereotype
    (tight, loose, balanced) as described in the report.
    """
    pot_odds = bet_to_call / (bet_to_call + pot) if bet_to_call > 0 else 0.0
    d = EHS - pot_odds

    # Constants tuned per opponent archetype
    params = {
        "tight":    {"a": 8,  "f1": 0.1,  "f2": 0.15, "fc": -0.05},
        "loose":    {"a": 5,  "f1": 0.0,  "f2": 0.05, "fc":  0.0 },
        "balanced": {"a": 6,  "f1": 0.05, "f2": 0.10, "fc": -0.02},
        "bluffer":  {"a": 4,  "f1": -0.1, "f2": 0.05, "fc":  0.05},
    }
    p = params.get(opponent_type, params["balanced"])

    prob_bet  = sigmoid( p["a"] * (d - p["f1"]))
    prob_fold = sigmoid(-p["a"] * (d + p["f2"]))
    prob_call = math.exp(-20 * (d + p["fc"]) ** 2)

    # Normalise so they sum to 1
    total = prob_bet + prob_fold + prob_call
    prob_bet  /= total
    prob_fold /= total
    prob_call /= total

    # Pick the highest-probability action
    actions = {"Raise/Bet": prob_bet, "Fold": prob_fold, "Call": prob_call}
    decision = max(actions, key=actions.get)

    return {
        "d":          round(d, 4),
        "pot_odds":   round(pot_odds, 4),
        "EHS":        EHS,
        "prob_raise": round(prob_bet,  3),
        "prob_fold":  round(prob_fold, 3),
        "prob_call":  round(prob_call, 3),
        "decision":   decision,
    }


# ─────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────

def get_remaining_deck(hole, board):
    """Return all 52 cards minus hole cards and board cards."""
    used = set(hole + board)
    all_cards = []
    for rank in "23456789TJQKA":
        for suit in "shdc":
            c = Card.new(rank + suit)
            if c not in used:
                all_cards.append(c)
    return all_cards

def make_cards(card_strings):
    """Convert list of strings like ['Ah','Kd'] to treys Card ints."""
    return [Card.new(c) for c in card_strings]

def hand_rank_label(hole, board):
    """Return a human-readable hand rank (e.g. 'Flush', 'Two Pair')."""
    if len(board) < 3:
        return "—"
    score = evaluator.evaluate(board, hole)
    return evaluator.class_to_string(evaluator.get_rank_class(score))


# ─────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────

def print_separator(title=""):
    print("\n" + "=" * 55)
    if title:
        print(f"  {title}")
        print("=" * 55)

def run_demo():
    """
    Walk through three hands at different streets,
    showing HS → EHS → decision, matching the worked
    example style in the project report.
    """
    print_separator("POKER ANALYTICS — EHS DEMO")
    print("Implementing: EHS = HS*(1-NPOT) + (1-HS)*PPOT")

    # ── Example 1: Flop (same board as report: Ad Qc / 3h 4c Jh) ──
    print_separator("Example 1 — Flop (strong hand)")
    hole  = make_cards(["Ad", "Qc"])
    board = make_cards(["3h", "4c", "Jh"])

    print(f"  Hole cards : Ad Qc")
    print(f"  Board      : 3h 4c Jh  (Flop)")
    print(f"  Hand rank  : {hand_rank_label(hole, board)}")

    EHS, HS, PPOT, NPOT = effective_hand_strength(hole, board)
    print(f"\n  HS         = {HS}   (current win probability)")
    print(f"  PPOT       = {PPOT}   (chance to improve if behind)")
    print(f"  NPOT       = {NPOT}   (chance opponent catches up)")
    print(f"  EHS        = {EHS}   ← final strength estimate")

    result = betting_decision(EHS, pot=10, bet_to_call=3)
    print(f"\n  Pot=$10, Bet to call=$3")
    print(f"  Pot odds   = {result['pot_odds']}  |  d = EHS - pot_odds = {result['d']}")
    print(f"  P(Raise)={result['prob_raise']}  P(Call)={result['prob_call']}  P(Fold)={result['prob_fold']}")
    print(f"  ➜ Decision : {result['decision']}")

    # ── Example 2: Flop (weak hand, high potential) ──
    print_separator("Example 2 — Flop (weak hand, flush draw)")
    hole  = make_cards(["5h", "2h"])
    board = make_cards(["Ah", "9h", "Kc"])

    print(f"  Hole cards : 5h 2h")
    print(f"  Board      : Ah 9h Kc  (Flop — flush draw)")
    print(f"  Hand rank  : {hand_rank_label(hole, board)}")

    EHS, HS, PPOT, NPOT = effective_hand_strength(hole, board)
    print(f"\n  HS         = {HS}")
    print(f"  PPOT       = {PPOT}")
    print(f"  NPOT       = {NPOT}")
    print(f"  EHS        = {EHS}   ← boosted by flush draw potential")

    result = betting_decision(EHS, pot=8, bet_to_call=4, opponent_type="loose")
    print(f"\n  Pot=$8, Bet to call=$4  |  Opponent type: loose")
    print(f"  Pot odds   = {result['pot_odds']}  |  d = {result['d']}")
    print(f"  P(Raise)={result['prob_raise']}  P(Call)={result['prob_call']}  P(Fold)={result['prob_fold']}")
    print(f"  ➜ Decision : {result['decision']}")

    # ── Example 3: River (no more potential) ──
    print_separator("Example 3 — River (no more cards)")
    hole  = make_cards(["Kh", "Kd"])
    board = make_cards(["Ks", "7c", "2d", "9h", "Jc"])

    print(f"  Hole cards : Kh Kd")
    print(f"  Board      : Ks 7c 2d 9h Jc  (River)")
    print(f"  Hand rank  : {hand_rank_label(hole, board)}")

    EHS, HS, PPOT, NPOT = effective_hand_strength(hole, board)
    print(f"\n  HS         = {HS}")
    print(f"  PPOT       = {PPOT}   (river — no cards left)")
    print(f"  NPOT       = {NPOT}")
    print(f"  EHS        = {EHS}   ← equals HS on river")

    result = betting_decision(EHS, pot=20, bet_to_call=0, opponent_type="tight")
    print(f"\n  Pot=$20, Bet to call=$0 (we act first)  |  Opponent: tight")
    print(f"  Pot odds   = {result['pot_odds']}  |  d = {result['d']}")
    print(f"  P(Raise)={result['prob_raise']}  P(Call)={result['prob_call']}  P(Fold)={result['prob_fold']}")
    print(f"  ➜ Decision : {result['decision']}")

    # ── EHS vs opponent type table ──
    print_separator("Opponent type comparison  (Example 1 hand)")
    hole  = make_cards(["Ad", "Qc"])
    board = make_cards(["3h", "4c", "Jh"])
    EHS, *_ = effective_hand_strength(hole, board)

    print(f"  Hand: Ad Qc on 3h 4c Jh   EHS={EHS}   Pot=$10  Call=$3\n")
    print(f"  {'Opponent':12}  {'d':>6}  {'P(Raise)':>9}  {'P(Call)':>8}  {'P(Fold)':>8}  Decision")
    print(f"  {'-'*65}")
    for opp_type in ["tight", "balanced", "loose", "bluffer"]:
        r = betting_decision(EHS, pot=10, bet_to_call=3, opponent_type=opp_type)
        print(f"  {opp_type:12}  {r['d']:>6}  {r['prob_raise']:>9}  "
              f"{r['prob_call']:>8}  {r['prob_fold']:>8}  {r['decision']}")

    print_separator()
    print("  Done. All formulas from the project report implemented.")
    print("=" * 55)


if __name__ == "__main__":
    random.seed(42)
    run_demo()