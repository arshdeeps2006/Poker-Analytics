# Hand Strength and Effective Hand Strength (EHS)

# HS   : probability our hand currently beats a random opponent hand without future cards
# PPOT : probability a currently-behind hand ends up ahead by the river
# NPOT : probability a currently-ahead hand ends up behind by the river
# EHS  : forward-looking hand value combining HS, PPOT, NPOT

import random
from itertools import combinations
from treys import Card, Evaluator

evaluator = Evaluator()

RANKS = "23456789TJQKA"
SUITS = "shdc"


def get_remaining_deck(hole, board):
    """All 52 cards minus the ones already in play (hole + board)."""
    used = set(hole + board)
    return [Card.new(r + s) for r in RANKS for s in SUITS if Card.new(r + s) not in used]


def hand_strength(hole, board):
    """
    HS = (W + T/2) / (W + T + L)
    Enumerates every possible 2-card opponent hand from the remaining deck.
    """
    deck = get_remaining_deck(hole, board)
    W, T, L = 0, 0, 0
    hero_score = evaluator.evaluate(board, hole)  # constant across the loop
    for opp in combinations(deck, 2):
        opp_score = evaluator.evaluate(board, list(opp))
        if hero_score < opp_score:      # lower score = stronger hand in treys
            W += 1
        elif hero_score == opp_score:
            T += 1
        else:
            L += 1
    total = W + T + L
    return (W + T / 2) / total if total else 0.0


def hand_potential(hole, board, samples=200, seed=None):

    # Monte Carlo estimate of PPOT and NPOT.
    # Returns (PPOT, NPOT), each rounded to 4 decimals.
    # On the river there are no future cards, so both are 0.0.

    if seed is not None:
        random.seed(seed)

    deck = get_remaining_deck(hole, board)
    cards_needed = 5 - len(board)
    if cards_needed <= 0:
        return 0.0, 0.0  # river: nothing left to draw

    ahead_total = behind_total = 0
    ahead_caught = behind_improve = 0

    for _ in range(samples):
        available = deck[:]
        random.shuffle(available)
        opp, rest = available[:2], available[2:]

        hero_now = evaluator.evaluate(board, hole)
        opp_now = evaluator.evaluate(board, opp)
        ahead_now = hero_now < opp_now
        behind_now = hero_now > opp_now

        full_board = board + rest[:cards_needed]
        hero_end = evaluator.evaluate(full_board, hole)
        opp_end = evaluator.evaluate(full_board, opp)

        if ahead_now:
            ahead_total += 1
            if opp_end < hero_end:
                ahead_caught += 1
        if behind_now:
            behind_total += 1
            if hero_end < opp_end:
                behind_improve += 1

    PPOT = behind_improve / behind_total if behind_total else 0.0
    NPOT = ahead_caught / ahead_total if ahead_total else 0.0
    return round(PPOT, 4), round(NPOT, 4)


def effective_hand_strength(hole, board, samples=200, seed=None):
    
    # EHS = HS * (1 - NPOT) + (1 - HS) * PPOT
    # Returns (EHS, HS, PPOT, NPOT).
    
    HS = hand_strength(hole, board)
    PPOT, NPOT = hand_potential(hole, board, samples, seed=seed)
    EHS = HS * (1 - NPOT) + (1 - HS) * PPOT
    return round(EHS, 4), round(HS, 4), PPOT, NPOT
