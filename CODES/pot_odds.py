# Chapter 4 — Pot Odds and Implied Odds
# Pot Odds = b / (b + p)
# Call is profitable iff EHS > Pot Odds  (derived from EV = EHS*p - (1-EHS)*b > 0)

def pot_odds(bet_to_call, pot):
    return bet_to_call / (bet_to_call + pot) if bet_to_call > 0 else 0.0


def is_call_profitable(EHS, bet_to_call, pot):
    return EHS > pot_odds(bet_to_call, pot)


def implied_pot_odds(bet_to_call, pot, expected_future_winnings=0.0):
    
    # Implied odds: extend the effective pot by expected money won on later
    # streets if a drawing hand completes.

    effective_pot = pot + expected_future_winnings
    return bet_to_call / (bet_to_call + effective_pot) if bet_to_call > 0 else 0.0
