# Player Profiling: VPIP, PFR, and Aggression Factor (AF)
# hands: list of dicts, each shaped like:
    # {'entered': bool, 'raised_pf': bool, 'bets_raises': int, 'calls': int}



def classify_player(vpip, pfr):
    loose = vpip > 25
    aggressive = pfr > 15
    if loose and aggressive:
        return "Loose-Aggressive (LAG)"
    if loose and not aggressive:
        return "Loose-Passive (LP)"
    if not loose and aggressive:
        return "Tight-Aggressive (TAG)"
    return "Tight-Passive (TP)"


def player_stats(hands):
    # Returns (vpip, pfr, af, archetype).
    n = len(hands)
    if n == 0:
        raise ValueError("hands list is empty — need at least one observed hand")

    vpip = 100 * sum(h['entered'] for h in hands) / n
    pfr = 100 * sum(h['raised_pf'] for h in hands) / n
    total_br = sum(h['bets_raises'] for h in hands)
    total_call = sum(h['calls'] for h in hands)
    af = total_br / total_call if total_call else float('inf')

    return round(vpip, 1), round(pfr, 1), round(af, 2), classify_player(vpip, pfr)


# Map an archetype string straight to the opponent_type key used by the
# betting decision engine (Chapter 9), so profiling plugs directly into it.
ARCHETYPE_TO_ENGINE_TYPE = {
    "Tight-Aggressive (TAG)": "tight",
    "Loose-Aggressive (LAG)": "bluffer",
    "Tight-Passive (TP)": "tight",
    "Loose-Passive (LP)": "loose",
}


def archetype_to_engine_type(archetype):
    return ARCHETYPE_TO_ENGINE_TYPE.get(archetype, "balanced")
