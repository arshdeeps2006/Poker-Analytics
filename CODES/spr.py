# Stack-to-Pot Ratio (SPR)
# SPR = min(stack_hero, stack_opp) / pot


def stack_to_pot_ratio(stack_hero, stack_opp, pot):
    # Returns (spr, tag) where tag is the strategic interpretation.
    spr = min(stack_hero, stack_opp) / pot
    if spr < 3:
        tag = "Low - commit with strong hands"
    elif spr <= 13:
        tag = "Medium - balance aggression and caution"
    else:
        tag = "High - drawing hands gain value"
    return round(spr, 2), tag
