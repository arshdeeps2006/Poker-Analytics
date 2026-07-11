# Positional Advantage
# EHS_adjusted = EHS * lambda(position)

POSITION_MULTIPLIER = {
    "UTG": 0.85, "MP": 0.92, "CO": 1.00,
    "BTN": 1.15, "SB": 0.80, "BB": 0.95,
}


def adjust_for_position(ehs, position):
    return round(ehs * POSITION_MULTIPLIER.get(position, 1.0), 4)
