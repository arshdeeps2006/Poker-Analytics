
# Minimum Defense Frequency and Bluff Frequency

# MDF = p / (p + b)
# Optimal Bluff Freq = b / (p + b)
# These always sum to 1.



def minimum_defense_frequency(pot, bet):
    mdf = pot / (pot + bet)
    bluff_freq = bet / (pot + bet)
    return round(mdf, 4), round(bluff_freq, 4)
