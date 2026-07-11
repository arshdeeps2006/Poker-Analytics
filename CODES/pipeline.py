# Chapter 10 — System Overview: the full pipeline, wired end-to-end.

# Public Game State -> Hand Evaluator -> EHS -> SPR / Pot Odds ->
# Player Profiling -> Positional Adjustment -> Betting Decision Engine -> Output

# This module is the integration layer the report describes in Figure 10.1
# but does not implement as a single callable function. It does that here.

from ehs import effective_hand_strength
from spr import stack_to_pot_ratio
from pot_odds import pot_odds, is_call_profitable
from profiling import player_stats, archetype_to_engine_type
from position import adjust_for_position
from mdf import minimum_defense_frequency
from decision_engine import betting_decision


def run_pipeline(
    hole,
    board,
    pot,
    bet_to_call,
    stack_hero,
    stack_opp,
    position="CO",
    opponent_hands=None,
    opponent_type_override=None,
    ehs_samples=200,
    seed=None,
):
    # Runs the complete decision pipeline for one decision point.

    # Required:
    #     hole, board : lists of treys Card ints (hole cards / community cards)
    #     pot : current pot size before the bet to call
    #     bet_to_call : amount hero must call (0 if facing a check)
    #     stack_hero, stack_opp : remaining stacks, for SPR

    # Optional:
    #     position : one of UTG/MP/CO/BTN/SB/BB (default CO -> neutral 1.0x)
    #     opponent_hands : list of hand-history dicts for player_stats();
    #                              if omitted, opponent_type_override or "balanced" is used
    #     opponent_type_override : directly force "tight"/"loose"/"balanced"/"bluffer",
    #                              skipping the profiling step (useful if you don't have
    #                              hand history yet)

    # Returns a dict with every intermediate value plus the final decision,
    # so you can inspect/debug each pipeline stage.

    # 1. EHS
    EHS, HS, PPOT, NPOT = effective_hand_strength(hole, board, samples=ehs_samples, seed=seed)

    # 2. SPR
    spr_value, spr_tag = stack_to_pot_ratio(stack_hero, stack_opp, pot)

    # 3. Pot odds / call profitability (pre-position-adjustment check, informational)
    odds = pot_odds(bet_to_call, pot)
    raw_call_profitable = is_call_profitable(EHS, bet_to_call, pot)

    # 4. Player profiling -> opponent archetype -> engine opponent_type
    if opponent_type_override is not None:
        opponent_type = opponent_type_override
        profile = None
    elif opponent_hands:
        vpip, pfr, af, archetype = player_stats(opponent_hands)
        opponent_type = archetype_to_engine_type(archetype)
        profile = {"vpip": vpip, "pfr": pfr, "af": af, "archetype": archetype}
    else:
        opponent_type = "balanced"
        profile = None

    # 5. Positional adjustment
    ehs_adjusted = adjust_for_position(EHS, position)

    # 6. Betting decision engine (uses the position-adjusted EHS)
    decision = betting_decision(ehs_adjusted, pot, bet_to_call, opponent_type=opponent_type)

    # 7. MDF / bluff frequency, informational sanity-check numbers
    mdf_value, bluff_freq = minimum_defense_frequency(pot, bet_to_call) if bet_to_call > 0 else (None, None)

    return {
        "hand_strength": {"HS": HS, "PPOT": PPOT, "NPOT": NPOT, "EHS": EHS},
        "spr": {"value": spr_value, "tag": spr_tag},
        "pot_odds": {"value": round(odds, 4), "raw_call_profitable": raw_call_profitable},
        "profiling": profile,
        "opponent_type_used": opponent_type,
        "position": position,
        "ehs_adjusted": ehs_adjusted,
        "mdf": {"mdf": mdf_value, "optimal_bluff_freq": bluff_freq},
        "decision": decision,
    }


def print_report(result):
    # print a run_pipeline() result for terminal inspection
    hs = result["hand_strength"]
    print("---- Hand Strength / EHS ----")
    print(f"HS={hs['HS']}  PPOT={hs['PPOT']}  NPOT={hs['NPOT']}  EHS={hs['EHS']}")

    print("\n---- SPR ----")
    print(f"SPR={result['spr']['value']}  ({result['spr']['tag']})")

    print("\n---- Pot Odds ----")
    print(f"Pot odds required={result['pot_odds']['value']}  "
          f"raw EHS>pot_odds? {result['pot_odds']['raw_call_profitable']}")

    if result["profiling"]:
        pr = result["profiling"]
        print("\n---- Player Profiling ----")
        print(f"VPIP={pr['vpip']}%  PFR={pr['pfr']}%  AF={pr['af']}  -> {pr['archetype']}")

    print(f"\n---- Position ({result['position']}) ----")
    print(f"EHS adjusted for position: {result['ehs_adjusted']}")

    if result["mdf"]["mdf"] is not None:
        print("\n---- MDF / Bluff Frequency ----")
        print(f"MDF={result['mdf']['mdf']}  Optimal Bluff Freq={result['mdf']['optimal_bluff_freq']}")

    d = result["decision"]
    print("\n---- FINAL DECISION ----")
    print(f"opponent_type used: {result['opponent_type_used']}")
    print(f"d = EHS - pot_odds = {d['d']}")
    print(f"P(Raise)={d['prob_raise']}  P(Call)={d['prob_call']}  P(Fold)={d['prob_fold']}")
    print(f"==> {d['decision']}")
