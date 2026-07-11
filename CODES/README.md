# Poker Analytics — Decision Engine (Implementation)

This package implements the full pipeline described in `Poker_Analytics_Report.pdf`,
Chapters 2 through 10. Every formula is taken directly from the report; the only
addition is `pipeline.py`, which wires all the separate modules into one callable
end-to-end function (the report specifies this pipeline conceptually in Figure 10.1
but never assembles it as runnable code).

## Files

| File | Report Chapter | What it does |
|---|---|---|
| `ehs.py` | 2 | Hand Strength, PPOT/NPOT (Monte Carlo), Effective Hand Strength |
| `spr.py` | 3 | Stack-to-Pot Ratio |
| `pot_odds.py` | 4 | Pot odds, implied odds, call-profitability check |
| `profiling.py` | 5 | VPIP / PFR / AF from hand history, archetype classification |
| `position.py` | 6 | Positional EHS multiplier |
| `mdf.py` | 8 | Minimum Defense Frequency, optimal bluff frequency |
| `decision_engine.py` | 9 | Sigmoid-based Fold/Call/Raise probability engine |
| `pipeline.py` | 10 | **Glue code**: runs all of the above as one pipeline |
| `tests/` | — | Scripts that validate each module against the report's own worked examples |

## Quick start

```python
import sys
sys.path.insert(0, "/path/to/poker_analytics")

from treys import Card
from pipeline import run_pipeline, print_report

hole = [Card.new("5h"), Card.new("2h")]
board = [Card.new("Ah"), Card.new("9h"), Card.new("Kc")]

result = run_pipeline(
    hole=hole, board=board,
    pot=10, bet_to_call=3,
    stack_hero=400, stack_opp=500,
    position="CO",
    opponent_type_override="balanced",  # or pass opponent_hands=[...] instead
)
print_report(result)
```

Cards use `treys` notation: rank (`2`-`9`,`T`,`J`,`Q`,`K`,`A`) + suit (`s`,`h`,`d`,`c`).
E.g. `Card.new("Ah")` = Ace of hearts.

## How to plug in real opponent data

`run_pipeline()` needs a `opponent_hands` list to do real profiling. Each entry is:

```python
{'entered': True, 'raised_pf': True, 'bets_raises': 2, 'calls': 0}
```

one dict per historical hand you've observed that opponent play. If you don't have
hand-history data yet, use `opponent_type_override="balanced"` (or `"tight"` /
`"loose"` / `"bluffer"`) to skip profiling and force an assumption.

## Validation performed

All of the following were checked against the report's own tables (see `tests/`):

- ✅ `hand_strength()` matches Table 2.1's exact HS (0.585 computed vs 0.59 in report)
- ✅ `hand_potential()` matches Table 2.1's PPOT/NPOT within Monte Carlo sampling noise
- ✅ `pot_odds()` matches Table 4.1 exactly (all 4 rows)
- ✅ `minimum_defense_frequency()` matches Table 8.1 exactly (all 4 rows)
- ✅ `stack_to_pot_ratio()` buckets match Table 3.1's SPR ranges
- ✅ `adjust_for_position()` matches Table 6.1's multipliers exactly
- ✅ `betting_decision()` reproduces the correct **decision** (Raise/Call/Raise) for
  all three rows of Table 9.1
- ✅ Full pipeline runs end-to-end and correctly changes its output when the
  opponent archetype changes, holding the hand/pot fixed (the core claim of Ch. 7)

## ⚠️ Known gaps / things to be aware of

1. **Table 9.2 discrepancy (pre-existing in the report, not in this code).**
   Re-running the report's own Listing 9.1 formula with its own constants at
   `d = 0.282` does **not** reproduce the exact percentages printed in Table 9.2
   (e.g. "balanced" gives P(Raise)=0.699 here vs 0.621 in the report). The
   *ranking* and *decision* are consistent, but the precise numbers in that table
   don't match the report's own stated formula. This isn't something this code
   got wrong — it's worth double-checking if the report's Table 9.2 was generated
   with a slightly different code version, before citing those exact figures.

2. **Archetype → opponent_type mapping was invented, not specified.**
   Chapter 5 classifies opponents as TAG/LAG/TP/LP. Chapter 9's engine expects
   `"tight"/"loose"/"balanced"/"bluffer"`. The report never bridges these two
   vocabularies. `profiling.archetype_to_engine_type()` makes a reasonable mapping
   (LAG→bluffer, TAG/TP→tight, LP→loose) but this is a judgment call you may want
   to revisit — e.g. you might prefer TAG opponents get their own distinct sigmoid
   constants instead of being folded into "tight".

3. **SPR isn't actually consumed by the decision engine.**
   The report computes SPR and explains its strategic meaning (Table 3.1), but
   `betting_decision()` doesn't take SPR as a parameter anywhere — it only uses
   EHS and pot odds. The pipeline here computes and reports SPR for visibility,
   but it doesn't change the sigmoid output. If you want low-SPR situations to
   actually push the model toward more committal decisions, that's an extension
   you'd need to design (e.g. another multiplier on `d`, similar to position).

4. **Implied odds aren't used anywhere in the engine.** Chapter 4 mentions implied
   odds conceptually; `implied_pot_odds()` exists as a function but `run_pipeline()`
   always uses raw pot odds, since estimating "expected future winnings" requires
   assumptions the report doesn't specify.

5. **Sigmoid constants (`a, f1, f2, fc` per archetype) are the report's hand-picked
   values, not fitted/learned from data.** They produce sensible-looking curves
   but aren't derived from any optimization — worth flagging if asked how they
   were chosen.

6. **EHS Monte Carlo sampling means results vary slightly run to run** unless you
   pass a fixed `seed`. Increase `ehs_samples` for more stable PPOT/NPOT estimates
   at the cost of speed (200 samples is fast but noisy; 2000+ is more stable).
