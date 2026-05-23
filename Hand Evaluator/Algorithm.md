# Perfect Hash Poker Evaluator: The Base-5 Split Method

Instead of generating a massive lookup table for all 133,784,560 possible 7-card combinations (which destroys CPU cache performance), this algorithm splits the problem into two distinct branches: **Flushes** and **Non-Flushes**. By doing so, it reduces the required memory lookup tables to just **8,192** and **49,205** entries respectively, while requiring a maximum of 13 CPU cycles to compute a perfect, collision-free hash.

## Why This Approach?

| Method | CPU Cycles | Hash Table Size | Cache Performance |
| :--- | :--- | :--- | :--- |
| **52-bit Binary Math** | Up to 52 | ~133,784,560 | Terrible |
| **Chunked 52-bit DP** | 4 | ~2,000,000 | Poor |
| **Split Base-5 Method (This Repo)** | **Max 13** | **~57,397 total** | **Excellent** |

---

## Core Algorithm Concepts

The complexity of poker evaluation comes from the interplay between card **suits** (Flushes) and card **ranks** (Pairs, Straights, etc.). If we separate these concerns, the math becomes vastly simpler.

### Case 1: The Flush Shortcut
In a 7-card hand, if you hold 5 or more cards of the same suit, the hand is guaranteed to be a Flush (or Straight Flush). It is mathematically impossible to also hold a Full House or Four-of-a-Kind.
*   **The Logic:** As we read the 7 cards, we keep a counter for each suit. If any suit hits 5, we instantly stop caring about the other cards. 
*   **The Hash:** We drop the suit information entirely and represent the remaining ranks as a 13-bit binary number (where `1` means we hold that rank).
*   **Max States:** A 13-bit binary number has a maximum value of $2^{13} = 8192$. This means our flush lookup table only needs **8,192 entries**.

### Case 2: The Non-Flush Quinary Hash
If a hand does not contain a flush, suits no longer matter. All that matters is *how many* of each rank we hold.
*   **The Logic:** We represent the hand as a 13-digit **quinary (Base-5)** number. Each digit represents a rank (2 through Ace), and its value (0, 1, 2, 3, or 4) represents how many of that card we hold. The sum of all 13 digits will always exactly equal 7.
*   **The Hash:** We need to find the exact lexicographical position of our specific Base-5 number among all valid combinations. We do this using a 3D Dynamic Programming (DP) array precomputed at startup.
*   **Max States:** There are exactly **49,205** valid non-flush combinations. 

---

## The Dynamic Programming (DP) Array

To quickly calculate the unique ID for a Base-5 hand without iterating through millions of combinations, we pre-populate a 3D array: `dp[l][n][k]`.

*   `l` **(0-4):** The most significant bit of the excluding endpoint.
*   `n` **(0-13):** The number of trailing zero bits (ranks remaining to process).
*   `k` **(0-7):** The remaining number of cards needed to reach 7.

When evaluating a hand, we loop through the 13 ranks. If we hold cards in that rank, we add the pre-calculated value from the DP array to our hash sum. This takes a maximum of 13 CPU cycles.

---

## C++ Implementation

```cpp
// *****THIS CODE IS STILL UNDER REVIEW AND HAS NOT BEEN FINALIZED******//

#include <bits/stdc++.h>

using namespace std;

class PokerEvaluator {
private:
    // dp[l][n][k] array sizing: 5 * 14 * 8
    int dp[5][14][8] = {0};

    // Precomputes the Perfect Hash offsets
    void init_dp() {
        for (int i = 0; i <= 4; i++) dp[1][1][i] = 1;
        for (int i = 5; i <= 7; i++) dp[1][1][i] = 0;

        for (int i = 2; i <= 13; i++) {
            dp[1][i][1] = i;
            dp[1][i][0] = 1;
        }

        for (int i = 2; i <= 13; i++) {
            for (int j = 2; j <= 7; j++) {
                int sum = 0;
                for (int k = 0; k <= 4; k++) {
                    if (j - k >= 0) sum += dp[1][i - 1][j - k];
                }
                dp[1][i][j] = sum;
            }
        }

        for (int l = 2; l <= 4; l++) {
            for (int i = 1; i <= 13; i++) {
                for (int j = 0; j <= 7; j++) {
                    int val = j - l + 1;
                    dp[l][i][j] = dp[l - 1][i][j] + (val >= 0 ? dp[1][i][val] : 0);
                }
            }
        }
    }

    // Computes the perfect hash ID for a non-flush hand
    int hash_quinary(unsigned char q[], int len, int k) {
        int sum = 0;
        for (int i = 0; i < len; i++) {
            if (q[i] > 0) {
                sum += dp[q[i]][len - i - 1][k];
                k -= q[i];
                if (k <= 0) break;
            }
        }
        return sum; 
    }

public:
    struct EvalResult {
        bool is_flush;
        int hash_key; 
    };

    PokerEvaluator() {
        init_dp(); // Run exactly once on startup
    }

    // Expects exactly 7 cards. 
    // Format: Rank = card % 13, Suit = card / 13
    EvalResult evaluate(const vector<int>& cards) {
        int suit_count[4] = {0};
        int suit_mask[4] = {0};
        unsigned char quinary_ranks[13] = {0};

        // 1. Parse the cards into logical buckets
        for (int card : cards) {
            int rank = card % 13;
            int suit = card / 13;

            suit_count[suit]++;
            suit_mask[suit] |= (1 << rank);
            quinary_ranks[rank]++;
        }

        // 2. Evaluate Flushes First
        for (int s = 0; s < 4; s++) {
            if (suit_count[s] >= 5) {
                return {true, suit_mask[s]}; // Returns a 13-bit mask (0 - 8191)
            }
        }

        // 3. Evaluate Non-Flushes via Dynamic Programming array
        int hash_key = hash_quinary(quinary_ranks, 13, 7);
        return {false, hash_key}; // Returns Quinary Hash (0 - 49204)
    }
};
```

---

## How to Complete the Evaluator

This code generates **Perfect Hashes** (unique IDs). To build a fully functional engine that tells you who won a hand, you must generate two lookup tables containing actual poker hand values (e.g., assigning an integer score where a Royal Flush > Full House > Two Pair).

1.  **Flush Lookup:** 
    Create `int flush_table[8192];`. Iterate through all 13-bit binary combinations. If the combination contains 5 or more `1`s, evaluate its poker rank and store it at that index.
2.  **Non-Flush Lookup:** 
    Create `int non_flush_table[49205];`. Iterate through all valid Base-5 quinary combinations that sum to 7. Evaluate the standard poker rank for each combination and store it at its generated `hash_key` index.

**Final usage in your game loop:**
```cpp
EvalResult res = evaluator.evaluate(player_hand);
int hand_strength = res.is_flush ? flush_table[res.hash_key] : non_flush_table[res.hash_key];
```
