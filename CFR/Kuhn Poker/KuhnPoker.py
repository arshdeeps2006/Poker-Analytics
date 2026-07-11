########################################################## Monte Carlo CFR #############################################################

import numpy as np
from random import shuffle
import time
import sys

# Blueprint("Kuhn Poker Trainer - MCCFR External Sampling")
class Kuhn:
    def __init__(self):
        self.nodeMap = {}
        self.n_cards = 3
        self.deck = np.array([0, 1, 2])
        self.n_actions = 2 # 0: pass, 1: bet

    def train(self, n_iterations=1000000):
        expected_game_value = 0
        
        for i in range(n_iterations):
            # We update one player at a time in External Sampling
            for update_player in [0, 1]: 
                shuffle(self.deck)
                util = self.mccfr_external('', update_player)
                
                # Keep track of EV from Player 1's perspective (when update_player is 0)
                if update_player == 0:
                    expected_game_value += util

            # After both players have been updated, we recalculate their strategies 
            # for the next iteration using Regret Matching.
            for i, v in self.nodeMap.items():
                v.strategy = v.get_strategy()

        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def mccfr_external(self, history, update_player): 
        n = len(history)
        current_player = n % 2
        is_player_1 = current_player == 0
        
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        opponent_card = self.deck[1] if is_player_1 else self.deck[0]

        # 1. Terminal State (Game Over)
        if self.is_terminal(history):
            return self.get_reward(history, player_card, opponent_card)

        # 2. Get Information Set
        node = self.get_node(player_card, history)
        strategy = node.strategy

        # 3. External Sampling Logic
        if current_player == update_player:
            # TRAVERSER NODE: Explore ALL actions to calculate exact regrets
            action_utils = np.zeros(self.n_actions)

            for act in range(self.n_actions):
                next_history = history + node.action_dict[act]
                action_utils[act] = -1 * self.mccfr_external(next_history, update_player)

            # Expected utility of this node
            util = sum(action_utils * strategy)
            
            # Regret = Utility of Action - Average Utility of Node
            regrets = action_utils - util
            node.regret_sum += regrets
            
            # We accumulate the strategy sum when we update the traverser
            node.strategy_sum += strategy
            
            return util
            
        else:
            # OPPONENT NODE: Sample exactly ONE action based on their strategy
            sampled_act = np.random.choice(self.n_actions, p=strategy)
            next_history = history + node.action_dict[sampled_act]
            
            # Only explore the branch of the sampled action
            return -1 * self.mccfr_external(next_history, update_player)

    @staticmethod
    def is_terminal(history):
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True
        return False

    @staticmethod
    def get_reward(history, player_card, opponent_card):
        terminal_pass = history[-1] == 'p' 
        double_bet = history[-2:] == "bb"  
        if terminal_pass:
            if history[-2:] == 'pp':
                return 1 if player_card > opponent_card else -1
            else:
                return 1
        elif double_bet:
            return 2 if player_card > opponent_card else -2

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            action_dict = {0: 'p', 1: 'b'}
            info_set = Node(key, action_dict)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]

class Node:
    def __init__(self, key, action_dict, n_actions=2):
        self.key = key
        self.n_actions = n_actions
        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)
        self.action_dict = action_dict
        self.strategy = np.repeat(1/self.n_actions, self.n_actions)

    def get_strategy(self):
        regrets = self.regret_sum.copy()
        regrets[regrets < 0] = 0 # Ignore negative regrets
        normalizing_sum = sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            # Default back to uniform random if all regrets are <= 0
            return np.repeat(1/self.n_actions, self.n_actions)

    def get_average_strategy(self):
        strategy = self.strategy_sum.copy()
        total = sum(strategy)
        if total > 0:
            strategy /= total
        else:
            strategy = np.repeat(1/self.n_actions, self.n_actions)
        return strategy

    def __str__(self):
        strategies = ['{:03.2f}'.format(x) for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)

def display_results(ev, i_map):
    print("--- EXPECTED VALUES (EV) ---")
    print("What this means: The average payoff per hand if both players play optimally.")
    print("In Kuhn Poker, Player 1 acts first and is at a slight disadvantage.")
    print("The theoretical perfect EV for Player 1 is -1/18 (approx. -0.0555).")

    print(f" Player 1 Expected Value: {ev:.5f}")
    print(f" Player 2 Expected Value: {-1 * ev:.5f}\n")
    print("----------------------------------------------------------------\n")

    print("--- OPTIMAL STRATEGIES ---")
    print("What this means: The percentage of time a player should Pass or Bet")
    print("based on the card they are holding and the current game history.\n")
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    print("Format: [Pass %,  Bet %]")
    print("PLAYER 1 STRATEGIES:")
    for i, v in filter(lambda x: len(x[0].split()[1] if len(x[0].split()) > 1 else "") % 2 == 0, sorted_items):
        print(f"  {v}")
        
    print("\nPLAYER 2 STRATEGIES:")
    for i, v in filter(lambda x: len(x[0].split()[1] if len(x[0].split()) > 1 else "") % 2 == 1, sorted_items):
        print(f"  {v}")
        
    print("\n================================================================")

# BOT MATCHES TO TEST THE TRAINED ALGORITHM
def simulate_bot_match(trainer, n_hands=10000, bot_type='random'):
    ev_sum = 0
    
    for rounds in range(n_hands):
        shuffle(trainer.deck)
        history = ''
        
        while not trainer.is_terminal(history):
            current_player = len(history) % 2
            is_player_1 = current_player == 0
            card = trainer.deck[0] if is_player_1 else trainer.deck[1]
            
            if is_player_1:
                # CFR Agent's turn (Player 1)
                node = trainer.get_node(card, history)
                strategy = node.get_average_strategy()
                action = np.random.choice([0, 1], p=strategy)
            else:
                # Bot's turn (Player 2)
                if bot_type == 'random':
                    action = np.random.choice([0, 1])
                elif bot_type == 'call':
                    action = 1  # Always bets/calls
                    
            history += trainer.nodeMap[list(trainer.nodeMap.keys())[0]].action_dict[action]
            
        # Reward is always from Player 1's perspective
        reward = trainer.get_reward(history, trainer.deck[0], trainer.deck[1])
        ev_sum += reward
        
    win_rate = ev_sum / n_hands
    print(f"CFR vs {bot_type.capitalize()} Bot ({n_hands} hands):")
    print(f"  Win Rate (Expected Value per hand): {win_rate:+.4f} chips")
    return win_rate  

# Testing the exploitability of the CFR trained model via "Empirical Best Response"
def compute_exploitability(trained_agent, n_iterations=10000):
    nemesis = Kuhn()
    exploited_ev = 0
    
    # Nemesis trains exclusively as Player 2 to exploit our trained Player 1
    for i in range(n_iterations):
        shuffle(nemesis.deck)
        history = ''
        util = mccfr_nemesis(nemesis, trained_agent, history, update_player=1)
        exploited_ev += util
        
        # Update Nemesis strategy
        for _, v in nemesis.nodeMap.items():
            v.strategy = v.get_strategy()
            
    # Calculate difference between theoretical EV (-0.055) and exploited EV
    final_ev = exploited_ev / n_iterations
    exploitability_metric = abs(-0.055 - final_ev)
    
    print("\n--- EXPLOITABILITY METRICS ---")
    print(f"  Nemesis Max Exploitation EV: {final_ev:+.4f}")
    print(f"  Exact Exploitability (Deviation from Nash): {exploitability_metric:.4f}")
    return exploitability_metric

def mccfr_nemesis(nemesis, trained_agent, history, update_player):
    current_player = len(history) % 2
    is_player_1 = current_player == 0
    
    player_card = nemesis.deck[0] if is_player_1 else nemesis.deck[1]
    opponent_card = nemesis.deck[1] if is_player_1 else nemesis.deck[0]

    if nemesis.is_terminal(history):
        return nemesis.get_reward(history, player_card, opponent_card)

    nemesis_node = nemesis.get_node(player_card, history)
    
    # If it's Player 1 (Our locked CFR agent), strictly sample from its trained strategy
    if current_player == 0:
        trained_node = trained_agent.get_node(player_card, history)
        trained_strategy = trained_node.get_average_strategy()
        sampled_act = np.random.choice(nemesis.n_actions, p=trained_strategy)
        next_history = history + nemesis_node.action_dict[sampled_act]
        return -1 * mccfr_nemesis(nemesis, trained_agent, next_history, update_player)
        
    # If it's Player 2 (The Nemesis), explore all actions to find the Best Response
    else:
        action_utils = np.zeros(nemesis.n_actions)
        strategy = nemesis_node.strategy
        
        for act in range(nemesis.n_actions):
            next_history = history + nemesis_node.action_dict[act]
            action_utils[act] = -1 * mccfr_nemesis(nemesis, trained_agent, next_history, update_player)

        util = sum(action_utils * strategy)
        regrets = action_utils - util
        nemesis_node.regret_sum += regrets
        nemesis_node.strategy_sum += strategy
        return util
    

if __name__ == "__main__":
    time1 = time.time()
    trainer = Kuhn()

    print("Training MCCFR (External Sampling) for Kuhn Poker... Please wait.\n")
    trainer.train(n_iterations=50000)

    print("\n--- SIMULATION AGAINST HEURISTIC BOTS ---")
    simulate_bot_match(trainer, n_hands=10000, bot_type='random')
    simulate_bot_match(trainer, n_hands=10000, bot_type='call')

    compute_exploitability(trainer, n_iterations=10000)
    
    print(f"\nExecution Time: {abs(time1 - time.time()):.2f} seconds")
    print(f"Memory Size of Trainer Object: {sys.getsizeof(trainer)} bytes")













################################################### Chanced Sampling CFR ###########################################################


# import numpy as np
# from random import shuffle
# import time
# import sys

# #################################################################################################################
# # Mentally, in one complete iteration:
# # shuffle cards

# # start root CFR

# #     visit node
# #     read current strategy
# #     recurse into pass subtree
# #     recurse into bet subtree
# #     get utilities
# #     compute EV
# #     compute regret
# #     update regret table
#     # return EV
# #################################################################################################################

# # Blueprint("Kuhn Poker Trainer")
# class Kuhn:
#     def __init__(self):
#         self.nodeMap = {} # For storing the information sets
#         self.expected_game_value = 0
#         self.n_cards = 3
#         self.nash_equilibrium = dict()
#         self.current_player = 0
#         self.deck = np.array([0, 1, 2])
#         self.n_actions = 2 # 0: pass, 1: bet

#     def train(self, n_iterations=50000):
#         expected_game_value = 0
#         for i in range(n_iterations):
#             shuffle(self.deck)
#             expected_game_value += self.cfr('', 1, 1) # Starts the CFR algorithm from the root of the game tree (empty history '')
#             for i, v in self.nodeMap.items():
#                 v.update_strategy()

#         expected_game_value /= n_iterations
#         display_results(expected_game_value, self.nodeMap)

#     def cfr(self, history, pr_1, pr_2): 
#         n = len(history)
#         is_player_1 = n % 2 == 0
#         player_card = self.deck[0] if is_player_1 else self.deck[1]

#         if self.is_terminal(history):
#             card_player = self.deck[0] if is_player_1 else self.deck[1]
#             card_opponent = self.deck[1] if is_player_1 else self.deck[0]
#             reward = self.get_reward(history, card_player, card_opponent)
#             return reward

#         # define the information set/state of the player
#         node = self.get_node(player_card, history)
#         strategy = node.strategy

#         # Counterfactual utility per action.
#         action_utils = np.zeros(self.n_actions)

#         for act in range(self.n_actions):
#             next_history = history + node.action_dict[act]
#             if is_player_1:
#                 action_utils[act] = -1 * self.cfr(next_history, pr_1 * strategy[act], pr_2)
#             else:
#                 action_utils[act] = -1 * self.cfr(next_history, pr_1, pr_2 * strategy[act])

#         # Utility of information set.
#         util = sum(action_utils * strategy)
#         regrets = action_utils - util
#         if is_player_1:
#             node.reach_pr += pr_1
#             node.regret_sum += pr_2 * regrets
#         else:
#             node.reach_pr += pr_2
#             node.regret_sum += pr_1 * regrets

#         return util

#     @staticmethod
#     def is_terminal(history):
#         # A terminal state is reached if:
#         # 1. The last two actions were both passes (pp)
#         # 2. The last two actions were both bets (bb)
#         # 3. The last two actions were a bet followed by a pass (bp)
#         if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
#             return True
#         # In all other cases, the game is still ongoing.
#         return False

#     @staticmethod
#     def get_reward(history, player_card, opponent_card):
#         terminal_pass = history[-1] == 'p' # true if the last action of opponent was a pass
#         double_bet = history[-2:] == "bb"  # true if the last two actions were bets
#         if terminal_pass:
#             if history[-2:] == 'pp':
#                 return 1 if player_card > opponent_card else -1
#             else:
#                 return 1
#         elif double_bet:
#             return 2 if player_card > opponent_card else -2

#     def get_node(self, card, history):
#         key = str(card) + " " + history
#         if key not in self.nodeMap:
#             action_dict = {0: 'p', 1: 'b'}
#             info_set = Node(key, action_dict)
#             self.nodeMap[key] = info_set
#             return info_set
#         return self.nodeMap[key]

# class Node:
#     def __init__(self, key, action_dict, n_actions=2):
#         self.key = key
#         self.n_actions = n_actions
#         self.regret_sum = np.zeros(self.n_actions)
#         self.strategy_sum = np.zeros(self.n_actions)
#         self.action_dict = action_dict
#         self.strategy = np.repeat(1/self.n_actions, self.n_actions)
#         self.reach_pr = 0
#         self.reach_pr_sum = 0

#     def update_strategy(self):
#         self.strategy_sum += self.reach_pr * self.strategy
#         self.reach_pr_sum += self.reach_pr
#         self.strategy = self.get_strategy()
#         self.reach_pr = 0

#     def get_strategy(self):
#         regrets = self.regret_sum
#         regrets[regrets < 0] = 0
#         normalizing_sum = sum(regrets)
#         if normalizing_sum > 0:
#             return regrets / normalizing_sum
#         else:
#             return np.repeat(1/self.n_actions, self.n_actions)

#     def get_average_strategy(self):
#         strategy = self.strategy_sum / self.reach_pr_sum
#         # Re-normalize
#         total = sum(strategy)
#         strategy /= total
#         return strategy

#     def __str__(self):
#         strategies = ['{:03.2f}'.format(x)
#                       for x in self.get_average_strategy()]
#         return '{} {}'.format(self.key.ljust(6), strategies)


# def display_results(ev, i_map):
#     print("--- EXPECTED VALUES (EV) ---")
#     print("What this means: The average payoff per hand if both players play optimally.")
#     print("In Kuhn Poker, Player 1 acts first and is at a slight disadvantage.")
#     print("The theoretical perfect EV for Player 1 is -1/18 (approx. -0.0555).")
#     print(f"\n Player 1 Expected Value: {ev:.5f}")
#     print(f" Player 2 Expected Value: {-1 * ev:.5f}\n")
#     print("----------------------------------------------------------------\n")

#     print("--- OPTIMAL STRATEGIES ---")
#     print("What this means: The percentage of time a player should Pass or Bet")
#     print("based on the card they are holding and the current game history.\n")
    
#     sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    
#     print("PLAYER 1 STRATEGIES (Acts on even-length histories):")
#     for i, v in filter(lambda x: len(x[0].split()[1] if len(x[0].split()) > 1 else "") % 2 == 0, sorted_items):
#         print(f"  {v}")
        
#     print("\nPLAYER 2 STRATEGIES (Acts on odd-length histories):")
#     for i, v in filter(lambda x: len(x[0].split()[1] if len(x[0].split()) > 1 else "") % 2 == 1, sorted_items):
#         print(f"  {v}")
        
#     print("\n================================================================")


# if __name__ == "__main__":
#     time1 = time.time()
#     trainer = Kuhn()

#     print("Training CFR Algorithm... Please wait.\n")
#     trainer.train(n_iterations=25000)
    
#     print(f"\nExecution Time: {abs(time1 - time.time()):.2f} seconds")
#     print(f"Memory Size of Trainer Object: {sys.getsizeof(trainer)} bytes")














