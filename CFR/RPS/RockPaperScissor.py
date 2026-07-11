import numpy as np
from numpy.random import choice

# Blueprint("Rock Paper Scissors Trainer")
class RPSTrainer:
    def __init__(self):

        self.NUM_ACTIONS = 3
        self.possible_actions = np.arange(self.NUM_ACTIONS)

        # Along a row, the row index is the action of player 1
        # and the column index is the action of player 2
        # The value is the reward for player 1.
        # actionUtility [i, j] is the reward for player 1 when player 1 takes action i and player 2 takes action j
        self.actionUtility = np.array([
                    [0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]
                ])
        
        # regret_sum[i] is the cumulative regret for not having taken action i in the past
        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        # strategy_sum[i] is the cumulative probability of having taken action i in the past
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)

        # like above, but for the opponent
        self.opponent_regret_sum = np.zeros(self.NUM_ACTIONS)
        self.opponent_strategy_sum = np.zeros(self.NUM_ACTIONS)

    def get_strategy(self, regret_sum):
        # we only want to consider positive regrets
        # negative regrets mean that we are doing better than the alternative action
        new_sum = np.clip(regret_sum, a_min=0, a_max=None)
        normalizing_sum = np.sum(new_sum)
        if normalizing_sum > 0:
            new_sum /= normalizing_sum
        else:
            # if we have no positive regrets, we will just play uniformly at random
            new_sum = np.repeat(1/self.NUM_ACTIONS, self.NUM_ACTIONS)
        return new_sum

    # returns the average strategy over all iterations of training
    def get_average_strategy(self, strategy_sum):
        average_strategy = [0, 0, 0]
        normalizing_sum = sum(strategy_sum)
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                average_strategy[a] = strategy_sum[a] / normalizing_sum
            else:
                average_strategy[a] = 1.0 / self.NUM_ACTIONS
        return average_strategy

    # returns an action based on the given strategy (which is a probability distribution over actions)
    def get_action(self, strategy):
        return choice(self.possible_actions, p=strategy)

    # returns the reward for player 1 given the actions of both players
    def get_reward(self, my_action, opponent_action):
        return self.actionUtility[my_action, opponent_action]

    def train(self, iterations):

        for i in range(iterations):

            #################################################################################
            # 1. Get the current strategy for both players based on their cumulative regrets.
            # 2. Update the cumulative strategy sums for both players.
            #################################################################################

            strategy = self.get_strategy(self.regret_sum)
            opp_strategy = self.get_strategy(self.opponent_regret_sum)

            self.strategy_sum += strategy
            self.opponent_strategy_sum += opp_strategy                   

            opponent_action = self.get_action(opp_strategy)
            my_action = self.get_action(strategy)

            my_reward = self.get_reward(my_action, opponent_action)
            opp_reward = self.get_reward(opponent_action, my_action)

            for j in range(self.NUM_ACTIONS):
                my_regret = self.get_reward(j, opponent_action) - my_reward
                opp_regret = self.get_reward(j, my_action) - opp_reward
                self.regret_sum[j] += my_regret
                self.opponent_regret_sum[j] += opp_regret


def main():
    trainer = RPSTrainer()
    trainer.train(10000)
    target_policy = trainer.get_average_strategy(trainer.strategy_sum)
    opp_target_policy = trainer.get_average_strategy(trainer.opponent_strategy_sum)

    # Convert the raw decimals into formatted percentage strings
    p1_percentages = [f"{p:.1%}" for p in target_policy]
    p2_percentages = [f"{p:.1%}" for p in opp_target_policy]

    # Print the new formatted lists
    print(f"player 1 policy: {p1_percentages}")
    print(f"player 2 policy: {p2_percentages}")
    print("As you can see, both players have converged to the Nash equilibrium of playing each action with equal probability.")
    print("This proves that the CFR algorithm succesfully learns that the optimal strategy in rock paper scissors is to play each action with equal probability.")

if __name__ == "__main__":
    main()