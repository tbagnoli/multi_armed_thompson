# -----------------------------------------------------------
# Class for implementing a Thompson sampling (TS) strategy
#
# Developer: Tullio Bagnoli
# Email: tullio.bagnoli@protonmail.com
# -----------------------------------------------------------

import numpy as np


class Thompson:

    def __init__(self,
                 success_probs: [float],
                 steps: int = None) -> None:
        """
        :param success_probs: an array of floats in the [0, 1] interval, the success P of each bandit
        :param steps: the iteration steps (optional, defaults to 1000)
        """

        # test validity of input probabilities
        try:
            self.n_bandits = len(success_probs)
        except TypeError:
            print("Success probabilities must be passed in array")
        self.success_probs = np.array(success_probs)
        self.max_prob = self.success_probs.max()
        try:
            assert np.all(self.success_probs >= 0)
            assert np.all(self.success_probs <= 1)
        except AssertionError:
            print("Elements of array of success probabilities must belong to [0, 1] interval.")

        # test validity of input steps
        if steps is None:
            self.steps = 1000
        else:
            try:
                assert ((steps == int(steps)) & (steps > 0))
            except AssertionError:
                print("Number of steps must be positive integer")
            self.steps = steps

        # variables storing the progression of rewards and penalties of each bandit
        self.rewards = None
        self.penalties = None
        # variables storing the global progression of the experiment
        self.choices = None
        self.cumsum_rewards = None
        self.cumsum_penalties = None
        self.total_rewards = None
        self.regret = None
        # variables to tweak the behaviour of the algorithm:
        # damping exploration/exploitation
        self.damp_alpha = 1
        self.damp_beta = 1
        # changing initial conditions
        self.alpha_init = 1
        self.beta_init = 1
        # use optimistic TS
        self.optimistic = False

    def draw_bandit(self,
                    k: int) -> (int, int):
        """
        Function to draw from one of the bandits k,
        with the probability of success being a Bernoulli distribution with parameter success_probs[k].
        :param k: the bandit to be drawn
        :return: the reward (0/1) and the regret
        """
        reward = np.random.binomial(1, self.success_probs[k])
        regret = self.max_prob - self.success_probs[k]

        return reward, regret

    def sampling(self) -> int:
        """
        Function to pick which bandit to draw from with TS
        :return: the bandit to draw
        """
        # randomly sample posterior distributions for each bandit
        thetas = [np.random.beta(self.alpha_init + alpha, self.beta_init + beta)
                  for (alpha, beta) in zip(self.rewards.sum(axis=1),
                                           self.penalties.sum(axis=1))]

        # introducing caution, by never allowing sampling to go below a minimum threshold = 0.1
        if self.optimistic:
            thetas = [max(t, 0.1) for t in thetas]

        # pick bandit with max probability
        k = np.argmax(thetas)

        return k

    def run_experiment(self) -> None:
        """
        Main function to run the experiment and store its results
        """
        # reset all variables to zero
        self.rewards = np.zeros((self.n_bandits, self.steps))
        self.penalties = np.zeros((self.n_bandits, self.steps))
        # variables storing the global progression of the experiment
        self.choices = np.zeros((self.n_bandits, self.steps))
        self.total_rewards = np.zeros(self.steps)
        self.regret = np.zeros(self.steps)

        # start experiment
        for t in range(self.steps):
            # pick a bandit, record the choice
            bandit = self.sampling()
            self.choices[bandit, t] = 1
            # draw from it
            reward, regret = self.draw_bandit(bandit)
            # update distribution of drawn bandit (and no other)
            if reward == 1:
                self.rewards[bandit, t] += 1 * self.damp_alpha
            else:
                self.penalties[bandit, t] += 1 * self.damp_beta
            # keep track of global progression of experiment
            self.regret[t] = regret

        # cumulative rewards and penalties gathered by each bandit over time
        self.cumsum_rewards = self.rewards.cumsum(axis=1)
        self.cumsum_penalties = self.penalties.cumsum(axis=1)
        # cumulative simulation rewards over time (all bandits)
        self.total_rewards = self.cumsum_rewards.sum(axis=0)
