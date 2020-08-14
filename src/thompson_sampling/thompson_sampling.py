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
                 steps: int = None,
                 alpha_damping: float = None,
                 beta_damping: float = None,
                 alpha_init: [float] = None,
                 beta_init: [float] = None,
                 optimistic: bool = None,
                 optimistic_threshold: float = None
                 ) -> None:
        """
        :param success_probs: an array of floats in the [0, 1] interval, the success probability of each bandit
        :param steps: the iteration steps (optional, defaults to 1000)
        :param alpha_damping: reduces the tendency of TS towards exploitation (defaults to 1, no damping)
        :param beta_damping: reduces the tendency of TS towards exploration (defaults to 1, no damping)
        :param alpha_init: an array of non-negative floats, the initial conditions of each bandit
            (defaults to 0, uniform priors)
        :param beta_init: an array of non-negative floats, the initial conditions of each bandit
            (defaults to 0, uniform priors)
        :param optimistic: whether to use an optimistic TS strategy, whereby a lower bound is put on sampling
            (defaults to False)
        :param optimistic_threshold: the lower bound for sampling in an optimistic strategy (defaults to 1.e-6)
        """

        # test validity of input variables
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

        if steps is None:
            self.steps = 1000
        else:
            try:
                assert ((steps == int(steps)) & (steps > 0))
            except AssertionError:
                print("Number of steps must be positive integer")
            self.steps = steps

        if alpha_damping is None:
            self.alpha_damping = 1
        else:
            try:
                assert ((alpha_damping >= 0) & (alpha_damping <= 1))
            except AssertionError:
                print("alpha_damping must belong to [0, 1] interval.")
            self.alpha_damping = alpha_damping

        if beta_damping is None:
            self.beta_damping = 1
        else:
            try:
                assert ((beta_damping >= 0) & (beta_damping <= 1))
            except AssertionError:
                print("beta_damping must belong to [0, 1] interval.")
            self.beta_damping = beta_damping

        if alpha_init is None:
            self.alpha_init = np.ones_like(success_probs)
        else:
            try:
                assert len(alpha_init) == self.n_bandits
            except AssertionError:
                print("Arrays alpha_init and success_probs must have equal length")
            try:
                assert np.all(alpha_init >= 0)
            except AssertionError:
                print("Elements of array alpha_init must be >= 0.")
            self.alpha_init = alpha_init

        if beta_init is None:
            self.beta_init = np.ones_like(success_probs)
        else:
            try:
                assert len(beta_init) == self.n_bandits
            except AssertionError:
                print("Arrays beta_init and success_probs must have equal length")
            try:
                assert np.all(beta_init >= 0)
            except AssertionError:
                print("Elements of array beta_init must be >= 0.")
            self.beta_init = beta_init

        if optimistic is None:
            self.optimistic = False
        else:
            try:
                assert isinstance(optimistic, bool)
            except AssertionError:
                raise AssertionError('parameter optimistic must be True / False')
            self.optimistic = optimistic
            if optimistic_threshold is None:
                self.optimistic_threshold = 1.e-6
            else:
                try:
                    assert ((optimistic_threshold >= 0) & (optimistic_threshold < 1))
                except AssertionError:
                    print("optimistic_threshold must belong to [0, 1[ interval.")
                self.optimistic_threshold = optimistic_threshold

        # variables storing the progression of rewards and penalties of each bandit
        self.rewards = None
        self.penalties = None
        # variables storing the global progression of the experiment
        self.choices = None
        self.cumsum_rewards = None
        self.cumsum_penalties = None
        self.total_rewards = None
        self.regret = None

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
        thetas = [np.random.beta(1 + alpha_init + alpha,
                                 1 + beta_init + beta)
                  for (alpha_init, alpha, beta_init, beta)
                  in zip(self.alpha_init, self.rewards.sum(axis=1),
                         self.beta_init, self.penalties.sum(axis=1))]

        # introducing caution, by never allowing sampling to go below a minimum threshold = 0.1
        if self.optimistic:
            thetas = [max(t, self.optimistic_threshold) for t in thetas]

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
                self.rewards[bandit, t] += 1 * self.alpha_damping
            else:
                self.penalties[bandit, t] += 1 * self.beta_damping
            # keep track of global progression of experiment
            self.regret[t] = regret

        # cumulative rewards and penalties gathered by each bandit over time
        self.cumsum_rewards = self.rewards.cumsum(axis=1)
        self.cumsum_penalties = self.penalties.cumsum(axis=1)
        # cumulative simulation rewards over time (all bandits)
        self.total_rewards = self.cumsum_rewards.sum(axis=0)
