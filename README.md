# Thompson sampling for the multi-armed bandit problem

This module contains the code necessary 
to implement a Thompson sampling strategy
in a multi-armed bandit setting.

## Python version

The module has been developed on Python 3.7.3.

## Python Libraries

Install requirements as:

    pip3 install -r requirements.txt

## Usage

To initialize the class:

    ts = Thompson(success_probs, steps)

where
* `success_probs`: an array of floats in the \[0, 1\] interval,
    the success probability of each bandit;
* `steps`: the number of steps to iterate over (optional, defaults to 1000).

## Theory recap

### Multi-armed bandit problem

The multi-armed bandit problem is an unsupervised-learning problem
in which a fixed set of limited resources
must be allocated between competing choices
without prior knowledge of the rewards offered by each of them,
which must be instead learned on the go.

The name refers to the hypothetical situation
in which a player must choose between a set of $K$ slot machines,
and at each round decide whether to continue play with the one
currently looking like the most promising choice
(based on the expectation of an "exploitation", i.e., a winning streak)
versus trying a different machine,
which might turn out to offer a higher reward ("exploration").
(The winning probability of each machine is assumed constant in time.)

Practically, the issue of the trade-off between exploitation and exploration
is one faced in many online reinforcement-learning problems,
such as offering a (new or relatively new) app user the optimal UX flow
(success being continued user engagement),
picking between different banner ads that can be displayed on a website
(success being a click or a conversion), etc.

### Thompson sampling and Bayes' rule

Thompson sampling (TS) is a solving approach to the aforementioned exploitation-exploration problem
that attempts to converge towards the optimal solution by finding a balance
between exploiting what is known to maximize immediate performance on the one hand,
and investing to accumulate new information that may improve future performance on the other.
The second aspect is what distinguishes this class of solutions from so-called
greedy approaches, which instead only maximize (based on the historical data at hand)
the immediate reward. Greedy approaches have a higher chance of getting stuck in local maxima,
and cannot thoroughly explore the available parameter space, which in simpler terms means
to try out options with larger uncertainties over their average outcomes.

While a greedy approach would assess at each round the expected success probability $\theta_k$
of each action $k$ and pick the maximum one among them,
in TS, at each round,
the probability of picking a bandit is equal to the probability of it being the optimal choice.
To practically emulate this behaviour,
a value for the success probability of each possible action is randomly sampled
from the posterior distribution of each one
(the maximum of which _is_ of course the expected success probability of that action),
and only then the maximum is picked.

The posterior probability distribution is then updated based on the observed result,
so becoming the prior distribution for the next round.
This update rule, mixing prior assumptions about the probability distribution and the empirical observations,
is what makes TS a Bayesian approach:
in Bayesian statistics, the probability of an event is based both on data
(the fresh empirical observations, which in our case are the results of each round of lever pulling)
and on prior information or beliefs about the event
(in our case, the rewards expected from each machine, 
initially guessed at random and then adjusted after every round of pulling).
This second aspect is what distinguishes Bayesian methods from purely frequentist ones,
which are only based on the observed data and do not incorporate the concept of priors.

### Beta-Bernoulli bandit

In the Beta-Bernoulli schematization of the bandit problem,
each time an action $k$ is selected,
a reward of 1 (success) is generated with probability $\theta_k$ 
\- otherwise, a reward of 0 (failure) is generated, with probability $(1 - \theta_k)$.
Bernoulli refers here to 0/1 being the only possible outcomes.

Statistically, the prior distribution of expected rewards for a Bernoulli random variable
is the so-called Beta distribution $Beta(\alpha_k, \beta_k)$,
its parameters $\alpha_k$ and $\beta_k$ sometimes being called pseudo-counts,
because the update rule increases them by one with each observed success or failure, respectively.
The posterior distribution we get after updating
the prior probability of the selected action (and of no other one)
depending on the reward obtained becomes then the prior distribution for the next round, and so on,
increasingly growing sharper and closer to its expected value.

A useful property of the distributions so built is that each $\theta_k$ is the probability
that the random estimate drawn from it exceeds those drawn for other actions.
Also, since these estimates are drawn from the posterior distributions,
they are equal to the probability that the corresponding action is optimal,
conditioned on observed history.

As we play and gather evidence, our posterior distributions become more concentrated
(more so, the more rewarding they are),
increasing our chance of maximizing the sum of the collective rewards:
exploration reduces over time, and exploitation is increasingly privileged as the uncertainties are reduced.
Meanwhile the regret, or the sum of the rewards under an optimal strategy
minus the actually collected rewards, asymptotically converge to zero.

### Author

Tullio Bagnoli

### License

This project is licensed under the MIT license.
