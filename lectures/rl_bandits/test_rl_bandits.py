"""
Python code for reproducing the figures of chapter "Multi-armed Bandits"
from the book "Reinforcement Learning: An Introduction" (2018) by R. Sutton and A. Barto.

Inspired by https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py
"""

import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from tqdm import trange


class Bandit:
    """A k-armed bandit"""

    def __init__(self, k) -> None:
        # Number of arms/actions
        self.k = k

        # True action values (Q-values). Shape: (k,)
        self.action_values = torch.normal(mean=0, std=1, size=(k,))

        # Action with best value
        self.best_action = torch.argmax(self.action_values)

    def pull(self, action) -> torch.Tensor:
        """Pull a bandit arm, returning the associated reward"""

        assert action >= 0 and action < self.k, (
            "Action must correspond to one of the bandit's arms"
        )

        # Reward is generated from a normal distribution centered on the true action value
        return torch.normal(mean=self.action_values[action], std=1)


class Policy(ABC):
    """Abstract Base Class for agent policies"""

    def __init__(
        self, n_actions: int, epsilon: float = 0.1, initial_estimate: float = 0
    ):
        # Number of possible actions/bandit arms
        self.n_actions = n_actions

        # Probability for exploration in epsilon-greedy algorithm
        self.epsilon = epsilon

        # Action value (Q-value) estimates for each action.
        # Initial estimate implements the optimistic initial action value technique
        self.action_estimates = torch.zeros(n_actions) + initial_estimate

    def choose_action(self, step: int) -> torch.Tensor:
        """Choose and return an action (the pulled bandit arm)"""

        if torch.rand(1) < self.epsilon:
            # Explore: choose a random action
            return torch.randint(low=0, high=self.n_actions, size=(1,))
        else:
            return self._choose_exploit_action(step=step)

    def _choose_exploit_action(self, step: int) -> torch.Tensor:
        """Exploit: return the most promising action"""

        # No need for step in default behavior
        _ = step

        # Choose the action with the highest estimated value
        return torch.argmax(self.action_estimates)

    @abstractmethod
    def update_estimate(self, action, reward) -> None:
        """Update estimate (Q-value) of an action after receiving its reward"""


class SampleAverages(Policy):
    """Policy using sample averages to update estimates"""

    def __init__(
        self, n_actions: int, epsilon: float = 0.1, initial_estimate: float = 0
    ):
        super().__init__(
            n_actions=n_actions, epsilon=epsilon, initial_estimate=initial_estimate
        )

        # Number of times an action was taken
        self.action_count = torch.zeros(n_actions)

    def update_estimate(self, action, reward) -> None:
        # Increment number of times n the chosen action was taken
        self.action_count[action] += 1

        # Update estimate for chosen action, using 1/n as update factor
        self.action_estimates[action] += (
            reward - self.action_estimates[action]
        ) / self.action_count[action]


class StepSize(Policy):
    """Policy using a constant step size to update estimates"""

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        initial_estimate: float = 0,
        step_size=0.1,
    ):
        super().__init__(
            n_actions=n_actions, epsilon=epsilon, initial_estimate=initial_estimate
        )

        # Step size for updating estimates (denoted alpha in S & B book)
        self.step_size = step_size

    def update_estimate(self, action, reward) -> None:
        # Update estimate for chosen action, using step size as update factor
        self.action_estimates[action] += self.step_size * (
            reward - self.action_estimates[action]
        )


class UCB(SampleAverages):
    """Policy using the Upper Confidence Bound to choose exploitation action"""

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        initial_estimate: float = 0,
        exploration_factor=2,
    ):
        super().__init__(
            n_actions=n_actions,
            epsilon=epsilon,
            initial_estimate=initial_estimate,
        )

        # Degree of exploration (denoted c in S & B book)
        self.exploration_factor = exploration_factor

    def _choose_exploit_action(self, step: int) -> torch.Tensor:
        # Compute UCB estimates for all actions
        ucb_estimates = self.action_estimates + self.exploration_factor * torch.sqrt(
            math.log(step + 1) / (self.action_count + 1e-5)
        )

        # Choose the action with the highest UCB-estimated value
        return torch.argmax(ucb_estimates)


def run(bandit: Bandit, policy: Policy, n_steps: int):
    """Run a policy against a bandit for n_steps steps"""

    # Rewards for each step
    rewards = torch.zeros(n_steps)

    # For each step, was the optimal action taken?
    optimal_actions_taken = torch.zeros(n_steps)

    for step in range(n_steps):
        # Pull a bandit arm and update associated reward estimate
        action = policy.choose_action(step=step)
        reward = bandit.pull(action=action)
        policy.update_estimate(action=action, reward=reward)

        rewards[step] = reward
        optimal_actions_taken[step] = action == bandit.best_action

    return rewards, optimal_actions_taken


def plot_figure_2_1(k=10) -> None:
    """Reproduce figure 2.1 of Sutton & Barto book: example bandit problem"""

    # Generate true Q-values for each action from a standard normal distribution N(0,1)
    q_true = torch.randn(k)
    # Generate rewards distributions for each action from a unit-variance normal distribution N(Q,1)
    data = [sorted(torch.normal(mean=q.item(), std=1.0, size=(200,))) for q in q_true]

    # Plot reward distributions for each action
    plt.violinplot(dataset=data, showmeans=True)

    # Plot horizontal line for mean of Q-values
    plt.plot([0, k + 1], [0, 0], "k--")

    # Configure x-axis
    plt.xlim([0, k + 1])
    plt.xlabel("Action")
    # Change x-axis labels to integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # Display all possible actions as integer ticks
    plt.xticks(range(1, k + 1))

    # Configure y-axis
    plt.ylabel("Reward distribution")

    plt.title(f"Example bandit problem (k={k})")
    plt.show()


def plot_figure_2_2(k=10, n_runs=200, n_steps=100) -> None:
    """Reproduce figure 2.2 of Sutton & Barto book: average performance of epsilon-greedy methods"""

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))
    fig.suptitle(
        f"Average performance of $\\epsilon$-greedy methods on a {k}-armed bandit ({n_runs} runs, sample averages)"
    )

    for epsilon, label in zip((0, 0.01, 0.1), ("(greedy)", "", "")):
        avg_rewards = torch.zeros(n_steps)
        optimal_action_percent = torch.zeros(n_steps)

        for _ in trange(n_runs):
            rewards, optimal_actions_taken = run(
                bandit=Bandit(k=k),
                policy=SampleAverages(n_actions=k, epsilon=epsilon),
                n_steps=n_steps,
            )

            avg_rewards += rewards
            optimal_action_percent += optimal_actions_taken

        # Average the previously cumulated values
        avg_rewards /= n_runs
        optimal_action_percent /= n_runs

        # Plot average rewards for the current value of epsilon
        ax1.plot(avg_rewards, label=f"$\\epsilon$ = {epsilon} {label}")
        ax1.set(ylabel="Average reward")
        ax1.legend()

        # Plot % of optimal action chosen for the current value of epsilon
        ax2.plot(optimal_action_percent, label=f"$\\epsilon$ = {epsilon} {label}")
        ax2.set(ylabel="% Optimal action")
        ax2.legend()

    plt.xlabel("Steps")
    plt.show()


def plot_figure_2_3(k=10, n_runs=200, n_steps=100, step_size=0.1) -> None:
    """Reproduce figure 2.3 of Sutton & Barto book: effect of optimistic initial action-value estimates"""

    plt.figure(figsize=(10, 5))
    plt.title(
        f"Effect of optimistic initial action-value estimates on a {k}-armed bandit ({n_runs} runs, $\\alpha$ = {step_size})"
    )

    for epsilon, initial_estimate, label in zip(
        (0, 0.1), (5, 0), ("Optimistic, greedy", "Realistic, $\\epsilon$-greedy")
    ):
        optimal_action_percent = torch.zeros(n_steps)

        for _ in trange(n_runs):
            _, optimal_actions_taken = run(
                bandit=Bandit(k=k),
                policy=StepSize(
                    n_actions=k,
                    epsilon=epsilon,
                    initial_estimate=initial_estimate,
                    step_size=step_size,
                ),
                n_steps=n_steps,
            )

            optimal_action_percent += optimal_actions_taken

        # Average the previously cumulated values
        optimal_action_percent /= n_runs

        # Plot % of optimal action chosen for the current value of epsilon
        plt.plot(
            optimal_action_percent,
            label=f"{label} ($\\epsilon$ = {epsilon}, $Q_1$ = {initial_estimate})",
        )

    plt.ylabel("% Optimal action")
    plt.legend()
    plt.xlabel("Steps")
    plt.show()


def plot_figure_2_4(
    k=10, n_runs=200, n_steps=100, epsilon=0.1, exploration_factor=2
) -> None:
    """Reproduce figure 2.4 of Sutton & Barto book: average performance of UCB action selection"""

    plt.figure(figsize=(10, 5))
    plt.title(
        f"Average performance of UCB action selection on a {k}-armed bandit ({n_runs} runs, sample averages)"
    )

    policies = (
        SampleAverages(n_actions=k, epsilon=epsilon),
        UCB(
            n_actions=k,
            epsilon=epsilon,
            exploration_factor=exploration_factor,
        ),
    )
    labels = (
        f"$\\epsilon$-greedy ($\\epsilon$ = {epsilon})",
        f"UCB ($c$ = {exploration_factor})",
    )

    for policy, label in zip(policies, labels):
        avg_rewards = torch.zeros(n_steps)

        for _ in trange(n_runs):
            rewards, _ = run(bandit=Bandit(k=k), policy=policy, n_steps=n_steps)

            avg_rewards += rewards

        # Average the previously cumulated values
        avg_rewards /= n_runs

        # Plot average rewards for the current value of epsilon
        plt.plot(avg_rewards, label=f"{label}")

    plt.ylabel("Average reward")
    plt.legend()
    plt.xlabel("Steps")
    plt.show()


# Standalone execution
if __name__ == "__main__":
    # Set the seed for generating random numbers in order to obtain reproducible results
    torch.manual_seed(6)

    # Use same hyperparameters as in Sutton & Barto book
    n_runs = 200
    n_steps = 1000

    # plot_figure_2_1()
    plot_figure_2_2(n_runs=n_runs, n_steps=n_steps)
    # plot_figure_2_3(n_runs=n_runs, n_steps=n_steps)
    plot_figure_2_4(n_runs=n_runs, n_steps=n_steps)
