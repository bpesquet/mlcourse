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
    """A multi-armed bandit"""

    def __init__(self, n_actions) -> None:
        # Number of arms/actions (denoted k in S & B book)
        self.n_actions = n_actions

        # True action values (Q-values). Shape: (n_actions,)
        self.action_values = torch.normal(mean=0, std=1, size=(n_actions,))

        # Action with best value
        self.best_action = torch.argmax(self.action_values)

    def pull(self, action) -> torch.Tensor:
        """Pull a bandit arm, returning the associated reward"""

        assert action >= 0 and action < self.n_actions, (
            "Action must correspond to one of the bandit's arms"
        )

        # Reward is generated from a normal distribution centered on the true action value
        return torch.normal(mean=self.action_values[action], std=1)


class Policy(ABC):
    """Abstract Base Class for policies"""

    def __init__(self, n_actions: int, initial_estimate: float):
        # Number of possible actions/bandit arms
        self.n_actions = n_actions

        # Optional initial value of action value estimates.
        # Used to implement the optimistic initial action value technique
        self.initial_estimate = initial_estimate

        self.reset()

    def reset(self) -> None:
        """Reset policy attributes. Must be called by overrides in subclasses"""

        # Action value (Q-value) estimates for each action
        self.action_estimates = torch.zeros(self.n_actions) + self.initial_estimate

    @abstractmethod
    def choose_action(self, step: int) -> torch.Tensor:
        """Choose and return an action (the pulled bandit arm)"""

    @abstractmethod
    def update_estimate(self, action, reward) -> None:
        """Update estimate (Q-value) of an action after receiving its reward"""


class SampleAverages(Policy):
    """Policy using sample averages to update estimates"""

    def __init__(
        self, n_actions: int, epsilon: float = 0.1, initial_estimate: float = 0
    ):
        super().__init__(n_actions=n_actions, initial_estimate=initial_estimate)

        # Probability for exploration in epsilon-greedy algorithm
        self.epsilon = epsilon

    def reset(self) -> None:
        super().reset()

        # Number of times an action was taken
        self.action_count = torch.zeros(self.n_actions)

    def choose_action(self, step: int) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            # Explore: choose a random action
            return torch.randint(low=0, high=self.n_actions, size=(1,))
        else:
            # Exploit: choose the action with the highest value estimate
            return torch.argmax(self.action_estimates)

    def update_estimate(self, action, reward) -> None:
        # Increment number of times n the chosen action was taken
        self.action_count[action] += 1

        # Update estimate for chosen action, using 1/n as factor
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
        super().__init__(n_actions=n_actions, initial_estimate=initial_estimate)

        # Probability for exploration in epsilon-greedy algorithm
        self.epsilon = epsilon

        # Step size for updating estimates (denoted alpha in S & B book)
        self.step_size = step_size

    def choose_action(self, step: int) -> torch.Tensor:
        if torch.rand(1) < self.epsilon:
            # Explore: choose a random action
            return torch.randint(low=0, high=self.n_actions, size=(1,))
        else:
            # Exploit: choose the action with the highest value estimate
            return torch.argmax(self.action_estimates)

    def update_estimate(self, action, reward) -> None:
        # Update estimate for chosen action, using step size as factor
        self.action_estimates[action] += self.step_size * (
            reward - self.action_estimates[action]
        )


class UCB(Policy):
    """Policy using the Upper Confidence Bound to choose exploitation action"""

    def __init__(
        self,
        n_actions: int,
        initial_estimate: float = 0,
        exploration_factor=2,
    ):
        super().__init__(
            n_actions=n_actions,
            initial_estimate=initial_estimate,
        )

        # Degree of exploration (denoted c in S & B book)
        self.exploration_factor = exploration_factor

        self.reset()

    def reset(self) -> None:
        super().reset()

        # Number of times an action was taken
        self.action_count = torch.zeros(self.n_actions)

    def choose_action(self, step: int) -> torch.Tensor:
        # Compute UCB estimates for all actions
        ucb_estimates = self.action_estimates + self.exploration_factor * torch.sqrt(
            math.log(step + 1) / (self.action_count + 1e-5)
        )

        # Choose the action with the highest UCB-estimated value
        return torch.argmax(ucb_estimates)

    def update_estimate(self, action, reward) -> None:
        # Increment number of times n the chosen action was taken
        self.action_count[action] += 1

        # Update estimate for chosen action, using 1/n as update factor
        self.action_estimates[action] += (
            reward - self.action_estimates[action]
        ) / self.action_count[action]


def run(
    bandit: Bandit, policy: Policy, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a policy against a bandit for n_steps steps"""

    # Rewards for each step
    rewards = torch.zeros(n_steps)

    # For each step, was the optimal action taken?
    optimal_actions_taken = torch.zeros(n_steps)

    policy.reset()

    for step in range(n_steps):
        # Pull a bandit arm and update associated reward estimate
        action = policy.choose_action(step=step)
        reward = bandit.pull(action=action)
        policy.update_estimate(action=action, reward=reward)

        rewards[step] = reward
        optimal_actions_taken[step] = action == bandit.best_action

    return rewards, optimal_actions_taken


def simulate(
    n_actions: int, policy: Policy, n_runs: int, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate n_runs runs of a policy against a bandit"""

    # Rewards for each run and step
    rewards = torch.zeros(size=(n_runs, n_steps))

    # For each run and step, was the optimal action taken?
    optimal_action_taken = torch.zeros(size=(n_runs, n_steps))

    for r in trange(n_runs):
        run_rewards, run_optimal_actions_taken = run(
            bandit=Bandit(n_actions=n_actions),
            policy=policy,
            n_steps=n_steps,
        )

        rewards[r, :] = run_rewards
        optimal_action_taken[r, :] = run_optimal_actions_taken

    # Return averaged values against all runs
    return rewards.mean(axis=0), optimal_action_taken.mean(axis=0)


def plot_figure_2_1(n_actions=10) -> None:
    """Reproduce figure 2.1 of Sutton & Barto book: example bandit problem"""

    # Generate true Q-values for each action from a standard normal distribution N(0,1)
    q_true = torch.randn(n_actions)

    # Generate rewards distributions for each action from a unit-variance normal distribution N(Q,1)
    data = [sorted(torch.normal(mean=q.item(), std=1.0, size=(200,))) for q in q_true]

    # Plot reward distributions for each action
    plt.violinplot(dataset=data, showmeans=True)

    # Plot horizontal line for mean of Q-values
    plt.plot([0, n_actions + 1], [0, 0], "k--")

    # Configure x-axis
    plt.xlim([0, n_actions + 1])
    plt.xlabel("Action")
    # Change x-axis labels to integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # Display all possible actions as integer ticks
    plt.xticks(range(1, n_actions + 1))

    # Configure y-axis
    plt.ylabel("Reward distribution")

    plt.title(f"Example bandit problem (k={n_actions})")
    plt.show()


def plot_figure_2_2(n_actions=10, n_runs=200, n_steps=100) -> None:
    """Reproduce figure 2.2 of Sutton & Barto book: average performance of epsilon-greedy methods"""

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))
    fig.suptitle(
        f"Average performance of $\\epsilon$-greedy methods on a {n_actions}-armed bandit ({n_runs} runs, sample averages)"
    )

    # Compared hyperparameters and associated plot labels
    epsilons = (0, 0.01, 0.1)
    labels = ("(greedy)", "", "")

    for epsilon, label in zip(epsilons, labels):
        avg_rewards, optimal_action_percent = simulate(
            n_actions=n_actions,
            policy=SampleAverages(n_actions=n_actions, epsilon=epsilon),
            n_runs=n_runs,
            n_steps=n_steps,
        )

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


def plot_figure_2_3(n_actions=10, n_runs=200, n_steps=100, step_size=0.1) -> None:
    """Reproduce figure 2.3 of Sutton & Barto book: effect of optimistic initial action-value estimates"""

    plt.figure(figsize=(10, 5))
    plt.title(
        f"Effect of optimistic initial action-value estimates on a {n_actions}-armed bandit ({n_runs} runs, $\\alpha$ = {step_size})"
    )

    # Compared hyperparameters and associated plot labels
    epsilons = (0, 0.1)
    initial_estimates = (5, 0)
    labels = ("Optimistic, greedy", "Realistic, $\\epsilon$-greedy")

    for epsilon, initial_estimate, label in zip(epsilons, initial_estimates, labels):
        _, optimal_action_percent = simulate(
            n_actions=n_actions,
            policy=StepSize(
                n_actions=n_actions,
                epsilon=epsilon,
                initial_estimate=initial_estimate,
                step_size=step_size,
            ),
            n_runs=n_runs,
            n_steps=n_steps,
        )

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
    n_actions=10, n_runs=200, n_steps=100, epsilon=0.1, exploration_factor=2
) -> None:
    """Reproduce figure 2.4 of Sutton & Barto book: average performance of UCB action selection"""

    plt.figure(figsize=(10, 5))
    plt.title(
        f"Average performance of UCB action selection on a {n_actions}-armed bandit ({n_runs} runs, sample averages)"
    )

    # Compared policies and associated plot labels
    policies = (
        SampleAverages(n_actions=n_actions, epsilon=epsilon),
        UCB(
            n_actions=n_actions,
            exploration_factor=exploration_factor,
        ),
    )
    labels = (
        f"$\\epsilon$-greedy ($\\epsilon$ = {epsilon})",
        f"UCB ($c$ = {exploration_factor})",
    )

    for policy, label in zip(policies, labels):
        avg_rewards, _ = simulate(
            n_actions=n_actions, policy=policy, n_runs=n_runs, n_steps=n_steps
        )

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
    n_runs = 2000
    n_steps = 1000

    plot_figure_2_1()
    plot_figure_2_2(n_runs=n_runs, n_steps=n_steps)
    plot_figure_2_3(n_runs=n_runs, n_steps=n_steps)
    plot_figure_2_4(n_runs=n_runs, n_steps=n_steps)
