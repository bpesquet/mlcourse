"""
Python code for reproducing the figures of chapter "Multi-armed Bandits"
from the book "Reinforcement Learning: An Introduction" (2018) by R. Sutton and A. Barto.

Inspired by https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py
"""

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


class Agent:
    """An agent playing with a bandit"""

    def __init__(
        self,
        bandit: Bandit,
        epsilon: float = 0.1,
        step_size: float = 0,
        initial_estimate: float = 0,
    ) -> None:
        self.bandit = bandit

        # Probability for exploration in epsilon-greedy algorithm
        self.epsilon = epsilon

        # Factor for updating estimates. If zero, the sample averages method is used
        self.step_size = step_size

        # Action value (Q-value) estimates for each action.
        # Initial estimate implements the optimistic initial action value technique
        self.action_estimates = torch.zeros(self.bandit.k) + initial_estimate

        # Number of times an action was taken. Used to update estimates in the sample averages method
        self.action_count = torch.zeros(self.bandit.k)

    def play(self, n_steps):
        """Play with the bandit for n_steps"""

        # Rewards for each step
        rewards = torch.zeros(n_steps)

        # For each step, was the optimal action taken?
        optimal_actions_taken = torch.zeros(n_steps)

        for step in range(n_steps):
            # Pull a bandit arm and update associated reward estimate
            action = self._choose_action()
            reward = self.bandit.pull(action=action)
            self._update_estimate(action=action, reward=reward)

            rewards[step] = reward
            optimal_actions_taken[step] = action == self.bandit.best_action

        return rewards, optimal_actions_taken

    def _choose_action(self) -> torch.Tensor:
        """Choose and return an action (the pulled bandit arm)"""

        if torch.rand(1) < self.epsilon:
            # Explore: choose a random action
            return torch.randint(low=0, high=self.bandit.k, size=(1,))
        else:
            # Exploit: choose the action with the highest estimated value
            return torch.argmax(self.action_estimates)

    def _update_estimate(self, action, reward) -> None:
        """Update estimate (Q-value) of an action after receiving its reward"""

        # Incremental update of the action value estimate
        self.action_count[action] += 1

        # Using sample averages if step size is zero
        update_factor = (
            self.step_size if self.step_size != 0 else 1 / self.action_count[action]
        )

        # Update estimate for chosen action
        self.action_estimates[action] += update_factor * (
            reward - self.action_estimates[action]
        )


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
            bandit = Bandit(k=k)
            agent = Agent(bandit=bandit, epsilon=epsilon)
            rewards, optimal_actions_taken = agent.play(n_steps=n_steps)

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
            bandit = Bandit(k=k)
            agent = Agent(
                bandit=bandit,
                epsilon=epsilon,
                step_size=step_size,
                initial_estimate=initial_estimate,
            )
            _, optimal_actions_taken = agent.play(n_steps=n_steps)

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
