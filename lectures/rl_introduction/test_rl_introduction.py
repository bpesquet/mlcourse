"""
Introduction to Reinforcement Learning.

Inspired by https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MDP:
    """A Markov Decision Process"""

    def __init__(
        self,
        possible_actions: list[list],
        transition_probabilities: list[list[list]],
        rewards: list[list[list]],
    ):
        # For each state, store the possible actions
        self.possible_actions = possible_actions

        # For each state and possible action, store the transition probas p(s'|s,a)
        self.transition_probabilities = transition_probabilities

        # For each state and possible action, store the rewards r(s,a,s')
        self.rewards = rewards


def init_q_values(possible_actions: list[list]) -> np.ndarray:
    """Init action-state values to 0 for all possible actions in all states"""

    q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
    for state, actions in enumerate(possible_actions):
        q_values[state, actions] = 0.0  # for all possible actions

    return q_values


def print_optimal_actions(q_values: np.ndarray, possible_actions: list[list]) -> None:
    """Print actions with maximum Q-value for each state"""

    # Find action with maximum Q-value for each state
    optimal_actions = np.argmax(q_values, axis=1)

    n_states = len(possible_actions)
    for s in range(n_states):
        print(f"Optimal action for state {s} is a{optimal_actions[s]}")


def q_value_iteration(n_iterations: int, mdp: MDP) -> tuple[np.ndarray, list]:
    """Implement the Q-Value iteration algorithm"""

    q_values = init_q_values(possible_actions=mdp.possible_actions)

    gamma = 0.9  # Discount factor - try changing it to 0.95
    n_states = len(mdp.possible_actions)

    history = []  # Store training history for plotting (later)
    for _ in range(n_iterations):
        Q_prev = q_values.copy()
        history.append(Q_prev)
        # Compute Q_k+1 for all states and actions
        for s in range(n_states):
            for a in mdp.possible_actions[s]:
                q_values[s, a] = np.sum(
                    [
                        mdp.transition_probabilities[s][a][sp]
                        * (mdp.rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                        for sp in range(n_states)
                    ]
                )

    history = np.array(history)

    return q_values, history


def step(state, action, mdp: MDP) -> tuple[int, int]:
    """Perform an action and return next state and reward"""

    n_states = len(mdp.possible_actions)

    probas = mdp.transition_probabilities[state][action]
    next_state = np.random.choice(range(n_states), p=probas)
    reward = mdp.rewards[state][action][next_state]
    return next_state, reward


def exploration_policy(state, possible_actions) -> int:
    """Explore the MDP, returning a random action"""

    # This basic exploration policy is sufficient for this simple problem
    return np.random.choice(possible_actions[state])


def q_learning(n_iterations: int, mdp: MDP) -> tuple[np.ndarray, list]:
    """Implement the Q-learning algorithm"""

    q_values = init_q_values(possible_actions=mdp.possible_actions)

    alpha0 = 0.05  # initial learning rate
    decay = 0.005  # learning rate decay
    gamma = 0.9  # discount factor
    state = 0  # initial state
    history = []  # Training history

    for iteration in range(n_iterations):
        history.append(q_values.copy())
        action = exploration_policy(state=state, possible_actions=mdp.possible_actions)
        next_state, reward = step(state=state, action=action, mdp=mdp)
        next_q_value = np.max(q_values[next_state])  # greedy policy at the next step
        alpha = alpha0 / (1 + iteration * decay)  # learning rate decay
        q_values[state, action] *= 1 - alpha
        q_values[state, action] += alpha * (reward + gamma * next_q_value)
        state = next_state

    history = np.array(history)

    return q_values, history


def plot_q_values(
    n_iterations_q_vi: int,
    n_iterations_q_learning: int,
    history_q_vi: np.ndarray,
    history_q_learning: np.ndarray,
):
    """Plot histories for Q-Value iteration and Q-learning algorithms"""

    # Final Q-value for s0 and a0
    true_q_value = history_q_vi[-1, 0, 0]

    # Plot training histories for Q-Value Iteration and Q-Learning methods
    _, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].set_ylabel("Q-Value$(s_0, a_0)$", fontsize=14)
    axes[0].set_title("Q-Value Iteration", fontsize=14)
    axes[1].set_title("Q-Learning", fontsize=14)
    for ax, width, history in zip(
        axes,
        (n_iterations_q_vi, n_iterations_q_learning),
        (history_q_vi, history_q_learning),
    ):
        ax.plot([0, width], [true_q_value, true_q_value], "k--")
        ax.plot(np.arange(width), history[:, 0, 0], "b-", linewidth=2)
        ax.set_xlabel("Iterations", fontsize=14)
        ax.axis([0, width, 0, 24])

    plt.show()


def test_rl_intro(show_plot=False) -> None:
    """Main test function"""

    mdp = MDP(
        possible_actions=[[0, 1, 2], [0, 2], [1]],
        transition_probabilities=[
            [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
            [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
            [None, [0.8, 0.1, 0.1], None],
        ],
        rewards=[
            [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
            [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],
        ],
    )

    # Run the Q-Value iteration algorithm
    n_iterations_q_vi = 50
    q_final_vi, history_q_vi = q_value_iteration(
        n_iterations=n_iterations_q_vi, mdp=mdp
    )
    # Show final action-state values
    print(q_final_vi)
    print_optimal_actions(q_values=q_final_vi, possible_actions=mdp.possible_actions)

    # Run the Q-learning algorithm
    n_iterations_q_learning = 10000
    q_final_learning, history_q_learning = q_learning(
        n_iterations=n_iterations_q_learning, mdp=mdp
    )
    # Show final action-state values
    print(q_final_learning)
    print_optimal_actions(
        q_values=q_final_learning, possible_actions=mdp.possible_actions
    )

    if show_plot:
        # Improve plots appearance
        sns.set_theme()

        plot_q_values(
            n_iterations_q_vi=n_iterations_q_vi,
            n_iterations_q_learning=n_iterations_q_learning,
            history_q_vi=history_q_vi,
            history_q_learning=history_q_learning,
        )


# Standalone execution
if __name__ == "__main__":
    test_rl_intro(show_plot=True)
