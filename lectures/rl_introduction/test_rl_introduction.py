"""
Introduction to Reinforcement Learning.

Inspired by https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
"""

import numpy as np


def create_mdp():
    """Create a Markov Decision Process"""

    # For each state, store the possible actions
    possible_actions = [[0, 1, 2], [0, 2], [1]]

    # For each state and possible action, store the transition probas p(s'|s,a)
    transition_probabilities = [
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None],
    ]

    # For each state and possible action, store the rewards r(s,a,s')
    rewards = [
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],
    ]

    return possible_actions, transition_probabilities, rewards


def init_q_values(possible_actions):
    """Init action-state values to 0 for all possible actions in all states"""

    q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
    for state, actions in enumerate(possible_actions):
        q_values[state, actions] = 0.0  # for all possible actions

    return q_values


def print_optimal_actions(q_values, possible_actions):
    """Print actions with maximum Q-value for each state"""

    # Find action with maximum Q-value for each state
    optimal_actions = np.argmax(q_values, axis=1)

    n_states = len(possible_actions)
    for s in range(n_states):
        print(f"Optimal action for state {s} is a{optimal_actions[s]}")


def q_value_iteration(
    n_iterations, possible_actions, transition_probabilities, rewards
):
    """Implement the Q-Value iteration algorithm"""

    q_values = init_q_values(possible_actions=possible_actions)

    gamma = 0.9  # Discount factor - try changing it to 0.95
    n_states = len(possible_actions)

    history = []  # Store training history for plotting (later)
    for _ in range(n_iterations):
        Q_prev = q_values.copy()
        history.append(Q_prev)
        # Compute Q_k+1 for all states and actions
        for s in range(n_states):
            for a in possible_actions[s]:
                q_values[s, a] = np.sum(
                    [
                        transition_probabilities[s][a][sp]
                        * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                        for sp in range(n_states)
                    ]
                )

    history = np.array(history)

    return q_values, history


def step(state, action, possible_actions, transition_probabilities, rewards):
    """Perform an action and receive next state and reward"""

    n_states = len(possible_actions)

    probas = transition_probabilities[state][action]
    next_state = np.random.choice(range(n_states), p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward


def exploration_policy(state, possible_actions):
    """Explore the MDP, returning a random action"""

    # This basic exploration policy is sufficient for this simple problem
    return np.random.choice(possible_actions[state])


def q_learning(n_iterations, possible_actions, transition_probabilities, rewards):
    """Implement the Q-learning algorithm"""

    q_values = init_q_values(possible_actions=possible_actions)

    alpha0 = 0.05  # initial learning rate
    decay = 0.005  # learning rate decay
    gamma = 0.9  # discount factor
    state = 0  # initial state
    history = []  # Training history

    for iteration in range(n_iterations):
        history.append(q_values.copy())
        action = exploration_policy(state=state, possible_actions=possible_actions)
        next_state, reward = step(
            state=state,
            action=action,
            possible_actions=possible_actions,
            transition_probabilities=transition_probabilities,
            rewards=rewards,
        )
        next_q_value = np.max(q_values[next_state])  # greedy policy at the next step
        alpha = alpha0 / (1 + iteration * decay)  # learning rate decay
        q_values[state, action] *= 1 - alpha
        q_values[state, action] += alpha * (reward + gamma * next_q_value)
        state = next_state

    history = np.array(history)

    return q_values, history


def test_rl_intro():
    """Main test function"""

    possible_actions, transition_probabilities, rewards = create_mdp()

    # Run the Q-Value iteration algorithm
    n_iterations_q_value = 50
    q_final_vi, history_q_vi = q_value_iteration(
        n_iterations=n_iterations_q_value,
        possible_actions=possible_actions,
        transition_probabilities=transition_probabilities,
        rewards=rewards,
    )
    # Show final action-state values
    print(q_final_vi)
    print_optimal_actions(q_values=q_final_vi, possible_actions=possible_actions)

    # Run the Q-learning algorithm
    n_iterations_q_learning = 10000
    q_final_learning, history_q_learning = q_learning(
        n_iterations=n_iterations_q_learning,
        possible_actions=possible_actions,
        transition_probabilities=transition_probabilities,
        rewards=rewards,
    )

    # Show final action-state values
    print(q_final_learning)
    print_optimal_actions(q_values=q_final_learning, possible_actions=possible_actions)


# Standalone execution
if __name__ == "__main__":
    test_rl_intro()
