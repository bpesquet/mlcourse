"""
Python code for reproducing the figures of gridworld example in chapter "Dynamic Programming"
of the book "Reinforcement Learning: An Introduction" (2018) by R. Sutton and A. Barto.

Adapted from https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/grid_world.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def draw_image(image, n_iterations: int):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(
            i,
            -1,
            width,
            height,
            text=i + 1,
            loc="right",
            edgecolor="none",
            facecolor="none",
        )
        tb.add_cell(
            -1,
            i,
            width,
            height / 2,
            text=i + 1,
            loc="center",
            edgecolor="none",
            facecolor="none",
        )
    ax.add_table(tb)

    plt.suptitle(f"State values after {n_iterations} iterations")


def compute_state_value(in_place=True, discount=1.0, draw_values=False):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        # For each state
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                # Compute the new value of the state by summing for each possible action
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (
                        reward + discount * state_values[next_i, next_j]
                    )
                new_state_values[i, j] = value

        iteration += 1

        if draw_values:
            draw_image(np.round(new_state_values, decimals=2), n_iterations=iteration)
            plt.show()

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

    return new_state_values, iteration


def figure_4_1():
    # While the authors suggest using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    _, async_iterations = compute_state_value(in_place=True)
    values, sync_iterations = compute_state_value(in_place=False)

    print(f"In-place: {async_iterations} iterations")
    print(f"Synchronous: {sync_iterations} iterations")

    draw_image(np.round(values, decimals=2), n_iterations=sync_iterations)
    plt.show()


if __name__ == "__main__":
    figure_4_1()
