import math
import matplotlib.pyplot as plt
import torch


def ground_truth(x):
    """Return the value for a specific input x of the function we're trying to fit"""

    # f(x) = sin(2 * pi * x)
    return torch.sin(2 * torch.pi * x)


def create_dataset(N):
    """Generate a sinusoidal dataset of N samples and targets"""

    # Generate a vector of N samples spaced uniformly in the [0,1] range
    samples = torch.linspace(start=0, end=1, steps=N, dtype=torch.float64)

    # Create a vector of N targets associated to samples by adding Gaussian noise to ground truth value
    targets = ground_truth(samples) + torch.randn(N) * 0.3

    return samples, targets


def add_powers(x, M):
    """Create a matrix with powers from 0 to M of x as columns"""

    # Adding biases (power=0 gives a column of ones) as first column simplifies later linear algebra computations
    return torch.vander(x=x, N=M + 1, increasing=True)


def fit(X, t, lambda_reg=0):
    """Fit a linear model to a dataset and return the optimal weights"""

    # Construct the regularization matrix
    reg_matrix = torch.eye(X.shape[1]) * lambda_reg
    reg_matrix[0, 0] = 0  # Don't regularize bias

    # Solve the X.w = t equations system in closed form.
    # A more efficient but less explicit alternative would be to use torch.linalg.lstsq or torch.linalg.solve
    term1 = (reg_matrix + X.T @ X).inverse()
    term2 = X.T @ t
    return term1 @ term2


def rms_error(X, w, t):
    """Compute the Root Mean Squared Error"""

    y = X @ w
    return torch.sqrt(((y - t) ** 2).mean())


def plot_fit(plot, x, w, t, M):
    """Plot the fit of a linear model to a dataset"""

    # Plot inputs and targets as dots
    plot.scatter(x, t, color="blue", label="Training samples")

    x_plot = torch.linspace(start=0, end=1, steps=100, dtype=torch.float64)

    # Plot underlying function
    plot.plot(
        x_plot,
        ground_truth(x_plot),
        color="green",
        linestyle="dashed",
        label="Underlying function",
    )

    # Plot model predictions
    X_plot = add_powers(x=x_plot, M=M)
    plot.plot(
        x_plot,
        X_plot @ w,
        color="red",
        label=f"Model fit (M={M})",
    )


def plot_figure_1_6(x, t):
    """Reproduce figure 1.6 of DLFC book: impact of model complexity on fitting"""

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.suptitle("Impact of model complexity on fitting")

    # Init subplots coordinates
    fig_row = fig_col = 0

    # Plot model fit for various orders of the polynomial model
    for M in (0, 1, 3, 9):
        # Create the training matrix
        X_train = add_powers(x=samples, M=M)

        # Fit model to training set without any regularization for now
        w_best = fit(X=X_train, t=targets)
        print(w_best)

        # Plot inputs and targets as dots
        axs[fig_row, fig_col].scatter(
            samples, targets, color="blue", label="Training samples"
        )

        # Create subplot with model fit for polynomial order M
        plot_fit(plot=axs[fig_row, fig_col], x=samples, w=w_best, t=t, M=M)
        axs[fig_row, fig_col].set_title(f"M = {M}")

        # Increment coordinates of next subplot
        fig_col += 1
        if fig_col == 2:
            fig_row += 1
            fig_col = 0

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_figure_1_7(x, t, N_test=100):
    """Reproduce figure 1.7 of DLFC book: impact of model complexity on training and test errors"""

    # Create the test set
    samples_test, targets_test = create_dataset(N=N_test)

    # Init histories
    history_train = []
    history_test = []

    # Compute training and test errors for various orders of the polynomial model
    for M in range(10):
        # Create the training matrix
        X_train = add_powers(x=x, M=M)

        # Fit model to training set without any regularization
        w_best = fit(X=X_train, t=t)

        # Record error on training set
        history_train.append(rms_error(X=X_train, w=w_best, t=t))

        # Create the test matrix
        X_test = add_powers(x=samples_test, M=M)

        # Record error on test set
        history_test.append(rms_error(X=X_test, w=w_best, t=targets_test))

    # Plot histories
    plt.plot(history_train, marker="o", label="Training")
    plt.plot(history_test, marker="^", label="Test")

    plt.xlabel("M")
    plt.ylabel("RMS error")
    plt.title("Impact of model complexity on training and test errors")
    plt.legend()
    plt.show()


def plot_figure_1_8(M=9):
    """Reproduce figure 1.8 of DLFC book: impact of dataset size on overfitting"""

    fig, axs = plt.subplots(ncols=2, figsize=(11, 5))
    fig.suptitle(f"Impact of dataset size on overfitting (M = {M})")

    # Init plot column
    fig_col = 0

    # Plot model fit for various dataset sizes
    for N in (10, 100):
        # Create the training set
        samples, targets = create_dataset(N=N)

        # Create the training matrix
        X_train = add_powers(x=samples, M=M)

        # Fit model to training set without any regularization for now
        w_best = fit(X=X_train, t=targets)

        # Create subplot with model fit for dataset size N
        plot_fit(plot=axs[fig_col], x=samples, w=w_best, t=targets, M=M)
        axs[fig_col].set_title(f"N = {N}")

        # Increment column of next plot
        fig_col += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_figure_1_9(x, t, M=9):
    """Reproduce figure 1.8 of DLFC book: impact of regularization on overfitting"""

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    fig.suptitle(f"Impact of regularization on overfitting (M = {M})")

    # Init plot column
    fig_col = 0

    # Create the training matrix
    X_train = add_powers(x=x, M=M)

    # Plot model fit for various values of regularization factor
    for lambda_reg in (0, math.exp(-10), 1):
        # Fit model to training set with regularization
        w_best = fit(X=X_train, t=t, lambda_reg=lambda_reg)

        # Create subplot with model fit for lambda value
        plot_fit(plot=axs[fig_col], x=samples, w=w_best, t=t, M=M)
        ln_lambda = f"{math.log(lambda_reg):.0f}" if lambda_reg > 0 else "-Infinity"
        axs[fig_col].set_title(f"ln λ = {ln_lambda}")

        # Increment column of next plot
        fig_col += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_figure_1_10(x, t, N_test=100, M=9):
    """Reproduce figure 1.10 of DLFC book: impact of regularization on training and test errors"""

    # Create the training and test sets
    samples_test, targets_test = create_dataset(N=N_test)

    # Create the training matrix
    X_train = add_powers(x=x, M=M)

    # Create the test matrix
    X_test = add_powers(x=samples_test, M=M)

    # Init histories
    history_lambda = []
    history_train = []
    history_test = []

    # Compute training and test errors for various values of regularization factor
    for lambda_exponent in range(-30, 1, 2):
        lambda_reg = math.exp(lambda_exponent)

        # Fit model to training set with regularization
        w_best = fit(X=X_train, t=t, lambda_reg=lambda_reg)

        # Record error on training set
        history_train.append(rms_error(X=X_train, w=w_best, t=t))

        # Record error on test set
        history_test.append(rms_error(X=X_test, w=w_best, t=targets_test))

        # Record value of regularization factor
        history_lambda.append(lambda_exponent)

    # Plot histories
    plt.plot(history_lambda, history_train, marker="o", label="Training")
    plt.plot(history_lambda, history_test, marker="^", label="Test")

    plt.xlabel("ln λ")
    plt.ylabel("RMS error")
    plt.title(f"Impact of regularization on training and test errors (M = {M})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Sets the seed for generating random numbers in order to obtain reproducible results
    torch.manual_seed(6)

    # Create the reference training set
    samples, targets = create_dataset(N=10)

    plot_figure_1_6(x=samples, t=targets)
    plot_figure_1_7(x=samples, t=targets)
    plot_figure_1_8()
    plot_figure_1_9(x=samples, t=targets)
    plot_figure_1_10(x=samples, t=targets)
