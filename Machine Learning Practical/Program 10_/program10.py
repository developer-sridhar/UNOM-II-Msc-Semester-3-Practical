import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

# Radial kernel (Gaussian) function
def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))

# Locally weighted regression function
def local_regression(x0, X, Y, tau):
    x0 = np.r_[1, x0]  # Add bias term
    X = np.c_[np.ones(len(X)), X]  # Add bias term to dataset

    # Compute weights using radial kernel
    xw = X.T * radial_kernel(x0, X, tau)  # Weighted X transpose

    # Compute beta using normal equations
    beta = np.linalg.pinv(xw @ X) @ xw @ Y  # Matrix multiplication

    # Predict value
    return x0 @ beta  # Matrix multiplication

# Generate dataset
n = 1000
X = np.linspace(-3, 3, num=n)
print("The Data Set (10 Samples) X:\n", X[1:10])

Y = np.log(np.abs(X ** 2 - 1) + 0.5)
print("The Fitting Curve Data Set (10 Samples) Y:\n", Y[1:10])

# Jitter X
X += np.random.normal(scale=0.1, size=n)
print("Normalized (10 Samples) X:\n", X[1:10])

# Domain for prediction
domain = np.linspace(-3, 3, num=300)
print("X0 Domain Space (10 Samples):\n", domain[1:10])

# Plotting function
def plot_lwr(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plot = figure(width=400, height=400)
    plot.title.text = f'tau={tau}'
    plot.scatter(X, Y, alpha=0.3)
    plot.line(domain, prediction, line_width=2, color='red')
    return plot

# Display plots for different tau values
show(gridplot([
    [plot_lwr(10.0), plot_lwr(1.0)],
    [plot_lwr(0.1), plot_lwr(0.01)]
]))