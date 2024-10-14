import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


def run_simple_describe():
    np.random.seed(42)

    # Init data
    ran_gen = np.random
    data = 50 * ran_gen.randn(1000) + 200
    print(data)

    # Data frame
    df = pd.DataFrame()

    # Describe data
    df['data'] = data
    df.describe()

    # Show distribution
    plt.hist(data, 18)

    plt.show()


def skew_distribution():
    _, ax = plt.subplots(1, 1)

    # Initialize distribution parameters:
    a = 4  # Parameter of the skewed normal distribution.
    _, _, _, _ = skewnorm.stats(a, moments='mvsk')

    # Create array x in 1% -> 99% percentile value
    x = np.linspace(skewnorm.ppf(0.01, a),
                    skewnorm.ppf(0.99, a), 100)

    # Calculate the probability density function (PDF) for values of x with parameter a
    ax.plot(x, skewnorm.pdf(x, a),
            'r-', lw=5, alpha=0.6, label='skewnorm pdf')

    # Creates a "frozen" skewed normal distribution object with parameter a
    rv = skewnorm(a)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
    np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a))

    r = skewnorm.rvs(a, size=1000) * 50 + 50
    plt.hist(r, 18)

    df = pd.DataFrame()
    df['data'] = r
    df.describe()

    plt.show()


skew_distribution()
