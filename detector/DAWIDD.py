# -*- coding: utf-8 -*-
import numpy as np
# -*- coding: utf-8 -*-
from scipy.stats import ttest_rel
from sklearn.svm import SVC


def svm_independence_test(X, y, n_itr=10, p_val=0.00001, n_sel=50):
    Z = np.concatenate((X, np.linspace(0, 1, X.shape[0]).reshape(-1, 1)), axis=1)
    svm = SVC(gamma=2, C=1, kernel="rbf")
    s1, s2 = [], []
    for _ in range(n_itr):
        sel = np.random.choice(range(Z.shape[0]), size=min(n_sel, int(2 * Z.shape[0] / 3)), replace=False)
        if len(np.unique(y[sel])) == 1:  # Number classes has to be greater than one!
            continue

        svm.fit(Z[sel], y[sel])
        s1.append(svm.score(Z, y))
        s2.append(svm.score(np.concatenate((X, np.random.random(X.shape[0]).reshape(-1, 1)), axis=1), y))

    if len(s1) == 0 or len(s2) == 0:
        return True
    elif (np.array(s1) - np.array(s2)).var() == 0:
        return abs(np.mean(s1) - np.mean(s2)) < 0.000001
    else:
        return ttest_rel(s1, s2)[1] > p_val


def test_independence(X, Y, Z=None):
    return svm_independence_test(X, Y)


class DAWIDD():
    """
    Implementation of the dynamic-adapting-window-independence-drift-detector (DAWIDD)
    
    Parameters
    ----------
    max_window_size : int, optional
        The maximal size of the window. When reaching the maximal size, the oldest sample is removed.

        The default is 90
    min_window_size : int, optional
        The minimal number of samples that is needed for computing the hypothesis test.

        The default is 70
    min_p_value : int, optional
        The threshold of the p-value - not every test outputs a p-value (sometimes only 1.0 <=> independent and 0.0 <=> not independent are returned)

        The default is 0.001
    """

    def __init__(self, max_window_size=90, min_window_size=70, min_p_value=0.001):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_p_value = min_p_value

        self.X = []
        self.n_items = 0
        self.min_n_items = self.min_window_size / 4.

        self.drift_detected = False

    # You have to overwrite this function if you want to use a different test for independence
    def _test_for_independence(self):
        t = np.array(range(self.n_items)) / (1. * self.n_items)
        t /= np.std(t)
        t = t.reshape(-1, 1)

        X = np.array(self.X)
        X_ = X[:, :-1].reshape(X.shape[0], -1)
        Y = X[:, -1].reshape(-1, 1)
        return test_independence(X_, Y.ravel())

    def set_input(self, x):
        self.add_batch(x)

        return self.detected_change()

    def add_batch(self, x):
        self.drift_detected = False

        # Add item
        self.X.append(x.flatten())
        self.n_items += 1

        # Is buffer full?
        if self.n_items > self.max_window_size:
            self.X.pop(0)
            self.n_items -= 1

        # Enough items for testing for drift?
        if self.n_items >= self.min_window_size:
            # Test for drift
            p = self._test_for_independence()

            if p <= self.min_p_value:
                self.drift_detected = True

                # Remove samples until no drift is present!
                while p <= self.min_p_value and self.n_items >= self.min_n_items:
                    # Remove old samples
                    self.X.pop(0)
                    self.n_items -= 1

    def detected_change(self):
        return self.drift_detected
