# Sweep label-noise rate and training size for a simple expression classifier.
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def make_dataset(n_samples, n_features=40, n_informative=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    means = np.zeros((2, n_features))
    means[1, :n_informative] = 1.5
    X = rng.normal(size=(n_samples, n_features)) + means[y]
    y_noisy = y.copy()
    if noise > 0:
        flip = rng.random(n_samples) < noise
        y_noisy[flip] = 1 - y_noisy[flip]
    return X, y_noisy, y


def evaluate(n_samples, noise, seed=0):
    X, y_train_labels, y_true = make_dataset(n_samples, noise=noise, seed=seed)
    X_tr, X_te, y_tr, _, _, y_te_true = train_test_split(
        X, y_train_labels, y_true, test_size=0.3, random_state=seed
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    # Score against the CLEAN test labels, because that is the real target.
    return clf.score(X_te, y_te_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    acc = evaluate(args.n_samples, args.noise, args.seed)
    print("n_samples=" + str(args.n_samples) + " noise=" + str(args.noise))
    print("clean-label test accuracy: " + str(round(acc, 4)))


if __name__ == "__main__":
    main()
