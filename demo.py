# Self-contained demo: label quality vs quantity for a toy expression classifier.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    X, y_labels, y_true = make_dataset(n_samples, noise=noise, seed=seed)
    X_tr, X_te, y_tr, _, _, y_te_true = train_test_split(
        X, y_labels, y_true, test_size=0.3, random_state=seed
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te_true)


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    sizes = [200, 500, 1000, 2000, 4000, 8000]
    noise_levels = [0.0, 0.1, 0.25, 0.4]

    records = []
    plt.figure(figsize=(8, 5))
    for noise in noise_levels:
        accs = []
        for n in sizes:
            reps = [evaluate(n, noise, seed=s) for s in range(5)]
            acc = float(np.mean(reps))
            accs.append(acc)
            records.append({"n_samples": n, "noise": noise, "clean_test_accuracy": round(acc, 4)})
        plt.plot(sizes, accs, marker="o", label="noise=" + str(noise))

    plt.xscale("log")
    plt.xlabel("Training samples (log scale)")
    plt.ylabel("Accuracy on clean test labels")
    plt.title("Scale does not buy back label quality")
    plt.legend(title="Label noise rate")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/demo.png", dpi=120)
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/summary.csv", index=False)

    print("Wrote figures/demo.png and results/summary.csv")
    ceiling = df[df["n_samples"] == max(sizes)]
    print("Accuracy ceiling at largest sample size, by noise level:")
    print(ceiling[["noise", "clean_test_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
