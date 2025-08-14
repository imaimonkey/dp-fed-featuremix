# experiments/analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def load_results(folder_path="experiments/results"):
    records = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            exp_name = file.replace(".csv", "")
            final_acc = (
                df["test_accuracy"].dropna().values[-1]
                if "test_accuracy" in df.columns
                else None
            )
            records.append((exp_name, final_acc))

    # 정확도 기준 내림차순 정렬
    records = [r for r in records if r[1] is not None]
    records.sort(key=lambda x: x[1], reverse=True)

    return records


def plot_results(results):
    exp_names = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(exp_names, accuracies, color="skyblue")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Comparison of Experimental Results")
    plt.ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 1,
            f"{acc:.2f}%",
            ha="center",
            fontsize=10,
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("experiments/results/accuracy_comparison.png")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    if results:
        plot_results(results)
    else:
        print("No results found.")
