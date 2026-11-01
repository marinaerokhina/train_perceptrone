"""Reproducible demonstration of a single-neuron binary classifier.

Run: python -m src.start --no-plot
"""
import argparse
import numpy as np

from src.perceptone import SimplePerceptone


def training_data():
    # Complete binary truth table: scheduled AND (kit available OR motivation).
    samples = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=float)
    labels = ((samples[:, 0] == 1)
              & ((samples[:, 1] == 1) | (samples[:, 2] == 1))).astype(int)
    return samples, labels


def main():
    parser = argparse.ArgumentParser(description="Single sigmoid neuron (educational)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable interactive Matplotlib graph")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save training-loss chart to a PNG file")
    args = parser.parse_args()

    samples, targets = training_data()
    model = SimplePerceptone(random_state=42)
    history = model.train(samples, targets, args.learning_rate, args.epochs)
    predictions = model.predict_classes(samples)

    print("Input -> expected / predicted")
    for features, target, predicted in zip(samples.astype(int), targets, predictions):
        print(f"{features.tolist()} -> {int(target)} / {int(predicted)}")
    print(f"Accuracy on the complete truth table: {np.mean(predictions == targets):.2%}")
    print(f"Binary cross-entropy: {history[0]:.5f} -> {history[-1]:.5f}")

    if args.save_plot or not args.no_plot:
        import matplotlib.pyplot as plt
        plt.plot(history, label="Binary cross-entropy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"Saved chart: {args.save_plot}")
        if not args.no_plot:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
