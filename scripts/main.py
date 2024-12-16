import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    args = parser.parse_args()

    if args.preprocess:
        subprocess.run(["python", "scripts/preprocess.py"])

    if args.train:
        subprocess.run(["python", "scripts/train.py"])

    if args.evaluate:
        subprocess.run(["python", "scripts/evaluate.py"])