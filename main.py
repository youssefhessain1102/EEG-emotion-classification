import yaml

from src.predict import run_inference
from src.train import run_training


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Train pipeline
    _, _, best_mask, final_model = run_training(config)
    print(best_mask[-14:])

    # Predict & submission pipeline
    run_inference(config, final_model, best_mask)


if __name__ == "__main__":
    main()