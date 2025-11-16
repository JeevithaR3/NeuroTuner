# tuner.py
import optuna, json, csv, os
from trainer import run_training
from utils import append_result

def objective(trial):
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 1
    }
    metrics = run_training(config)
    score = metrics["accuracy"] - 0.001 * metrics["latency_ms"]  # tradeoff factor
    result = {**config, **metrics, "score": score}
    append_result(result)   # write to results.csv/json
    print(f"Trial {trial.number}: {result}")
    return score

def run_tuning(n_trials=5):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params
    # Save best
    with open("best_config.json", "w") as f:
        json.dump({"best": best, "value": study.best_value}, f)
    return best

if __name__ == "__main__":
    print(run_tuning(5))
