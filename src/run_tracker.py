import pandas
import json
import os
from datetime import datetime

RESULTS_FILE = "run_history.csv"

def save_run(results: dict, notes: str = ""):
    #Append a single model evaluation result to the CSV history
    # Flatten the confusion matrix to a JSON string so it fits in one cell
    row = {key: value for key, value in results.items() if key != "confusion_matrix"}
    row["confusion_matrix"] = json.dumps(results.get("confusion_matrix", []))
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["notes"] = notes

    dataFrame_new = pandas.DataFrame([row])

    if os.path.exists(RESULTS_FILE):
        dataFrame_existing = pandas.read_csv(RESULTS_FILE)
        dataFrame_combined = pandas.concat([dataFrame_existing, dataFrame_new], ignore_index=True)
    else:
        dataFrame_combined = dataFrame_new

    dataFrame_combined.to_csv(RESULTS_FILE, index=False)
    print(f"Run saved to {RESULTS_FILE}")

def load_runs() -> pandas.DataFrame:
    # Load all historical runs into a DataFrame
    if not os.path.exists(RESULTS_FILE):
        print("No run history found.")
        return pandas.DataFrame()
    return pandas.read_csv(RESULTS_FILE)