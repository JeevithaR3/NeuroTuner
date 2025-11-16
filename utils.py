# utils.py
import csv, json, os

CSV_FILE = "results.csv"
JSON_FILE = "results.json"

def append_result(d):
    # to CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(d.keys()))
        if not file_exists: writer.writeheader()
        writer.writerow(d)
    # to JSON (append list)
    if os.path.isfile(JSON_FILE):
        with open(JSON_FILE, "r") as f: data = json.load(f)
    else:
        data = []
    data.append(d)
    with open(JSON_FILE, "w") as f: json.dump(data, f, indent=2)
