import json
import csv
import os

def save_metrics_to_json(metrics, filename):
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing the metrics to save.
        filename (str): The name of the file to save the metrics to.
    """
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print('Metrics saved to {}'.format(filename))

def save_metrics_to_csv(metrics, filename):
    """
    Save metrics to a CSV file.

    Args:
        metrics (dict): Dictionary containing the metrics to save.
        filename (str): The name of the file to save the metrics to.
    """
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print('Metrics saved to {}'.format(filename))