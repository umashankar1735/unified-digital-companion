# scripts/7_logging.py
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "escalations.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)
if not LOG.exists():
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","query","confidence","retrieved_snippet"])

def log_escalation(query, confidence, snippet):
    import datetime
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.utcnow().isoformat(), query, confidence, snippet])