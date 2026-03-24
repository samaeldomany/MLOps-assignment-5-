import mlflow
import sys
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

if accuracy < 0.85:
    print(f"Accuracy {accuracy:.4f} failed")
    sys.exit(1)
else:
    print(f"Accuracy {accuracy:.4f} passed")
    sys.exit(0)