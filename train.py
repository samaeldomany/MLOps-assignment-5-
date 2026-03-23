import random

print("Training model and logging to MLflow...")

# Simulate a unique MLflow Run ID
mock_run_id = f"run_{random.randint(1000, 9999)}"

# Export the Run ID to a text file for the next job
with open("model_info.txt", "w") as f:
    f.write(mock_run_id)

print(f"Success! Run ID {mock_run_id} saved to model_info.txt")