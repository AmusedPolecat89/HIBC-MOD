import json
import numpy as np

NUM_VECTORS = 100
VECTOR_DIM = 512
OUTPUT_FILE = "data.jsonl"

print(f"Generating {NUM_VECTORS} records for testing...")

with open(OUTPUT_FILE, "w") as f:
    for i in range(NUM_VECTORS):
        record = {
            "id": f"doc_{i:03}",
            "vector": np.random.rand(VECTOR_DIM).astype(np.float32).tolist(),
            "metadata": {
                "source": f"source_{i % 10}",
                "timestamp": 1678886400 + i
            }
        }
        f.write(json.dumps(record) + "\n")

print(f"✅ Sample data written to '{OUTPUT_FILE}'.")
