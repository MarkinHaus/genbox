"""
genbox — local AI generation toolkit
usage: from genbox import pipeline, models, config
"""
from genbox.config import cfg
from genbox import models
# pipeline imported lazily to avoid heavy ML imports at CLI startup


if __name__ == "__main__":
    from safetensors import safe_open

    path = r"C:\Users\Markin\AppData\Roaming\.genbox\models\sd15\Lakis-Flux-000002.safetensors"
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    print([k for k in keys if "cond" in k or "clip" in k.lower() or "text" in k.lower()][:5])
    print("Total keys:", len(keys))
    print(keys[:10])