
import json
import os

PATH = r"D:\VMTIDE\MTI-EVO\data\cortex_dump.json"

if not os.path.exists(PATH):
    print("File not found.")
    exit()

with open(PATH, 'r') as f:
    data = json.load(f)

if isinstance(data, list):
    print(f"Found LIST with {len(data)} items. Converting to DICT...")
    new_data = {}
    for item in data:
        seed = str(item.get("seed", 0))
        # Ensure 'weights' exist or are shimmed?
        # Core expects 'weights'. If missing, it uses default.
        if "weights" not in item:
            item["weights"] = [0.0] * 384 # Shim
        new_data[seed] = item
    
    with open(PATH, 'w') as f:
        json.dump(new_data, f, indent=2)
    print("✅ Conversion Complete. Saved as DICT.")
else:
    print("Already a DICT. No action needed.")
