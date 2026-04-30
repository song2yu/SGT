# check_text_encoder_keys.py
from safetensors.torch import load_file

ckpt_path = "experiments_8gpus/ft_panoptic/checkpoint-1500/model.safetensors"
state_dict = load_file(ckpt_path)

print("=== All text_encoder keys ===")
te_keys = [k for k in state_dict.keys() if k.startswith("text_encoder.")]

# Count distinct prefixes
prefixes = {}
for k in te_keys:
    # Drop the first segment after `text_encoder.`
    rest = k[len("text_encoder."):]
    parts = rest.split(".")
    prefix = ".".join(parts[:3])  # Take the first three levels
    if prefix not in prefixes:
        prefixes[prefix] = 0
    prefixes[prefix] += 1

print(f"\nTotal text_encoder keys: {len(te_keys)}")
print("\n=== Key prefixes (first 3 levels) ===")
for prefix, count in sorted(prefixes.items()):
    print(f"  {prefix}: {count}")

# Check whether there are any `visual`-related entries
print("\n=== Visual related keys ===")
visual_keys = [k for k in te_keys if "visual" in k.lower()]
print(f"Count: {len(visual_keys)}")
for k in visual_keys[:10]:
    print(f"  {k}")