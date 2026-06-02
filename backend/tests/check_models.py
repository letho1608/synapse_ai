import sys
sys.path.insert(0, 'D:/Code/synapse_ai')

from synapse.model_list import HF_MODELS, HF_MODEL_LAYERS, resolve_hf_id

print(f"Total models: {len(HF_MODELS)}")
print()
print("Sample models:")
for k,v in list(HF_MODELS.items())[20:35]:
    print(f"  {k!r} -> {v!r}")
print()

for inp in ['Qwen/Qwen2.5-0.5B-Instruct', 'qwen2.5:0.5b', 'HuggingFaceTB/SmolLM-135M-Instruct', 'Qwen/Qwen2.5-0.5B']:
    r = resolve_hf_id(inp)
    in_hf = r in HF_MODELS
    layers = HF_MODEL_LAYERS.get(r, "N/A")
    print(f"resolve_hf_id({inp!r}) = {r!r}")
    print(f"  in HF_MODELS: {in_hf}, layers: {layers}")