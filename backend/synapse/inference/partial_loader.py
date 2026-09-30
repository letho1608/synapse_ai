"""
Load only the parts of a causal LM a node serves.

`from_pretrained` materializes every layer, so before this every node held
the whole model and "pooling memory" was not real. Here the module skeleton
is built on the meta device (free), and only what a shard needs is
allocated and filled from the checkpoint's safetensors files:

- decoder layers [start, end]      (always)
- token (+ GPT-2 position) embedding (first stage)
- final norm + LM head               (last stage)
- rotary embedding                   (tiny, rebuilt from the config)

Layers are added on demand (`ensure`), so the same holder serves a new
range after a re-partition or a replica request, and `release` frees layers
the node no longer serves.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from synapse.helpers import DEBUG


def _hub_file(repo_id: str, filename: str) -> Optional[str]:
    """Local path of one repo file: cache first, then download unless offline."""
    if os.path.isdir(repo_id):
        p = os.path.join(repo_id, filename)
        return p if os.path.exists(p) else None
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except Exception:
        pass
    if os.getenv("HF_HUB_OFFLINE") == "1":
        return None
    try:
        return hf_hub_download(repo_id, filename)
    except Exception:
        return None


def checkpoint_index(repo_id: str) -> Optional[Dict[str, str]]:
    """tensor name -> local safetensors file, or None (no safetensors)."""
    idx = _hub_file(repo_id, "model.safetensors.index.json")
    if idx:
        with open(idx, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        out: Dict[str, str] = {}
        paths: Dict[str, Optional[str]] = {}
        for name, fn in weight_map.items():
            if fn not in paths:
                paths[fn] = _hub_file(repo_id, fn)
            if paths[fn] is None:
                return None
            out[name] = paths[fn]
        return out
    single = _hub_file(repo_id, "model.safetensors")
    if not single:
        return None
    from safetensors import safe_open
    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


class PartialModel:
    """The pieces of one causal LM that this node has materialized."""

    def __init__(self, repo_id: str, dtype: torch.dtype, device: str,
                 attn_implementation: str = "eager", quant_mode: str = "none"):
        from transformers import AutoConfig
        self.repo_id = repo_id
        self.dtype = dtype
        self.device = device
        self.quant_mode = quant_mode
        self.config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        try:
            self.config._attn_implementation = attn_implementation
        except Exception:
            pass
        self.n_layers = int(self.config.num_hidden_layers)
        self.index = checkpoint_index(repo_id)
        if self.index is None:
            raise FileNotFoundError(f"{repo_id}: no safetensors checkpoint")
        self.layers: Dict[int, nn.Module] = {}
        self.embed: Optional[nn.Module] = None
        self.pos_embed: Optional[nn.Module] = None
        self.final_norm: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.rotary: Optional[nn.Module] = None
        self._skeleton = self._build_skeleton(1, None)
        self._layers_path = self._find_layers_path(self._skeleton)
        self._build_rotary()

    # -- skeleton / naming -------------------------------------------------
    def _build_skeleton(self, n_layers: int, layer_ids: Optional[List[int]]):
        from transformers import AutoModelForCausalLM
        cfg = copy.deepcopy(self.config)
        cfg.num_hidden_layers = n_layers
        types = getattr(cfg, "layer_types", None)
        if isinstance(types, list) and layer_ids is not None:
            cfg.layer_types = [types[i] for i in layer_ids]
        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    @staticmethod
    def _find_layers_path(model) -> str:
        best = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                if best is None or name.count(".") < best.count("."):
                    best = name
        if best is None:
            raise RuntimeError("cannot find the decoder layer list")
        return best

    def _ckpt_name(self, name: str) -> Optional[str]:
        """Map a module-path tensor name to the checkpoint's name."""
        if name in self.index:
            return name
        prefix = getattr(self._skeleton, "base_model_prefix", "") or ""
        if prefix and name.startswith(prefix + "."):
            alt = name[len(prefix) + 1:]
            if alt in self.index:
                return alt
        if prefix and (prefix + "." + name) in self.index:
            return prefix + "." + name
        return None

    def _load(self, module: nn.Module, module_path: str, rename=None) -> None:
        """to_empty + fill every parameter/persistent buffer from the checkpoint."""
        from safetensors import safe_open
        module.to_empty(device=self.device)
        module.to(dtype=self.dtype)  # floating tensors only
        state = {}
        by_file: Dict[str, List[tuple]] = {}
        for local in module.state_dict().keys():
            full = f"{module_path}.{local}" if module_path else local
            if rename is not None:
                full = rename(full)
            ck = self._ckpt_name(full)
            if ck is None:
                raise KeyError(f"{self.repo_id}: tensor {full!r} not in checkpoint")
            by_file.setdefault(self.index[ck], []).append((local, ck))
        for path, pairs in by_file.items():
            with safe_open(path, "pt") as f:
                for local, ck in pairs:
                    state[local] = f.get_tensor(ck)
        ref = module.state_dict()
        for k, v in state.items():
            state[k] = v.to(dtype=ref[k].dtype)
        module.load_state_dict(state, strict=True)
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    def _sub(self, model, path: str):
        mod = model
        for part in path.split("."):
            mod = getattr(mod, part)
        return mod

    def _path_of(self, model, target) -> Optional[str]:
        for name, mod in model.named_modules():
            if mod is target:
                return name
        return None

    def _build_rotary(self) -> None:
        sk = self._skeleton
        rot = None
        for name, mod in sk.named_modules():
            if name.endswith("rotary_emb"):
                rot = mod
                break
        if rot is None:
            return
        try:
            self.rotary = type(rot)(config=self.config, device=self.device)
        except TypeError:
            self.rotary = type(rot)(self.config)
            self.rotary.to(self.device)

    # -- public API -------------------------------------------------------
    def ensure(self, start: int, end: int, first: bool, last: bool) -> None:
        """Materialize layers [start, end] plus the ends this shard needs."""
        missing = [i for i in range(start, end + 1) if i not in self.layers]
        if missing:
            sk = self._build_skeleton(len(missing), missing)
            layer_list = self._sub(sk, self._layers_path)
            for local, absolute in enumerate(missing):
                layer = layer_list[local]
                for mod in layer.modules():
                    if hasattr(mod, "layer_idx"):
                        mod.layer_idx = absolute  # KV cache slot = absolute index
                src = f"{self._layers_path}.{local}"
                dst = f"{self._layers_path}.{absolute}"
                self._load(layer, src, rename=lambda n, s=src, d=dst: d + n[len(s):])
                if self.quant_mode != "none":
                    from synapse.inference.weight_quant import quantize_module
                    quantize_module(layer, self.quant_mode)
                self.layers[absolute] = layer
            if DEBUG >= 1:
                print(f"[PartialModel] loaded layers {missing[0]}-{missing[-1]} "
                      f"({len(missing)}/{self.n_layers}) quant={self.quant_mode}")
        sk = self._skeleton
        if first and self.embed is None:
            emb = sk.get_input_embeddings()
            self._load(emb, self._path_of(sk, emb))
            self.embed = emb
            wpe = getattr(getattr(sk, "transformer", None), "wpe", None)
            if wpe is not None:
                self._load(wpe, self._path_of(sk, wpe))
                self.pos_embed = wpe
        if last and self.lm_head is None:
            norm = self._final_norm_module(sk)
            if norm is not None:
                self._load(norm, self._path_of(sk, norm))
                self.final_norm = norm
            head = sk.get_output_embeddings()
            head_path = self._path_of(sk, head)
            tied = self._ckpt_name(f"{head_path}.weight") is None
            if tied and self.embed is not None and getattr(head, "bias", None) is None:
                head.weight = self.embed.weight  # share, don't copy
            elif tied:
                # Tied embeddings: the head reuses the input embedding matrix.
                emb_path = self._path_of(sk, sk.get_input_embeddings())
                self._load(head, head_path,
                           rename=lambda n, hp=head_path, ep=emb_path: ep + n[len(hp):])
            else:
                self._load(head, head_path)
            self.lm_head = head

    @staticmethod
    def _final_norm_module(sk):
        base = getattr(sk, sk.base_model_prefix, None) if getattr(sk, "base_model_prefix", None) else None
        for owner in (base, sk):
            if owner is None:
                continue
            for attr in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):
                mod = getattr(owner, attr, None)
                if isinstance(mod, nn.Module):
                    return mod
        return None

    def release(self, keep: Iterable[int]) -> List[int]:
        """Drop layers not in `keep`; returns the freed indices."""
        keep = set(keep)
        freed = [i for i in list(self.layers) if i not in keep]
        for i in freed:
            del self.layers[i]
        return freed

    def loaded_bytes(self) -> int:
        from synapse.inference.weight_quant import module_bytes
        mods = list(self.layers.values()) + [m for m in (
            self.embed, self.pos_embed, self.final_norm, self.lm_head, self.rotary) if m is not None]
        seen, total = set(), 0
        for m in mods:
            for t in list(m.parameters()) + list(m.buffers()):
                if id(t) in seen:
                    continue
                seen.add(id(t))
                total += t.numel() * t.element_size()
        return total

    def layer_param_count(self) -> int:
        """Parameters in one decoder layer (from a meta skeleton, free)."""
        layer = self._sub(self._skeleton, self._layers_path)[0]
        return sum(p.numel() for p in layer.parameters())


class LayerView:
    """Sequence-like view over loaded layers indexed by absolute position."""

    def __init__(self, pm: PartialModel):
        self._pm = pm

    def __len__(self) -> int:
        return self._pm.n_layers

    def __getitem__(self, i: int) -> nn.Module:
        try:
            return self._pm.layers[i]
        except KeyError:
            raise RuntimeError(f"layer {i} is not loaded on this node") from None
