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

Downloads follow the same rule: for a sharded checkpoint only the
safetensors files that hold this node's tensors are fetched, when they are
first needed (a new range after a re-plan fetches just what is missing).
A single-file checkpoint that is not cached yet is read with HTTP range
requests: the header, then only the byte ranges of this node's tensors
(RangeFile). SYNAPSE_RANGE_FETCH=0 downloads whole files instead.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, Iterable, List, Optional

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


# safetensors dtype tag -> torch dtype
_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
# Ranges closer than this are fetched in one request (fewer round trips).
_RANGE_MERGE_GAP = 1 << 20


def _remote_url(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo_id, filename)


class RangeFile:
    """A remote safetensors file read with HTTP range requests.

    Only the header and the byte ranges of requested tensors cross the
    network; nothing is written to disk (the node keeps the tensors it
    loads in memory, a re-plan fetches the new range again)."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = dict(headers or {})
        self.bytes_fetched = 0
        head = self._get(0, 7)
        n = int.from_bytes(head, "little")
        meta = json.loads(self._get(8, 8 + n - 1))
        meta.pop("__metadata__", None)
        self.base = 8 + n
        self.meta = meta  # name -> {dtype, shape, data_offsets}

    def _get(self, start: int, end: int) -> bytes:
        import requests
        r = requests.get(self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"},
                         timeout=60, allow_redirects=True)
        r.raise_for_status()
        if r.status_code != 206:
            raise IOError(f"{self.url}: server ignored the range request")
        data = r.content
        if len(data) != end - start + 1:
            raise IOError(f"{self.url}: short range read {len(data)} != {end - start + 1}")
        self.bytes_fetched += len(data)
        return data

    def keys(self) -> List[str]:
        return list(self.meta)

    def get_tensors(self, names: List[str]) -> Dict[str, torch.Tensor]:
        """Fetch several tensors, merging nearby ranges into one request."""
        spans = sorted((self.meta[n]["data_offsets"][0], self.meta[n]["data_offsets"][1], n)
                       for n in names)
        out: Dict[str, torch.Tensor] = {}
        i = 0
        while i < len(spans):
            a, b, _ = spans[i]
            j = i
            while j + 1 < len(spans) and spans[j + 1][0] - b <= _RANGE_MERGE_GAP:
                j += 1
                b = max(b, spans[j][1])
            blob = self._get(self.base + a, self.base + b - 1) if b > a else b""
            for sa, sb, name in spans[i:j + 1]:
                info = self.meta[name]
                dtype = _ST_DTYPES[info["dtype"]]
                raw = bytearray(blob[sa - a:sb - a])
                t = torch.frombuffer(raw, dtype=dtype) if raw else torch.empty(0, dtype=dtype)
                out[name] = t.reshape(info["shape"]).clone()
            i = j + 1
        return out


def _range_file(repo_id: str, filename: str) -> Optional[RangeFile]:
    """RangeFile for a hub file that is not cached locally, or None (local
    dir, cached, offline, disabled, or the server cannot serve ranges)."""
    if os.path.isdir(repo_id) or os.getenv("HF_HUB_OFFLINE") == "1" \
            or os.getenv("SYNAPSE_RANGE_FETCH", "1") == "0":
        return None
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id, filename, local_files_only=True)
        return None  # already cached: read it from disk
    except Exception:
        pass
    try:
        try:
            from huggingface_hub.utils import build_hf_headers
            headers = build_hf_headers()
        except Exception:
            headers = {}
        return RangeFile(_remote_url(repo_id, filename), headers)
    except Exception as e:
        if DEBUG >= 1:
            print(f"[PartialModel] range reads unavailable for {repo_id}/{filename} ({e}); "
                  "downloading the whole file")
        return None


def checkpoint_index(repo_id: str, remote: Optional[Dict[str, "RangeFile"]] = None
                     ) -> Optional[Dict[str, str]]:
    """tensor name -> checkpoint file name (relative to the repo), or None
    (no safetensors). Reads only the index; shard files are not fetched.
    A single uncached file is opened for range reads (stored in `remote`)
    instead of being downloaded just to list its tensors."""
    idx = _hub_file(repo_id, "model.safetensors.index.json")
    if idx:
        with open(idx, encoding="utf-8") as f:
            return dict(json.load(f).get("weight_map", {}))
    rf = _range_file(repo_id, "model.safetensors")
    if rf is not None:
        if remote is not None:
            remote["model.safetensors"] = rf
        return {k: "model.safetensors" for k in rf.keys()}
    single = _hub_file(repo_id, "model.safetensors")
    if not single:
        return None
    from safetensors import safe_open
    with safe_open(single, "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


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
        # checkpoint file name -> RangeFile, for files read over HTTP ranges
        self.remote: Dict[str, RangeFile] = {}
        self.index = checkpoint_index(repo_id, self.remote)
        if self.index is None:
            raise FileNotFoundError(f"{repo_id}: no safetensors checkpoint")
        # checkpoint file name -> local path, for the files fetched so far
        self.files: Dict[str, str] = {}
        self.layers: Dict[int, nn.Module] = {}
        self.embed: Optional[nn.Module] = None
        self.pos_embed: Optional[nn.Module] = None
        self.final_norm: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.rotary: Optional[nn.Module] = None
        # True when the checkpoint has no separate head (output = input
        # embedding); stage training keeps split copies in sync.
        self.head_tied = False
        self._skeleton = self._build_skeleton(1, None)
        self._layers_path = self._find_layers_path(self._skeleton)
        self._build_rotary()
        head = self._skeleton.get_output_embeddings()
        head_path = self._path_of(self._skeleton, head) if head is not None else None
        self.head_tied = bool(head_path) and self._ckpt_name(f"{head_path}.weight") is None

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

    def _file(self, name: str) -> str:
        """Local path of one checkpoint file, fetched on first use."""
        path = self.files.get(name)
        if path is None:
            path = _hub_file(self.repo_id, name)
            if path is None:
                raise FileNotFoundError(f"{self.repo_id}: cannot fetch {name}")
            self.files[name] = path
        return path

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
        for name, pairs in by_file.items():
            rf = self.remote.get(name)
            if rf is not None:
                got = rf.get_tensors([ck for _, ck in pairs])
                for local, ck in pairs:
                    state[local] = got[ck]
                continue
            with safe_open(self._file(name), "pt") as f:
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
                    from synapse.inference.weight_quant import layer_mode, quantize_module
                    quantize_module(layer, layer_mode(self.quant_mode, absolute, self.n_layers))
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
            tied = self.head_tied
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

    def files_for(self, start: int, end: int, first: bool, last: bool) -> List[str]:
        """Checkpoint files holding a shard's tensors, without loading any."""
        sk = self._skeleton
        lp = self._layers_path
        names: List[str] = []
        layer0 = self._sub(sk, lp)[0]
        for i in range(start, end + 1):
            names += [f"{lp}.{i}.{k}" for k in layer0.state_dict().keys()]
        mods = []
        if first:
            mods.append(sk.get_input_embeddings())
            wpe = getattr(getattr(sk, "transformer", None), "wpe", None)
            if wpe is not None:
                mods.append(wpe)
        if last:
            norm = self._final_norm_module(sk)
            if norm is not None:
                mods.append(norm)
            # a tied head reads the input embedding
            mods.append(sk.get_input_embeddings() if self.head_tied else sk.get_output_embeddings())
        for mod in mods:
            path = self._path_of(sk, mod)
            names += [f"{path}.{k}" for k in mod.state_dict().keys()]
        files = {self.index[ck] for ck in map(self._ckpt_name, names) if ck is not None}
        return sorted(files)

    def prefetch(self, start: int, end: int, first: bool, last: bool) -> List[str]:
        """Download (into the HF cache) only the files a shard needs. Files
        read over HTTP ranges are skipped: their bytes are fetched at load."""
        files = self.files_for(start, end, first, last)
        for name in files:
            if name not in self.remote:
                self._file(name)
        return files

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
        seen: set = set()
        return sum(module_bytes(m, seen) for m in mods)

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
