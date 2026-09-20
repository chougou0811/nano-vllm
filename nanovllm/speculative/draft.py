"""Use the pinned EAGLE reference decoder, not a reimplemented draft network."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open


CHECKPOINT_SHA256 = "12d9f436fc08e6fffc580c5e4a71ecefcb44bef450e500df3fac698e9f530cbf"
REFERENCE_HASHES = {
    'cnets.py': 'b0e2512db4529c9689ffd768e9a2812059a5715d786686a131a751c56fbf3292',
    'configs.py': '14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5',
    'utils_c.py': 'fa7bee5ea870200756bd48bb27c1dbfb2e8f1d42f4e9c832d6b5d1681ca97f58',
    'choices.py': '5ab95137f63b61f756057927b2af427ffa0bc351384d99e334a9da853fcbfa7e',
}


class ReferenceDraft:
    def __init__(self, path, reference_path, target_path, target_config, device):
        path, reference_path = Path(path), Path(reference_path)
        manifest = json.loads((path/"download_manifest.json").read_text())
        if manifest.get("sha256") != CHECKPOINT_SHA256 or not manifest.get("verified"):
            raise ValueError("Expected pinned and verified AngelSlim checkpoint")
        with (path/"pytorch_model.bin").open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Draft weight digest mismatch")
        for name,digest in REFERENCE_HASHES.items():
            if hashlib.sha256((reference_path/'eagle'/'model'/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pinned reference digest mismatch: {name}")
        sys.path.insert(0,str(reference_path))
        from eagle.model.cnets import Model
        from eagle.model.configs import EConfig
        if Path(sys.modules[Model.__module__].__file__).resolve() != (reference_path/'eagle/model/cnets.py').resolve():
            raise ValueError('A different EAGLE module is already imported')
        cfg = EConfig.from_pretrained(str(path),local_files_only=True)
        for key in ["hidden_size","vocab_size","rope_theta","head_dim"]:
            if getattr(cfg,key) != getattr(target_config,key):
                raise ValueError(f"Draft/target mismatch: {key}")
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            self.model = Model(cfg,load_emb=False,bias=False).eval()
        finally:
            torch.set_default_dtype(old_dtype)
        state = torch.load(path/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True)
        result = self.model.load_state_dict(state,strict=False)
        if result.missing_keys != ["embed_tokens.weight"] or result.unexpected_keys:
            raise ValueError(f"Unexpected reference state dict: {result}")
        index = json.loads((Path(target_path)/"model.safetensors.index.json").read_text())
        with safe_open(Path(target_path)/index["weight_map"]["model.embed_tokens.weight"],framework="pt",device="cpu") as f:
            self.model.embed_tokens.weight.data = f.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
        mapping = self.model.d2t+torch.arange(cfg.draft_vocab_size)
        if mapping.unique().numel() != cfg.draft_vocab_size or mapping.min() < 0 or mapping.max() >= cfg.vocab_size:
            raise ValueError("Invalid reduced-vocabulary mapping")
        if not torch.equal(mapping.sort().values, self.model.t2d.nonzero().flatten()):
            raise ValueError("d2t/t2d disagree")
        self.model.to(device=device,dtype=torch.bfloat16)
        self.model.reset()
        self.model.reset_kv()
        self.device = device
        self.provenance = dict(checkpoint=manifest,reference=str(reference_path),
            checkpoint_dtype="float16",runtime_dtype="bfloat16",missing_embedding="borrowed pinned target embedding",
            reference_files={str(p.relative_to(reference_path)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in reference_path.rglob("*.py")})

    @torch.inference_mode()
    def propose(self, features, tokens, k):
        if len(tokens) != features.shape[0]+1 or k < 1:
            raise ValueError("Features must precede the committed pending token")
        self.model.reset()
        self.model.reset_kv()
        ids = torch.tensor(tokens[1:],device=self.device,dtype=torch.long).unsqueeze(0)
        hidden, past = self.model(features.unsqueeze(0),input_ids=ids,use_cache=True)
        proposals = []
        for step in range(k):
            logits = self.model.lm_head(self.model.norm(hidden[:,-1:]))
            index = logits[0,-1].argmax()
            token = int((index+self.model.d2t[index]).item())
            proposals.append(token)
            if step+1 < k:
                hidden,past = self.model(hidden[:,-1:],input_ids=torch.tensor([[token]],device=self.device),
                                        past_key_values=past,use_cache=True)
        # No proposal-derived cache survives. Next call rebuilds from target features.
        del hidden,past
        return proposals
