"""Independent HF reference for saved scheduler teacher-prefix diagnostics."""
import argparse
import json
from pathlib import Path

from .report import write_json


def main():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostic-dir", required=True)
    args = parser.parse_args()
    output = Path(args.diagnostic_dir)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    devices = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    devices.update({f"model.layers.{i}": int(i >= config.num_hidden_layers // 2)
                    for i in range(config.num_hidden_layers)})
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=devices, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = tokenizer.encode("The quick brown fox jumps over the lazy dog. A language model predicts the next token.")
    if (output / "correctness-inputs.json").exists():
        prompts = json.loads((output / "correctness-inputs.json").read_text())["prompts"]
    else:
        prompts = [(text * (length // len(text) + 1))[:length] for length in [31,255,256,257,511,512]]
    references = {}
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            ids = torch.tensor([prompt + [7, 13, 19]], device="cuda:0")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=4).logits[0].float().cpu()
            for j, row in enumerate(logits):
                references[i, j] = row
    torch.save(references, output / "hf-reference-logits.pt")
    results = {"device_map": devices, "reference": "HF BF16 eager, full context, no KV reuse", "policies": {}}
    for path in sorted(output.glob("correctness-logits-*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        rows = []
        for key, actual in saved.items():
            ref = references[key]
            rows.append(dict(request=key[0], token=key[1], max_abs=(ref-actual).abs().max().item(),
                top1_equal=ref.argmax().item() == actual.argmax().item(),
                probability_l1=(ref.softmax(-1)-actual.softmax(-1)).abs().sum().item()))
        results["policies"][path.stem] = rows
        print(path.stem, "max_abs", max(r["max_abs"] for r in rows),
              "top1", sum(r["top1_equal"] for r in rows), "/", len(rows), flush=True)
    write_json(output / "hf-reference.json", results)


if __name__ == "__main__":
    main()
