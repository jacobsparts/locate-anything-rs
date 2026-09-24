"""Tensor name mapping: upstream HuggingFace name -> engine tensor name.

Names are byte-identical to the existing C++ engine's GGUF schema (see
locate-anything.cpp/scripts/gguf_keys.py) so tensors can be diffed 1:1.

The engine schema is deliberately close to the llama.cpp/ggml conventions the
existing C++ engine uses, so the two can be compared tensor-for-tensor.

Returns None for a tensor that should not be stored.
"""
import re

ARCH = "locateanything"

TENSOR_RULES = [
    # LM (Qwen2)
    (r"^language_model\.model\.embed_tokens\.weight$", "lm.tok_embd.weight"),
    (r"^language_model\.model\.layers\.(\d+)\.self_attn\.([qkv])_proj\.(weight|bias)$", r"lm.blk.\1.attn_\2.\3"),
    (r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"lm.blk.\1.attn_o.weight"),
    (r"^language_model\.model\.layers\.(\d+)\.input_layernorm\.weight$", r"lm.blk.\1.attn_norm.weight"),
    (r"^language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"lm.blk.\1.ffn_norm.weight"),
    (r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", r"lm.blk.\1.ffn_\2.weight"),
    (r"^language_model\.model\.norm\.weight$", "lm.output_norm.weight"),
    (r"^language_model\.lm_head\.weight$", "lm.output.weight"),
    # multimodal projector
    (r"^mlp1\.(\d+)\.(weight|bias)$", r"proj.\1.\2"),
    # vision (MoonViT)
    (r"^vision_model\.patch_embed\.proj\.weight$", "vit.patch_embed.weight"),
    (r"^vision_model\.patch_embed\.proj\.bias$", "vit.patch_embed.bias"),
    (r"^vision_model\.patch_embed\.pos_emb\.weight$", "vit.pos_emb.weight"),
    (r"^vision_model\.encoder\.blocks\.(\d+)\.(norm0|norm1)\.(weight|bias)$", r"vit.blk.\1.\2.\3"),
    (r"^vision_model\.encoder\.blocks\.(\d+)\.(wqkv|wo)\.(weight|bias)$", r"vit.blk.\1.\2.\3"),
    (r"^vision_model\.encoder\.blocks\.(\d+)\.mlp\.(fc0|fc1)\.(weight|bias)$", r"vit.blk.\1.\2.\3"),
    (r"^vision_model\.encoder\.final_layernorm\.(weight|bias)$", r"vit.final_norm.\1"),
]


def rename_tensor(name: str):
    for pat, rep in TENSOR_RULES:
        m = re.match(pat, name)
        if m:
            return m.expand(rep)
    return None
