import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

# reuse the mask function from qwen implementation
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)



class Qwen3InputPipe(nn.Module):
    """
    Replic of HF Qwen3Model.forward function's input token process part
      - support input_ids or inputs_embeds
      - when use_cache=True and no pkv, create DynamicCache
      - calculate cache_position / position_ids(optional)
      - (optional) precaculate rotary position_embeddings and pass down
      - (optional) turn attention_mask into HF 的 causal_mask_mapping(dict)
    put all the control params into dict, use _normalize_tuple to make kwargs perfect forwarding
    """
    def __init__(self,
                 config,
                 tied_embed: nn.Embedding,
                 rotary_emb_module,       # passin qwen3_model.model.rotary_emb
                 has_sliding_layers: bool = False,
                 build_mask: bool = True):     # wether turn mask into mapping(dict)
        super().__init__()
        self.config = config
        self.embed_tokens = tied_embed
        self.rotary_emb = rotary_emb_module
        self.has_sliding_layers = has_sliding_layers
        self.build_mask = build_mask

    def forward(self, batch: dict):
        # === 1) process：input_ids / inputs_embeds  ===
        input_ids      = batch.get("input_ids", None)
        inputs_embeds  = batch.get("inputs_embeds", None)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # === 2) cache / position related ===
        use_cache       = bool(batch.get("use_cache", False))
        past_key_values = batch.get("past_key_values", None)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        cache_position = batch.get("cache_position", None)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            # [S] on device
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        position_ids = batch.get("position_ids", None)
        if position_ids is None:
            # [1, S]
            position_ids = cache_position.unsqueeze(0)

        # === 3) attention mask ===
        attention_mask = batch.get("attention_mask", None)

        # if there is no dict（mapping, create one
        causal_mask_mapping = attention_mask
        if self.build_mask and not isinstance(attention_mask, dict):
            if create_causal_mask is None:
                # pass down as it was
                causal_mask_mapping = attention_mask
            else:
                mask_kwargs = dict(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                if self.has_sliding_layers and create_sliding_window_causal_mask is not None:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # === 4) calculate RoPE  position_embeddings（ ===
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # === 5) create perfect forwarding dict ===
        x = dict(batch)  # shallow copy batch
        x.update({
            "hidden_states": inputs_embeds,          # requried 
            "attention_mask": causal_mask_mapping,   # dict（mapping）or mask tensor
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,  # passdow to all the layers
        })
        # clear raw inputs
        x.pop("input_ids", None)
        x.pop("inputs_embeds", None)

        return x
