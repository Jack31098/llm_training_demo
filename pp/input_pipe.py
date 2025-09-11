import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

# reuse the mask function from qwen implementation
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    Qwen3RotaryEmbedding,
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
    def __init__(self, config, vocab_size: int, d_model: int, pad_id: int,
                 has_sliding_layers: bool = False, build_mask: bool = True, rope_eps: float = 1e-6):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # expose top-level weight for TiedLayerSpec
        # self.weight = self.embed_tokens.weight
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.has_sliding_layers = has_sliding_layers
        self.build_mask = build_mask

    def forward(self, batch):
        # DeepSpeed loader feeds (input_ids, attention_mask)
        tup = list(batch) if isinstance(batch, (tuple, list)) else [batch]
        input_ids = tup[0] if len(tup) >= 1 else None
        attention_mask = tup[1] if len(tup) >= 2 else None
        inputs_embeds = None
        position_ids = None
        use_cache_flag = torch.zeros(1, dtype=torch.long, device=input_ids.device if isinstance(input_ids, torch.Tensor) else None)
        cache_position = None

        # 1) input_ids / inputs_embeds
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            # use tied weight via top-level attribute to ensure tying works even if modules differ
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = F.embedding(
            #     input_ids,
            #     self.weight,
            #     padding_idx=self.embed_tokens.padding_idx,
            # )

        # 2) cache / pos
        # training path: do not transport past_key_values across pipe; represent as flag tensor only
        past_key_values = None
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 3) build nothing non-tensor for cross-stage comm; pass padding mask through, later layers build per-layer masks

        # 4) RoPE is computed inside each decoder layer; avoid transporting floats here

        # 5) pack tensors-only 8-slot tuple for transport compatibility
        rsvd1 = torch.zeros(1, dtype=torch.int, device=inputs_embeds.device)
        rsvd2 = torch.zeros(1, dtype=torch.int, device=inputs_embeds.device)
        return (
            inputs_embeds,      # 0 hidden_states  [B, S, D]
            attention_mask,     # 1 padding mask   [B, S]
            position_ids,       # 2 [B, S]
            cache_position,     # 5 [S]
            rsvd1,              # 6 reserved tensor
            rsvd2,              # 7 reserved tensor
        )
