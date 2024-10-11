import torch
from torch import Tensor

from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, timestep_embedding
from comfy.ldm.flux.math import attention


def _flux_dsb_forward_negpip(_self: DoubleStreamBlock, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, mask: Tensor):
    img_mod1, img_mod2 = _self.img_mod(vec)
    txt_mod1, txt_mod2 = _self.txt_mod(vec)

    # prepare image for attention
    img_modulated = _self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = _self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, _self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = _self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = _self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = _self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, _self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_v *= mask[:, None, :, None]
    txt_q, txt_k = _self.txt_attn.norm(txt_q, txt_k, txt_v)

    # run actual attention
    attn = attention(torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2), pe=pe)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * _self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * _self.img_mlp((1 + img_mod2.scale) * _self.img_norm2(img) + img_mod2.shift)

    # calculate the txt bloks
    txt += txt_mod1.gate * _self.txt_attn.proj(txt_attn)
    txt += txt_mod2.gate * _self.txt_mlp((1 + txt_mod2.scale) * _self.txt_norm2(txt) + txt_mod2.shift)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt


def _flux_ssb_forward_negpip(_self: SingleStreamBlock, x: Tensor, vec: Tensor, pe: Tensor, mask: Tensor) -> Tensor:
    mod, _ = _self.modulation(vec)
    x_mod = (1 + mod.scale) * _self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(_self.linear1(x_mod), [3 * _self.hidden_size, _self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, _self.num_heads, -1).permute(2, 0, 3, 1, 4)
    v[:, :, : mask.shape[1]] *= mask[:, None, :, None]
    q, k = _self.norm(q, k, v)

    # compute attention
    attn = attention(q, k, v, pe=pe)
    # compute activation in mlp stream, cat again and run second linear layer
    output = _self.linear2(torch.cat((attn, _self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def _flux_forward_orig_negpip(
    _self: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = _self.img_in(img)
    vec = _self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if _self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + _self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + _self.vector_in(y[:, : _self.params.vec_in_dim])

    txt_ids = txt_ids[:, 0::2, :]
    txt_negpip = txt[:, 1::2, :]
    txt = txt[:, 0::2, :]
    mask = (txt == txt_negpip).max(dim=-1).values.int()
    mask[mask == 0] = -1

    txt = _self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = _self.pe_embedder(ids)

    for i, block in enumerate(_self.double_blocks):
        img, txt = _flux_dsb_forward_negpip(block, img=img, txt=txt, vec=vec, pe=pe, mask=mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(_self.single_blocks):
        img = _flux_ssb_forward_negpip(block, img, vec=vec, pe=pe, mask=mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = _self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img
