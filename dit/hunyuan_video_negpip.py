import torch
from torch import Tensor
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.flux.layers import timestep_embedding
from .flux_negpip import _flux_dsb_forward_negpip, _flux_ssb_forward_negpip


def _hunyuan_video_forward_orig_negpip(
    self: HunyuanVideo,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    txt_mask: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})

    initial_shape = list(img.shape)
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])

    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    txt_ids = txt_ids[:, 0::2, :]
    txt_negpip = txt[:, 1::2, :]
    txt = txt[:, 0::2, :]
    negpip_mask = (txt == txt_negpip).all(dim=-1).int()
    negpip_mask[negpip_mask == 0] = -1

    if txt_mask is not None and not torch.is_floating_point(txt_mask):
        txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

    txt = self.txt_in(txt, timesteps, txt_mask)

    ids = torch.cat((img_ids, txt_ids), dim=1)
    pe = self.pe_embedder(ids)

    img_len = img.shape[1]
    if txt_mask is not None:
        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
        attn_mask[:, 0, img_len:] = txt_mask
    else:
        attn_mask = None

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = _flux_dsb_forward_negpip(
                    block, img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], negpip_mask=negpip_mask
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap}
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = _flux_dsb_forward_negpip(block, img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, negpip_mask=negpip_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((img, txt), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = _flux_ssb_forward_negpip(
                    block, args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], negpip_mask=negpip_mask
                )
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
            img = out["img"]
        else:
            img = _flux_ssb_forward_negpip(block, img, vec=vec, pe=pe, attn_mask=attn_mask, negpip_mask=negpip_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, :img_len] += add

    img = img[:, :img_len]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    shape = initial_shape[-3:]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
    img = img.reshape(initial_shape)
    return img
