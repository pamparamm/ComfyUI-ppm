import torch
from torch import Tensor
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.flux.layers import timestep_embedding
from comfy.text_encoders.hunyuan_video import HunyuanVideoClipModel
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
                    block, args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], negpip_mask=negpip_mask, flipped_img_txt=True
                )
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
            img = out["img"]
        else:
            img = _flux_ssb_forward_negpip(block, img, vec=vec, pe=pe, attn_mask=attn_mask, negpip_mask=negpip_mask, flipped_img_txt=True)

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


def _hunyuan_video_clip_encode_token_weights_negpip(self: HunyuanVideoClipModel, token_weight_pairs):
    token_weight_pairs_l = token_weight_pairs["l"]
    token_weight_pairs_llama = token_weight_pairs["llama"]

    llama_out_negpip, llama_pooled, llama_extra_out = self.llama.encode_token_weights(token_weight_pairs_llama)
    llama_out = llama_out_negpip[:, 0::2, :]

    template_end = 0
    for i, v in enumerate(token_weight_pairs_llama[0]):
        if v[0] == 128007:  # <|end_header_id|>
            template_end = i

    if llama_out.shape[1] > (template_end + 2):
        if token_weight_pairs_llama[0][template_end + 1][0] == 271:
            template_end += 2
    llama_out_negpip = llama_out_negpip[:, template_end * 2 :]
    llama_extra_out["attention_mask"] = llama_extra_out["attention_mask"][:, template_end:]
    if llama_extra_out["attention_mask"].sum() == torch.numel(llama_extra_out["attention_mask"]):
        llama_extra_out.pop("attention_mask")  # attention mask is useless if no masked elements

    l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
    return llama_out_negpip, l_pooled, llama_extra_out
