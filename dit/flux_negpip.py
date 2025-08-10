import torch
from torch import Tensor

from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, timestep_embedding
from comfy.ldm.flux.math import attention
from comfy.ldm.flux.model import Flux


def flux_dsb_forward_negpip(
    self: DoubleStreamBlock,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask: Tensor | None = None,
    negpip_mask: Tensor | None = None,
):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

    if negpip_mask is not None:
        txt_v *= negpip_mask[:, None, :, None]

    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    if hasattr(self, "flipped_img_txt") and self.flipped_img_txt:
        # run actual attention
        attn = attention(
            torch.cat((img_q, txt_q), dim=2),
            torch.cat((img_k, txt_k), dim=2),
            torch.cat((img_v, txt_v), dim=2),
            pe=pe,
            mask=attn_mask,
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
    else:
        # run actual attention
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
            mask=attn_mask,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

    # calculate the txt bloks
    txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt


def flux_ssb_forward_negpip(
    self: SingleStreamBlock,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask: Tensor | None = None,
    negpip_mask: Tensor | None = None,
    flipped_img_txt: bool = False,
) -> Tensor:
    mod, _ = self.modulation(vec)
    qkv, mlp = torch.split(
        self.linear1((1 + mod.scale) * self.pre_norm(x) + mod.shift),
        [3 * self.hidden_size, self.mlp_hidden_dim],
        dim=-1,
    )

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

    if negpip_mask is not None:
        if flipped_img_txt:
            v[:, :, -negpip_mask.shape[1] :] *= negpip_mask[:, None, :, None]
        else:
            v[:, :, : negpip_mask.shape[1]] *= negpip_mask[:, None, :, None]

    q, k = self.norm(q, k, v)

    # compute attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def flux_forward_orig_negpip(
    self: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor | None = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])

    txt_ids = txt_ids[:, 0::2, :]
    txt_negpip = txt[:, 1::2, :]
    txt = txt[:, 0::2, :]
    negpip_mask = (txt == txt_negpip).all(dim=-1).int()
    negpip_mask[negpip_mask == 0] = -1

    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = flux_dsb_forward_negpip(
                    block,
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    negpip_mask=negpip_mask,
                )
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                },
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = flux_dsb_forward_negpip(
                block,
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                negpip_mask=negpip_mask,
            )

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = flux_ssb_forward_negpip(
                    block,
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    negpip_mask=negpip_mask,
                    flipped_img_txt=False,
                )
                return out

            out = blocks_replace[("single_block", i)](
                {
                    "img": img,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                },
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = flux_ssb_forward_negpip(
                block,
                img,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                negpip_mask=negpip_mask,
                flipped_img_txt=False,
            )

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img
