from comfy.sd import CLIP


class TokenCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "debug_print": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    OUTPUT_NODE = True

    def encode(self, clip: CLIP, text: str, debug_print: bool):
        lengths = []
        blocks = []
        special_tokens = set(clip.cond_stage_model.clip_l.special_tokens.values())
        vocab = clip.tokenizer.clip_l.inv_vocab
        prompts = text.split("BREAK")
        for prompt in prompts:
            if len(prompt) > 0:
                tokens_pure = clip.tokenize(prompt)
                tokens_concat = sum(tokens_pure["l"], [])
                block = [t for t in tokens_concat if t[0] not in special_tokens]
                blocks.append(block)

        if len(blocks) > 0:
            lengths = [str(len(b)) for b in blocks]
            if debug_print:
                print(f"Token count: {' + '.join(lengths)}")
                print("--start--")
                print(" + ".join(f"'{vocab[t[0]]}'" for b in blocks for t in b))
                print("--finish--")
        else:
            lengths = ["0"]
        return (lengths,)
