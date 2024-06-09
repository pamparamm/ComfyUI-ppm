import random


class RandomPromptGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "length": ("INT", {"default": 10, "min": 1, "max": 20}),
                "weight": ("FLOAT", {"default": 1.4, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    OUTPUT_NODE = True

    def generate(self, seed, length, weight):
        generator = random.Random(seed)
        alphabet = "abcdefghijklmnopqrstupvxyz1234567890"
        prompt = "".join(generator.choice(alphabet) for _ in range(length))
        prompt_weighted = f"({prompt}:{weight})" if weight != 1.0 else prompt
        return (prompt_weighted,)
