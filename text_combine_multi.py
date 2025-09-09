"""
Text Combine Multi Node
A node that allows dynamic text input combination with configurable separators
"""

class TextCombineMulti:
    @classmethod
    def INPUT_TYPES(s):
        # 动态生成所有可能的text输入参数（text_3到text_20）
        optional_inputs = {}
        for i in range(3, 21):  # text_3 到 text_20
            optional_inputs[f"text_{i}"] = ("STRING", {"default": ""})
        
        return {
            "required": {
                "text_1": ("STRING", {"default": ""}),
                "text_2": ("STRING", {"default": ""}),
                "separator": ("STRING", {"default": "", "multiline": False}),
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "combine_texts"
    CATEGORY = "VVL-Tools/text"
    DESCRIPTION = """
Dynamic text combination node that allows you to combine multiple text inputs.
You can set how many text inputs the node has with the **inputcount** parameter and clicking update.
All text inputs support direct typing and multiline text.
Use the separator parameter to control how texts are joined together.
"""

    def combine_texts(self, inputcount, separator, text_1="", text_2="", **kwargs):
        """
        Combine multiple text inputs using the specified separator
        """
        combined_texts = []
        
        # Collect text_1 and text_2 (always available)
        if text_1.strip():
            combined_texts.append(text_1)
        if text_2.strip():
            combined_texts.append(text_2)
            
        # Collect additional dynamic text inputs
        for i in range(3, inputcount + 1):
            text_key = f"text_{i}"
            text_value = kwargs.get(text_key, "")
            
            # Only add non-empty texts to avoid unnecessary separators
            if text_value.strip():
                combined_texts.append(text_value)
        
        # Join all texts with the separator
        result = separator.join(combined_texts)
        
        return (result,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TextCombineMulti": TextCombineMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCombineMulti": "VVL Text Combine Multi",
}