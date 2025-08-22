"""
VVL Loop Control Nodes

This module provides async loop control nodes for ComfyUI:
1. VVLForLoopStart(Async) - Start a for loop with specified iterations
2. VVLForLoopEnd(Async) - End a for loop and handle iteration

Based on the original forLoopStart and forLoopEnd from ComfyUI-Easy-Use.

Author: VVL Test
Version: 1.0.0
"""

from comfy.comfy_types.node_typing import IO
from comfy_execution.graph_utils import GraphBuilder


# Use the same MAX_FLOW_NUM as original
MAX_FLOW_NUM = 2

# Create a proxy type similar to original any_type
class AlwaysEqualProxy(str):
    def __ne__(self, __value):
        return False

any_type = AlwaysEqualProxy("*")

# Create a bypass tuple similar to original ByPassTypeTuple
class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            return any_type
        return super().__getitem__(index)


class VVLForLoopStartAsync:
    """
    VVL version of forLoopStart with original logic preserved.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + ["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_start"
    CATEGORY = "VVL/Loop"

    def for_loop_start(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in
                          range(1, MAX_FLOW_NUM)}
        while_open = graph.node("easy whileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class VVLForLoopEndAsync:
    """
    VVL version of forLoopEnd with original logic preserved.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "for_loop_end"
    CATEGORY = "VVL/Loop"

    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        while_open = flow[0]
        total = None

        # 兼容：将内部id映射到display id，避免 NodeNotFound（最小必要改动）
        try:
            open_display_id = dynprompt.get_display_node_id(while_open) if dynprompt is not None else while_open
        except Exception:
            open_display_id = while_open

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(open_display_id)
        if forstart_node['class_type'] == 'VVL forLoopStart':
            inputs = forstart_node['inputs']
            total = inputs['total']
        elif forstart_node['class_type'] == 'easy loadImagesForLoop':
            inputs = forstart_node['inputs']
            limit = inputs['limit']
            start_index = inputs['start_index']
            # Filter files by extension
            directory = inputs['directory']
            total = graph.node('easy imagesCountInDirectory', directory=directory, limit=limit, start_index=start_index, extension='*').out(0)

        sub = graph.node("easy mathInt", operation="add", a=[open_display_id, 1], b=1)
        cond = graph.node("easy compare", a=sub.out(0), b=total, comparison='a < b')
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in
                        range(1, MAX_FLOW_NUM)}
        # 规范化flow链接为显示ID
        flow_link = [open_display_id, flow[1]] if isinstance(flow, (list, tuple)) and len(flow) > 1 else [open_display_id, 0]
        while_close = graph.node("easy whileLoopEnd",
                                 flow=flow_link,
                                 condition=cond.out(0),
                                 initial_value0=sub.out(0),
                                 **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }






# Node mappings for registration - only the main for loop nodes
NODE_CLASS_MAPPINGS = {
    "VVL forLoopStart": VVLForLoopStartAsync,
    "VVL forLoopEnd": VVLForLoopEndAsync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL forLoopStart": "VVL For Loop Start (Async)",
    "VVL forLoopEnd": "VVL For Loop End (Async)",
}