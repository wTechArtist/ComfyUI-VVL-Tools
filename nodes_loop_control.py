"""
VVL Loop Control Nodes

原封不动搬运ComfyUI-Easy-Use中的forLoopStart和forLoopEnd节点，只改节点名称。

Author: VVL Test
Version: 1.0.0
"""

from typing import Iterator, List, Tuple, Dict, Any, Union, Optional
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.graph import is_link

# 原始常量和类型定义
DEFAULT_FLOW_NUM = 2
MAX_FLOW_NUM = 20

class AlwaysEqualProxy(str):
    def __ne__(self, __value):
        return False

any_type = AlwaysEqualProxy("*")

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            return any_type
        return super().__getitem__(index)


class VVLForLoopStartAsync:
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
        while_open = graph.node("VVL whileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class VVLForLoopEndAsync:
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

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(while_open)
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

        sub = graph.node("VVL mathInt", operation="add", a=[while_open, 1], b=1)
        cond = graph.node("VVL compare", a=sub.out(0), b=total, comparison='a < b')
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in
                        range(1, MAX_FLOW_NUM)}
        while_close = graph.node("VVL whileLoopEnd",
                                 flow=flow,
                                 condition=cond.out(0),
                                 initial_value0=sub.out(0),
                                 **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }


class VVLWhileLoopStartAsync:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_open"

    CATEGORY = "VVL/Loop"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)


class VVLWhileLoopEndAsync:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_close"

    CATEGORY = "VVL/Loop"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ['VVL forLoopEnd', 'VVL whileLoopEnd']:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                    if '.' in parent_id:
                        arr = parent_id.split('.')
                        arr[len(arr)-1] = output_id
                        upstream[parent_id].append('.'.join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None,**kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        # Get the list of all output nodes between the open and close nodes
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            # 这里需要导入ALL_NODE_CLASS_MAPPINGS，暂时注释掉
            # class_def = ALL_NODE_CLASS_MAPPINGS[class_type]
            # if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
            #     for k, v in node['inputs'].items():
            #         if is_link(v):
            #             output_nodes[id] = v

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class VVLMathIntAsync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["add", "subtract", "multiply", "divide", "modulo"], {"default": "add"}),
                "a": ("INT", {"default": 0}),
                "b": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "math_operation"
    CATEGORY = "VVL/Math"

    def math_operation(self, operation, a, b):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b if b != 0 else 0,)
        elif operation == "modulo":
            return (a % b if b != 0 else 0,)
        else:
            return (a,)


class VVLCompareAsync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
                "comparison": (["a == b", "a != b", "a < b", "a <= b", "a > b", "a >= b"], {"default": "a < b"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "compare"
    CATEGORY = "VVL/Logic"

    def compare(self, a, b, comparison):
        try:
            if comparison == "a == b":
                return (a == b,)
            elif comparison == "a != b":
                return (a != b,)
            elif comparison == "a < b":
                return (a < b,)
            elif comparison == "a <= b":
                return (a <= b,)
            elif comparison == "a > b":
                return (a > b,)
            elif comparison == "a >= b":
                return (a >= b,)
            else:
                return (False,)
        except:
            return (False,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "VVL forLoopStart": VVLForLoopStartAsync,
    "VVL forLoopEnd": VVLForLoopEndAsync,
    "VVL whileLoopStart": VVLWhileLoopStartAsync,
    "VVL whileLoopEnd": VVLWhileLoopEndAsync,
    "VVL mathInt": VVLMathIntAsync,
    "VVL compare": VVLCompareAsync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL forLoopStart": "VVL For Loop Start (Async)",
    "VVL forLoopEnd": "VVL For Loop End (Async)",
    "VVL whileLoopStart": "VVL While Loop Start (Async)",
    "VVL whileLoopEnd": "VVL While Loop End (Async)",
    "VVL mathInt": "VVL Math Int (Async)",
    "VVL compare": "VVL Compare (Async)",
}