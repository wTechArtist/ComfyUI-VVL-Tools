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
                "parallel": ("BOOLEAN", {"default": True}),
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

    def for_loop_start(self, total, parallel=True, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in
                          range(1, MAX_FLOW_NUM)}
        # 这里创建一个占位 whileOpen 节点，供 End 节点通过 rawLink 获取到 open 的内部节点 id
        # 注意：并发模式下我们仍保留该占位节点，以便 End 能定位循环体边界
        while_open = graph.node("VVL whileLoopStart", condition=True, initial_value0=i, **initial_values)
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

    def _explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
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
                    self._explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def _resolve_int_input(self, dynprompt, val):
        """尽量把 val 解析成编译期可用的 int；失败返回 None"""
        # 已经是整数
        if isinstance(val, int):
            return val

        # 是 link：尝试顺藤摸瓜
        try:
            if is_link(val):
                src_id, _ = val[0], val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else (val[0], 0)
                node = dynprompt.get_node(src_id)
                ct = node.get("class_type", "")

                # 1) 解析 Length Any 类节点（名字可能因插件不同而异，只要有 "any" 输入即可）
                if "length" in ct.lower() or ("length" in node.get("label", "").lower()):
                    any_in = node.get("inputs", {}).get("any", None)
                    # 如果 any 是 link，进一步解析其常量列表长度
                    length = None
                    if is_link(any_in):
                        any_src = dynprompt.get_node(any_in[0])
                        any_ct = any_src.get("class_type", "")

                        # 1.1) String List：从文本解析
                        if "string" in any_ct.lower() and "list" in any_ct.lower():
                            cfg = any_src.get("inputs", {})
                            text = str(cfg.get("list", "") or "")
                            use_nl = bool(cfg.get("new_line_as_separator", True))
                            sep = str(cfg.get("separator", ","))
                            items = [s for s in (text.splitlines() if use_nl else text.split(sep)) if str(s).strip() != ""]
                            length = len(items)

                        # 1.2) VVL listConstruct：统计 itemN 个数
                        elif "listconstruct" in any_ct.lower():
                            cnt = 0
                            for i in range(0, 1024):
                                if f"item{i}" in any_src.get("inputs", {}):
                                    cnt += 1
                                else:
                                    break
                            length = cnt

                    # 如果 any 直接是常量 list（极少见），也可直接 len()
                    if length is None:
                        any_val = node.get("inputs", {}).get("any", None)
                        if isinstance(any_val, list):
                            length = len(any_val)

                    if isinstance(length, int):
                        return length

                # 2) 解析 VVL mathInt（递归解析 a、b）
                if "math" in ct.lower() and "int" in ct.lower():
                    op = node.get("inputs", {}).get("operation", "add")
                    a = node.get("inputs", {}).get("a", 0)
                    b = node.get("inputs", {}).get("b", 0)
                    ai = self._resolve_int_input(dynprompt, a)
                    bi = self._resolve_int_input(dynprompt, b)
                    if isinstance(ai, int) and isinstance(bi, int):
                        if op == "add":      return ai + bi
                        if op == "subtract": return ai - bi
                        if op == "multiply": return ai * bi
                        if op == "divide":   return (ai // bi) if bi != 0 else 0
                        if op == "modulo":   return (ai % bi) if bi != 0 else 0

        except Exception:
            pass
        return None

    def _build_parallel(self, flow, dynprompt, unique_id, kwargs):
        graph = GraphBuilder()

        while_open = flow[0]
        total = None
        initial_from_start = {}
        parallel_flag = False

        # 通过 dynprompt 获取 start 节点设置
        forstart_node = dynprompt.get_node(while_open)
        if forstart_node['class_type'] == 'VVL forLoopStart':
            inputs = forstart_node['inputs']
            total = inputs.get('total', None)
            # 新增：尝试把 total 解析为 int
            total = self._resolve_int_input(dynprompt, total)
            parallel_flag = inputs.get('parallel', False)
            for i in range(MAX_FLOW_NUM):
                key = f"initial_value{i}"
                if key in inputs:
                    initial_from_start[key] = inputs[key]

        # 仅当 total 为 int 且 parallel 为 True 时并发展开
        if not isinstance(total, int) or not parallel_flag or total <= 0:
            return None

        # 收集 open 与 end 之间的子图
        upstream = {}
        parent_ids = []
        self._explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        contained = {}
        open_node = flow[0]
        self._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        contained_ids = list(contained.keys())

        # 为每个迭代索引克隆子图
        clone_map = {}
        for idx in range(total):
            clone_map[idx] = {}
            for node_id in contained_ids:
                original_node = dynprompt.get_node(node_id)
                # 创建克隆节点（不覆写 display_id，避免冲突）
                clone = graph.node(original_node["class_type"], f"{node_id}__{idx}")
                clone_map[idx][node_id] = clone

        # 连接每个克隆的输入
        for idx in range(total):
            for node_id in contained_ids:
                original_node = dynprompt.get_node(node_id)
                clone = clone_map[idx][node_id]
                inputs_dict = original_node.get("inputs", {})
                for k, v in inputs_dict.items():
                    if is_link(v) and v[0] in contained:
                        parent = clone_map[idx][v[0]]
                        clone.set_input(k, parent.out(v[1]))
                    else:
                        # 若该输入来自 forLoopStart 且为索引，注入当前 idx
                        if is_link(v):
                            try:
                                parent_display_id = dynprompt.get_display_node_id(v[0])
                                parent_display_node = dynprompt.get_node(parent_display_id)
                                parent_class_type = parent_display_node["class_type"]
                            except Exception:
                                parent_class_type = None

                            if parent_class_type == 'VVL forLoopStart' and (k == 'index' or original_node["class_type"] == 'VVL listGetItem'):
                                clone.set_input(k, idx)
                                continue

                        clone.set_input(k, v)

                # 特殊处理 forLoopStart：为每个克隆设置不同的 index（initial_value0）
                if original_node["class_type"] == 'VVL forLoopStart':
                    clone.set_input("initial_value0", idx)
                    for i in range(1, MAX_FLOW_NUM):
                        key = f"initial_value{i}"
                        if key in initial_from_start:
                            clone.set_input(key, initial_from_start[key])

        # 根据 forLoopEnd 的输入，聚合每个需要输出的值为列表
        results = []
        for i in range(1, MAX_FLOW_NUM):
            key = f"initial_value{i}"
            v = kwargs.get(key, None)
            if v is None or not is_link(v):
                # 未连接，返回 None
                results.append(None)
                continue

            src_node_id, src_out = v[0], v[1]

            # 对每个 idx，取对应克隆的输出，打包为列表
            pack_inputs = {}
            for idx in range(total):
                if src_node_id not in clone_map[idx]:
                    # 如果链接源不在子图中，降级为原值
                    pack_inputs[f"item{idx}"] = v
                else:
                    pack_inputs[f"item{idx}"] = clone_map[idx][src_node_id].out(src_out)

            pack_node = graph.node("VVL listConstruct", **pack_inputs)
            results.append(pack_node.out(0))

        return {
            "result": tuple(results),
            "expand": graph.finalize(),
        }

    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        # 优先尝试并发展开
        parallel_result = self._build_parallel(flow, dynprompt, unique_id, kwargs)
        if parallel_result is not None:
            return parallel_result
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


class VVLListConstructAsync:
    @classmethod
    def INPUT_TYPES(cls):
        # 预留足够多的可选 item 输入，满足常见并发规模
        optional_items = {f"item{i}": (any_type,) for i in range(0, 1024)}
        return {
            "required": {},
            "optional": optional_items,
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("list",)
    FUNCTION = "build"
    CATEGORY = "VVL/Utils"

    def build(self, **kwargs):
        items = []
        # 保证按索引顺序聚合
        for i in range(0, 1024):
            key = f"item{i}"
            if key in kwargs:
                items.append(kwargs[key])
            else:
                break
        return (items,)


class VVLListGetItemAsync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": (any_type,),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("item",)
    FUNCTION = "get_item"
    CATEGORY = "VVL/Utils"

    def get_item(self, list, index):
        try:
            # 标准列表或可下标序列
            if hasattr(list, '__getitem__'):
                length = None
                try:
                    length = len(list)
                except Exception:
                    length = None

                if length is not None and length > 0:
                    if index >= length:
                        index = index % length
                return (list[index],)

            # 可迭代但不可直接下标，尝试转为列表
            try:
                materialized = list if isinstance(list, list.__class__) else [x for x in list]
            except Exception:
                materialized = None
            if materialized is not None and len(materialized) > 0:
                if index >= len(materialized):
                    index = index % len(materialized)
                return (materialized[index],)
        except Exception:
            pass
        # 回退：返回原值
        return (list,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "VVL forLoopStart": VVLForLoopStartAsync,
    "VVL forLoopEnd": VVLForLoopEndAsync,
    "VVL whileLoopStart": VVLWhileLoopStartAsync,
    "VVL whileLoopEnd": VVLWhileLoopEndAsync,
    "VVL mathInt": VVLMathIntAsync,
    "VVL compare": VVLCompareAsync,
    "VVL listConstruct": VVLListConstructAsync,
    "VVL listGetItem": VVLListGetItemAsync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL forLoopStart": "VVL For Loop Start (Async)",
    "VVL forLoopEnd": "VVL For Loop End (Async)",
    "VVL whileLoopStart": "VVL While Loop Start (Async)",
    "VVL whileLoopEnd": "VVL While Loop End (Async)",
    "VVL mathInt": "VVL Math Int (Async)",
    "VVL compare": "VVL Compare (Async)",
    "VVL listConstruct": "VVL List Construct (Async)",
    "VVL listGetItem": "VVL List Get Item (Async)",
}