"""
VVL Loop Control Nodes

原封不动搬运ComfyUI-Easy-Use中的forLoopStart和forLoopEnd节点，只改节点名称。

Author: VVL Test
Version: 1.0.0
"""

import logging
from typing import Iterator, List, Tuple, Dict, Any, Union, Optional
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.graph import is_link

# 设置调试日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


class FlexibleOptionalInputType(dict):
    """用于可变可选输入的类型容器，支持动态输入端口。
    
    - 任何未显式声明的 key，在被访问时都返回 (type,) 作为其类型定义
    - __contains__ 始终返回 True，从而允许前端动态增加可选输入端口
    """

    def __init__(self, type_def, data=None):
        self.type_def = type_def
        self.data = data or {}
        # 将初始数据映射到自身，使其在 UI 上可见
        for k, v in self.data.items():
            self[k] = v

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.type_def,)

    def __contains__(self, key):
        return True


class VVLForLoopStartAsync:
    """
    VVL For Loop Start节点 - 支持动态输入端口
    
    功能特点：
    1. 支持动态添加/删除initial_value输入端口
    2. 自动调整输出端口数量以匹配输入
    3. 支持并行和顺序执行模式
    4. 保持与原有实现的兼容性
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "parallel": ("BOOLEAN", {"default": True}),
            },
            "optional": FlexibleOptionalInputType(any_type, {
                "initial_value1": (any_type,),
            }),
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
            logger.debug(f"[VVL] for_loop_start: unique_id={unique_id}, initial_value0={i}, total={total}")

        # 动态收集所有initial_value输入
        initial_values = {}
        outputs = []
        
        # 首先收集所有传入的initial_value参数
        for key, value in kwargs.items():
            if key.startswith("initial_value") and key != "initial_value0":
                initial_values[key] = value
                
        # 按数字顺序排序并生成输出
        sorted_keys = sorted([k for k in initial_values.keys()], 
                           key=lambda x: int(x.replace("initial_value", "") or "0"))
        
        # 补齐到MAX_FLOW_NUM以保持兼容性
        for num in range(1, MAX_FLOW_NUM):
            key = f"initial_value{num}"
            if key not in initial_values:
                initial_values[key] = None
                
        # 生成输出（按顺序）
        for num in range(1, MAX_FLOW_NUM):
            key = f"initial_value{num}"
            outputs.append(initial_values.get(key, None))
        
        logger.debug(f"[VVL] for_loop_start: 动态输入数量={len([k for k in kwargs.keys() if k.startswith('initial_value') and k != 'initial_value0'])}")
        
        # 这里创建一个占位 whileOpen 节点，供 End 节点通过 rawLink 获取到 open 的内部节点 id
        # 注意：并发模式下我们仍保留该占位节点，以便 End 能定位循环体边界
        while_open = graph.node("VVL whileLoopStart", condition=True, initial_value0=i, **initial_values)
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class VVLForLoopEndAsync:
    """
    VVL For Loop End节点 - 支持动态输入端口
    
    功能特点：
    1. 支持动态添加/删除initial_value输入端口
    2. 自动调整输出端口数量以匹配输入
    3. 支持并行展开和顺序执行
    4. 与ForLoopStart节点配合工作
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": FlexibleOptionalInputType(any_type, {
                "initial_value1": (any_type, {"rawLink": True}),
            }),
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
        try:
            node_info = dynprompt.get_node(node_id)
        except Exception:
            # 尝试将克隆 display_id 还原为原始 id（形如 16__2 -> 16）
            try:
                base_id = str(node_id).split("__")[0]
                node_info = dynprompt.get_node(base_id)
                node_id = base_id
            except Exception:
                # 如果节点仍不存在（例如外层克隆在当前上下文不可见），跳过
                return
        
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                try:
                    display_id = dynprompt.get_display_node_id(parent_id)
                    try:
                        display_node = dynprompt.get_node(display_id)
                    except Exception:
                        # 回退到原始 id（去掉 __idx 后缀）
                        base_disp = str(display_id).split("__")[0]
                        display_node = dynprompt.get_node(base_disp)
                        display_id = base_disp
                    class_type = display_node["class_type"]
                except Exception:
                    # 父节点不存在或不可访问，跳过这个依赖
                    continue
                
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
        logger.debug(f"[VVL] _resolve_int_input: 输入值 val={val}, 类型={type(val)}")
        
        # 已经是整数
        if isinstance(val, int):
            logger.debug(f"[VVL] _resolve_int_input: 直接返回整数 {val}")
            return val

        # 是 link：尝试顺藤摸瓜
        try:
            if is_link(val):
                # 记录源节点与输出口索引
                src_id = val[0]
                src_out = 0
                try:
                    if isinstance(val, (list, tuple)) and len(val) > 1:
                        src_out = int(val[1])
                except Exception:
                    src_out = 0
                logger.debug(f"[VVL] _resolve_int_input: 检测到 link，src_id={src_id}")
                node = dynprompt.get_node(src_id)
                ct = node.get("class_type", "")
                logger.debug(f"[VVL] _resolve_int_input: 源节点类型={ct}")

                # 1) 解析 Length Any 类节点（名字可能因插件不同而异，只要有 "any" 输入即可）
                if "length" in ct.lower() or ("length" in node.get("label", "").lower()):
                    logger.debug(f"[VVL] _resolve_int_input: 检测到 Length 节点")
                    inputs_dict = node.get("inputs", {})
                    # 兼容多种常见键名
                    candidate_keys = ("any", "anything", "array", "list", "values", "items", "input")
                    chosen_key = None
                    for k in candidate_keys:
                        if k in inputs_dict:
                            chosen_key = k
                            break
                    any_in = inputs_dict.get(chosen_key, None)
                    logger.debug(f"[VVL] _resolve_int_input: Length 节点的 {chosen_key or 'any'} 输入={any_in}")
                    # 如果 any 是 link，进一步解析其常量列表长度
                    length = None
                    if is_link(any_in):
                        try:
                            any_src = dynprompt.get_node(any_in[0])
                        except Exception:
                            # 尝试处理克隆节点的ID
                            base_id = str(any_in[0]).split("__")[0]
                            try:
                                any_src = dynprompt.get_node(base_id)
                            except Exception as e:
                                logger.debug(f"[VVL] _resolve_int_input: 无法获取源节点 {any_in[0]}: {e}")
                                any_src = None
                        
                        if any_src:
                            any_ct = any_src.get("class_type", "")

                            # 1.0) 特殊情况：listGetItem从chunks列表获取chunk
                            if "listgetitem" in any_ct.lower():
                                logger.debug(f"[VVL] _resolve_int_input: Length的输入是listGetItem")
                                getitem_inputs = any_src.get("inputs", {})
                                list_input = getitem_inputs.get("list", None)
                                index_input = getitem_inputs.get("index", None)
                                
                                # 尝试获取index值（可能是常量或来自forLoopStart）
                                index_val = None
                                if isinstance(index_input, int):
                                    index_val = index_input
                                elif is_link(index_input):
                                    # 如果index来自forLoopStart，在并行展开时会被注入实际值
                                    # 尝试从节点ID推断index（如果是克隆节点）
                                    # 例如：9.0.0.7__2 -> index=2
                                    try:
                                        node_id_str = str(any_in[0])  # listGetItem的ID
                                        if "__" in node_id_str:
                                            index_val = int(node_id_str.split("__")[-1])
                                            logger.debug(f"[VVL] _resolve_int_input: 从节点ID {node_id_str} 推断index={index_val}")
                                    except Exception as e:
                                        logger.debug(f"[VVL] _resolve_int_input: 无法从节点ID推断index: {e}")
                                
                                # 检查list_input是否来自listChunk的chunks输出
                                if is_link(list_input):
                                    try:
                                        chunk_src = dynprompt.get_node(list_input[0])
                                    except Exception:
                                        base_id = str(list_input[0]).split("__")[0]
                                        try:
                                            chunk_src = dynprompt.get_node(base_id)
                                        except Exception:
                                            chunk_src = None
                                    
                                    if chunk_src and "listchunk" in chunk_src.get("class_type", "").lower():
                                        logger.debug(f"[VVL] _resolve_int_input: listGetItem的list来自listChunk")
                                        # 获取listChunk的输入
                                        chunk_inputs = chunk_src.get("inputs", {})
                                        chunk_list = chunk_inputs.get("list", None)
                                        chunk_size = chunk_inputs.get("size", None)
                                        
                                        # 解析chunk_size
                                        size_val = self._resolve_int_input(dynprompt, chunk_size)
                                        
                                        # 解析原始列表长度
                                        list_len = None
                                        if is_link(chunk_list):
                                            try:
                                                list_src = dynprompt.get_node(chunk_list[0])
                                            except Exception:
                                                base_id = str(chunk_list[0]).split("__")[0]
                                                try:
                                                    list_src = dynprompt.get_node(base_id)
                                                except Exception:
                                                    list_src = None
                                            
                                            if list_src:
                                                list_ct = list_src.get("class_type", "")
                                                # String List
                                                if "string" in list_ct.lower() and "list" in list_ct.lower():
                                                    cfg = list_src.get("inputs", {})
                                                    text = str(cfg.get("list", "") or "")
                                                    use_nl = bool(cfg.get("new_line_as_separator", True))
                                                    sep = str(cfg.get("separator", ","))
                                                    items = [s for s in (text.splitlines() if use_nl else text.split(sep)) if str(s).strip() != ""]
                                                    list_len = len(items)
                                                # VVL listConstruct
                                                elif "listconstruct" in list_ct.lower():
                                                    cnt = 0
                                                    for i in range(0, 1024):
                                                        if f"item{i}" in list_src.get("inputs", {}):
                                                            cnt += 1
                                                        else:
                                                            break
                                                    list_len = cnt
                                        
                                        # 如果成功获取到原始列表长度和chunk大小
                                        if isinstance(list_len, int) and isinstance(size_val, int) and size_val > 0:
                                            # 计算chunk的实际长度
                                            # 注意：在并行展开时，index会被注入实际值（0, 1, 2...）
                                            # 这里我们无法知道具体的index，但可以根据模式推断
                                            # 对于嵌套循环，内层的Length通常需要当前chunk的实际长度
                                            
                                            # 计算总chunk数
                                            num_chunks = (list_len + size_val - 1) // size_val
                                            
                                            # 如果我们知道具体的index，可以计算准确的chunk长度
                                            if isinstance(index_val, int):
                                                if index_val < num_chunks - 1:
                                                    length = size_val
                                                else:
                                                    # 最后一个chunk
                                                    length = list_len - (num_chunks - 1) * size_val
                                                logger.debug(f"[VVL] _resolve_int_input: chunk[{index_val}]长度={length} (总长{list_len}, chunk_size={size_val})")
                                            else:
                                                # 不知道具体index时，尝试更智能的推断
                                                # 如果总长度不是chunk_size的整数倍，最后一个chunk会更小
                                                # 为了安全起见，返回最小可能的chunk长度
                                                last_chunk_size = list_len - (num_chunks - 1) * size_val
                                                if last_chunk_size < size_val:
                                                    # 有一个较小的最后chunk，保守地返回较小值
                                                    # 这样可以避免越界访问
                                                    length = min(size_val, last_chunk_size)
                                                    logger.debug(f"[VVL] _resolve_int_input: 未知index，保守估计chunk长度={length} (可能是最后一个chunk)")
                                                else:
                                                    length = size_val
                                                    logger.debug(f"[VVL] _resolve_int_input: 未知index，假设chunk长度={size_val}")
                                            return length

                            # 1.1) String List：从文本解析
                            elif "string" in any_ct.lower() and "list" in any_ct.lower():
                                logger.debug(f"[VVL] _resolve_int_input: 检测到 String List 节点")
                                cfg = any_src.get("inputs", {})
                                text = str(cfg.get("list", "") or "")
                                use_nl = bool(cfg.get("new_line_as_separator", True))
                                sep = str(cfg.get("separator", ","))
                                items = [s for s in (text.splitlines() if use_nl else text.split(sep)) if str(s).strip() != ""]
                                length = len(items)
                                logger.debug(f"[VVL] _resolve_int_input: String List 解析出 {length} 个项目: {items[:5]}...")

                            # 1.2) VVL listConstruct：统计 itemN 个数
                            elif "listconstruct" in any_ct.lower():
                                logger.debug(f"[VVL] _resolve_int_input: 检测到 VVL listConstruct 节点")
                                cnt = 0
                                for i in range(0, 1024):
                                    if f"item{i}" in any_src.get("inputs", {}):
                                        cnt += 1
                                    else:
                                        break
                                length = cnt
                                logger.debug(f"[VVL] _resolve_int_input: VVL listConstruct 统计出 {length} 个项目")



                            # 1.4) JsonExtractSubjectNamesScales：尝试静态解析 JSON 获取 objects 长度
                            elif "jsonextractsubjectnamesscales" in any_ct.lower():
                                logger.debug(f"[VVL] _resolve_int_input: 检测到 JsonExtractSubjectNamesScales 节点，尝试静态解析 JSON")
                                length = self._try_resolve_length_from_jsonextract(dynprompt, any_src)
                                logger.debug(f"[VVL] _resolve_int_input: JsonExtractSubjectNamesScales 静态解析长度={length}")

                            # 1.5) ApplyUrlsToJson：可从其输入 JSON 推断 objects 长度
                            elif "applyurlstojson" in any_ct.lower():
                                logger.debug(f"[VVL] _resolve_int_input: 检测到 ApplyUrlsToJson 节点，尝试从 deduplicated_json 常量推断长度")
                                try:
                                    dedup = any_src.get("inputs", {}).get("deduplicated_json", None)
                                    s = self._resolve_constant_string(dynprompt, dedup)
                                    if isinstance(s, str) and s:
                                        import json as _json
                                        length = len((_json.loads(s) or {}).get("objects", []))
                                except Exception as e:
                                    logger.debug(f"[VVL] _resolve_int_input: 解析 ApplyUrlsToJson 时异常: {e}")

                            # 1.6) 通用 JSON 数组输入键：json_array / array / list
                            else:
                                try:
                                    for key in ("json_array", "array", "list"):
                                        if key in any_src.get("inputs", {}):
                                            raw = any_src.get("inputs", {}).get(key)
                                            s = self._resolve_constant_string(dynprompt, raw)
                                            if isinstance(s, str) and s:
                                                import json as _json
                                                try:
                                                    arr = _json.loads(s)
                                                    if isinstance(arr, list):
                                                        length = len(arr)
                                                        logger.debug(f"[VVL] _resolve_int_input: 通用 {key} 常量 JSON 数组长度={length}")
                                                        break
                                                except Exception:
                                                    pass
                                except Exception as e:
                                    logger.debug(f"[VVL] _resolve_int_input: 通用 JSON 数组键解析异常: {e}")

                    # 如果 any 直接是常量 list（极少见），也可直接 len()
                    if length is None:
                        any_val = inputs_dict.get(chosen_key or "any", None)
                        if isinstance(any_val, list):
                            length = len(any_val)

                    if isinstance(length, int):
                        logger.debug(f"[VVL] _resolve_int_input: Length 节点最终解析结果: {length}")
                        return length
                    else:
                        logger.debug(f"[VVL] _resolve_int_input: Length 节点解析失败，length={length}")

                # 2) 解析 VVL listChunk：当 total 连接到其 num_chunks 输出时可以静态计算
                if "listchunk" in ct.lower():
                    logger.debug(f"[VVL] _resolve_int_input: 检测到 VVL listChunk 节点，尝试静态解析")
                    try:
                        inputs = node.get("inputs", {})
                        list_in = inputs.get("list", None)
                        size_in = inputs.get("size", None)

                        # 解析 size
                        size_val = self._resolve_int_input(dynprompt, size_in)

                        # 解析 list 的长度
                        def _resolve_list_length(val_obj):
                            try:
                                if is_link(val_obj):
                                    any_src = dynprompt.get_node(val_obj[0])
                                    any_ct = any_src.get("class_type", "")
                                    # String List
                                    if "string" in any_ct.lower() and "list" in any_ct.lower():
                                        cfg = any_src.get("inputs", {})
                                        text = str(cfg.get("list", "") or "")
                                        use_nl = bool(cfg.get("new_line_as_separator", True))
                                        sep = str(cfg.get("separator", ","))
                                        items = [s for s in (text.splitlines() if use_nl else text.split(sep)) if str(s).strip() != ""]
                                        return len(items)
                                    # VVL listConstruct
                                    if "listconstruct" in any_ct.lower():
                                        cnt = 0
                                        for i in range(0, 1024):
                                            if f"item{i}" in any_src.get("inputs", {}):
                                                cnt += 1
                                            else:
                                                break
                                        return cnt
                                # 常量 python 列表
                                if isinstance(val_obj, list):
                                    return len(val_obj)
                            except Exception:
                                pass
                            return None

                        list_len = _resolve_list_length(list_in)
                        logger.debug(f"[VVL] _resolve_int_input: listChunk 静态解析 list_len={list_len}, size={size_val}, src_out={src_out}")

                        # 只有当请求的输出口为 num_chunks(=1) 且两者可解析时才返回
                        if src_out == 1 and isinstance(list_len, int) and isinstance(size_val, int) and size_val > 0:
                            num_chunks = (list_len + size_val - 1) // size_val
                            logger.debug(f"[VVL] _resolve_int_input: 计算得到 num_chunks={num_chunks}")
                            return num_chunks
                    except Exception as e:
                        logger.debug(f"[VVL] _resolve_int_input: 解析 listChunk 异常: {e}")

                # 2) 解析 VVL mathInt（递归解析 a、b）
                if "math" in ct.lower() and "int" in ct.lower():
                    logger.debug(f"[VVL] _resolve_int_input: 检测到 Math Int 节点")
                    op = node.get("inputs", {}).get("operation", "add")
                    a = node.get("inputs", {}).get("a", 0)
                    b = node.get("inputs", {}).get("b", 0)
                    logger.debug(f"[VVL] _resolve_int_input: Math Int 操作={op}, a={a}, b={b}")
                    ai = self._resolve_int_input(dynprompt, a)
                    bi = self._resolve_int_input(dynprompt, b)
                    logger.debug(f"[VVL] _resolve_int_input: Math Int 递归解析结果 ai={ai}, bi={bi}")
                    if isinstance(ai, int) and isinstance(bi, int):
                        result = None
                        if op == "add":      result = ai + bi
                        elif op == "subtract": result = ai - bi
                        elif op == "multiply": result = ai * bi
                        elif op == "divide":   result = (ai // bi) if bi != 0 else 0
                        elif op == "modulo":   result = (ai % bi) if bi != 0 else 0
                        logger.debug(f"[VVL] _resolve_int_input: Math Int 计算结果: {ai} {op} {bi} = {result}")
                        return result

        except Exception as e:
            logger.debug(f"[VVL] _resolve_int_input: 解析过程中发生异常: {e}")
            pass
        
        logger.debug(f"[VVL] _resolve_int_input: 无法解析，返回 None")
        return None

    def _resolve_constant_string(self, dynprompt, val):
        """尽量将一个输入解析为编译期可用的常量字符串。解析失败返回 None。"""
        try:
            if isinstance(val, str):
                return val
            if is_link(val):
                src = dynprompt.get_node(val[0])
                ct = src.get("class_type", "")
                # JsonMarkdownCleaner: 复用其清理逻辑
                if "jsonmarkdowncleaner" in ct.lower():
                    nested = src.get("inputs", {}).get("json_text", None)
                    s = self._resolve_constant_string(dynprompt, nested)
                    if isinstance(s, str):
                        txt = s.strip()
                        if txt.startswith("```json"):
                            txt = txt[7:].strip()
                        elif txt.startswith("```"):
                            txt = txt[3:].strip()
                        if txt.endswith("```"):
                            txt = txt[:-3].strip()
                        return txt
                # 一般性透传：尝试常见键
                for key in ("text", "string", "json_text", "value"):
                    v = src.get("inputs", {}).get(key, None)
                    if isinstance(v, str) and not is_link(v):
                        return v
        except Exception as e:
            logger.debug(f"[VVL] _resolve_constant_string: 解析异常: {e}")
        return None

    def _try_resolve_length_from_jsonextract(self, dynprompt, node):
        """针对 JsonExtractSubjectNamesScales，在编译期从其 json_text 常量中解析 objects 长度。失败返回 None。"""
        try:
            inputs = node.get("inputs", {})
            json_val = inputs.get("json_text", None)
            json_text = self._resolve_constant_string(dynprompt, json_val)
            logger.debug(f"[VVL] _try_resolve_length_from_jsonextract: 解析到 json_text 常量={True if isinstance(json_text, str) else False}")
            if isinstance(json_text, str) and json_text:
                import json as _json
                try:
                    data = _json.loads(json_text)
                except Exception as e:
                    logger.debug(f"[VVL] _try_resolve_length_from_jsonextract: JSON 解析失败: {e}")
                    return None
                objects = data.get("objects", [])
                if isinstance(objects, list):
                    return len(objects)
                return 0
        except Exception as e:
            logger.debug(f"[VVL] _try_resolve_length_from_jsonextract: 解析异常: {e}")
        return None

    def _build_parallel(self, flow, dynprompt, unique_id, kwargs):
        logger.debug(f"[VVL] _build_parallel: 开始并行展开检查")
        graph = GraphBuilder()

        while_open = flow[0]
        total = None
        initial_from_start = {}
        parallel_flag = False

        # 改进的嵌套循环检测：
        # 1. 检查是否已经是多层嵌套（避免过深的递归）
        # 2. 允许更深层级的并行展开（提高到4层）
        nest_level = str(unique_id).count('.')
        is_cloned = '__' in str(unique_id)
        
        # 只在嵌套层级非常深时跳过并行展开（提高限制到4层）
        if nest_level > 4:
            logger.debug(f"[VVL] _build_parallel: 嵌套层级过深 (level={nest_level})，跳过并行展开")
            return None
        
        # 如果是克隆节点但嵌套层级合理，仍然尝试并行展开
        if is_cloned and nest_level <= 4:
            logger.debug(f"[VVL] _build_parallel: 检测到克隆节点 (level={nest_level})，但层级合理，继续尝试并行展开")

        # 通过 dynprompt 获取 start 节点设置
        try:
            # 对于克隆节点，需要处理可能的 display_id
            if is_cloned:
                # 尝试获取原始节点ID（去掉 __idx 后缀）
                base_id = str(while_open).split("__")[0]
                try:
                    forstart_node = dynprompt.get_node(while_open)
                except Exception:
                    forstart_node = dynprompt.get_node(base_id)
            else:
                forstart_node = dynprompt.get_node(while_open)
        except Exception as e:
            logger.debug(f"[VVL] _build_parallel: 无法获取 forLoopStart 节点: {e}，跳过")
            return None
            
        logger.debug(f"[VVL] _build_parallel: forLoopStart 节点类型={forstart_node.get('class_type')}")
        if forstart_node['class_type'] == 'VVL forLoopStart':
            inputs = forstart_node['inputs']
            total = inputs.get('total', None)
            logger.debug(f"[VVL] _build_parallel: 原始 total 值={total}, 类型={type(total)}")
            # 新增：尝试把 total 解析为 int
            total = self._resolve_int_input(dynprompt, total)
            logger.debug(f"[VVL] _build_parallel: 解析后 total 值={total}, 类型={type(total)}")
            parallel_flag = inputs.get('parallel', False)
            logger.debug(f"[VVL] _build_parallel: parallel 标志={parallel_flag}")
            for i in range(MAX_FLOW_NUM):
                key = f"initial_value{i}"
                if key in inputs:
                    initial_from_start[key] = inputs[key]

        # 仅当 total 为 int 且 parallel 为 True 时并发展开
        if not isinstance(total, int) or not parallel_flag or total <= 0:
            logger.debug(f"[VVL] _build_parallel: 不满足并行条件，返回 None - total={total}, parallel={parallel_flag}")
            return None
        
        logger.debug(f"[VVL] _build_parallel: 满足并行条件，开始并行展开 {total} 个迭代")

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
        logger.debug(f"[VVL] _build_parallel: 开始克隆子图，包含 {len(contained_ids)} 个节点")
        clone_map = {}
        for idx in range(total):
            clone_map[idx] = {}
            for node_id in contained_ids:
                original_node = dynprompt.get_node(node_id)
                # 创建克隆节点，并显式设置 display_id，确保运行期 unique_id 与 prompt 键一致
                clone_display_id = f"{node_id}__{idx}"
                original_ct = original_node["class_type"]
                clone = graph.node(original_ct, clone_display_id)
                try:
                    clone.set_override_display_id(clone_display_id)
                except Exception:
                    pass
                clone_map[idx][node_id] = clone
        logger.debug(f"[VVL] _build_parallel: 克隆子图完成")

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

                            # 对于依赖 forLoopStart.index 的输入，需要区分外层和内层
                            if parent_class_type == 'VVL forLoopStart':
                                # 检查是否是外层循环的forLoopStart
                                try:
                                    parent_node_id = v[0]
                                    if parent_node_id == while_open:
                                        # 这是外层循环的index，注入当前idx
                                        if k == 'index' or v[1] == 1:  # index输入或index输出端口
                                            clone.set_input(k, idx)
                                            continue
                                except Exception:
                                    pass
                                # 否则保持原始链接（内层循环的index）

                        clone.set_input(k, v)

                # 特殊处理 forLoopStart：为每个克隆设置不同的 index（initial_value0）
                # 但需要区分外层和内层循环
                if original_node["class_type"] == 'VVL forLoopStart':
                    # 检查是否是外层循环的forLoopStart（即我们正在并行展开的那个）
                    # 外层循环的forLoopStart节点ID应该与while_open匹配
                    if node_id == open_node:
                        # 这是外层循环，设置不同的idx
                        clone.set_input("initial_value0", idx)
                        for i in range(1, MAX_FLOW_NUM):
                            key = f"initial_value{i}"
                            if key in initial_from_start:
                                clone.set_input(key, initial_from_start[key])
                    else:
                        # 这是内层循环，保持原始的initial_value0（通常为0）
                        pass  # 不修改，保持原始输入

        # 根据 forLoopEnd 的输入，聚合每个需要输出的值为列表（动态处理）
        results = []
        
        # 首先收集所有实际传入的initial_value参数
        actual_inputs = {}
        for key, value in kwargs.items():
            if key.startswith("initial_value") and value is not None:
                actual_inputs[key] = value
        
        # 按MAX_FLOW_NUM处理，以保持兼容性
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

        logger.debug(f"[VVL] _build_parallel: 并行展开完成，返回 {len(results)} 个结果")
        return {
            "result": tuple(results),
            "expand": graph.finalize(),
        }

    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        logger.debug(f"[VVL] for_loop_end: 开始处理，unique_id={unique_id}")
        # 优先尝试并发展开
        parallel_result = self._build_parallel(flow, dynprompt, unique_id, kwargs)
        if parallel_result is not None:
            logger.debug(f"[VVL] for_loop_end: 使用并行展开结果")
            return parallel_result
        
        logger.debug(f"[VVL] for_loop_end: 并行展开失败，使用顺序 while 循环")
        graph = GraphBuilder()
        while_open = flow[0]
        total = None

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(while_open)
        if forstart_node['class_type'] == 'VVL forLoopStart':
            inputs = forstart_node['inputs']
            total_raw = inputs['total']
            # 在顺序模式下也要解析 total 值
            total = self._resolve_int_input(dynprompt, total_raw)
            logger.debug(f"[VVL] for_loop_end: 顺序模式解析 total: {total_raw} -> {total}")
            # 如果解析失败，回退到原始值
            if total is None:
                total = total_raw
            
            # 检查initial_value0，确保内层循环从0开始
            initial_idx = inputs.get('initial_value0', 0)
            if isinstance(initial_idx, list) and is_link(initial_idx):
                # 如果initial_value0是一个link，可能来自外层循环
                # 对于内层循环，应该使用0
                logger.debug(f"[VVL] for_loop_end: 检测到initial_value0是link，可能是嵌套循环")
                # 内层循环应该从0开始
                initial_idx = 0
            logger.debug(f"[VVL] for_loop_end: initial_value0={initial_idx}")
        elif forstart_node['class_type'] == 'easy loadImagesForLoop':
            inputs = forstart_node['inputs']
            limit = inputs['limit']
            start_index = inputs['start_index']
            # Filter files by extension
            directory = inputs['directory']
            total = graph.node('easy imagesCountInDirectory', directory=directory, limit=limit, start_index=start_index, extension='*').out(0)

        # 获取当前循环的起始值（通常为0）
        current_index = [while_open, 1]  # 从forLoopStart的index输出获取当前索引
        
        # 下一个索引 = 当前索引 + 1
        sub = graph.node("VVL mathInt", operation="add", a=current_index, b=1)
        
        # 比较：下一个索引 < total
        cond = graph.node("VVL compare", a=sub.out(0), b=total, comparison='a < b')
        
        # 动态收集input_values，支持任意数量的initial_value输入
        input_values = {}
        for key, value in kwargs.items():
            if key.startswith("initial_value"):
                input_values[key] = value
        
        # 补齐到MAX_FLOW_NUM以保持兼容性
        for i in range(1, MAX_FLOW_NUM):
            key = f"initial_value{i}"
            if key not in input_values:
                input_values[key] = None
        
        while_close = graph.node("VVL whileLoopEnd",
                                 flow=flow,
                                 condition=cond.out(0),
                                 initial_value0=sub.out(0),
                                 **input_values)
        logger.debug(f"[VVL] for_loop_end: 顺序 while 循环完成")
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
        try:
            node_info = dynprompt.get_node(node_id)
        except Exception:
            # 回退到原始 id（去掉 __idx）
            try:
                base_id = str(node_id).split("__")[0]
                node_info = dynprompt.get_node(base_id)
                node_id = base_id
            except Exception:
                return
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                try:
                    display_node = dynprompt.get_node(display_id)
                except Exception:
                    base_disp = str(display_id).split("__")[0]
                    display_node = dynprompt.get_node(base_disp)
                    display_id = base_disp
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


class VVLListLengthAsync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (any_type,),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_length"
    CATEGORY = "VVL/Utils"

    def get_length(self, any):
        try:
            # 优先尝试 len()
            try:
                return (int(len(any)),)
            except Exception:
                pass

            # 可迭代物化
            try:
                materialized = [x for x in any]
                return (int(len(materialized)),)
            except Exception:
                pass

            # String List 源常量
            if isinstance(any, str):
                lines = [s for s in any.splitlines() if s.strip() != ""]
                return (len(lines),)
        except Exception:
            pass
        return (0,)



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
    "VVL listLength": VVLListLengthAsync,
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
    "VVL listLength": "VVL List Length (Async)",
}