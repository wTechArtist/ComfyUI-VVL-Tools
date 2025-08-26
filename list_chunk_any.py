"""
VVL List Chunk (Async)

将任意列表/可迭代对象按给定大小切分为块列表：[[...size...], [...], ...]

特点：
- 输入为任意类型（any_type），兼容普通 list/tuple、ComfyUI 中的张量/对象列表等
- size <= 0 时回退为整列表单块
- 支持非下标可迭代对象：会物化为列表再分块
"""

from typing import Any, List

try:
    from py.libs.utils import AlwaysEqualProxy
    any_type = AlwaysEqualProxy("*")
except Exception:
    class _AlwaysEqualProxy(str):
        def __ne__(self, __value):
            return False
    any_type = _AlwaysEqualProxy("*")


class VVLListChunkAsync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": (any_type,),
                "size": ("INT", {"default": 5, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("chunks", "num_chunks")
    FUNCTION = "chunk"
    CATEGORY = "VVL/Utils"

    def _to_list(self, value: Any) -> List[Any]:
        # 已是 list
        if isinstance(value, list):
            return value
        # 元组 → 列表
        if isinstance(value, tuple):
            return list(value)
        # 可下标且能取 len
        try:
            if hasattr(value, "__getitem__"):
                try:
                    length = len(value)
                except Exception:
                    length = None
                if length is not None:
                    return [value[i] for i in range(length)]
        except Exception:
            pass
        # 一般可迭代
        try:
            return [x for x in value]
        except Exception:
            # 不是序列，就当作单元素列表
            return [value]

    def chunk(self, list, size):
        items = self._to_list(list)
        if size is None or size <= 0:
            size = max(1, len(items))

        chunks: List[Any] = []
        if len(items) == 0:
            return ([], 0)

        for i in range(0, len(items), size):
            chunks.append(items[i:i + size])

        return (chunks, len(chunks))


NODE_CLASS_MAPPINGS = {
    "VVL listChunk": VVLListChunkAsync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL listChunk": "VVL List Chunk (Async)",
}

