"""
Dynamic Batch Anything Node

动态批处理任意类型输入；根据现有输入端口的使用情况在前端自动添加/移除输入端口，
无需按钮与 inputcount 参数。
"""

import torch
import comfy.utils

# 尝试从py.libs.utils导入所需的工具
try:
    from py.libs.utils import AlwaysEqualProxy, compare_revision
    any_type = AlwaysEqualProxy("*")
except ImportError:
    # 如果无法导入，使用简单的字符串替代
    any_type = "*"


class FlexibleOptionalInputType(dict):
    """用于可变可选输入的类型容器。

    - 任何未显式声明的 key，在被访问时都返回 (type,) 作为其类型定义
    - __contains__ 始终返回 True，从而允许前端动态增加可选输入端口
    """

    def __init__(self, type, data=None):
        self.type = type
        self.data = data or {}
        # 将初始数据映射到自身，使其在 UI 上可见
        for k, v in self.data.items():
            self[k] = v

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True

class DynamicBatchAnything:
    """
    动态批处理任意类型数据的节点
    
    功能特点：
    1. 使用inputcount参数控制输入数量
    2. 自动处理不同类型的数据批处理
    3. 支持图像、潜在向量、字符串、数字等类型
    4. 当输入为None时自动跳过
    5. 最多支持1000个输入端口
    """
    
    DESCRIPTION = """
Dynamic batch processing for any type of data.
Automatically grows/shrinks input ports based on usage.
"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 使用可变可选输入；提供两个初始输入端口
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(any_type, {
                "input_1": (any_type,),
                "input_2": (any_type,),
            }),
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch_output",)
    FUNCTION = "dynamic_batch"
    CATEGORY = "VVL/Logic"
    
    def latent_batch(self, latent_1, latent_2):
        """批处理潜在向量"""
        samples_out = latent_1.copy()
        s1 = latent_1["samples"]
        s2 = latent_2["samples"]

        # 如果尺寸不匹配，调整到相同尺寸
        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        
        # 拼接样本
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        
        # 处理批次索引
        samples_out["batch_index"] = latent_1.get("batch_index", 
                                                 [x for x in range(0, s1.shape[0])]) + latent_2.get(
            "batch_index", [x for x in range(0, s2.shape[0])])

        return samples_out
    
    def batch_two_items(self, item_1, item_2):
        """批处理两个项目"""
        if item_1 is None:
            return item_2
        elif item_2 is None:
            return item_1
            
        # 处理张量类型
        if isinstance(item_1, torch.Tensor) and isinstance(item_2, torch.Tensor):
            # 如果尺寸不匹配，调整第二个张量的尺寸
            if item_1.shape[1:] != item_2.shape[1:]:
                item_2 = comfy.utils.common_upscale(
                    item_2.movedim(-1, 1), 
                    item_1.shape[2], 
                    item_1.shape[1], 
                    "bilinear", 
                    "center"
                ).movedim(1, -1)
            return torch.cat((item_1, item_2), 0)
        
        # 处理基本类型 (字符串、浮点数、整数)
        elif isinstance(item_1, (str, float, int)):
            if isinstance(item_2, tuple):
                return item_2 + (item_1,)
            elif isinstance(item_2, list):
                return item_2 + [item_1]
            return [item_1, item_2]
        
        elif isinstance(item_2, (str, float, int)):
            if isinstance(item_1, tuple):
                return item_1 + (item_2,)
            elif isinstance(item_1, list):
                return item_1 + [item_2]
            return [item_1, item_2]
        
        # 处理潜在向量 (字典类型，包含'samples'键)
        elif isinstance(item_1, dict) and 'samples' in item_1:
            if isinstance(item_2, dict) and 'samples' in item_2:
                return self.latent_batch(item_1, item_2)
            return item_1
        
        elif isinstance(item_2, dict) and 'samples' in item_2:
            return item_2
        
        # 处理列表/元组类型
        elif isinstance(item_1, (list, tuple)) and isinstance(item_2, (list, tuple)):
            if isinstance(item_1, tuple) and isinstance(item_2, tuple):
                return item_1 + item_2
            else:
                return list(item_1) + list(item_2)
        
        elif isinstance(item_1, (list, tuple)):
            if isinstance(item_1, tuple):
                return item_1 + (item_2,)
            else:
                return item_1 + [item_2]
        
        elif isinstance(item_2, (list, tuple)):
            if isinstance(item_2, tuple):
                return (item_1,) + item_2
            else:
                return [item_1] + item_2
        
        # 默认情况：尝试连接
        else:
            try:
                return item_1 + item_2
            except:
                # 如果无法连接，返回列表
                return [item_1, item_2]

    def dynamic_batch(self, **kwargs):
        """动态批处理函数"""
        # 收集所有非None的输入（按 input_编号 排序）
        valid_inputs = []
        input_keys = [k for k in kwargs.keys() if isinstance(k, str) and k.startswith("input_")]
        def _key_to_index(k: str) -> int:
            try:
                return int(k.split("_")[1])
            except Exception:
                return 0
        for input_key in sorted(input_keys, key=_key_to_index):
            if kwargs.get(input_key, None) is not None:
                valid_inputs.append(kwargs[input_key])
        
        # 如果没有有效输入，返回None
        if not valid_inputs:
            return (None,)
        
        # 如果只有一个输入，直接返回
        if len(valid_inputs) == 1:
            return (valid_inputs[0],)
        
        # 批处理多个输入
        result = valid_inputs[0]
        for i in range(1, len(valid_inputs)):
            result = self.batch_two_items(result, valid_inputs[i])
        
        return (result,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "DynamicBatchAnything": DynamicBatchAnything,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicBatchAnything": "VVL Dynamic Batch Any",
}