"""
ExecutionBlocker utilities for VVL nodes
"""

# 直接使用Easy-Use的类型定义
import asyncio
import random


try:
    import sys
    import os
    # 添加Easy-Use路径
    easy_use_path = os.path.join(os.path.dirname(__file__), '..', 'ComfyUI-Easy-Use')
    if easy_use_path not in sys.path:
        sys.path.append(easy_use_path)
    
    from py.libs.utils import AlwaysEqualProxy
    any_type = AlwaysEqualProxy("*")
except ImportError:
    # 如果Easy-Use不可用，使用简单的字符串
    any_type = "*"

class async_sleep_v1:
    """
    检测ExecutionBlocker并自动切换到备用输入
    专门配合Easy-Use A or B节点使用
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test1": (any_type,),   
                "sleep_mode": (["constant", "random"], {
                    "default": "constant",
                    "tooltip": "睡眠模式：常量时间或随机时间"
                }),
                "sleep_time": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 60.0, 
                    "step": 0.1,
                    "tooltip": "常量模式下的睡眠时间，或随机模式下的最小时间，单位秒"
                }),
                "max_sleep_time": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.1, 
                    "max": 60.0, 
                    "step": 0.1,
                    "tooltip": "随机模式下的最大睡眠时间，单位秒（仅在随机模式下生效）"
                }),
                "add_url_suffix": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否为字符串添加'_url'后缀"
                }),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "VVL/utils"
    
    async def process(self, test1, sleep_mode, sleep_time, max_sleep_time, add_url_suffix):
        # 根据睡眠模式计算实际睡眠时间
        if sleep_mode == "random":
            # 确保最小时间不大于最大时间
            min_time = min(sleep_time, max_sleep_time)
            max_time = max(sleep_time, max_sleep_time)
            # 生成随机浮点数时间
            actual_sleep_time = random.uniform(min_time, max_time)
        else:
            # 常量模式
            actual_sleep_time = sleep_time
        
        # 执行异步睡眠
        await asyncio.sleep(actual_sleep_time)
    
        # 根据开关决定是否为字符串添加"_url"后缀
        if add_url_suffix and isinstance(test1, str):
            return (test1 + "_url",)
        else:
            return (test1,)



# Node class mappings
NODE_CLASS_MAPPINGS = {
    "async_sleep_v1": async_sleep_v1,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "async_sleep_v1": "VVL async_sleep_v1",
}