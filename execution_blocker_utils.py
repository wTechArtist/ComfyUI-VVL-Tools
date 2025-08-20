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
                "sleep_time": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "睡眠时间，单位秒"
                }),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "VVL/utils"
    
    async def process(self, test1, sleep_time):

        await asyncio.sleep(sleep_time)
        # await asyncio.sleep()
    
        # 如果test1是字符串，则添加"_url"后缀；否则直接返回原值
        if isinstance(test1, str):
            return (test1 + "_url",)
        else:
            return (test1,)

        # return (test1 + "_url",)



# Node class mappings
NODE_CLASS_MAPPINGS = {
    "async_sleep_v1": async_sleep_v1,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "async_sleep_v1": "VVL async_sleep_v1",
}