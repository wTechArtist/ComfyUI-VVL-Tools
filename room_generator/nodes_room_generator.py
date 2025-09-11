"""
ComfyUI节点：房间边界框生成器
基于 generate_room_bbox.py 的 RoomGenerator 类实现
"""

import json
from typing import Dict, Any, Tuple
from .generate_room_bbox import RoomGenerator, CollisionError


class RoomBboxGeneratorNode:
    """
    ComfyUI节点：根据房间内部尺寸生成3D房间结构JSON
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length_m": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "房间内部长度（X轴），单位：米"
                }),
                "width_m": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "房间内部宽度（Y轴），单位：米"
                }),
                "height_m": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "房间内部高度（Z轴），单位：米"
                }),
                "wall_thickness_cm": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "墙体厚度，单位：厘米"
                }),
            },
            "optional": {
                "floor_thickness_cm": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "地板厚度，单位：厘米。如果不指定，默认使用墙体厚度"
                }),
                "ceiling_thickness_cm": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "天花板厚度，单位：厘米。如果不指定，默认使用墙体厚度"
                }),
                "subject": ("STRING", {
                    "default": "modern style",
                    "multiline": False,
                    "tooltip": "场景主题描述"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("room_json_formatted",)
    FUNCTION = "generate_room"
    CATEGORY = "VVL-Tools/3D Room Generator"
    
    def generate_room(self, length_m, width_m, height_m, wall_thickness_cm, 
                     floor_thickness_cm=None, ceiling_thickness_cm=None, 
                     subject="modern style"):
        """
        生成房间JSON数据
        
        Args:
            length_m: 房间内部长度（米）
            width_m: 房间内部宽度（米）
            height_m: 房间内部高度（米）
            wall_thickness_cm: 墙体厚度（厘米）
            floor_thickness_cm: 地板厚度（厘米）
            ceiling_thickness_cm: 天花板厚度（厘米）
            subject: 场景主题描述
            
        Returns:
            str: 格式化的room_json字符串
        """
        try:
            # 创建房间生成器
            generator = RoomGenerator(
                length_m=length_m,
                width_m=width_m,
                height_m=height_m,
                wall_thickness_cm=wall_thickness_cm,
                floor_thickness_cm=floor_thickness_cm,
                ceiling_thickness_cm=ceiling_thickness_cm,
                safety_margin_cm=0.0,  # 使用默认安全间距
                scene_style=subject  # 使用subject作为scene_style传递给RoomGenerator
            )
            
            # 生成房间JSON
            room_data = generator.generate_room_json()
            
            # 转换为格式化的JSON字符串
            room_json_formatted = json.dumps(room_data, indent=4, ensure_ascii=False)
            
            return (room_json_formatted,)
            
        except (ValueError, CollisionError) as e:
            error_msg = f"房间生成错误: {str(e)}"
            print(f"[RoomBboxGeneratorNode] {error_msg}")
            # 返回错误信息作为格式化JSON
            error_data = {"error": error_msg}
            return (json.dumps(error_data, indent=4, ensure_ascii=False),)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            print(f"[RoomBboxGeneratorNode] {error_msg}")
            error_data = {"error": error_msg}
            return (json.dumps(error_data, indent=4, ensure_ascii=False),)


# 导出类供外部使用
__all__ = ['RoomBboxGeneratorNode']