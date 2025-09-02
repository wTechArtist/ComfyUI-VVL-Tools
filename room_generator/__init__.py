"""
房间生成器模块
包含房间边界框生成和ComfyUI节点
"""

from .generate_room_bbox import RoomGenerator, CollisionError
from .nodes_room_generator import RoomBboxGeneratorNode, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['RoomGenerator', 'CollisionError', 'RoomBboxGeneratorNode', 'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']