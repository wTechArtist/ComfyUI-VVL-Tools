"""
ComfyUI VVL Test Nodes

This package contains custom nodes for various utilities:
1. JsonObjectDeduplicator - Remove duplicate objects based on name and scale
2. JsonObjectMerger - Merge processed JSON with removed duplicates
3. DynamicBatchAnything - Dynamic batch processing for any type of data
4. VVL Loop Control Nodes - For/While loop control with async support
5. TensorListPreview - Preview special format tensor lists with shape [1, H, W, 3]

Author: VVL Test
Version: 1.0.0
"""

from .nodes_json_object_utils import NODE_CLASS_MAPPINGS as JSON_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as JSON_DISPLAY_MAPPINGS
from .execution_blocker_utils import NODE_CLASS_MAPPINGS as BLOCKER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BLOCKER_DISPLAY_MAPPINGS
from .batch_anything_dynamic import NODE_CLASS_MAPPINGS as BATCH_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BATCH_DISPLAY_MAPPINGS
from .enhanced_lambert_renderer_node import NODE_CLASS_MAPPINGS as RENDERER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as RENDERER_DISPLAY_MAPPINGS
from .nodes_loop_control import NODE_CLASS_MAPPINGS as LOOP_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOOP_DISPLAY_MAPPINGS
from .tensor_list_preview import NODE_CLASS_MAPPINGS as TENSOR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TENSOR_DISPLAY_MAPPINGS
# 合并所有节点映射
NODE_CLASS_MAPPINGS = {**JSON_MAPPINGS, **BLOCKER_MAPPINGS, **BATCH_MAPPINGS, **RENDERER_MAPPINGS, **LOOP_MAPPINGS, **TENSOR_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**JSON_DISPLAY_MAPPINGS, **BLOCKER_DISPLAY_MAPPINGS, **BATCH_DISPLAY_MAPPINGS, **RENDERER_DISPLAY_MAPPINGS, **LOOP_DISPLAY_MAPPINGS, **TENSOR_DISPLAY_MAPPINGS}

# 让前端自动加载 web/ 下的脚本（如 js 扩展）
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']