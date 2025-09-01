"""
ComfyUI-VVL-Tools
A collection of custom nodes for ComfyUI
"""

import logging
import os
logger = logging.getLogger(__name__)

# 导入节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 声明web目录以支持前端扩展
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# 从各个模块导入节点
try:
    from .nodes_loop_control import NODE_CLASS_MAPPINGS as LOOP_MAPPINGS
    from .nodes_loop_control import NODE_DISPLAY_NAME_MAPPINGS as LOOP_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(LOOP_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LOOP_DISPLAY_MAPPINGS)
    logger.info("Loaded loop control nodes")
except ImportError as e:
    logger.warning(f"Failed to load loop control nodes: {e}")

try:
    from .list_chunk_any import NODE_CLASS_MAPPINGS as CHUNK_MAPPINGS
    from .list_chunk_any import NODE_DISPLAY_NAME_MAPPINGS as CHUNK_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(CHUNK_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CHUNK_DISPLAY_MAPPINGS)
    logger.info("Loaded list chunk node")
except ImportError as e:
    logger.warning(f"Failed to load list chunk node: {e}")

try:
    from .execution_blocker_utils import NODE_CLASS_MAPPINGS as BLOCKER_MAPPINGS
    from .execution_blocker_utils import NODE_DISPLAY_NAME_MAPPINGS as BLOCKER_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(BLOCKER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BLOCKER_DISPLAY_MAPPINGS)
    logger.info("Loaded execution blocker utils")
except ImportError as e:
    logger.warning(f"Failed to load execution blocker utils: {e}")

try:
    from .batch_anything_dynamic import NODE_CLASS_MAPPINGS as BATCH_MAPPINGS
    from .batch_anything_dynamic import NODE_DISPLAY_NAME_MAPPINGS as BATCH_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(BATCH_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BATCH_DISPLAY_MAPPINGS)
    logger.info("Loaded batch anything dynamic node")
except ImportError as e:
    logger.warning(f"Failed to load batch anything dynamic node: {e}")

try:
    from .enhanced_lambert_renderer_node import NODE_CLASS_MAPPINGS as RENDERER_MAPPINGS
    from .enhanced_lambert_renderer_node import NODE_DISPLAY_NAME_MAPPINGS as RENDERER_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(RENDERER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(RENDERER_DISPLAY_MAPPINGS)
    logger.info("Loaded enhanced Lambert 3D renderer node")
except ImportError as e:
    logger.warning(f"Failed to load enhanced Lambert 3D renderer node: {e}")

try:
    from .tensor_list_preview import NODE_CLASS_MAPPINGS as TENSOR_PREVIEW_MAPPINGS
    from .tensor_list_preview import NODE_DISPLAY_NAME_MAPPINGS as TENSOR_PREVIEW_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(TENSOR_PREVIEW_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(TENSOR_PREVIEW_DISPLAY_MAPPINGS)
    logger.info("Loaded tensor list preview node")
except ImportError as e:
    logger.warning(f"Failed to load tensor list preview node: {e}")

try:
    from .nodes_json_object_utils import NODE_CLASS_MAPPINGS as JSON_UTILS_MAPPINGS
    from .nodes_json_object_utils import NODE_DISPLAY_NAME_MAPPINGS as JSON_UTILS_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(JSON_UTILS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(JSON_UTILS_DISPLAY_MAPPINGS)
    logger.info("Loaded JSON object utils nodes")
except ImportError as e:
    logger.warning(f"Failed to load JSON object utils nodes: {e}")

# 导出给ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']