"""
ComfyUI-VVL-Tools
A collection of custom nodes for ComfyUI
"""

import logging
import os

logger = logging.getLogger(__name__)

# 声明web目录以支持前端扩展
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# 导入所有节点
try:
    from .nodes_loop_control import *
    from .list_chunk_any import *
    from .execution_blocker_utils import *
    from .batch_anything_dynamic import *
    from .enhanced_lambert_renderer_node import *
    from .tensor_list_preview import *
    from .nodes_json_object_utils import *
    from .room_generator.nodes_room_generator import *
    from .room_generator.generate_room_bbox import RoomGenerator, CollisionError
    from .batch_text_loader import *
    from .text_combine_multi import *
    from .blender.processor_3d import *
except ImportError as e:
    print(f"\033[34mVVL Tools: \033[92mFailed to load some nodes: {e}\033[0m")

# 节点映射
NODE_CLASS_MAPPINGS = {
    # Loop Control节点 (9个)
    "VVL forLoopStart": VVLForLoopStartAsync,
    "VVL forLoopEnd": VVLForLoopEndAsync,
    "VVL whileLoopStart": VVLWhileLoopStartAsync,
    "VVL whileLoopEnd": VVLWhileLoopEndAsync,
    "VVL mathInt": VVLMathIntAsync,
    "VVL compare": VVLCompareAsync,
    "VVL listConstruct": VVLListConstructAsync,
    "VVL listGetItem": VVLListGetItemAsync,
    "VVL listLength": VVLListLengthAsync,
    
    # List Chunk节点 (1个)
    "VVL listChunk": VVLListChunkAsync,
    
    # Execution Blocker节点 (1个)
    "async_sleep_v1": async_sleep_v1,
    
    # Batch Anything Dynamic节点 (1个)
    "DynamicBatchAnything": DynamicBatchAnything,
    
    # 3D Renderer节点 (1个)
    "Enhanced3DRenderer": Enhanced3DRenderer,
    
    # Tensor Preview节点 (1个)
    "TensorListPreview": TensorListPreview,
    
    # JSON工具节点 (11个)
    "JsonObjectDeduplicator": JsonObjectDeduplicator,
    "JsonObjectMerger": JsonObjectMerger,
    "JsonExtractSubjectNamesScales": JsonExtractSubjectNamesScales,
    "ApplyUrlsToJson": ApplyUrlsToJson,
    "JsonMarkdownCleaner": JsonMarkdownCleaner,
    "IndexUrlPairDeduplicator": IndexUrlPairDeduplicator,
    "JsonArrayElementFieldExtractor": JsonArrayElementFieldExtractor,
    "JsonRotationScaleAdjuster": JsonRotationScaleAdjuster,
    "JsonScaleMaxAdjuster": JsonScaleMaxAdjuster,
    "JsonCompressor": JsonCompressor,
    "DimensionReorderAndScale": DimensionReorderAndScale,
    
    # Room Generator节点 (1个)
    "RoomBboxGeneratorNode": RoomBboxGeneratorNode,
    
    # Text工具节点 (2个)
    "VVL_Load_Text_Batch": VVL_Load_Text_Batch,
    "TextCombineMulti": TextCombineMulti,
    
    # Blender节点 (1个)
    "BlenderSmartModelScaler": BlenderSmartModelScaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loop Control节点
    "VVL forLoopStart": "VVL For Loop Start (Async)",
    "VVL forLoopEnd": "VVL For Loop End (Async)",
    "VVL whileLoopStart": "VVL While Loop Start (Async)",
    "VVL whileLoopEnd": "VVL While Loop End (Async)",
    "VVL mathInt": "VVL Math Int (Async)",
    "VVL compare": "VVL Compare (Async)",
    "VVL listConstruct": "VVL List Construct (Async)",
    "VVL listGetItem": "VVL List Get Item (Async)",
    "VVL listLength": "VVL List Length (Async)",
    
    # List Chunk节点
    "VVL listChunk": "VVL List Chunk (Async)",
    
    # Execution Blocker节点
    "async_sleep_v1": "VVL async_sleep_v1",
    
    # Batch Anything Dynamic节点
    "DynamicBatchAnything": "VVL Batch Anything Dynamic",
    
    # 3D Renderer节点
    "Enhanced3DRenderer": "VVL Enhanced Lambert 3D Renderer",
    
    # Tensor Preview节点
    "TensorListPreview": "VVL Tensor List Preview",
    
    # JSON工具节点
    "JsonObjectDeduplicator": "VVL JSON Object Deduplicator",
    "JsonObjectMerger": "VVL JSON Object Merger",
    "JsonExtractSubjectNamesScales": "VVL JSON Extract: subject, names, scales",
    "ApplyUrlsToJson": "VVL Apply URLs to JSON",
    "JsonMarkdownCleaner": "VVL JSON Markdown Cleaner",
    "IndexUrlPairDeduplicator": "VVL Index-URL Pair Deduplicator",
    "JsonArrayElementFieldExtractor": "VVL JSON Array Element Field Extractor",
    "JsonRotationScaleAdjuster": "VVL JSON Rotation & Scale Adjuster",
    "JsonScaleMaxAdjuster": "VVL JSON Scale Max Value Adjuster",
    "JsonCompressor": "VVL JSON Compressor",
    "DimensionReorderAndScale": "VVL Dimension Reorder and Scale",
    
    # Room Generator节点
    "RoomBboxGeneratorNode": "VVL 3D Room Box Json Generator",
    
    # Text工具节点
    "VVL_Load_Text_Batch": "VVL Batch Text Loader",
    "TextCombineMulti": "VVL Text Combine Multi",
    
    # Blender节点
    "BlenderSmartModelScaler": "VVL Blender Smart Model Scaler",
}

print("------------------------------------------")    
print("\033[34mVVL Tools: \033[92m 30+ Nodes Loaded\033[0m")
print("------------------------------------------") 

# 导出给ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']