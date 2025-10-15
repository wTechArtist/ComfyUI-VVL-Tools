import json
import re
import ast
import copy
from comfy.comfy_types.node_typing import IO


class JsonObjectDeduplicator:
    """
    Remove duplicate objects from JSON based on name (excluding numeric suffix) and scale.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("deduplicated_json", "removed_duplicates")
    FUNCTION = "deduplicate_objects"
    CATEGORY = "VVL/json"
    
    def deduplicate_objects(self, json_text, **kwargs):
        try:
            # Parse input JSON
            data = json.loads(json_text)
            
            # Check if 'objects' field exists
            if 'objects' not in data:
                return json_text, json.dumps({"removed_objects": []})
            
            objects = data['objects']
            unique_objects = []
            removed_objects = []
            kept_original_indices = []
            
            # Track unique combinations of (base_name, scale)
            seen_combinations = {}
            
            for i, obj in enumerate(objects):
                # Extract base name (remove numeric suffix)
                name = obj.get('name', '')
                base_name = re.sub(r'\d+$', '', name).strip()
                
                # Get scale as tuple for comparison
                scale = obj.get('scale', [])
                scale_tuple = tuple(scale) if isinstance(scale, list) else scale
                
                # Create combination key
                combination_key = (base_name, scale_tuple)
                
                if combination_key not in seen_combinations:
                    # First occurrence - keep it
                    seen_combinations[combination_key] = len(unique_objects)
                    unique_objects.append(obj)
                    kept_original_indices.append(i)
                else:
                    # Duplicate - add to removed list
                    removed_obj = obj.copy()
                    removed_obj['_original_index'] = i
                    removed_obj['_duplicate_of_index'] = seen_combinations[combination_key]
                    removed_objects.append(removed_obj)
            
            # Update data with deduplicated objects
            data['objects'] = unique_objects
            
            # Create result JSONs
            deduplicated_json = json.dumps(data, ensure_ascii=False, indent=2)
            removed_duplicates = json.dumps({
                "removed_objects": removed_objects,
                "original_count": len(objects),
                "deduplicated_count": len(unique_objects),
                "kept_original_indices": kept_original_indices
            }, ensure_ascii=False, indent=2)
            
            return deduplicated_json, removed_duplicates
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON input: {str(e)}"
            return json.dumps({"error": error_msg}), json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Error processing JSON: {str(e)}"
            return json.dumps({"error": error_msg}), json.dumps({"error": error_msg})


class JsonObjectMerger:
    """
    Merge processed JSON with removed duplicates, copying processed fields (output.url, rotation, etc.) 
    to objects with same name and scale.
    
    Updates restored duplicate objects with fields from their corresponding processed objects:
    - output: Always copied if exists in processed object (including output.url)
    - 3d_url: For backward compatibility, also copied if exists
    - rotation: Always copied if exists in processed object
    - Additional fields can be easily added as needed
    
    This ensures all objects with the same base name and scale have consistent processed values.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_json": (IO.STRING, {"multiline": True, "default": ""}),
                "removed_duplicates": (IO.STRING, {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("merged_json",)
    FUNCTION = "merge_objects"
    CATEGORY = "VVL/json"
    
    def _copy_nested_field(self, source_obj, target_obj, field_path):
        """
        从源对象复制嵌套字段到目标对象
        支持点号分隔的路径，如 "output.url"
        """
        if '.' not in field_path:
            # 简单字段，直接复制
            if field_path in source_obj:
                target_obj[field_path] = source_obj[field_path]
                return True
            return False
        
        # 分割路径
        path_parts = field_path.split('.')
        
        # 检查源对象中是否存在该路径
        source_current = source_obj
        for part in path_parts[:-1]:
            if part not in source_current or not isinstance(source_current[part], dict):
                return False
            source_current = source_current[part]
        
        final_key = path_parts[-1]
        if final_key not in source_current:
            return False
        
        # 在目标对象中创建路径并复制值
        target_current = target_obj
        for part in path_parts[:-1]:
            if part not in target_current:
                target_current[part] = {}
            elif not isinstance(target_current[part], dict):
                # 如果中间路径不是字典，无法继续
                return False
            target_current = target_current[part]
        
        # 复制值
        target_current[final_key] = source_current[final_key]
        return True
    
    def merge_objects(self, processed_json, removed_duplicates, **kwargs):
        try:
            # Parse input JSONs
            processed_data = json.loads(processed_json)
            removed_data = json.loads(removed_duplicates)
            
            # Check for errors in input
            if "error" in processed_data or "error" in removed_data:
                return (json.dumps({"error": "Input contains errors"}),)
            
            # Check if required fields exist
            if 'objects' not in processed_data or 'removed_objects' not in removed_data:
                return (processed_json,)
            
            processed_objects = processed_data['objects']
            removed_objects = removed_data['removed_objects']
            kept_original_indices = removed_data.get('kept_original_indices', [])
            original_count = removed_data.get('original_count', len(processed_objects) + len(removed_objects))
            
            # Create a mapping of (base_name, scale) to processed object data from processed objects
            object_data_mapping = {}
            for obj in processed_objects:
                name = obj.get('name', '')
                base_name = re.sub(r'\d+$', '', name).strip()
                scale = obj.get('scale', [])
                scale_tuple = tuple(scale) if isinstance(scale, list) else scale
                
                combination_key = (base_name, scale_tuple)
                # Store the entire processed object for reference
                object_data_mapping[combination_key] = obj
            
            # Restore removed objects with updated fields from corresponding processed objects
            restored_objects = []
            copy_count = 0
            
            for removed_obj in removed_objects:
                # Remove metadata fields
                obj = {k: v for k, v in removed_obj.items() 
                      if not k.startswith('_')}
                
                # Find matching processed object
                name = obj.get('name', '')
                base_name = re.sub(r'\d+$', '', name).strip()
                scale = obj.get('scale', [])
                scale_tuple = tuple(scale) if isinstance(scale, list) else scale
                
                combination_key = (base_name, scale_tuple)
                if combination_key in object_data_mapping:
                    processed_obj = object_data_mapping[combination_key]
                    
                    # Copy specific fields from processed object to restored object
                    # output - 复制整个 output 对象（包括 output.url）
                    if 'output' in processed_obj:
                        obj['output'] = copy.deepcopy(processed_obj['output'])
                        copy_count += 1
                    
                    # 3d_url - 为了向后兼容，也复制此字段（如果存在）
                    if '3d_url' in processed_obj:
                        obj['3d_url'] = processed_obj['3d_url']
                    
                    # rotation - copy if exists in processed object
                    # if 'rotation' in processed_obj:
                    #     obj['rotation'] = processed_obj['rotation']
                    
                    # You can add more fields here as needed
                    # Example: if 'position' in processed_obj:
                    #     obj['position'] = processed_obj['position']
                
                restored_objects.append(obj)
            
            # Reconstruct original order using kept_original_indices and _original_index
            full_objects = [None] * max(original_count, len(processed_objects) + len(restored_objects))

            # Place processed (kept) objects back to their original indices
            for idx, obj in enumerate(processed_objects):
                target_index = kept_original_indices[idx] if idx < len(kept_original_indices) else idx
                if 0 <= target_index < len(full_objects):
                    full_objects[target_index] = obj
                else:
                    # Fallback append if index out of bounds
                    full_objects.append(obj)

            # Place restored (previously removed) objects at their original indices
            for removed_obj, restored in zip(removed_objects, restored_objects):
                target_index = removed_obj.get('_original_index')
                if isinstance(target_index, int) and 0 <= target_index < len(full_objects):
                    full_objects[target_index] = restored
                else:
                    full_objects.append(restored)

            # Remove any None slots while preserving order
            all_objects = [obj for obj in full_objects if obj is not None]

            # Update data with all objects
            processed_data['objects'] = all_objects
            
            # Create final JSON
            merged_json = json.dumps(processed_data, ensure_ascii=False, indent=2)
            
            # 输出处理统计
            print(f"JsonObjectMerger 完成:")
            print(f"  • 恢复的对象数: {len(restored_objects)}")
            print(f"  • 复制 output 字段: {copy_count} 个")
            print(f"  • 最终对象总数: {len(all_objects)}")
            
            return (merged_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON input: {str(e)}"
            return (json.dumps({"error": error_msg}),)
        except Exception as e:
            error_msg = f"Error merging JSON: {str(e)}"
            return (json.dumps({"error": error_msg}),)


# Extract subject, names list, and scales list from JSON
class JsonExtractSubjectNamesScales:
    """
    Extracts subject, names list, scales list, and style from the input JSON.

    - names_json: JSON string array of names, optionally stripping numeric suffixes
    - scales_json: JSON string array of scale triplets (e.g., [[0.7,0.5,0.75], ...])
    - style: Style string extracted from scene.style field
    Returning JSON strings keeps strong compatibility with community nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": ""}),
                "strip_numeric_suffix": (IO.BOOLEAN, {"default": True, "label_on": "strip", "label_off": "keep"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.ANY, IO.ANY, IO.STRING)
    RETURN_NAMES = ("subject", "names_list", "scales_list", "style")
    FUNCTION = "extract_fields"
    CATEGORY = "VVL/json"

    def extract_fields(self, json_text, strip_numeric_suffix=True, **kwargs):
        try:
            data = json.loads(json_text)
            subject = data.get("subject", "")
            objects = data.get("objects", [])
            
            # Extract style from scene field
            scene = data.get("scene", {})
            style = scene.get("style", "") if isinstance(scene, dict) else ""

            names = []
            scales = []
            for obj in objects:
                name = obj.get("name", "")
                if strip_numeric_suffix:
                    name = re.sub(r"\d+$", "", name).strip()
                names.append(name)
                scales.append(obj.get("scale", []))

            return (
                subject,
                names,
                scales,
                style,
            )
        except json.JSONDecodeError as e:
            msg = json.dumps({"error": f"Invalid JSON input: {str(e)}"})
            return (msg, msg, msg, msg)
        except Exception as e:
            msg = json.dumps({"error": f"Error extracting fields: {str(e)}"})
            return (msg, msg, msg, msg)


class ApplyUrlsToJson:
    """
    Apply a list of (index, url) pairs to JSON objects with nested field path support.

    - Supports nested field paths like "output.url" or simple fields like "3d_url"
    - Automatically creates intermediate objects if they don't exist
    - Robust to inputs shaped like [[idx, url], ...]
    - Skips entries with None url (configurable)
    - Ignores out-of-bounds indices (configurable)
    
    Example nested field paths:
    - "output.url" → objects[idx].output.url = value
    - "3d_url" → objects[idx].3d_url = value (backward compatible)
    - "data.result.file_path" → objects[idx].data.result.file_path = value
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "deduplicated_json": (IO.STRING, {"multiline": True, "default": ""}),
                "idx_url_list": (IO.ANY, {}),
            },
            "optional": {
                "field_name": (IO.STRING, {"default": "output.url", "tooltip": "字段路径\n• 支持嵌套路径: 'output.url'\n• 支持简单字段: '3d_url'\n• 自动创建中间对象"}),
                "skip_none_url": (IO.BOOLEAN, {"default": True, "label_on": "skip", "label_off": "write None"}),
                "ignore_oob_index": (IO.BOOLEAN, {"default": True, "label_on": "ignore", "label_off": "error"}),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("processed_json",)
    FUNCTION = "apply"
    CATEGORY = "VVL/json"

    def _normalize_pairs(self, idx_url_list):
        # Debug: print what we actually received
        print(f"ApplyUrlsToJson received: {type(idx_url_list)} = {idx_url_list}")
        
        pairs = []
        if idx_url_list is None:
            return pairs

        # Handle various input formats from different ComfyUI node outputs
        container = idx_url_list
        if not isinstance(container, (list, tuple)):
            container = [container]

        # Special case: check if container itself is a single [idx, url] pair
        if (isinstance(container, (list, tuple)) and len(container) == 2 and
            not isinstance(container[0], (list, tuple, str)) and  # first element is not nested
            isinstance(container[1], str)):  # second element is string (url)
            try:
                idx = int(container[0])
                url = container[1]
                pairs.append((idx, url))
                print(f"ApplyUrlsToJson detected single pair: [{idx}, '{url}']")
                print(f"ApplyUrlsToJson normalized to: {pairs}")
                return pairs
            except (ValueError, TypeError):
                # If conversion fails, fall through to normal processing
                pass

        for item in container:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # Direct [idx, url] pair
                pairs.append((item[0], item[1]))
            elif isinstance(item, str) and "," in item:
                # CR_IntertwineLists style: "idx, url"
                parts = item.split(",", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0].strip())
                        url = parts[1].strip()
                        pairs.append((idx, url))
                    except ValueError:
                        continue
            elif hasattr(item, '__iter__') and not isinstance(item, str):
                # Nested containers - recursive flatten
                for subitem in item:
                    if isinstance(subitem, (list, tuple)) and len(subitem) >= 2:
                        pairs.append((subitem[0], subitem[1]))
        
        print(f"ApplyUrlsToJson normalized to: {pairs}")
        return pairs
    
    def _set_nested_field(self, obj, field_path, value):
        """
        设置嵌套字段值，支持点号分隔的路径
        例如: "output.url" 会设置 obj["output"]["url"] = value
        如果中间对象不存在，会自动创建
        """
        if not isinstance(obj, dict):
            return False
        
        # 如果没有点号，直接设置
        if '.' not in field_path:
            obj[field_path] = value
            return True
        
        # 分割路径
        path_parts = field_path.split('.')
        current = obj
        
        # 遍历到倒数第二个部分，创建/访问中间对象
        for part in path_parts[:-1]:
            if part not in current:
                # 创建中间对象
                current[part] = {}
            elif not isinstance(current[part], dict):
                # 中间路径存在但不是字典，无法继续
                print(f"ApplyUrlsToJson: 字段路径 '{field_path}' 中的 '{part}' 不是对象类型，无法设置嵌套值")
                return False
            current = current[part]
        
        # 设置最后的字段
        final_key = path_parts[-1]
        current[final_key] = value
        return True

    def apply(self, deduplicated_json, idx_url_list, field_name="output.url", skip_none_url=True, ignore_oob_index=True, **kwargs):
        try:
            data = json.loads(deduplicated_json)
        except json.JSONDecodeError as e:
            return (json.dumps({"error": f"Invalid JSON input: {str(e)}"}),)

        if not isinstance(data, dict) or "objects" not in data or not isinstance(data["objects"], list):
            return (deduplicated_json,)

        objects = data["objects"]

        pairs = self._normalize_pairs(idx_url_list)
        success_count = 0
        failed_count = 0
        
        for raw_idx, url in pairs:
            try:
                idx = int(raw_idx)
            except Exception:
                failed_count += 1
                continue

            if idx < 0 or idx >= len(objects):
                if ignore_oob_index:
                    failed_count += 1
                    continue
                else:
                    return (json.dumps({"error": f"Index out of bounds: {idx}"}, ensure_ascii=False),)

            if url is None and skip_none_url:
                continue

            obj = objects[idx]
            if isinstance(obj, dict):
                if self._set_nested_field(obj, field_name, url):
                    success_count += 1
                else:
                    failed_count += 1
        
        # 输出处理统计
        print(f"ApplyUrlsToJson 完成:")
        print(f"  • 字段路径: '{field_name}'")
        print(f"  • 成功设置: {success_count} 个")
        if failed_count > 0:
            print(f"  • 失败/跳过: {failed_count} 个")

        processed_json = json.dumps(data, ensure_ascii=False, indent=2)
        return (processed_json,)




class JsonMarkdownCleaner:
    """
    清理 JSON 字符串中的 Markdown 代码块标记。
    
    移除：
    - 开头的 ```json 或 ```
    - 结尾的 ```
    - 保持 JSON 内容完整
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("cleaned_json",)
    FUNCTION = "clean_json"
    CATEGORY = "VVL/json"
    
    def clean_json(self, json_text, **kwargs):
        """清理 JSON 字符串中的 Markdown 代码块标记"""
        try:
            cleaned_text = json_text.strip()
            
            # 移除开头的 ```json 或 ```
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:].strip()
            
            # 移除结尾的 ```
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()
            
            # 验证清理后的文本是否为有效 JSON
            try:
                json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                # 如果不是有效 JSON，返回警告但不阻止处理
                print(f"Warning: Cleaned text may not be valid JSON: {str(e)}")
            
            return (cleaned_text,)
            
        except Exception as e:
            error_msg = f"清理 JSON 时出错: {str(e)}"
            print(f"JsonMarkdownCleaner error: {error_msg}")
            return (json_text,)  # 出错时返回原始文本


class IndexUrlPairDeduplicator:
    """
    移除 (index, url) 对列表中的重复项。
    
    支持三种策略：
    - remove_all: 完全删除所有重复项（包括第一次出现的）
    - keep_first: 保留第一个出现的重复项
    - keep_last: 保留最后一个出现的重复项
    
    支持多种输入格式，如 [[idx, url], ...] 或 ["idx,url", ...]
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "idx_url_list": (IO.ANY, {}),
            },
            "optional": {
                "preserve_order": (IO.BOOLEAN, {"default": True, "label_on": "保持顺序", "label_off": "不保证顺序"}),
                "duplicate_strategy": (["keep_first", "keep_last", "remove_all"], {"default": "remove_all"}),
            },
        }
    
    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("deduplicated_pairs",)
    FUNCTION = "deduplicate_pairs"
    CATEGORY = "VVL/json"
    
    def _normalize_pairs(self, idx_url_list):
        """规范化输入数据为 (index, url) 对的列表"""
        pairs = []
        if idx_url_list is None:
            return pairs

        container = idx_url_list
        if not isinstance(container, (list, tuple)):
            container = [container]

        for item in container:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # 直接的 [idx, url] 对
                pairs.append((item[0], item[1]))
            elif isinstance(item, str) and "," in item:
                # CR_IntertwineLists 风格: "idx, url"
                parts = item.split(",", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0].strip())
                        url = parts[1].strip()
                        pairs.append((idx, url))
                    except ValueError:
                        continue
            elif hasattr(item, '__iter__') and not isinstance(item, str):
                # 嵌套容器 - 递归展开
                for subitem in item:
                    if isinstance(subitem, (list, tuple)) and len(subitem) >= 2:
                        pairs.append((subitem[0], subitem[1]))
        
        return pairs
    
    def deduplicate_pairs(self, idx_url_list, preserve_order=True, duplicate_strategy="remove_all", **kwargs):
        """去除重复的 (index, url) 对"""
        try:
            pairs = self._normalize_pairs(idx_url_list)
            
            if not pairs:
                return (idx_url_list,)
            
            # 统计每个键出现的次数
            key_counts = {}
            for idx, url in pairs:
                key = (idx, url)
                key_counts[key] = key_counts.get(key, 0) + 1
            
            if duplicate_strategy == "remove_all":
                # 移除所有重复项（包括第一次出现的）
                if preserve_order:
                    deduplicated = []
                    for idx, url in pairs:
                        key = (idx, url)
                        # 只保留出现次数为1的项
                        if key_counts[key] == 1:
                            deduplicated.append([idx, url])
                else:
                    # 不保证顺序，直接过滤
                    unique_keys = {k: [k[0], k[1]] for k, count in key_counts.items() if count == 1}
                    deduplicated = list(unique_keys.values())
                    
            elif duplicate_strategy == "keep_first":
                # 保留第一个出现的
                if preserve_order:
                    seen = set()
                    deduplicated = []
                    for idx, url in pairs:
                        key = (idx, url)
                        if key not in seen:
                            seen.add(key)
                            deduplicated.append([idx, url])
                else:
                    seen = {}
                    for idx, url in pairs:
                        key = (idx, url)
                        if key not in seen:
                            seen[key] = [idx, url]
                    deduplicated = list(seen.values())
                    
            elif duplicate_strategy == "keep_last":
                # 保留最后一个出现的
                if preserve_order:
                    seen = set()
                    deduplicated = []
                    for idx, url in pairs:
                        key = (idx, url)
                        if key in seen:
                            # 移除之前添加的相同项
                            deduplicated = [pair for pair in deduplicated 
                                          if (pair[0], pair[1]) != key]
                        seen.add(key)
                        deduplicated.append([idx, url])
                else:
                    # 不保证顺序，保留最后一个
                    seen = {}
                    for idx, url in pairs:
                        key = (idx, url)
                        seen[key] = [idx, url]
                    deduplicated = list(seen.values())
            
            print(f"IndexUrlPairDeduplicator: 输入 {len(pairs)} 对，输出 {len(deduplicated)} 对")
            print(f"策略: {duplicate_strategy}")
            print(f"去重前: {pairs}")
            print(f"去重后: {deduplicated}")
            
            # 输出重复项统计信息
            duplicates = {k: count for k, count in key_counts.items() if count > 1}
            if duplicates:
                print(f"发现的重复项: {duplicates}")
            
            return (deduplicated,)
            
        except Exception as e:
            error_msg = f"去重处理错误: {str(e)}"
            print(f"IndexUrlPairDeduplicator error: {error_msg}")
            return (idx_url_list,)


class JsonArrayElementFieldExtractor:
    """
    从JSON数组中提取指定索引元素的指定字段值。
    
    📋 支持输入格式：
    • JSON字符串数组: '[{"name":"file1","url":"path1"}, ...]'
    • Python列表对象: [{"name":"file1","url":"path1"}, ...]
    
    🎯 索引用法详解：
    • 正数索引: 从数组开头计数
      - 0 = 第1个元素
      - 1 = 第2个元素  
      - 2 = 第3个元素 ...
    
    • 负数索引: 从数组末尾计数
      - -1 = 最后1个元素
      - -2 = 倒数第2个元素
      - -3 = 倒数第3个元素 ...
    
    💡 实用示例：
    数组: ["A", "B", "C", "D", "E"]
    索引对照表:
    ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    │ 正索引  │    0    │    1    │    2    │    3    │    4    │
    │ 元素值  │   "A"   │   "B"   │   "C"   │   "D"   │   "E"   │
    │ 负索引  │   -5    │   -4    │   -3    │   -2    │   -1    │
    └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
    
    ⚙️ 可配置参数：
    • index: 数组索引位置（支持正负数）
    • field_name: 要提取的字段名称
    
    ⚠️ 错误处理：
    • 出错时统一返回字符串 "None"
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_array": (IO.ANY, {}),
            },
            "optional": {
                "index": (IO.INT, {"default": 0, "min": -999999, "max": 999999, "tooltip": "数组索引位置\n• 正数: 从头开始 (0=第1个, 1=第2个, ...)\n• 负数: 从尾开始 (-1=最后1个, -2=倒数第2个, ...)\n• 示例: 有5个元素的数组\n  索引 0,1,2,3,4 对应 第1,2,3,4,5个元素\n  索引 -1,-2,-3,-4,-5 对应 最后1,2,3,4,5个元素"}),
                "field_name": (IO.STRING, {"default": "url", "tooltip": "要提取的字段名称\n• 默认: 'url'\n• 常用字段: name, cover, id, path 等\n• 支持任意对象属性名"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("field_value",)
    FUNCTION = "extract_field_value"
    CATEGORY = "VVL/json"
    
    def extract_field_value(self, json_array, index=0, field_name="url", **kwargs):
        """从数组中提取指定索引元素的字段值"""
        try:
            # 如果输入是字符串 "None"，直接返回 "None"
            if json_array == "None":
                return ("None",)
            
            # 处理输入数据
            data = json_array
            
            # 如果输入是字符串，尝试解析为JSON
            if isinstance(json_array, str):
                try:
                    data = json.loads(json_array)
                except json.JSONDecodeError:
                    return ("None",)
            
            # 检查是否为列表/数组
            if not isinstance(data, list):
                return ("None",)
            
            # 检查数组是否为空
            if len(data) == 0:
                return ("None",)
            
            # 检查索引是否在有效范围内
            try:
                # Python支持负数索引，但我们需要检查边界
                if index >= len(data) or index < -len(data):
                    return ("None",)
                
                # 获取指定索引的元素
                target_element = data[index]
                
            except IndexError:
                return ("None",)
            
            # 检查目标元素是否为字典
            if not isinstance(target_element, dict):
                return ("None",)
            
            # 提取指定字段的值
            field_value = target_element.get(field_name, "")
            
            # 确保返回字符串类型
            if field_value is None:
                field_value = ""
            elif not isinstance(field_value, str):
                field_value = str(field_value)
            
            # 显示有用的调试信息
            actual_index = index if index >= 0 else len(data) + index
            print(f"JsonArrayElementFieldExtractor: 从索引 {index} (实际位置: {actual_index}) 提取字段 '{field_name}': {field_value}")
            
            return (field_value,)
            
        except Exception as e:
            error_msg = f"提取字段值时出错: {str(e)}"
            print(f"JsonArrayElementFieldExtractor error: {error_msg}")
            return ("None",)


class JsonRotationScaleAdjuster:
    """
    JSON旋转值和缩放调整器
    
    对JSON数据中所有对象的rotation和scale值进行批量调整：
    • 支持对X、Y、Z轴旋转值分别进行加减操作
    • 支持对X、Y、Z轴缩放值分别进行乘法操作
    • 可处理场景JSON文件中的所有objects
    • 保持原有的数据结构和格式
    
    🔄 功能特性：
    • 批量处理：一次性调整所有对象的旋转值和缩放值
    • 轴向控制：分别控制X、Y、Z轴的旋转偏移和缩放因子
    • 旋转范围：每个轴的调整范围为-360°到+360°
    • 缩放范围：每个轴的缩放因子范围为0.001到1000
    • 安全处理：自动跳过没有相应字段的对象
    
    ⚙️ 参数说明：
    • rotation_order: 旋转值顺序重排（XYZ, XZY, YXZ, YZX, ZXY, ZYX）
    • rotation_x_offset: X轴旋转偏移量（度）
    • rotation_y_offset: Y轴旋转偏移量（度）  
    • rotation_z_offset: Z轴旋转偏移量（度）
    • scale_x_multiplier: X轴缩放乘法因子
    • scale_y_multiplier: Y轴缩放乘法因子
    • scale_z_multiplier: Z轴缩放乘法因子
    
    📝 使用场景：
    • 场景整体旋转调整
    • 视角方向修正
    • 批量对象朝向调整
    • 坐标系转换补偿
    • 批量尺寸缩放调整
    • 比例修正和适配
    
    💡 注意事项：
    • 只处理包含相应字段且格式正确的对象
    • rotation和scale字段必须是长度≥3的数组
    • rotation顺序调换在偏移量计算之前执行
    • 角度计算不会自动规范化到0-360范围
    • 缩放使用乘法，1.0表示保持原尺寸
    
    🔄 旋转顺序说明：
    • XYZ: [x, y, z] - 默认顺序
    • XZY: [x, z, y] - X不变，Y和Z互换
    • YXZ: [y, x, z] - X和Y互换，Z不变
    • YZX: [y, z, x] - Y→X, Z→Y, X→Z
    • ZXY: [z, x, y] - Z→X, X→Y, Y→Z
    • ZYX: [z, y, x] - Z和X互换，Y不变
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "包含objects数组的JSON数据\n支持场景描述文件格式\n将对所有对象的rotation和scale值进行调整"})
            },
            "optional": {
                "rotation_order": (["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"], {
                    "default": "YZX",
                    "tooltip": "旋转值的顺序重排\n• XYZ: [x, y, z] - 默认顺序\n• XZY: [x, z, y] - Y和Z互换\n• YXZ: [y, x, z] - X和Y互换\n• YZX: [y, z, x] - 循环右移\n• ZXY: [z, x, y] - 循环左移\n• ZYX: [z, y, x] - X和Z互换\n注意：重排在偏移量计算之前执行"
                }),
                "rotation_x_offset": (IO.FLOAT, {
                    "default": 0.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "X轴旋转偏移量（度）\n正值：绕X轴正方向旋转\n负值：绕X轴负方向旋转\n范围：-360° 到 +360°\n注意：在rotation_order重排后应用"
                }),
                "rotation_y_offset": (IO.FLOAT, {
                    "default": -90.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "Y轴旋转偏移量（度）\n正值：绕Y轴正方向旋转\n负值：绕Y轴负方向旋转\n范围：-360° 到 +360°\n注意：在rotation_order重排后应用"
                }),
                "rotation_z_offset": (IO.FLOAT, {
                    "default": 0.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "Z轴旋转偏移量（度）\n正值：绕Z轴正方向旋转\n负值：绕Z轴负方向旋转\n范围：-360° 到 +360°\n注意：在rotation_order重排后应用"
                }),
                "scale_x_multiplier": (IO.FLOAT, {
                    "default": 0.01, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "X轴缩放乘法因子\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍\n范围：0.001 到 1000"
                }),
                "scale_y_multiplier": (IO.FLOAT, {
                    "default": 0.01, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "Y轴缩放乘法因子\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍\n范围：0.001 到 1000"
                }),
                "scale_z_multiplier": (IO.FLOAT, {
                    "default": 0.01, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "Z轴缩放乘法因子\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍\n范围：0.001 到 1000"
                }),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("adjusted_json",)
    FUNCTION = "adjust_rotations_and_scales"
    CATEGORY = "VVL/json"
    
    def _reorder_rotation(self, rotation_values, order):
        """根据指定顺序重新排列rotation值"""
        x, y, z = rotation_values[0], rotation_values[1], rotation_values[2]
        
        order_map = {
            "XYZ": [x, y, z],  # 默认顺序
            "XZY": [x, z, y],  # Y和Z互换
            "YXZ": [y, x, z],  # X和Y互换
            "YZX": [y, z, x],  # 循环右移：Y→X, Z→Y, X→Z
            "ZXY": [z, x, y],  # 循环左移：Z→X, X→Y, Y→Z
            "ZYX": [z, y, x],  # X和Z互换
        }
        
        return order_map.get(order, [x, y, z])

    def adjust_rotations_and_scales(self, json_text, rotation_order="XYZ", rotation_x_offset=0.0, rotation_y_offset=0.0, rotation_z_offset=0.0, 
                                   scale_x_multiplier=1.0, scale_y_multiplier=1.0, scale_z_multiplier=1.0, **kwargs):
        """调整JSON中所有对象的rotation和scale值"""
        try:
            # 解析输入JSON
            data = json.loads(json_text)
            
            # 检查是否有objects字段
            if 'objects' not in data or not isinstance(data['objects'], list):
                print("JsonRotationScaleAdjuster: 未找到有效的objects数组")
                return (json_text,)
            
            objects = data['objects']
            rotation_processed = 0
            scale_processed = 0
            skipped_count = 0
            
            # 遍历所有对象并调整rotation和scale值
            for i, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    skipped_count += 1
                    continue
                
                obj_processed = False
                
                # 处理rotation字段
                if 'rotation' in obj:
                    rotation = obj['rotation']
                    
                    # 检查rotation是否为有效的列表格式
                    if isinstance(rotation, list) and len(rotation) >= 3:
                        try:
                            # 确保原始值是数值类型
                            original_values = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
                            
                            # 1. 首先按照指定顺序重新排列rotation值
                            reordered_values = self._reorder_rotation(original_values, rotation_order)
                            
                            # 2. 然后在重排后的值上应用偏移量
                            new_x = reordered_values[0] + rotation_x_offset
                            new_y = reordered_values[1] + rotation_y_offset
                            new_z = reordered_values[2] + rotation_z_offset
                            
                            # 更新rotation值
                            obj['rotation'][0] = new_x
                            obj['rotation'][1] = new_y
                            obj['rotation'][2] = new_z
                            
                            rotation_processed += 1
                            obj_processed = True
                            
                        except (ValueError, TypeError) as e:
                            print(f"JsonRotationScaleAdjuster: 对象 {i} 的rotation值转换失败: {e}")
                    else:
                        print(f"JsonRotationScaleAdjuster: 对象 {i} 的rotation格式无效: {rotation}")
                
                # 处理scale字段
                if 'scale' in obj:
                    scale = obj['scale']
                    
                    # 检查scale是否为有效的列表格式
                    if isinstance(scale, list) and len(scale) >= 3:
                        try:
                            # 确保原始值是数值类型
                            original_x = float(scale[0])
                            original_y = float(scale[1]) 
                            original_z = float(scale[2])
                            
                            # 计算新的缩放值
                            new_x = original_x * scale_x_multiplier
                            new_y = original_y * scale_y_multiplier
                            new_z = original_z * scale_z_multiplier
                            
                            # 更新scale值
                            obj['scale'][0] = new_x
                            obj['scale'][1] = new_y
                            obj['scale'][2] = new_z
                            
                            scale_processed += 1
                            obj_processed = True
                            
                        except (ValueError, TypeError) as e:
                            print(f"JsonRotationScaleAdjuster: 对象 {i} 的scale值转换失败: {e}")
                    else:
                        print(f"JsonRotationScaleAdjuster: 对象 {i} 的scale格式无效: {scale}")
                
                # 如果对象没有被处理，增加跳过计数
                if not obj_processed:
                    skipped_count += 1
            
            # 生成处理后的JSON
            adjusted_json = json.dumps(data, ensure_ascii=False, indent=2)
            
            # 输出处理统计信息
            print(f"JsonRotationScaleAdjuster 处理完成:")
            print(f"  • Rotation处理: {rotation_processed} 个对象")
            print(f"  • Scale处理: {scale_processed} 个对象")
            print(f"  • 跳过处理: {skipped_count} 个对象")
            print(f"  • 旋转顺序: {rotation_order}")
            print(f"  • 旋转偏移: X={rotation_x_offset}°, Y={rotation_y_offset}°, Z={rotation_z_offset}°")
            print(f"  • 缩放因子: X={scale_x_multiplier}, Y={scale_y_multiplier}, Z={scale_z_multiplier}")
            
            return (adjusted_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            print(f"JsonRotationScaleAdjuster error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
        except Exception as e:
            error_msg = f"处理旋转值和缩放值时出错: {str(e)}"
            print(f"JsonRotationScaleAdjuster error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class JsonScaleMaxAdjuster:
    """
    JSON对象逐个缩放维度选择性调整器
    
    对JSON数据中前N个对象进行逐个处理，在每个对象内部找出scale的最小维度并保护，调整其他维度：
    • 逐个分析每个对象的scale值（X、Y、Z三个维度）
    • 在每个对象内部找出最小的维度值
    • 只对该对象内其他两个较大的维度进行增量调整
    • 保持每个对象最小维度不变，维持原有数据结构
    
    🎯 功能特性：
    • 对象范围：可设置处理前几个对象（默认前6个）
    • 逐对象处理：每个对象独立分析和调整
    • 维度保护：保护每个对象的最小维度不被修改
    • 安全处理：自动跳过没有scale字段或格式错误的对象
    
    ⚙️ 参数说明：
    • object_count: 处理前几个对象（默认6个）
    • max_value_increment: 给非最小维度增加的数值
    
    📝 使用场景：
    • 保持对象最细维度不变，拉伸其他维度
    • 创建非均匀缩放效果
    • 调整对象比例，突出长宽维度
    • 保护对象厚度或高度等关键维度
    
    💡 处理逻辑：
    1. 遍历前N个对象的scale字段
    2. 对每个对象分别处理：
       a. 获取X、Y、Z三个维度值
       b. 找出当前对象内的最小维度值
       c. 对其他两个维度进行增量调整
       d. 保持最小维度不变
    3. 保持JSON结构和其他数据不变
    
    🔍 注意事项：
    • 只处理包含有效scale字段的对象
    • scale字段必须是长度≥3的数组
    • 如果前N个对象不足，则处理实际存在的对象
    • 增量可以为负数（减少非最小维度）
    • 如果对象内三个维度相等，则全部调整
    
    📊 处理示例：
    对象scale=[1.0, 0.5, 0.8], 增量=0.1
    → 最小值0.5保持不变
    → 调整后: [1.1, 0.5, 0.9]
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "包含objects数组的JSON数据\n将处理前N个对象的scale值\n找出最大值并进行调整"})
            },
            "optional": {
                "object_count": (IO.INT, {
                    "default": 6, 
                    "min": 1, 
                    "max": 100,
                    "tooltip": "处理前几个对象\n• 默认: 6个对象\n• 范围: 1到100\n• 如果对象不足则处理所有可用对象\n• 逐个处理每个对象的scale维度"
                }),
                "max_value_increment": (IO.FLOAT, {
                    "default": 0.1, 
                    "min": -1000.0, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "非最小维度增量\n• 正数: 增加每个对象内除最小维度外的值\n• 负数: 减少每个对象内除最小维度外的值\n• 0: 不进行调整\n• 示例: 0.1表示给非最小维度都加0.1\n• 示例: -0.05表示给非最小维度都减0.05\n• 每个对象的最小维度始终保持不变"
                }),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("adjusted_json",)
    FUNCTION = "adjust_scale_max_value"
    CATEGORY = "VVL/json"
    
    def adjust_scale_max_value(self, json_text, object_count=6, max_value_increment=0.1, **kwargs):
        """调整前N个对象中scale值的最大值"""
        try:
            # 解析输入JSON
            data = json.loads(json_text)
            
            # 检查是否有objects字段
            if 'objects' not in data or not isinstance(data['objects'], list):
                print("JsonScaleMaxAdjuster: 未找到有效的objects数组")
                return (json_text,)
            
            objects = data['objects']
            
            # 确定实际处理的对象数量
            actual_count = min(object_count, len(objects))
            if actual_count == 0:
                print("JsonScaleMaxAdjuster: 没有可处理的对象")
                return (json_text,)
            
            # 逐个对象处理scale值
            valid_objects = 0
            total_adjusted = 0
            total_scale_values = 0
            
            for i in range(actual_count):
                obj = objects[i]
                if not isinstance(obj, dict):
                    continue
                
                if 'scale' in obj:
                    scale = obj['scale']
                    
                    # 检查scale是否为有效的列表格式
                    if isinstance(scale, list) and len(scale) >= 3:
                        try:
                            # 获取当前对象的X、Y、Z三个维度的值
                            x_val = float(scale[0])
                            y_val = float(scale[1])
                            z_val = float(scale[2])
                            
                            # 在当前对象内找出最小值
                            min_val_in_obj = min(x_val, y_val, z_val)
                            
                            # 对每个维度进行处理：如果不是最小值则调整
                            obj_adjusted = 0
                            if x_val != min_val_in_obj:
                                scale[0] = x_val + max_value_increment
                                obj_adjusted += 1
                            if y_val != min_val_in_obj:
                                scale[1] = y_val + max_value_increment
                                obj_adjusted += 1
                            if z_val != min_val_in_obj:
                                scale[2] = z_val + max_value_increment
                                obj_adjusted += 1
                            
                            # 特殊情况：如果三个值都相等，则全部调整
                            if obj_adjusted == 0 and x_val == y_val == z_val:
                                scale[0] = x_val + max_value_increment
                                scale[1] = y_val + max_value_increment
                                scale[2] = z_val + max_value_increment
                                obj_adjusted = 3
                            
                            total_adjusted += obj_adjusted
                            total_scale_values += 3
                            valid_objects += 1
                            
                            print(f"  对象[{i}]: scale=[{x_val:.3f}, {y_val:.3f}, {z_val:.3f}], 最小值={min_val_in_obj:.3f}, 调整了{obj_adjusted}个维度")
                            
                        except (ValueError, TypeError) as e:
                            print(f"JsonScaleMaxAdjuster: 对象 {i} 的scale值转换失败: {e}")
                    else:
                        print(f"JsonScaleMaxAdjuster: 对象 {i} 的scale格式无效: {scale}")
            
            if valid_objects == 0:
                print("JsonScaleMaxAdjuster: 未找到有效的scale值")
                return (json_text,)
            
            # 生成处理后的JSON
            adjusted_json = json.dumps(data, ensure_ascii=False, indent=2)
            
            # 输出处理统计信息
            print(f"JsonScaleMaxAdjuster 处理完成:")
            print(f"  • 处理对象范围: 前 {actual_count} 个对象")
            print(f"  • 有效对象数量: {valid_objects} 个")
            print(f"  • 总scale维度: {total_scale_values} 个")
            print(f"  • 调整的维度数: {total_adjusted} 个")
            print(f"  • 增量调整: +{max_value_increment}")
            print(f"  • 处理策略: 每个对象内部排除最小值，调整其他维度")
            
            return (adjusted_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            print(f"JsonScaleMaxAdjuster error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
        except Exception as e:
            error_msg = f"处理scale最大值时出错: {str(e)}"
            print(f"JsonScaleMaxAdjuster error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class JsonCompressor:
    """
    JSON压缩节点
    
    将格式化的JSON压缩为紧凑格式，移除所有非必要的空白字符：
    • 移除所有缩进空格和制表符
    • 移除所有换行符
    • 移除冒号和逗号后的额外空格
    • 保持JSON数据结构和内容完全不变
    
    🎯 功能特性：
    • 高效压缩：显著减小JSON文件大小
    • 完全兼容：保持JSON语法和数据完整性
    • 安全处理：验证输入JSON格式有效性
    • 错误处理：提供详细的错误信息反馈
    
    📊 压缩效果：
    • 移除缩进：节省大量空间
    • 移除换行：减少文件行数
    • 紧凑分隔符：最小化语法字符占用
    • 保持数据：确保数据内容不丢失
    
    📝 使用场景：
    • 减小JSON文件传输大小
    • 优化存储空间占用
    • 提高网络传输效率
    • 生成用于API的紧凑JSON
    
    💡 处理逻辑：
    1. 验证输入JSON格式有效性
    2. 解析JSON为Python对象
    3. 使用紧凑模式重新序列化
    4. 移除所有非必要空白字符
    5. 返回压缩后的JSON字符串
    
    🔍 注意事项：
    • 压缩是无损的，不会改变数据内容
    • 压缩后的JSON仍然是有效的JSON格式
    • 可以通过格式化工具还原为可读格式
    • 压缩程度取决于原始JSON的格式化程度
    
    📈 压缩示例：
    压缩前 (228字符):
    {
      "camera": {
        "position": [50, 0, 0],
        "rotation": [10, 0, 0]
      }
    }
    
    压缩后 (67字符):
    {"camera":{"position":[50,0,0],"rotation":[10,0,0]}}
    
    压缩率: 70.6%
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "要压缩的JSON数据\n支持任何有效的JSON格式\n包括带缩进、换行的格式化JSON\n将被压缩为紧凑的单行格式"})
            },
            "optional": {
                "show_compression_stats": (IO.BOOLEAN, {"default": True, "label_on": "显示统计", "label_off": "隐藏统计", "tooltip": "是否显示压缩统计信息\n• 开启: 显示压缩前后字符数和压缩率\n• 关闭: 仅输出压缩结果\n统计信息会在控制台中显示"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("compressed_json",)
    FUNCTION = "compress_json"
    CATEGORY = "VVL/json"
    
    def compress_json(self, json_text, show_compression_stats=True, **kwargs):
        """压缩JSON为紧凑格式"""
        try:
            # 验证并解析输入JSON
            data = json.loads(json_text)
            
            # 使用紧凑模式序列化JSON
            # separators=(',', ':') 移除逗号和冒号后的空格
            # ensure_ascii=False 保持非ASCII字符
            compressed_json = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
            
            # 计算压缩统计信息
            if show_compression_stats:
                original_size = len(json_text)
                compressed_size = len(compressed_json)
                space_saved = original_size - compressed_size
                compression_ratio = (space_saved / original_size * 100) if original_size > 0 else 0
                
                print(f"JsonCompressor 压缩完成:")
                print(f"  • 原始大小: {original_size} 字符")
                print(f"  • 压缩后大小: {compressed_size} 字符")
                print(f"  • 节省空间: {space_saved} 字符")
                print(f"  • 压缩率: {compression_ratio:.1f}%")
                
                # 显示压缩效果示例（前100个字符）
                if original_size > 100:
                    print(f"  • 压缩前预览: {json_text[:100]}...")
                    print(f"  • 压缩后预览: {compressed_json[:100]}...")
            
            return (compressed_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON格式错误: {str(e)}"
            print(f"JsonCompressor error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
        except Exception as e:
            error_msg = f"压缩JSON时出错: {str(e)}"
            print(f"JsonCompressor error: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)


class DimensionReorderAndScale:
    """
    三维数据重新排序和缩放节点
    
    处理格式如 [10, 10, 0.3] 的三维数据：
    • 支持任意调换长(length)、宽(width)、高(height)的位置
    • 提供缩放因子控制整体大小
    • 支持最小值和最大值限制（最后检验阶段）
    • 输出时默认移除方括号，返回逗号分隔的字符串
    
    📐 输入格式支持：
    • 列表: [10, 10, 0.3]
    • JSON字符串: "[10, 10, 0.3]"
    • 逗号分隔字符串: "10, 10, 0.3"
    
    🔄 排序选项：
    • LWH (长宽高) - 默认顺序
    • LHW (长高宽)
    • WLH (宽长高)
    • WHL (宽高长)
    • HLW (高长宽)
    • HWL (高宽长)
    
    ⚙️ 缩放控制：
    • scale_factor: 全局缩放因子 (默认 1.0)
    • 所有三个维度都会乘以此因子
    
    🔒 数值限制：
    • min_value: 最小值限制 (默认 1.0)
    • max_value: 最大值限制 (默认 2000.0)
    • 任何小于最小值的数值会被强制设为最小值
    • 任何大于最大值的数值会被强制设为最大值
    • 限制在最后检验阶段执行
    
    📤 输出格式：
    • 默认: "10.0,10.0,0.3" (逗号分隔，无空格)
    • 可选: "10.0, 10.0, 0.3" (保留空格)
    • 可选: "[10.0,10.0,0.3]" (保留方括号)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dimension_data": (IO.ANY, {"tooltip": "三维数据输入\n支持格式:\n• [10, 10, 0.3]\n• '[10, 10, 0.3]'\n• '10, 10, 0.3'"}),
            },
            "optional": {
                "reorder_pattern": (["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"], {
                    "default": "LWH",
                    "tooltip": "维度重排模式\n• L=长度(Length)\n• W=宽度(Width) \n• H=高度(Height)\n\n例如:\n• LWH: 长-宽-高 (默认)\n• WHL: 宽-高-长\n• HLW: 高-长-宽"
                }),
                "scale_factor": (IO.FLOAT, {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.01,
                    "tooltip": "全局缩放因子\n所有维度都乘以此值\n• 1.0 = 保持原尺寸\n• 0.5 = 缩小一半\n• 2.0 = 放大一倍"
                }),
                "decimal_places": (IO.INT, {
                    "default": 2, 
                    "min": 0, 
                    "max": 6,
                    "tooltip": "小数位数\n控制输出数值的精度\n• 0: 整数 '10, 10, 0'\n• 2: 两位小数 '10.00, 10.00, 0.30'\n• 3: 三位小数 '10.000, 10.000, 0.300'"
                }),
                "keep_brackets": (IO.BOOLEAN, {
                    "default": False, 
                    "label_on": "保留[]", 
                    "label_off": "移除[]",
                    "tooltip": "输出格式控制\n• 关闭: '10.0,10.0,0.3'\n• 开启: '[10.0,10.0,0.3]'"
                }),
                "remove_spaces": (IO.BOOLEAN, {
                    "default": False, 
                    "label_on": "移除空格", 
                    "label_off": "保留空格",
                    "tooltip": "是否移除输出中的所有空格\n• 开启: '10.00,10.00,0.30'\n• 关闭: '10.00, 10.00, 0.30'"
                }),
                "min_value": (IO.FLOAT, {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 999999.0, 
                    "step": 0.01,
                    "tooltip": "最小值限制\n任何小于此值的维度都会被强制设为此值\n默认: 1.0"
                }),
                "max_value": (IO.FLOAT, {
                    "default": 2000.0, 
                    "min": 0.001, 
                    "max": 999999.0, 
                    "step": 0.01,
                    "tooltip": "最大值限制\n任何大于此值的维度都会被强制设为此值\n默认: 2000.0"
                }),
            },
        }
    
    RETURN_TYPES = ("*", "*", "*", "*")
    RETURN_NAMES = ("reordered_dimensions", "dimension_1", "dimension_2", "dimension_3")
    FUNCTION = "reorder_and_scale"
    CATEGORY = "VVL/json"
    
    def _parse_dimension_data(self, dimension_data):
        """解析输入的维度数据为三个数值的列表"""
        if dimension_data is None:
            return None
            
        # 如果是列表或元组，直接使用
        if isinstance(dimension_data, (list, tuple)) and len(dimension_data) >= 3:
            return [float(dimension_data[0]), float(dimension_data[1]), float(dimension_data[2])]
        
        # 如果是字符串，尝试解析
        if isinstance(dimension_data, str):
            # 移除可能的方括号
            clean_data = dimension_data.strip()
            if clean_data.startswith('[') and clean_data.endswith(']'):
                clean_data = clean_data[1:-1]
            
            # 尝试按逗号分割
            parts = clean_data.split(',')
            if len(parts) >= 3:
                try:
                    return [float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())]
                except ValueError:
                    pass
            
            # 尝试作为JSON解析
            try:
                parsed = json.loads(dimension_data)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                    return [float(parsed[0]), float(parsed[1]), float(parsed[2])]
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _reorder_dimensions(self, lwh_values, pattern):
        """根据指定模式重新排列维度"""
        length, width, height = lwh_values
        
        reorder_map = {
            "LWH": [length, width, height],   # 长宽高 (默认)
            "LHW": [length, height, width],   # 长高宽
            "WLH": [width, length, height],   # 宽长高
            "WHL": [width, height, length],   # 宽高长
            "HLW": [height, length, width],   # 高长宽
            "HWL": [height, width, length],   # 高宽长
        }
        
        return reorder_map.get(pattern, [length, width, height])
    
    def _clamp_values(self, values, min_value, max_value):
        """将数值限制在最小值和最大值之间"""
        clamped_values = []
        for value in values:
            if value < min_value:
                clamped_values.append(min_value)
            elif value > max_value:
                clamped_values.append(max_value)
            else:
                clamped_values.append(value)
        return clamped_values
    
    def reorder_and_scale(self, dimension_data, reorder_pattern="LWH", scale_factor=1.0, keep_brackets=False, decimal_places=2, remove_spaces=True, min_value=1.0, max_value=2000.0, **kwargs):
        """重新排序和缩放三维数据"""
        try:
            # 解析输入数据
            parsed_data = self._parse_dimension_data(dimension_data)
            
            if parsed_data is None:
                error_msg = "无法解析输入的三维数据。期望格式: [10, 10, 0.3] 或 '10, 10, 0.3'"
                print(f"DimensionReorderAndScale error: {error_msg}")
                return (error_msg, 0.0, 0.0, 0.0)
            
            # 应用缩放因子
            scaled_data = [value * scale_factor for value in parsed_data]
            
            # 重新排列维度
            reordered_data = self._reorder_dimensions(scaled_data, reorder_pattern)
            
            # 应用最小值最大值限制（最后检验）
            clamped_data = self._clamp_values(reordered_data, min_value, max_value)
            
            # 格式化输出和单独数值
            if decimal_places == 0:
                # 整数格式
                formatted_values = [str(int(round(value))) for value in clamped_data]
                formatted_numbers = [int(round(value)) for value in clamped_data]
            else:
                # 小数格式
                format_str = f"{{:.{decimal_places}f}}"
                formatted_values = [format_str.format(value) for value in clamped_data]
                # 单独数值也应用相同的小数精度
                formatted_numbers = [round(value, decimal_places) for value in clamped_data]
            
            # 组装最终字符串
            separator = "," if remove_spaces else ", "
            if keep_brackets:
                result = f"[{separator.join(formatted_values)}]"
            else:
                result = separator.join(formatted_values)
            
            # 输出调试信息
            print(f"DimensionReorderAndScale:")
            print(f"  输入: {dimension_data}")
            print(f"  解析后: {parsed_data}")
            print(f"  缩放因子: {scale_factor}")
            print(f"  缩放后: {scaled_data}")
            print(f"  重排模式: {reorder_pattern}")
            print(f"  重排后: {reordered_data}")
            print(f"  数值限制: [{min_value}, {max_value}]")
            print(f"  限制后: {clamped_data}")
            print(f"  最终输出: {result}")
            print(f"  单独输出: {formatted_numbers[0]}, {formatted_numbers[1]}, {formatted_numbers[2]}")
            
            return (result, formatted_numbers[0], formatted_numbers[1], formatted_numbers[2])
            
        except Exception as e:
            error_msg = f"处理三维数据时出错: {str(e)}"
            print(f"DimensionReorderAndScale error: {error_msg}")
            return (error_msg, 0.0, 0.0, 0.0)


class JsonObjectSplitter:
    """
    JSON对象数组比例拆分器
    
    将JSON数据中的objects数组按比例拆分成两份：
    • 保持其他所有字段不变（camera、subject、scene等）
    • 只拆分objects数组为两部分
    • 支持自定义拆分比例（如 "3:2"、"1:1"）
    • 适用于需要分批处理场景对象的场景
    
    🎯 功能特性：
    • 比例拆分：按指定比例分配objects
    • 结构保持：除objects外的所有字段完全保留
    • 智能分配：自动处理不能整除的情况
    • 错误处理：提供详细的错误信息反馈
    
    📊 拆分规则（以 "3:2" 为例）：
    • 10个对象 → 6个 + 4个（3:2 = 60%:40%）
    • 5个对象 → 3个 + 2个
    • 7个对象 → 4个 + 3个（向上取整）
    
    📝 比例格式：
    • "1:1" - 平均拆分（默认）
    • "3:2" - 60% vs 40%
    • "2:1" - 66.7% vs 33.3%
    • "4:1" - 80% vs 20%
    
    📝 使用场景：
    • 分批处理大量场景对象
    • 将场景拆分为前景和背景
    • 按重要性分组管理对象
    • 优化渲染和加载性能
    
    💡 处理逻辑：
    1. 解析输入的JSON数据和比例参数
    2. 提取objects数组
    3. 根据比例计算分割点
    4. 创建两份完整的JSON，分别包含相应比例的objects
    5. 保持其他所有字段不变
    
    🔍 注意事项：
    • 输入JSON必须包含objects字段
    • objects必须是数组类型
    • 如果objects为空，两个输出相同
    • 拆分保持原始顺序不变
    • 比例格式错误时使用默认的 1:1
    
    📈 拆分示例：
    输入：10个objects，比例 "3:2"
    输出1：前6个objects（60%）
    输出2：后4个objects（40%）
    
    输入：5个objects，比例 "1:1"
    输出1：前3个objects
    输出2：后2个objects
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "包含objects数组的JSON数据\n将按比例拆分objects数组\n其他字段（camera、subject等）保持不变"})
            },
            "optional": {
                "ratio": (IO.STRING, {"default": "1:1", "tooltip": "拆分比例\n格式: \"数字:数字\"\n• \"1:1\" - 平均拆分（默认）\n• \"3:2\" - 60% vs 40%\n• \"2:1\" - 66.7% vs 33.3%\n• \"4:1\" - 80% vs 20%\n第一个数字对应第一份输出"})
            }
        }
    
    RETURN_TYPES = (IO.STRING, IO.STRING, IO.INT, IO.INT)
    RETURN_NAMES = ("json_part1", "json_part2", "part1_count", "part2_count")
    FUNCTION = "split_objects"
    CATEGORY = "VVL/json"
    
    def split_objects(self, json_text, ratio="1:1", **kwargs):
        """将JSON中的objects数组按比例拆分成两份"""
        try:
            # 解析输入JSON
            data = json.loads(json_text)
            
            # 检查是否有objects字段
            if 'objects' not in data:
                error_msg = "未找到objects字段"
                print(f"JsonObjectSplitter error: {error_msg}")
                return (
                    json.dumps({"error": error_msg}, ensure_ascii=False), 
                    json.dumps({"error": error_msg}, ensure_ascii=False),
                    0,
                    0
                )
            
            if not isinstance(data['objects'], list):
                error_msg = "objects字段不是数组类型"
                print(f"JsonObjectSplitter error: {error_msg}")
                return (
                    json.dumps({"error": error_msg}, ensure_ascii=False),
                    json.dumps({"error": error_msg}, ensure_ascii=False),
                    0,
                    0
                )
            
            objects = data['objects']
            total_count = len(objects)
            
            # 如果没有对象，返回两个相同的JSON
            if total_count == 0:
                print("JsonObjectSplitter: objects数组为空")
                return (json_text, json_text, 0, 0)
            
            # 解析比例参数
            ratio_parts = [1, 1]  # 默认 1:1
            try:
                ratio_str = str(ratio).strip()
                if ':' in ratio_str:
                    parts = ratio_str.split(':')
                    if len(parts) == 2:
                        ratio_parts[0] = float(parts[0].strip())
                        ratio_parts[1] = float(parts[1].strip())
                        
                        # 验证比例值是否有效
                        if ratio_parts[0] <= 0 or ratio_parts[1] <= 0:
                            print(f"JsonObjectSplitter: 比例值无效 ({ratio_str})，使用默认 1:1")
                            ratio_parts = [1, 1]
                    else:
                        print(f"JsonObjectSplitter: 比例格式错误 ({ratio_str})，使用默认 1:1")
                else:
                    print(f"JsonObjectSplitter: 比例格式错误 ({ratio_str})，使用默认 1:1")
            except Exception as e:
                print(f"JsonObjectSplitter: 解析比例时出错 ({ratio}): {e}，使用默认 1:1")
                ratio_parts = [1, 1]
            
            # 计算分割点
            # 根据比例计算第一份应该包含的对象数
            ratio_sum = ratio_parts[0] + ratio_parts[1]
            part1_ratio = ratio_parts[0] / ratio_sum
            split_point = int(total_count * part1_ratio + 0.5)  # 四舍五入
            
            # 确保分割点在有效范围内
            split_point = max(0, min(split_point, total_count))
            
            # 创建两份数据（使用深拷贝避免引用问题）
            data1 = copy.deepcopy(data)
            data1['objects'] = objects[:split_point]
            
            data2 = copy.deepcopy(data)
            data2['objects'] = objects[split_point:]
            
            # 转换为JSON字符串
            json_part1 = json.dumps(data1, ensure_ascii=False, indent=2)
            json_part2 = json.dumps(data2, ensure_ascii=False, indent=2)
            
            # 获取拆分后的数量
            part1_count = len(data1['objects'])
            part2_count = len(data2['objects'])
            
            # 输出处理统计信息
            part1_percent = (part1_count / total_count * 100) if total_count > 0 else 0
            part2_percent = (part2_count / total_count * 100) if total_count > 0 else 0
            
            print(f"JsonObjectSplitter 拆分完成:")
            print(f"  • 输入总对象数: {total_count}")
            print(f"  • 设置比例: {ratio_parts[0]}:{ratio_parts[1]} ({ratio})")
            print(f"  • 第一份对象数: {part1_count} ({part1_percent:.1f}%)")
            print(f"  • 第二份对象数: {part2_count} ({part2_percent:.1f}%)")
            print(f"  • 分割点索引: {split_point}")
            
            # 显示拆分详情
            if total_count <= 10:
                print(f"  • 第一份对象: {[obj.get('name', f'对象{i}') for i, obj in enumerate(data1['objects'])]}")
                print(f"  • 第二份对象: {[obj.get('name', f'对象{i}') for i, obj in enumerate(data2['objects'])]}")
            
            return (json_part1, json_part2, part1_count, part2_count)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            print(f"JsonObjectSplitter error: {error_msg}")
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                0
            )
        except Exception as e:
            error_msg = f"拆分JSON时出错: {str(e)}"
            print(f"JsonObjectSplitter error: {error_msg}")
            import traceback
            traceback.print_exc()
            return (
                json.dumps({"error": error_msg}, ensure_ascii=False),
                json.dumps({"error": error_msg}, ensure_ascii=False),
                0,
                0
            )


class IndexOffsetAdjuster:
    """
    索引偏移调整器
    
    调整字符串列表中每个元素的索引编号：
    • 处理格式如 "0,value" 的字符串列表
    • 给每个元素的第一个数字（索引）添加偏移量
    • 保持逗号后的值部分不变
    • 适用于需要重新编号或合并多个列表的场景
    
    🎯 功能特性：
    • 索引偏移：给所有索引添加固定偏移量
    • 格式保持：保持原始字符串格式
    • 批量处理：一次性处理整个列表
    • 灵活输入：支持多种输入格式
    
    📊 处理示例（offset=10）：
    输入：["0,url1", "1,url2", "2,url3"]
    输出：["10,url1", "11,url2", "12,url3"]
    
    输入：["5,data_a", "6,data_b", "7,data_c"]
    输出：["15,data_a", "16,data_b", "17,data_c"]
    
    📝 使用场景：
    • 合并多个批次的数据时重新编号
    • 在循环中为每个批次设置不同的起始索引
    • 将多个数据源的索引统一到同一命名空间
    • 避免不同批次之间的索引冲突
    
    💡 处理逻辑：
    1. 接收输入列表和偏移量
    2. 遍历每个元素
    3. 提取逗号前的索引数字
    4. 给索引加上偏移量
    5. 保持逗号后的内容不变
    6. 重新组合字符串
    
    🔍 注意事项：
    • 输入列表中的元素应该包含逗号分隔符
    • 逗号前的部分应该是数字或可转换为数字的字符串
    • 如果某个元素格式不正确，会跳过并保持原样
    • 偏移量可以是负数（但要注意结果索引不要小于0）
    
    📈 工作流示例：
    第一批：["0,url_a", "1,url_b"] → 循环处理
    第二批：["0,url_c", "1,url_d"] → offset=2 → ["2,url_c", "3,url_d"] → 循环处理
    这样可以避免两批数据的索引冲突
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": (IO.ANY, {"tooltip": "输入的列表数据\n格式: [\"索引,值\", ...]\n例如: [\"0,url1\", \"1,url2\", \"2,url3\"]"}),
                "offset": (IO.INT, {"default": 0, "min": -99999, "max": 99999, "step": 1, "tooltip": "索引偏移量\n将添加到每个元素的索引上\n• 正数: 增加索引值\n• 负数: 减少索引值\n• 0: 保持不变"})
            }
        }
    
    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("adjusted_list",)
    FUNCTION = "adjust_index_offset"
    CATEGORY = "VVL/json"
    
    def adjust_index_offset(self, input_list, offset, **kwargs):
        """调整列表中每个元素的索引偏移量"""
        try:
            # 处理输入数据
            if input_list is None:
                print("IndexOffsetAdjuster: 输入为空")
                return ([],)
            
            # 如果输入是字符串，尝试解析为列表
            if isinstance(input_list, str):
                try:
                    import json
                    input_list = json.loads(input_list)
                except Exception:
                    print("IndexOffsetAdjuster: 无法将字符串解析为列表")
                    return ([],)
            
            # 确保输入是列表类型
            if not isinstance(input_list, (list, tuple)):
                print(f"IndexOffsetAdjuster: 输入不是列表类型，类型={type(input_list)}")
                return ([input_list],)
            
            # 处理列表
            adjusted_list = []
            processed_count = 0
            skipped_count = 0
            
            for i, item in enumerate(input_list):
                try:
                    # 确保元素是字符串
                    item_str = str(item) if not isinstance(item, str) else item
                    
                    # 检查是否包含逗号
                    if ',' not in item_str:
                        # 没有逗号，保持原样
                        adjusted_list.append(item)
                        skipped_count += 1
                        continue
                    
                    # 分割字符串
                    parts = item_str.split(',', 1)  # 只分割第一个逗号
                    
                    if len(parts) != 2:
                        # 格式不正确，保持原样
                        adjusted_list.append(item)
                        skipped_count += 1
                        continue
                    
                    # 提取索引和值部分
                    index_str = parts[0].strip()
                    value_part = parts[1]
                    
                    # 尝试将索引转换为数字
                    try:
                        original_index = int(index_str)
                        new_index = original_index + offset
                        
                        # 重新组合字符串
                        adjusted_item = f"{new_index},{value_part}"
                        adjusted_list.append(adjusted_item)
                        processed_count += 1
                        
                    except ValueError:
                        # 索引部分不是数字，保持原样
                        adjusted_list.append(item)
                        skipped_count += 1
                        
                except Exception as e:
                    # 处理单个元素时出错，保持原样
                    print(f"IndexOffsetAdjuster: 处理元素 {i} 时出错: {e}")
                    adjusted_list.append(item)
                    skipped_count += 1
            
            # 输出处理统计信息
            print(f"IndexOffsetAdjuster 处理完成:")
            print(f"  • 输入元素数: {len(input_list)}")
            print(f"  • 偏移量: {offset}")
            print(f"  • 成功处理: {processed_count} 个")
            print(f"  • 跳过/保持: {skipped_count} 个")
            
            # 显示处理示例（前3个元素）
            if processed_count > 0 and len(input_list) <= 10:
                print(f"  • 处理示例:")
                for i in range(min(3, len(input_list))):
                    if i < len(adjusted_list):
                        original = str(input_list[i]) if i < len(input_list) else ""
                        adjusted = str(adjusted_list[i])
                        if original != adjusted:
                            print(f"    [{i}] {original} → {adjusted}")
            
            return (adjusted_list,)
            
        except Exception as e:
            error_msg = f"调整索引偏移量时出错: {str(e)}"
            print(f"IndexOffsetAdjuster error: {error_msg}")
            import traceback
            traceback.print_exc()
            return ([],)


# Node class mappings
NODE_CLASS_MAPPINGS = {
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
    "JsonObjectSplitter": JsonObjectSplitter,
    "IndexOffsetAdjuster": IndexOffsetAdjuster
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
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
    "JsonObjectSplitter": "VVL JSON Object Splitter",
    "IndexOffsetAdjuster": "VVL Index Offset Adjuster",
}