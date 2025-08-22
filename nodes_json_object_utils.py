import json
import re
import ast
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
    Merge processed JSON with removed duplicates, using the same 3d_url for objects with same name and scale.
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
            
            # Create a mapping of (base_name, scale) to 3d_url from processed objects
            url_mapping = {}
            for obj in processed_objects:
                name = obj.get('name', '')
                base_name = re.sub(r'\d+$', '', name).strip()
                scale = obj.get('scale', [])
                scale_tuple = tuple(scale) if isinstance(scale, list) else scale
                
                combination_key = (base_name, scale_tuple)
                if '3d_url' in obj:
                    url_mapping[combination_key] = obj['3d_url']
            
            # Restore removed objects with correct 3d_url
            restored_objects = []
            for removed_obj in removed_objects:
                # Remove metadata fields
                obj = {k: v for k, v in removed_obj.items() 
                      if not k.startswith('_')}
                
                # Find matching 3d_url
                name = obj.get('name', '')
                base_name = re.sub(r'\d+$', '', name).strip()
                scale = obj.get('scale', [])
                scale_tuple = tuple(scale) if isinstance(scale, list) else scale
                
                combination_key = (base_name, scale_tuple)
                if combination_key in url_mapping:
                    obj['3d_url'] = url_mapping[combination_key]
                
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
    Extracts subject, names list, and scales list from the input JSON.

    - names_json: JSON string array of names, optionally stripping numeric suffixes
    - scales_json: JSON string array of scale triplets (e.g., [[0.7,0.5,0.75], ...])
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

    RETURN_TYPES = (IO.STRING, IO.ANY, IO.ANY)
    RETURN_NAMES = ("subject", "names_list", "scales_list")
    FUNCTION = "extract_fields"
    CATEGORY = "VVL/json"

    def extract_fields(self, json_text, strip_numeric_suffix=True, **kwargs):
        try:
            data = json.loads(json_text)
            subject = data.get("subject", "")
            objects = data.get("objects", [])

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
            )
        except json.JSONDecodeError as e:
            msg = json.dumps({"error": f"Invalid JSON input: {str(e)}"})
            return (msg, msg, msg)
        except Exception as e:
            msg = json.dumps({"error": f"Error extracting fields: {str(e)}"})
            return (msg, msg, msg)


class ApplyUrlsToJson:
    """
    Apply a list of (index, url) pairs to deduplicated_json.objects[index].3d_url.

    - Robust to inputs shaped like [[idx, url], ...]
    - Skips entries with None url (configurable)
    - Ignores out-of-bounds indices (configurable)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "deduplicated_json": (IO.STRING, {"multiline": True, "default": ""}),
                "idx_url_list": (IO.ANY, {}),
            },
            "optional": {
                "field_name": (IO.STRING, {"default": "3d_url"}),
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

    def apply(self, deduplicated_json, idx_url_list, field_name="3d_url", skip_none_url=True, ignore_oob_index=True, **kwargs):
        try:
            data = json.loads(deduplicated_json)
        except json.JSONDecodeError as e:
            return (json.dumps({"error": f"Invalid JSON input: {str(e)}"}),)

        if not isinstance(data, dict) or "objects" not in data or not isinstance(data["objects"], list):
            return (deduplicated_json,)

        objects = data["objects"]

        pairs = self._normalize_pairs(idx_url_list)
        for raw_idx, url in pairs:
            try:
                idx = int(raw_idx)
            except Exception:
                continue

            if idx < 0 or idx >= len(objects):
                if ignore_oob_index:
                    continue
                else:
                    return (json.dumps({"error": f"Index out of bounds: {idx}"}, ensure_ascii=False),)

            if url is None and skip_none_url:
                continue

            obj = objects[idx]
            if isinstance(obj, dict):
                obj[field_name] = url

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
    • return_empty_on_error: 错误处理方式
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
                "return_empty_on_error": (IO.BOOLEAN, {"default": True, "label_on": "返回空值", "label_off": "返回错误信息", "tooltip": "出错时的处理方式\n• 开启: 出错返回空字符串 ''\n• 关闭: 出错返回错误信息描述\n推荐开启以保持工作流稳定性"}),
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("field_value",)
    FUNCTION = "extract_field_value"
    CATEGORY = "VVL/json"
    
    def extract_field_value(self, json_array, index=0, field_name="url", return_empty_on_error=True, **kwargs):
        """从数组中提取指定索引元素的字段值"""
        try:
            # 处理输入数据
            data = json_array
            
            # 如果输入是字符串，尝试解析为JSON
            if isinstance(json_array, str):
                try:
                    data = json.loads(json_array)
                except json.JSONDecodeError:
                    if return_empty_on_error:
                        return ("",)
                    else:
                        return (f"错误: 无效的JSON字符串",)
            
            # 检查是否为列表/数组
            if not isinstance(data, list):
                if return_empty_on_error:
                    return ("",)
                else:
                    return (f"错误: 输入不是数组格式",)
            
            # 检查数组是否为空
            if len(data) == 0:
                if return_empty_on_error:
                    return ("",)
                else:
                    return (f"错误: 数组为空",)
            
            # 检查索引是否在有效范围内
            try:
                # Python支持负数索引，但我们需要检查边界
                if index >= len(data) or index < -len(data):
                    if return_empty_on_error:
                        return ("",)
                    else:
                        return (f"错误: 索引 {index} 超出数组范围 (0 到 {len(data)-1})",)
                
                # 获取指定索引的元素
                target_element = data[index]
                
            except IndexError:
                if return_empty_on_error:
                    return ("",)
                else:
                    return (f"错误: 索引 {index} 超出数组范围",)
            
            # 检查目标元素是否为字典
            if not isinstance(target_element, dict):
                if return_empty_on_error:
                    return ("",)
                else:
                    return (f"错误: 索引 {index} 处的元素不是对象格式",)
            
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
            
            if return_empty_on_error:
                return ("",)
            else:
                return (error_msg,)


class DimensionReorderAndScale:
    """
    三维数据重新排序和缩放节点
    
    处理格式如 [10, 10, 0.3] 的三维数据：
    • 支持任意调换长(length)、宽(width)、高(height)的位置
    • 提供缩放因子控制整体大小
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
            },
        }
    
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("reordered_dimensions",)
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
    
    def reorder_and_scale(self, dimension_data, reorder_pattern="LWH", scale_factor=1.0, keep_brackets=False, decimal_places=2, remove_spaces=True, **kwargs):
        """重新排序和缩放三维数据"""
        try:
            # 解析输入数据
            parsed_data = self._parse_dimension_data(dimension_data)
            
            if parsed_data is None:
                error_msg = "无法解析输入的三维数据。期望格式: [10, 10, 0.3] 或 '10, 10, 0.3'"
                print(f"DimensionReorderAndScale error: {error_msg}")
                return (error_msg,)
            
            # 应用缩放因子
            scaled_data = [value * scale_factor for value in parsed_data]
            
            # 重新排列维度
            reordered_data = self._reorder_dimensions(scaled_data, reorder_pattern)
            
            # 格式化输出
            if decimal_places == 0:
                # 整数格式
                formatted_values = [str(int(round(value))) for value in reordered_data]
            else:
                # 小数格式
                format_str = f"{{:.{decimal_places}f}}"
                formatted_values = [format_str.format(value) for value in reordered_data]
            
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
            print(f"  最终输出: {result}")
            
            return (result,)
            
        except Exception as e:
            error_msg = f"处理三维数据时出错: {str(e)}"
            print(f"DimensionReorderAndScale error: {error_msg}")
            return (error_msg,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "JsonObjectDeduplicator": JsonObjectDeduplicator,
    "JsonObjectMerger": JsonObjectMerger,
    "JsonExtractSubjectNamesScales": JsonExtractSubjectNamesScales,
    "ApplyUrlsToJson": ApplyUrlsToJson,
    "JsonMarkdownCleaner": JsonMarkdownCleaner,
    "IndexUrlPairDeduplicator": IndexUrlPairDeduplicator,
    "JsonArrayElementFieldExtractor": JsonArrayElementFieldExtractor,
    "DimensionReorderAndScale": DimensionReorderAndScale
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
    "DimensionReorderAndScale": "VVL Dimension Reorder and Scale",
}