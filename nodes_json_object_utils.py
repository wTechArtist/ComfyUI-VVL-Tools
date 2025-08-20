import json
import re
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



# Node class mappings
NODE_CLASS_MAPPINGS = {
    "JsonObjectDeduplicator": JsonObjectDeduplicator,
    "JsonObjectMerger": JsonObjectMerger,
    "JsonExtractSubjectNamesScales": JsonExtractSubjectNamesScales,
    "ApplyUrlsToJson": ApplyUrlsToJson,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonObjectDeduplicator": "VVL JSON Object Deduplicator",
    "JsonObjectMerger": "VVL JSON Object Merger",
    "JsonExtractSubjectNamesScales": "VVL JSON Extract: subject, names, scales",
    "ApplyUrlsToJson": "VVL Apply URLs to JSON",
}