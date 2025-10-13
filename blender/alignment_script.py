"""
Blender批量对齐脚本
被ComfyUI节点调用，在Blender后台执行模型对齐和导出
直接调用 alignment_ops.py 中的核心函数
"""
import bpy
import sys
import os
import json
import math
import tempfile
import re
from mathutils import Vector, Matrix

# 导入 alignment_ops 中的核心函数
import importlib.util

# 动态加载 alignment_ops 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
alignment_ops_path = os.path.join(script_dir, "alignment_ops.py")

spec = importlib.util.spec_from_file_location("alignment_ops", alignment_ops_path)
alignment_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment_ops)

# 导入需要使用的核心函数
get_bbox_directions_static = alignment_ops.get_bbox_directions_static
check_models_alignment = alignment_ops.check_models_alignment

# ==================== 辅助函数 ====================

def download_and_import_model(obj_config, output_dir):
    """下载并导入模型"""
    try:
        # 获取URL
        url = obj_config.get('3d_url') or \
              obj_config.get('output', {}).get('url') or \
              obj_config.get('url', '')
        
        if not url:
            return None, None
        
        print(f"[下载模型] URL: {url}")
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        filename = url.split('/')[-1].split('?')[0]
        if not filename:
            filename = "model.glb"
        
        # 检测文件格式
        file_ext = os.path.splitext(filename)[1].lower()
        if not file_ext:
            file_ext = '.glb'
        
        temp_path = os.path.join(temp_dir, filename)
        
        # 使用requests下载
        try:
            import requests
            import warnings
            try:
                from requests.packages.urllib3.exceptions import InsecureRequestWarning
                warnings.simplefilter('ignore', InsecureRequestWarning)
            except:
                pass
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
            }
            
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            if 'dreammaker.netease.com' in parsed_url.netloc:
                headers['X-Auth-User'] = 'blender-alignment-tool'
            
            response = requests.get(url, headers=headers, timeout=300, verify=False)
            
            if response.status_code == 200:
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                print(f"[下载模型] 成功: {len(response.content)/(1024*1024):.2f} MB")
            else:
                raise Exception(f"HTTP错误 {response.status_code}")
        
        except ImportError:
            raise Exception("需要安装requests库")
        
        # 导入模型
        imported_obj = None
        
        if file_ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=temp_path)
            imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        elif file_ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=temp_path)
            imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        elif file_ext == '.obj':
            bpy.ops.import_scene.obj(filepath=temp_path)
            imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        else:
            bpy.ops.import_scene.fbx(filepath=temp_path)
            imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        
        if imported_obj:
            imported_obj.name = obj_config.get('name', filename)
            print(f"[导入模型] 成功: {imported_obj.name}")
        
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        
        return imported_obj, file_ext
        
    except Exception as e:
        print(f"[下载/导入失败] {str(e)}")
        return None, None

def create_reference_box(obj_config):
    """创建参考Box"""
    try:
        name = obj_config.get('name', 'ReferenceBox')
        rotation = obj_config.get('rotation', [0, 0, 0])
        scale = obj_config.get('scale', [1, 1, 1])
        
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        box = bpy.context.active_object
        box.name = f"{name}_RefBox"
        
        # 设置旋转（度转弧度）
        box.rotation_euler = [r * math.pi / 180 for r in rotation]
        
        # 设置缩放
        box.scale = scale
        
        print(f"[创建参考Box] {box.name}, 旋转: {rotation}°, 缩放: {scale}")
        
        return box
        
    except Exception as e:
        print(f"[创建参考Box失败] {str(e)}")
        return None

def execute_rotation_align(ref_obj, target_obj):
    """执行旋转对齐"""
    ref_directions = get_bbox_directions_static(ref_obj)
    target_directions = get_bbox_directions_static(target_obj)
    
    ref_loc, ref_rot, ref_scale = ref_obj.matrix_world.decompose()
    ref_dims = ref_directions['dimensions']
    tgt_dims = target_directions['dimensions']
    
    ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
    tgt_sizes = [tgt_dims.x, tgt_dims.y, tgt_dims.z]
    
    ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
    tgt_sorted_indices = sorted(range(3), key=lambda i: tgt_sizes[i])
    
    axis_mapping = [None, None, None]
    for rank in range(3):
        tgt_axis = tgt_sorted_indices[rank]
        ref_axis = ref_sorted_indices[rank]
        axis_mapping[tgt_axis] = ref_axis
    
    perm_matrix = Matrix((
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0)
    ))
    
    for i in range(3):
        perm_matrix[axis_mapping[i]][i] = 1.0
    
    det = perm_matrix.determinant()
    if det < 0:
        last_mapping = axis_mapping[2]
        perm_matrix[last_mapping][2] *= -1
    
    final_rotation_matrix = ref_rot.to_matrix() @ perm_matrix
    location, old_rotation, scale = target_obj.matrix_world.decompose()
    new_rotation = final_rotation_matrix.to_quaternion()
    target_obj.matrix_world = Matrix.LocRotScale(location, new_rotation, scale)
    
    bpy.context.view_layer.update()

def execute_perfect_align(ref_obj, target_obj):
    """执行完美重合对齐"""
    ref_dirs = get_bbox_directions_static(ref_obj)
    target_dirs = get_bbox_directions_static(target_obj)
    
    ref_dims = ref_dirs['dimensions']
    target_dims = target_dirs['dimensions']
    
    ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
    target_sizes = [target_dims.x, target_dims.y, target_dims.z]
    
    ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
    target_sorted_indices = sorted(range(3), key=lambda i: target_sizes[i])
    
    scale_factors = [1.0, 1.0, 1.0]
    for rank in range(3):
        ref_axis_idx = ref_sorted_indices[rank]
        target_axis_idx = target_sorted_indices[rank]
        
        ref_size = ref_sizes[ref_axis_idx]
        target_size = target_sizes[target_axis_idx]
        
        if target_size > 1e-6:
            scale_factors[target_axis_idx] = ref_size / target_size
        else:
            scale_factors[target_axis_idx] = 1.0
    
    current_scale = target_obj.scale.copy()
    new_scale = Vector((
        current_scale.x * scale_factors[0],
        current_scale.y * scale_factors[1],
        current_scale.z * scale_factors[2]
    ))
    
    target_obj.scale = new_scale
    bpy.context.view_layer.update()
    
    ref_matrix = ref_obj.matrix_world
    ref_bbox_local_center = sum((Vector(corner) for corner in ref_obj.bound_box), Vector()) / 8
    ref_bbox_world_center = ref_matrix @ ref_bbox_local_center
    
    target_matrix = target_obj.matrix_world
    target_bbox_local_center = sum((Vector(corner) for corner in target_obj.bound_box), Vector()) / 8
    target_bbox_world_center = target_matrix @ target_bbox_local_center
    
    offset = ref_bbox_world_center - target_bbox_world_center
    target_obj.location += offset
    
    bpy.context.view_layer.update()

def apply_coordinate_transform(model_obj, target_engine, export_format):
    """应用坐标系转换"""
    if target_engine == 'NONE':
        print(f"  [坐标转换] 无转换模式")
        return
    
    elif target_engine == 'UE':
        # GLB/GLTF格式时跳过缩放（导出器会自动处理单位转换）
        if export_format and export_format.lower() in ['.glb', '.gltf']:
            print(f"  [坐标转换] UE模式 + GLB导出：跳过场景内缩放（GLB导出器自动处理）")
            return
        
        # FBX等其他格式需要手动缩放100倍（UE使用厘米，Blender使用米）
        UE_SCALE_FACTOR = 100.0
        print(f"  [坐标转换] UE模式 + FBX导出：应用{UE_SCALE_FACTOR}x缩放")
        current_scale = model_obj.scale.copy()
        model_obj.scale = Vector((
            current_scale.x * UE_SCALE_FACTOR,
            current_scale.y * UE_SCALE_FACTOR,
            current_scale.z * UE_SCALE_FACTOR
        ))
    
    bpy.context.view_layer.update()

def export_model(model_obj, obj_config, output_dir, file_format, export_format):
    """导出模型"""
    try:
        model_name = obj_config.get('name', 'model')
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        
        if export_format == 'GLB':
            output_filename = f"{safe_name}.glb"
            output_path = os.path.join(output_dir, output_filename)
            
            bpy.ops.object.select_all(action='DESELECT')
            model_obj.select_set(True)
            for child in model_obj.children_recursive:
                child.select_set(True)
            
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                use_selection=True,
                export_texcoords=True,
                export_normals=True,
                export_materials='EXPORT',
                export_image_format='AUTO',
            )
        else:  # FBX
            output_filename = f"{safe_name}.fbx"
            output_path = os.path.join(output_dir, output_filename)
            
            bpy.ops.object.select_all(action='DESELECT')
            model_obj.select_set(True)
            for child in model_obj.children_recursive:
                child.select_set(True)
            
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                use_selection=True,
                apply_scale_options='FBX_SCALE_ALL',
                axis_forward='-Z',
                axis_up='Y',
                bake_space_transform=True,
                object_types={'MESH', 'EMPTY'},
                use_mesh_modifiers=True,
                mesh_smooth_type='OFF',
                use_tspace=True,
                embed_textures=True,
                path_mode='COPY',
                batch_mode='OFF',
            )
        
        print(f"[导出模型] 成功: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[导出模型失败] {str(e)}")
        return None

# ==================== 主函数 ====================

def main():
    """主处理函数"""
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    if len(argv) < 5:
        print("错误: 参数不足")
        sys.exit(1)
    
    input_json_path = argv[0]
    output_dir = argv[1]
    export_format = argv[2]
    target_engine = argv[3]
    output_json_path = argv[4]
    
    print(f"\n[Blender对齐脚本] 开始处理")
    print(f"  输入JSON: {input_json_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  导出格式: {export_format}")
    print(f"  目标引擎: {target_engine}")
    
    # 读取输入JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    objects = config.get('objects', [])
    print(f"  共 {len(objects)} 个对象\n")
    
    # 清空场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 处理每个对象
    processed_objects = []
    success_count = 0
    failed_count = 0
    
    for idx, obj_config in enumerate(objects):
        try:
            obj_name = obj_config.get('name', f'object_{idx}')
            print(f"[{idx+1}/{len(objects)}] 处理: {obj_name}")
            
            # 下载并导入模型
            model_obj, file_format = download_and_import_model(obj_config, output_dir)
            if not model_obj:
                print(f"  ✗ 失败: 无法导入模型")
                failed_count += 1
                continue
            
            # 设置位置为原点
            model_obj.location = Vector((0, 0, 0))
            
            # 创建参考Box
            ref_box = create_reference_box(obj_config)
            if not ref_box:
                print(f"  ✗ 失败: 无法创建参考Box")
                failed_count += 1
                continue
            
            # 应用JSON中的旋转到模型
            rotation = obj_config.get('rotation', [0, 0, 0])
            model_obj.rotation_euler = [r * math.pi / 180 for r in rotation]
            bpy.context.view_layer.update()
            
            # 检查是否已对齐
            tolerance_rad = 1.0 * math.pi / 180.0  # 1度容差
            is_aligned, alignment_info = check_models_alignment(ref_box, model_obj, tolerance_rad)
            
            # 检查是否有错误（当函数执行异常时，返回 {'error': ...}）
            if 'error' in alignment_info:
                print(f"  ⚠ 对齐检查异常: {alignment_info['error']}，默认执行对齐")
                execute_rotation_align(ref_box, model_obj)
            elif not is_aligned:
                angle_diff = alignment_info.get('max_angle_diff_deg', 0)
                print(f"  → 旋转对齐（角度差: {angle_diff:.2f}°）")
                execute_rotation_align(ref_box, model_obj)
            else:
                print(f"  → 已对齐，跳过旋转")
            
            # 执行完美重合对齐
            print(f"  → 完美重合对齐")
            execute_perfect_align(ref_box, model_obj)
            
            # 删除参考Box
            bpy.data.objects.remove(ref_box, do_unlink=True)
            
            # 应用坐标系转换
            # 注意：这里要传入用户选择的导出格式，而不是输入文件格式
            # GLB格式在UE模式下会跳过缩放（GLB导出器会自动处理单位转换）
            export_ext = '.glb' if export_format == 'GLB' else '.fbx'
            apply_coordinate_transform(model_obj, target_engine, export_ext)
            
            # 确保位置为原点
            model_obj.location = Vector((0, 0, 0))
            bpy.context.view_layer.update()
            
            # 导出模型
            exported_path = export_model(model_obj, obj_config, output_dir, file_format, export_format)
            
            if not exported_path:
                print(f"  ✗ 失败: 导出失败")
                failed_count += 1
                continue
            
            # 创建新的对象配置
            new_obj_config = obj_config.copy()
            new_obj_config['rotation'] = [0, 0, 0]
            
            # 更新URL字段
            if '3d_url' in obj_config:
                new_obj_config['3d_url'] = exported_path
            elif 'output' in obj_config and 'url' in obj_config['output']:
                if 'output' not in new_obj_config:
                    new_obj_config['output'] = {}
                new_obj_config['output']['url'] = exported_path
            else:
                if 'output' not in new_obj_config:
                    new_obj_config['output'] = {}
                new_obj_config['output']['url'] = exported_path
            
            processed_objects.append(new_obj_config)
            success_count += 1
            print(f"  ✓ 成功")
            
            # 删除模型对象
            bpy.data.objects.remove(model_obj, do_unlink=True)
            
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # 生成输出JSON
    output_config = config.copy()
    output_config['objects'] = processed_objects
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Blender对齐脚本] 处理完成")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {failed_count} 个")
    print(f"  输出: {output_json_path}")

# 执行主函数
if __name__ == "__main__":
    main()
