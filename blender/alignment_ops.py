import bpy
import bmesh
import os
import json

from math import radians
from mathutils import Vector, Matrix
from bpy.types import Operator
from bpy.props import StringProperty

# 获取插件包名
def get_addon_name():
    return __name__.partition('.')[0]

# 数值稳定性阈值
_EPS = 1e-8

# 对齐检查的默认容差（角度，弧度）
_ALIGNMENT_TOLERANCE = 0.017453292  # 约1度

def get_bbox_directions_static(obj):
    """静态函数版本的get_bbox_directions，用于检查对齐状态"""
    # 获取局部坐标系下的尺寸
    local_dims = obj.dimensions.copy()
    
    # 获取物体的旋转矩阵（仅旋转部分）
    loc, rot, scale = obj.matrix_world.decompose()
    rot_matrix = rot.to_matrix()
    
    # 局部坐标系的轴向量
    local_x = Vector((1, 0, 0))
    local_y = Vector((0, 1, 0))
    local_z = Vector((0, 0, 1))
    
    # 转换到世界坐标系
    world_x = rot_matrix @ local_x
    world_y = rot_matrix @ local_y
    world_z = rot_matrix @ local_z
    
    # 获取世界坐标系下的中心点
    center = obj.matrix_world @ Vector((0, 0, 0))
    
    return {
        'x': world_x,
        'y': world_y,
        'z': world_z,
        'center': center,
        'dimensions': local_dims
    }


# ==================== 批量预览共用函数 ====================

def format_elapsed_time(elapsed_time):
    """格式化耗时为易读的字符串"""
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    if hours > 0:
        return f"{hours}小时{minutes}分钟{seconds:.2f}秒"
    elif minutes > 0:
        return f"{minutes}分钟{seconds:.2f}秒"
    else:
        return f"{seconds:.2f}秒"


def print_processing_summary(prefs, operation_name, success_count, failed_count, exported_count, elapsed_time, processing_mode=None):
    """打印处理完成统计信息"""
    time_str = format_elapsed_time(elapsed_time)
    
    if prefs.show_debug_info:
        print(f"\n{'='*80}")
        print(f"[{operation_name}] 处理完成统计:")
        if processing_mode:
            print(f"  处理模式: {processing_mode}")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {failed_count} 个")
        if exported_count >= 0:
            print(f"  导出: {exported_count} 个")
        print(f"  总耗时: {time_str} ({elapsed_time:.2f}秒)")
        print(f"{'='*80}\n")
    
    return time_str


def download_and_import_model_helper(obj_config, auth_token, prefs, report_func):
    """下载并导入模型的辅助函数"""
    try:
        import tempfile
        # 兼容多种URL字段格式
        url = obj_config.get('3d_url') or \
              obj_config.get('output', {}).get('url') or \
              obj_config.get('url', '')
        
        if not url:
            report_func({'WARNING'}, f"对象缺少URL: {obj_config.get('name')}")
            return None, None
        
        if prefs.show_debug_info:
            print(f"[批量预览] 下载模型: {url}")
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        filename = url.split('/')[-1].split('?')[0]
        if not filename:
            filename = "model.fbx"
        
        # 检测文件格式
        file_ext = os.path.splitext(filename)[1].lower()
        if not file_ext:
            file_ext = '.fbx'
        
        temp_path = os.path.join(temp_dir, filename)
        
        # 使用requests库下载
        try:
            import requests
            import warnings
            try:
                from requests.packages.urllib3.exceptions import InsecureRequestWarning
                warnings.simplefilter('ignore', InsecureRequestWarning)
            except:
                pass
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
            }
            
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            
            # 为dreammaker域名添加特定的请求头
            if 'dreammaker.netease.com' in parsed_url.netloc:
                headers['Referer'] = 'https://dreammaker.netease.com/'
                headers['Origin'] = 'https://dreammaker.netease.com'
                headers['X-Requested-With'] = 'XMLHttpRequest'
                headers['X-Auth-User'] = 'blender-alignment-tool'
            
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            response = requests.get(url, headers=headers, timeout=300, verify=False)
            
            if response.status_code == 200:
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                if prefs.show_debug_info:
                    file_size_mb = len(response.content) / (1024 * 1024)
                    print(f"[批量预览] 下载成功: {file_size_mb:.2f} MB")
            else:
                raise Exception(f"HTTP错误 {response.status_code}: {response.reason}")
            
        except ImportError:
            report_func({'ERROR'}, "需要安装requests库")
            return None, None
        
        # 根据文件格式导入模型
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
            if prefs.show_debug_info:
                print(f"[批量预览] 导入成功: {imported_obj.name}")
        
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        
        return imported_obj, file_ext
        
    except Exception as e:
        report_func({'WARNING'}, f"下载或导入模型失败: {str(e)}")
        if prefs.show_debug_info:
            import traceback
            traceback.print_exc()
        return None, None


def apply_transform_to_model_helper(model_obj, obj_config, prefs, report_func):
    """应用JSON中的rotation和position到模型"""
    try:
        # 应用rotation
        rotation = obj_config.get('rotation', [0, 0, 0])
        rotation_rad = [r * 3.14159 / 180 for r in rotation]
        
        # 根据对象的旋转模式正确设置旋转
        if model_obj.rotation_mode == 'QUATERNION':
            # 四元数模式：先转换为四元数
            from mathutils import Euler
            euler = Euler(rotation_rad, 'XYZ')
            model_obj.rotation_quaternion = euler.to_quaternion()
        elif 'AXIS_ANGLE' in model_obj.rotation_mode:
            # 轴角模式：转换为轴角
            from mathutils import Euler
            euler = Euler(rotation_rad, 'XYZ')
            quat = euler.to_quaternion()
            axis, angle = quat.to_axis_angle()
            model_obj.rotation_axis_angle = [angle, axis.x, axis.y, axis.z]
        else:
            # 欧拉角模式：直接设置
            model_obj.rotation_euler = rotation_rad
        
        # 应用position（所有模式）
        position = obj_config.get('position', [0, 0, 0])
        if prefs.target_engine == 'UE':
            # UE模式：厘米→米，×0.01
            position = [p * 0.01 for p in position]
        
        model_obj.location = Vector(position)
        
        bpy.context.view_layer.update()
        
        if prefs.show_debug_info:
            print(f"[批量预览] 应用变换到模型:")
            print(f"  rotation: {rotation}°")
            print(f"  position: {position} {'(UE模式已转换)' if prefs.target_engine == 'UE' else ''}")
        
    except Exception as e:
        report_func({'WARNING'}, f"应用变换失败: {str(e)}")


def create_reference_box_with_position_helper(obj_config, prefs, report_func):
    """创建带position的参考Box"""
    try:
        name = obj_config.get('name', 'ReferenceBox')
        rotation = obj_config.get('rotation', [0, 0, 0])
        scale = obj_config.get('scale', [1, 1, 1])
        position = obj_config.get('position', [0, 0, 0])
        
        # UE模式下position转换
        if prefs.target_engine == 'UE':
            position = [p * 0.01 for p in position]
        
        # 创建单位立方体
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=Vector(position)
        )
        box = bpy.context.active_object
        box.name = f"box_{name}"
        
        # 设置旋转
        box.rotation_euler = [r * 3.14159 / 180 for r in rotation]
        
        # 设置缩放
        box.scale = scale
        
        if prefs.show_debug_info:
            print(f"[批量预览] 创建参考Box: {box.name}")
            print(f"  位置: {position} {'(UE模式已转换)' if prefs.target_engine == 'UE' else ''}")
            print(f"  旋转: {rotation}°")
            print(f"  缩放: {scale}")
        
        return box
        
    except Exception as e:
        report_func({'WARNING'}, f"创建参考Box失败: {str(e)}")
        return None


def align_model_to_box_preview_helper(model_obj, ref_box, context, prefs, report_func):
    """将模型对齐到参考Box（预览模式：跳过旋转对齐，直接完美重合）"""
    try:
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        
        # 选择模型和Box
        model_obj.select_set(True)
        ref_box.select_set(True)
        
        # 设置模型为活动对象
        context.view_layer.objects.active = model_obj
        
        if prefs.show_debug_info:
            print(f"[批量预览] 开始对齐: {model_obj.name} → {ref_box.name}")
            print(f"[批量预览] 跳过对齐检查和旋转对齐，直接执行完美重合")
        
        # 直接执行完美重合对齐（跳过旋转对齐）
        execute_perfect_align_batch_helper(ref_box, model_obj, prefs)
        
        if prefs.show_debug_info:
            print(f"[批量预览] 对齐完成: {model_obj.name}")
        
    except Exception as e:
        report_func({'WARNING'}, f"对齐失败: {str(e)}")
        if prefs.show_debug_info:
            import traceback
            traceback.print_exc()


def get_world_bbox_size(obj):
    """计算物体在世界坐标系下的轴对齐包围盒（AABB）尺寸"""
    # 获取物体的8个包围盒顶点在世界坐标系下的位置
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # 计算世界坐标系下的最小和最大边界
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    
    # 返回世界坐标系下的AABB尺寸
    return Vector((max_x - min_x, max_y - min_y, max_z - min_z))


def execute_perfect_align_batch_helper(ref_obj, target_obj, prefs):
    """批量模式的完美重合对齐（考虑旋转，按世界坐标系AABB匹配）"""
    if prefs.show_debug_info:
        print(f"[完美重合对齐] 开始处理:")
        print(f"  参考BOX: {ref_obj.name}")
        print(f"  目标模型: {target_obj.name}")
    
    # 获取世界坐标系下的真实AABB尺寸
    ref_world_size = get_world_bbox_size(ref_obj)
    target_world_size_before = get_world_bbox_size(target_obj)
    
    if prefs.show_debug_info:
        print(f"  参考BOX世界AABB: X={ref_world_size.x:.3f} Y={ref_world_size.y:.3f} Z={ref_world_size.z:.3f}")
        print(f"  目标模型世界AABB: X={target_world_size_before.x:.3f} Y={target_world_size_before.y:.3f} Z={target_world_size_before.z:.3f}")
    
    # 获取目标模型的旋转矩阵
    loc, rot, scale = target_obj.matrix_world.decompose()
    rot_matrix = rot.to_matrix()
    
    # 将世界坐标系的缩放需求转换到局部坐标系
    # 计算每个局部轴在世界坐标系下占据的空间
    local_x_in_world = rot_matrix @ Vector((1, 0, 0))
    local_y_in_world = rot_matrix @ Vector((0, 1, 0))
    local_z_in_world = rot_matrix @ Vector((0, 0, 1))
    
    # 计算局部轴对世界AABB各轴的贡献（绝对值）
    # 局部X轴对世界AABB的贡献
    local_x_contrib = Vector((
        abs(local_x_in_world.x),
        abs(local_x_in_world.y),
        abs(local_x_in_world.z)
    ))
    # 局部Y轴对世界AABB的贡献
    local_y_contrib = Vector((
        abs(local_y_in_world.x),
        abs(local_y_in_world.y),
        abs(local_y_in_world.z)
    ))
    # 局部Z轴对世界AABB的贡献
    local_z_contrib = Vector((
        abs(local_z_in_world.x),
        abs(local_z_in_world.y),
        abs(local_z_in_world.z)
    ))
    
    # 获取当前模型的局部尺寸（未缩放的基础尺寸）
    current_scale = target_obj.scale.copy()
    if abs(current_scale.x) > 1e-6:
        base_dim_x = target_obj.dimensions.x / current_scale.x
    else:
        base_dim_x = 0.0
    
    if abs(current_scale.y) > 1e-6:
        base_dim_y = target_obj.dimensions.y / current_scale.y
    else:
        base_dim_y = 0.0
    
    if abs(current_scale.z) > 1e-6:
        base_dim_z = target_obj.dimensions.z / current_scale.z
    else:
        base_dim_z = 0.0
    
    if prefs.show_debug_info:
        print(f"  模型基础尺寸: X={base_dim_x:.3f} Y={base_dim_y:.3f} Z={base_dim_z:.3f}")
        print(f"  当前缩放: X={current_scale.x:.3f} Y={current_scale.y:.3f} Z={current_scale.z:.3f}")
    
    # 构建从世界AABB到局部缩放的转换矩阵
    # world_aabb = M * local_scale * base_dim
    # 其中 M 是旋转贡献矩阵
    import numpy as np
    M = np.array([
        [local_x_contrib.x * base_dim_x, local_y_contrib.x * base_dim_y, local_z_contrib.x * base_dim_z],
        [local_x_contrib.y * base_dim_x, local_y_contrib.y * base_dim_y, local_z_contrib.y * base_dim_z],
        [local_x_contrib.z * base_dim_x, local_y_contrib.z * base_dim_y, local_z_contrib.z * base_dim_z]
    ])
    
    target_world = np.array([ref_world_size.x, ref_world_size.y, ref_world_size.z])
    
    # 求解局部缩放: M * local_scale = target_world
    try:
        new_scale_array = np.linalg.lstsq(M, target_world, rcond=None)[0]
        new_scale = Vector((
            max(new_scale_array[0], 0.001),  # 避免负值或零
            max(new_scale_array[1], 0.001),
            max(new_scale_array[2], 0.001)
        ))
    except:
        # 如果求解失败，回退到简单匹配
        if prefs.show_debug_info:
            print(f"  警告: 矩阵求解失败，使用简化方法")
        
        # 简化方法：直接按世界AABB比例缩放
        if target_world_size_before.x > 1e-6:
            scale_x = ref_world_size.x / target_world_size_before.x
        else:
            scale_x = 1.0
        
        if target_world_size_before.y > 1e-6:
            scale_y = ref_world_size.y / target_world_size_before.y
        else:
            scale_y = 1.0
        
        if target_world_size_before.z > 1e-6:
            scale_z = ref_world_size.z / target_world_size_before.z
        else:
            scale_z = 1.0
        
        new_scale = Vector((
            current_scale.x * scale_x,
            current_scale.y * scale_y,
            current_scale.z * scale_z
        ))
    
    if prefs.show_debug_info:
        print(f"  计算新缩放: X={new_scale.x:.3f} Y={new_scale.y:.3f} Z={new_scale.z:.3f}")
    
    # 应用新缩放
    target_obj.scale = new_scale
    bpy.context.view_layer.update()
    
    # 验证结果
    target_world_size_after = get_world_bbox_size(target_obj)
    if prefs.show_debug_info:
        print(f"  应用缩放后世界AABB: X={target_world_size_after.x:.3f} Y={target_world_size_after.y:.3f} Z={target_world_size_after.z:.3f}")
        print(f"  dimensions: X={target_obj.dimensions.x:.3f} Y={target_obj.dimensions.y:.3f} Z={target_obj.dimensions.z:.3f}")
    
    # 对齐中心点
    ref_matrix = ref_obj.matrix_world
    ref_bbox_local_center = sum((Vector(corner) for corner in ref_obj.bound_box), Vector()) / 8
    ref_bbox_world_center = ref_matrix @ ref_bbox_local_center
    
    target_matrix = target_obj.matrix_world
    target_bbox_local_center = sum((Vector(corner) for corner in target_obj.bound_box), Vector()) / 8
    target_bbox_world_center = target_matrix @ target_bbox_local_center
    
    offset = ref_bbox_world_center - target_bbox_world_center
    
    if prefs.show_debug_info:
        print(f"  BOX包围盒中心: X={ref_bbox_world_center.x:.3f} Y={ref_bbox_world_center.y:.3f} Z={ref_bbox_world_center.z:.3f}")
        print(f"  模型包围盒中心: X={target_bbox_world_center.x:.3f} Y={target_bbox_world_center.y:.3f} Z={target_bbox_world_center.z:.3f}")
        print(f"  计算偏移量: X={offset.x:.3f} Y={offset.y:.3f} Z={offset.z:.3f}")
    
    target_obj.location += offset
    
    if prefs.show_debug_info:
        print(f"  新位置: X={target_obj.location.x:.3f} Y={target_obj.location.y:.3f} Z={target_obj.location.z:.3f}")
        print(f"[完美重合对齐] ✓ 完成")
    
    bpy.context.view_layer.update()


def sanitize_materials_for_export(obj, prefs):
    """清理材质节点以避免导出时的贴图问题（增强版：处理超大纹理、损坏贴图）"""
    try:
        cleaned_count = 0
        removed_count = 0
        downscaled_count = 0
        
        # 贴图尺寸安全阈值（像素）
        MAX_SAFE_TEXTURE_SIZE = 1024  # 超过4K认为是超大纹理
        MAX_TEXTURE_PIXELS = 1048576  # 1024x1024 = 1M像素
        
        # 遍历对象及其子对象的所有材质
        all_objects = [obj] + list(obj.children_recursive)
        
        for current_obj in all_objects:
            if not hasattr(current_obj, 'material_slots'):
                continue
                
            for mat_slot in current_obj.material_slots:
                if not mat_slot.material:
                    continue
                    
                mat = mat_slot.material
                if not mat.use_nodes:
                    continue
                
                # 检查材质节点树中的图像纹理节点
                nodes_to_remove = []
                
                for node in mat.node_tree.nodes:
                    if node.type != 'TEX_IMAGE' or not node.image:
                        continue
                    
                    image = node.image
                    should_remove = False
                    should_downscale = False
                    
                    # === 1. 检查文件路径问题 ===
                    if image.filepath and not os.path.exists(bpy.path.abspath(image.filepath)):
                        if prefs.show_debug_info:
                            print(f"[材质清理] 发现缺失贴图: {image.name} -> {image.filepath}")
                        # 尝试移除路径引用，避免导出器访问不存在的文件
                        should_remove = True
                        cleaned_count += 1
                    
                    # === 2. 检查贴图尺寸 ===
                    try:
                        width = image.size[0]
                        height = image.size[1]
                        total_pixels = width * height
                        
                        if width > MAX_SAFE_TEXTURE_SIZE or height > MAX_SAFE_TEXTURE_SIZE or total_pixels > MAX_TEXTURE_PIXELS:
                            if prefs.show_debug_info:
                                print(f"[材质清理] 发现超大纹理: {image.name} ({width}x{height}, {total_pixels/1000000:.1f}M像素)")
                            
                            # 尝试降级处理
                            should_downscale = True
                    except Exception as size_error:
                        if prefs.show_debug_info:
                            print(f"[材质清理] 无法读取贴图尺寸: {image.name} -> {str(size_error)}")
                        # 无法读取尺寸的贴图可能已损坏
                        should_remove = True
                        cleaned_count += 1
                    
                    # === 3. 检查贴图格式和通道数 ===
                    try:
                        # 检查是否为高风险格式（EXR、TIFF 16位、S3TC压缩等）
                        if image.file_format in ['OPEN_EXR', 'OPEN_EXR_MULTILAYER', 'HDR']:
                            if prefs.show_debug_info:
                                print(f"[材质清理] 发现高风险格式贴图: {image.name} (格式: {image.file_format})")
                            # HDR/EXR格式容易触发OpenImageIO崩溃
                            should_downscale = True
                        
                        # 检查是否为空贴图
                        if image.pixels is None or len(image.pixels) == 0:
                            if prefs.show_debug_info:
                                print(f"[材质清理] 发现空贴图: {image.name}")
                            should_remove = True
                            cleaned_count += 1
                    except Exception as format_error:
                        if prefs.show_debug_info:
                            print(f"[材质清理] 检查贴图格式失败: {image.name} -> {str(format_error)}")
                        should_remove = True
                        cleaned_count += 1
                    
                    # === 4. 执行降级处理 ===
                    if should_downscale and not should_remove:
                        try:
                            # 创建降级版本（缩小到安全尺寸）
                            original_width = image.size[0]
                            original_height = image.size[1]
                            
                            # 计算安全缩放比例
                            scale_factor = min(
                                MAX_SAFE_TEXTURE_SIZE / original_width,
                                MAX_SAFE_TEXTURE_SIZE / original_height,
                                1.0  # 不放大
                            )
                            
                            if scale_factor < 1.0:
                                new_width = int(original_width * scale_factor)
                                new_height = int(original_height * scale_factor)
                                
                                if prefs.show_debug_info:
                                    print(f"[材质清理] 降级纹理: {image.name} {original_width}x{original_height} -> {new_width}x{new_height}")
                                
                                # 创建新的降级图像
                                downscaled_image = image.copy()
                                downscaled_image.name = f"{image.name}_safe"
                                downscaled_image.scale(new_width, new_height)
                                
                                # 转换为PNG格式（最安全）
                                downscaled_image.file_format = 'PNG'
                                
                                # 打包到blend文件中
                                downscaled_image.pack()
                                
                                # 替换节点中的图像引用
                                node.image = downscaled_image
                                
                                downscaled_count += 1
                                
                                if prefs.show_debug_info:
                                    print(f"[材质清理] 已替换为降级版本: {downscaled_image.name}")
                            else:
                                # 尺寸在安全范围内，但格式不安全，尝试转换格式
                                if image.file_format != 'PNG':
                                    image.file_format = 'PNG'
                                    if prefs.show_debug_info:
                                        print(f"[材质清理] 转换格式为PNG: {image.name}")
                                
                                # 打包图像
                                if not image.packed_file:
                                    image.pack()
                                    if prefs.show_debug_info:
                                        print(f"[材质清理] 已打包贴图: {image.name}")
                        
                        except Exception as downscale_error:
                            if prefs.show_debug_info:
                                print(f"[材质清理] 降级失败，将移除节点: {image.name} -> {str(downscale_error)}")
                            should_remove = True
                    
                    # === 5. 移除有问题的节点 ===
                    if should_remove:
                        nodes_to_remove.append(node)
                        if prefs.show_debug_info:
                            print(f"[材质清理] 标记移除节点: {node.name} (贴图: {image.name})")
                    
                    # === 6. 安全打包（未降级且未移除的正常贴图） ===
                    elif not should_downscale:
                        if not image.packed_file and image.filepath:
                            try:
                                # 尝试打包图像以避免路径问题
                                image.pack()
                                if prefs.show_debug_info:
                                    print(f"[材质清理] 已打包贴图: {image.name}")
                            except Exception as pack_error:
                                if prefs.show_debug_info:
                                    print(f"[材质清理] 打包失败，将移除节点: {image.name} -> {str(pack_error)}")
                                nodes_to_remove.append(node)
                
                # 移除标记的节点
                for node in nodes_to_remove:
                    try:
                        mat.node_tree.nodes.remove(node)
                        removed_count += 1
                        if prefs.show_debug_info:
                            print(f"[材质清理] 已移除问题节点: {node.name}")
                    except Exception as remove_error:
                        if prefs.show_debug_info:
                            print(f"[材质清理] 移除节点失败: {node.name} -> {str(remove_error)}")
        
        # 输出统计信息
        if prefs.show_debug_info:
            print(f"[材质清理] 清理完成:")
            print(f"  - 发现问题贴图: {cleaned_count} 个")
            print(f"  - 降级纹理: {downscaled_count} 个")
            print(f"  - 移除节点: {removed_count} 个")
        
    except Exception as e:
        if prefs.show_debug_info:
            print(f"[材质清理] 清理过程出错: {str(e)}")
            import traceback
            traceback.print_exc()


def export_model_glb_helper(report_func, model_obj, obj_config, output_dir, prefs):
    """导出模型为GLB格式，贴图嵌入（优化参数避免OpenImageIO崩溃）"""
    try:
        model_name = obj_config.get('name', 'model')
        # 清理文件名，移除不合法字符
        import re
        safe_name = re.sub(r'[<>:\"/\\|?*]', '_', model_name)
        output_filename = f"{safe_name}.glb"
        output_path = os.path.join(output_dir, output_filename)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 准备导出GLB: {model_name} -> {output_path}")
            print(f"[批量对齐] 目标引擎检查: target_engine = '{prefs.target_engine}'")
        
        # ★ 导出前清理材质，避免贴图问题
        sanitize_materials_for_export(model_obj, prefs)
        
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        
        # 选择要导出的模型（包括子对象）
        model_obj.select_set(True)
        
        # 同时选择所有子对象
        for child in model_obj.children_recursive:
            child.select_set(True)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 选中对象: {model_obj.name} 及其 {len(model_obj.children_recursive)} 个子对象")
        
        # 导出为GLB格式（优化参数以避免OpenImageIO崩溃）
        try:
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                use_selection=True,
                export_texcoords=True,
                export_normals=True,
                export_materials='EXPORT',
                export_animations=False,
                # ★ 关键优化：强制转换贴图为PNG，避免OpenImageIO读取原始格式时崩溃
                export_image_format='PNG',
                export_keep_originals=False,
                # ★ 嵌入贴图到GLB
                export_texture_dir='',
                # ★ 压缩优化
                export_draco_mesh_compression_enable=False,  # 禁用Draco压缩，避免额外的兼容性问题
                # ★ 其他安全选项
                export_yup=True,
                check_existing=False,
            )
            
            if prefs.show_debug_info:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[批量对齐] GLB导出完成: {output_path} ({file_size_mb:.2f} MB)")
            
            return output_path
            
        except Exception as export_error:
            # 如果PNG导出失败，尝试使用JPEG作为备选方案
            if prefs.show_debug_info:
                print(f"[批量对齐] PNG导出失败，尝试JPEG格式...")
            
            try:
                bpy.ops.export_scene.gltf(
                    filepath=output_path,
                    export_format='GLB',
                    use_selection=True,
                    export_texcoords=True,
                    export_normals=True,
                    export_materials='EXPORT',
                    export_animations=False,
                    export_image_format='JPEG',  # 备选：JPEG
                    export_keep_originals=False,
                    export_texture_dir='',
                    export_draco_mesh_compression_enable=False,
                    export_yup=True,
                    check_existing=False,
                )
                
                if prefs.show_debug_info:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"[批量对齐] GLB导出完成（JPEG格式）: {output_path} ({file_size_mb:.2f} MB)")
                
                return output_path
                
            except Exception as jpeg_error:
                # 如果JPEG也失败，尝试不导出贴图
                if prefs.show_debug_info:
                    print(f"[批量对齐] JPEG导出也失败，尝试不导出材质...")
                
                try:
                    bpy.ops.export_scene.gltf(
                        filepath=output_path,
                        export_format='GLB',
                        use_selection=True,
                        export_texcoords=True,
                        export_normals=True,
                        export_materials='PLACEHOLDER',  # 仅导出占位符材质
                        export_animations=False,
                        export_draco_mesh_compression_enable=False,
                        export_yup=True,
                        check_existing=False,
                    )
                    
                    if prefs.show_debug_info:
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        print(f"[批量对齐] GLB导出完成（无材质）: {output_path} ({file_size_mb:.2f} MB)")
                        print(f"[批量对齐] ⚠ 警告：由于贴图问题，该模型仅包含几何体")
                    
                    return output_path
                    
                except Exception as final_error:
                    raise final_error
    
    except Exception as e:
        report_func({'WARNING'}, f"导出GLB失败: {str(e)}")
        if prefs.show_debug_info:
            import traceback
            traceback.print_exc()
        return None


def export_model_fbx_helper(report_func, model_obj, obj_config, output_dir, prefs):
    """导出模型为FBX格式，贴图嵌入"""
    try:
        model_name = obj_config.get('name', 'model')
        # 清理文件名，移除不合法字符
        import re
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        output_filename = f"{safe_name}.fbx"
        output_path = os.path.join(output_dir, output_filename)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 准备导出: {model_name} -> {output_path}")
        
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        
        # 选择要导出的模型（包括子对象）
        model_obj.select_set(True)
        
        # 同时选择所有子对象
        for child in model_obj.children_recursive:
            child.select_set(True)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 选中对象: {model_obj.name} 及其 {len(model_obj.children_recursive)} 个子对象")
        
        # 导出为FBX格式（UE专用轴向：Forward=-Y, Up=Z）
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            apply_scale_options='FBX_SCALE_ALL',
            axis_forward='-Y',  # 前向轴（Blender标准到UE）
            axis_up='Z',  # 上向轴（Blender标准到UE）
            bake_space_transform=True,
            object_types={'MESH', 'EMPTY'},
            use_mesh_modifiers=True,
            mesh_smooth_type='OFF',
            use_tspace=True,
            embed_textures=True,
            path_mode='COPY',
            batch_mode='OFF',
        )
        
        if prefs.show_debug_info:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[批量对齐] 导出完成: {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
        
    except Exception as e:
        report_func({'WARNING'}, f"导出FBX失败: {str(e)}")
        if prefs.show_debug_info:
            import traceback
            traceback.print_exc()
        return None


def export_model_obj_helper(report_func, model_obj, obj_config, output_dir, prefs):
    """导出模型为OBJ格式"""
    try:
        model_name = obj_config.get('name', 'model')
        # 清理文件名，移除不合法字符
        import re
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        output_filename = f"{safe_name}.obj"
        output_path = os.path.join(output_dir, output_filename)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 准备导出OBJ: {model_name} -> {output_path}")
        
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        
        # 选择要导出的模型（包括子对象）
        model_obj.select_set(True)
        
        # 同时选择所有子对象
        for child in model_obj.children_recursive:
            child.select_set(True)
        
        if prefs.show_debug_info:
            print(f"[批量对齐] 选中对象: {model_obj.name} 及其 {len(model_obj.children_recursive)} 个子对象")
        
        # 导出为OBJ格式
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=True,
            use_materials=True,
            use_triangles=False,
            use_normals=True,
            use_uvs=True,
            path_mode='COPY',
        )
        
        if prefs.show_debug_info:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[批量对齐] OBJ导出完成: {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
        
    except Exception as e:
        report_func({'WARNING'}, f"导出OBJ失败: {str(e)}")
        if prefs.show_debug_info:
            import traceback
            traceback.print_exc()
        return None


# ==================== 批量预览操作符 ====================

# 批量Box对齐预览操作符
class OBJECT_OT_batch_box_preview(Operator):
    """批量Box对齐预览：下载模型并应用position/rotation，创建BOX预览对齐效果，不导出"""
    bl_idname = "alignment.batch_box_preview"
    bl_label = "批量Box对齐预览"
    bl_options = {'REGISTER', 'UNDO'}
    
    json_config: StringProperty(
        name="JSON配置",
        description="批量对齐预览的JSON配置",
        default=""
    )
    
    def execute(self, context):
        import time
        start_time = time.time()  # 记录开始时间
        
        try:
            # 获取插件偏好设置
            prefs = context.preferences.addons[get_addon_name()].preferences
            
            # 解析JSON配置
            json_text = self.json_config if self.json_config else prefs.batch_json_config
            if not json_text:
                self.report({'ERROR'}, "请输入JSON配置")
                return {'CANCELLED'}
            
            try:
                config = json.loads(json_text)
            except json.JSONDecodeError as e:
                self.report({'ERROR'}, f"JSON解析失败: {str(e)}")
                return {'CANCELLED'}
            
            # 认证信息
            auth_token = None
            
            # 获取objects列表
            objects = config.get('objects', [])
            if not objects:
                self.report({'ERROR'}, "JSON配置中没有objects")
                return {'CANCELLED'}
            
            self.report({'INFO'}, f"开始批量预览处理 {len(objects)} 个对象...")
            
            success_count = 0
            failed_count = 0
            
            for idx, obj_config in enumerate(objects):
                try:
                    self.report({'INFO'}, f"处理第 {idx+1}/{len(objects)} 个对象: {obj_config.get('name', '未命名')}")
                    
                    # 下载并导入模型
                    model_obj, file_format = download_and_import_model_helper(obj_config, auth_token, prefs, self.report)
                    if not model_obj:
                        failed_count += 1
                        continue
                    
                    # 应用JSON中的rotation和position到模型
                    apply_transform_to_model_helper(model_obj, obj_config, prefs, self.report)
                    
                    # 创建带position的参考Box
                    ref_box = create_reference_box_with_position_helper(obj_config, prefs, self.report)
                    if not ref_box:
                        failed_count += 1
                        continue
                    
                    # 执行完美重合对齐
                    align_model_to_box_preview_helper(model_obj, ref_box, context, prefs, self.report)
                    
                    # 保留Box，不删除
                    if prefs.show_debug_info:
                        print(f"[批量预览] ✓ 保留BOX: {ref_box.name}")
                    
                    success_count += 1
                    
                except Exception as e:
                    self.report({'WARNING'}, f"处理对象 {idx+1} 失败: {str(e)}")
                    failed_count += 1
                    if prefs.show_debug_info:
                        import traceback
                        traceback.print_exc()
            
            # 计算并输出总耗时
            elapsed_time = time.time() - start_time
            time_str = print_processing_summary(prefs, "批量预览", success_count, failed_count, -1, elapsed_time)
            
            self.report({'INFO'}, f"批量预览完成: 成功 {success_count} 个, 失败 {failed_count} 个")
            self.report({'INFO'}, f"总耗时: {time_str}")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"批量预览失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


# 批量Box对齐预览+导出操作符
class OBJECT_OT_batch_box_preview_export(Operator):
    """批量Box对齐预览+导出：应用position/rotation，创建BOX，完美重合对齐，导出模型（position归零），保留BOX"""
    bl_idname = "alignment.batch_box_preview_export"
    bl_label = "批量Box对齐预览+导出"
    bl_options = {'REGISTER', 'UNDO'}
    
    json_config: StringProperty(
        name="JSON配置",
        description="批量对齐预览+导出的JSON配置",
        default=""
    )
    
    def execute(self, context):
        import time
        start_time = time.time()  # 记录开始时间
        
        try:
            # 显式启用 glTF 和 FBX 导入/导出插件
            self.ensure_importers_enabled()
            
            # 获取插件偏好设置
            prefs = context.preferences.addons[get_addon_name()].preferences
            
            # 解析JSON配置
            json_text = self.json_config if self.json_config else prefs.batch_json_config
            if not json_text:
                self.report({'ERROR'}, "请输入JSON配置")
                return {'CANCELLED'}
            
            try:
                config = json.loads(json_text)
            except json.JSONDecodeError as e:
                self.report({'ERROR'}, f"JSON解析失败: {str(e)}")
                return {'CANCELLED'}
            
            # 认证信息
            auth_token = None
            
            # 获取objects列表
            objects = config.get('objects', [])
            if not objects:
                self.report({'ERROR'}, "JSON配置中没有objects")
                return {'CANCELLED'}
            
            # 创建输出目录
            output_dir = prefs.batch_output_dir
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 创建输出目录: {output_dir}")
                except Exception as e:
                    self.report({'ERROR'}, f"创建输出目录失败: {str(e)}")
                    return {'CANCELLED'}
            
            self.report({'INFO'}, f"开始批量预览+导出处理 {len(objects)} 个对象...")
            if prefs.show_debug_info:
                print(f"[批量预览+导出] 处理模式: {prefs.processing_mode}")
            
            success_count = 0
            failed_count = 0
            processed_objects = []
            
            # 根据处理模式选择不同的执行路径
            if prefs.processing_mode == 'STREAM':
                # 流式处理：逐个处理并导出
                success_count, failed_count, processed_objects = self.execute_stream_mode(
                    context, objects, auth_token, output_dir, prefs
                )
            else:
                # 一次性处理：保持原有逻辑
                success_count, failed_count, processed_objects = self.execute_batch_mode(
                    context, objects, auth_token, output_dir, prefs
                )
            
            # 生成输出JSON
            output_config = config.copy()
            output_config['objects'] = processed_objects
            
            # 保存JSON到文件
            json_output_path = os.path.join(output_dir, "output_config.json")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_config, f, indent=2, ensure_ascii=False)
                
                prefs.batch_output_json = json.dumps(output_config, indent=2, ensure_ascii=False)
                
                self.report({'INFO'}, f"输出JSON已保存: {json_output_path}")
                
                if prefs.show_debug_info:
                    print(f"[批量预览+导出] 输出JSON:\n{json.dumps(output_config, indent=2, ensure_ascii=False)}")
                
            except Exception as e:
                self.report({'WARNING'}, f"保存JSON失败: {str(e)}")
            
            # 计算并输出总耗时
            elapsed_time = time.time() - start_time
            time_str = print_processing_summary(prefs, "批量预览+导出", success_count, failed_count, len(processed_objects), elapsed_time, prefs.processing_mode)
            
            self.report({'INFO'}, f"批量预览+导出完成: 成功 {success_count} 个, 失败 {failed_count} 个, 导出 {len(processed_objects)} 个")
            self.report({'INFO'}, f"总耗时: {time_str}")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"批量预览+导出失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def ensure_importers_enabled(self):
        """确保glTF和FBX导入/导出插件已启用"""
        try:
            # 启用 glTF 2.0 插件
            if not bpy.context.preferences.addons.get('io_scene_gltf2'):
                bpy.ops.preferences.addon_enable(module='io_scene_gltf2')
            
            # 启用 FBX 插件
            if not bpy.context.preferences.addons.get('io_scene_fbx'):
                bpy.ops.preferences.addon_enable(module='io_scene_fbx')
        except:
            pass  # 如果已经启用或无法启用，忽略错误
    
    def execute_stream_mode(self, context, objects, auth_token, output_dir, prefs):
        """流式处理模式：逐个处理并导出后删除模型"""
        success_count = 0
        failed_count = 0
        processed_objects = []
        
        for idx, obj_config in enumerate(objects):
            try:
                self.report({'INFO'}, f"处理第 {idx+1}/{len(objects)} 个对象: {obj_config.get('name', '未命名')}")
                
                # 下载并导入模型
                model_obj, file_format = download_and_import_model_helper(obj_config, auth_token, prefs, self.report)
                if not model_obj:
                    failed_count += 1
                    continue
                
                # 应用JSON中的rotation和position到模型
                apply_transform_to_model_helper(model_obj, obj_config, prefs, self.report)
                
                # 创建带position的参考Box
                ref_box = create_reference_box_with_position_helper(obj_config, prefs, self.report)
                if not ref_box:
                    failed_count += 1
                    continue
                
                # 执行完美重合对齐
                align_model_to_box_preview_helper(model_obj, ref_box, context, prefs, self.report)
                
                # 保留Box
                if prefs.show_debug_info:
                    print(f"[批量预览+导出] ✓ 保留BOX: {ref_box.name}")
                
                # 使用用户设置的导出格式
                export_format = '.glb' if prefs.export_format == 'GLB' else '.fbx'
                
                # 立即导出模型
                exported_path = self.export_single_model(model_obj, obj_config, output_dir, export_format, prefs, context)
                
                if exported_path:
                    # 创建新的对象配置，rotation和position保持原始值
                    new_obj_config = obj_config.copy()
                    
                    # 打印输出JSON信息
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 输出JSON:")
                        print(f"  position: {new_obj_config.get('position')} (保持原始)")
                        print(f"  rotation: {new_obj_config.get('rotation')} (保持原始，模型已归零导出)")
                        print(f"  scale: {new_obj_config.get('scale')} (保持原始)")
                    
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
                    
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 导出成功: {exported_path}")
                
                # 删除模型对象以释放资源（保留BOX用于调试）
                try:
                    bpy.data.objects.remove(model_obj, do_unlink=True)
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 已删除模型: {obj_config.get('name', '未命名')}")
                except Exception as e:
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 删除模型失败: {str(e)}")
                
            except Exception as e:
                self.report({'WARNING'}, f"处理对象 {idx+1} 失败: {str(e)}")
                failed_count += 1
                if prefs.show_debug_info:
                    import traceback
                    traceback.print_exc()
        
        return success_count, failed_count, processed_objects
    
    def execute_batch_mode(self, context, objects, auth_token, output_dir, prefs):
        """一次性处理模式：保持原有逻辑"""
        success_count = 0
        failed_count = 0
        processed_objects = []
        model_box_pairs = []  # 存储(模型, BOX, 配置, 导出格式)
        
        for idx, obj_config in enumerate(objects):
            try:
                self.report({'INFO'}, f"处理第 {idx+1}/{len(objects)} 个对象: {obj_config.get('name', '未命名')}")
                
                # 下载并导入模型
                model_obj, file_format = download_and_import_model_helper(obj_config, auth_token, prefs, self.report)
                if not model_obj:
                    failed_count += 1
                    continue
                
                # 应用JSON中的rotation和position到模型
                apply_transform_to_model_helper(model_obj, obj_config, prefs, self.report)
                
                # 创建带position的参考Box
                ref_box = create_reference_box_with_position_helper(obj_config, prefs, self.report)
                if not ref_box:
                    failed_count += 1
                    continue
                
                # 执行完美重合对齐
                align_model_to_box_preview_helper(model_obj, ref_box, context, prefs, self.report)
                
                # 保留Box，不删除
                if prefs.show_debug_info:
                    print(f"[批量预览+导出] ✓ 保留BOX: {ref_box.name}")
                
                # 使用用户设置的导出格式
                export_format = '.glb' if prefs.export_format == 'GLB' else '.fbx'
                
                # 存储模型、BOX、配置和导出格式
                model_box_pairs.append((model_obj, ref_box, obj_config, export_format))
                success_count += 1
                
            except Exception as e:
                self.report({'WARNING'}, f"处理对象 {idx+1} 失败: {str(e)}")
                failed_count += 1
                if prefs.show_debug_info:
                    import traceback
                    traceback.print_exc()
        
        # 导出所有成功处理的模型
        format_name = 'GLB' if prefs.export_format == 'GLB' else 'FBX'
        self.report({'INFO'}, f"开始导出 {len(model_box_pairs)} 个模型（格式：{format_name}）...")
        
        for model_obj, ref_box, obj_config, file_format in model_box_pairs:
            try:
                # 保存模型当前变换（和BOX一样的位置、旋转和缩放）
                original_position = model_obj.location.copy()
                original_scale = model_obj.scale.copy()
                
                # 保存旋转（根据模式）
                original_rotation_quat = None
                original_rotation_axis_angle = None
                original_rotation = None
                
                if model_obj.rotation_mode == 'QUATERNION':
                    original_rotation_quat = model_obj.rotation_quaternion.copy()
                elif 'AXIS_ANGLE' in model_obj.rotation_mode:
                    original_rotation_axis_angle = model_obj.rotation_axis_angle[:]
                else:
                    original_rotation = model_obj.rotation_euler.copy()
                
                # 关键：导出前将模型position和rotation归零
                if prefs.show_debug_info:
                    print(f"[批量预览+导出] 导出前归零position和rotation: {model_obj.name}")
                    print(f"  当前位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                    
                    # 根据旋转模式显示当前旋转
                    if model_obj.rotation_mode == 'QUATERNION':
                        euler = model_obj.rotation_quaternion.to_euler()
                        print(f"  当前旋转(从四元数): X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
                    else:
                        print(f"  当前旋转: X={model_obj.rotation_euler.x*180/3.14159:.1f}° Y={model_obj.rotation_euler.y*180/3.14159:.1f}° Z={model_obj.rotation_euler.z*180/3.14159:.1f}°")
                
                model_obj.location = Vector((0, 0, 0))
                
                # 根据旋转模式正确归零旋转
                if model_obj.rotation_mode == 'QUATERNION':
                    model_obj.rotation_quaternion = (1, 0, 0, 0)
                elif 'AXIS_ANGLE' in model_obj.rotation_mode:
                    model_obj.rotation_axis_angle = [0, 0, 0, 1]  # angle=0, axis=[0,0,1]
                else:
                    model_obj.rotation_euler = (0, 0, 0)
                
                bpy.context.view_layer.update()
                
                if prefs.show_debug_info:
                    print(f"  归零后位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                    print(f"  归零后旋转: X=0.0° Y=0.0° Z=0.0°")
                
                # 应用坐标系转换（针对目标引擎）
                self.apply_coordinate_transform(
                    model_obj,
                    prefs.target_engine,
                    prefs,
                    file_format
                )
                
                # 导出模型
                exported_path = self.export_model(model_obj, obj_config, output_dir, file_format, prefs)
                
                # 导出后恢复模型位置、旋转和缩放（用于Blender预览对比）
                model_obj.location = original_position
                model_obj.scale = original_scale
                
                # 恢复旋转（根据模式）
                if model_obj.rotation_mode == 'QUATERNION' and original_rotation_quat:
                    model_obj.rotation_quaternion = original_rotation_quat
                elif 'AXIS_ANGLE' in model_obj.rotation_mode and original_rotation_axis_angle:
                    model_obj.rotation_axis_angle = original_rotation_axis_angle
                elif original_rotation:
                    model_obj.rotation_euler = original_rotation
                
                bpy.context.view_layer.update()
                
                if prefs.show_debug_info:
                    print(f"[批量预览+导出] 导出后恢复变换: {model_obj.name}")
                    print(f"  恢复位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                    
                    # 根据旋转模式显示恢复后的旋转
                    if model_obj.rotation_mode == 'QUATERNION':
                        euler = model_obj.rotation_quaternion.to_euler()
                        print(f"  恢复旋转(从四元数): X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
                    else:
                        print(f"  恢复旋转: X={model_obj.rotation_euler.x*180/3.14159:.1f}° Y={model_obj.rotation_euler.y*180/3.14159:.1f}° Z={model_obj.rotation_euler.z*180/3.14159:.1f}°")
                    
                    print(f"  恢复缩放: X={model_obj.scale.x:.3f} Y={model_obj.scale.y:.3f} Z={model_obj.scale.z:.3f}")
                
                if exported_path:
                    # 创建新的对象配置，rotation和position保持原始值
                    new_obj_config = obj_config.copy()
                    # rotation保持原始值，不修改（模型已归零导出，在引擎中通过JSON的rotation还原）
                    # position保持原始值，不修改
                    
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 输出JSON:")
                        print(f"  position: {new_obj_config.get('position')} (保持原始)")
                        print(f"  rotation: {new_obj_config.get('rotation')} (保持原始，模型已归零导出)")
                        print(f"  scale: {new_obj_config.get('scale')} (保持原始)")
                    
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
                    
                    if prefs.show_debug_info:
                        print(f"[批量预览+导出] 导出成功: {exported_path}")
                
            except Exception as e:
                self.report({'WARNING'}, f"导出模型 {obj_config.get('name', '未命名')} 失败: {str(e)}")
                if prefs.show_debug_info:
                    import traceback
                    traceback.print_exc()
        
        return success_count, failed_count, processed_objects
    
    def export_single_model(self, model_obj, obj_config, output_dir, file_format, prefs, context):
        """导出单个模型（归零、导出、恢复）- 用于流式处理"""
        try:
            # 保存模型当前变换（和BOX一样的位置、旋转和缩放）
            original_position = model_obj.location.copy()
            original_scale = model_obj.scale.copy()
            
            # 保存旋转（根据模式）
            original_rotation_quat = None
            original_rotation_axis_angle = None
            original_rotation = None
            
            if model_obj.rotation_mode == 'QUATERNION':
                original_rotation_quat = model_obj.rotation_quaternion.copy()
            elif 'AXIS_ANGLE' in model_obj.rotation_mode:
                original_rotation_axis_angle = model_obj.rotation_axis_angle[:]
            else:
                original_rotation = model_obj.rotation_euler.copy()
            
            # 关键：导出前将模型position和rotation归零
            if prefs.show_debug_info:
                print(f"[批量预览+导出] 导出前归零position和rotation: {model_obj.name}")
                print(f"  当前位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                
                # 根据旋转模式显示当前旋转
                if model_obj.rotation_mode == 'QUATERNION':
                    euler = model_obj.rotation_quaternion.to_euler()
                    print(f"  当前旋转(从四元数): X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
                else:
                    print(f"  当前旋转: X={model_obj.rotation_euler.x*180/3.14159:.1f}° Y={model_obj.rotation_euler.y*180/3.14159:.1f}° Z={model_obj.rotation_euler.z*180/3.14159:.1f}°")
            
            model_obj.location = Vector((0, 0, 0))
            
            # 根据旋转模式正确归零旋转
            if model_obj.rotation_mode == 'QUATERNION':
                model_obj.rotation_quaternion = (1, 0, 0, 0)
            elif 'AXIS_ANGLE' in model_obj.rotation_mode:
                model_obj.rotation_axis_angle = [0, 0, 0, 1]  # angle=0, axis=[0,0,1]
            else:
                model_obj.rotation_euler = (0, 0, 0)
            
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                print(f"  归零后位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                print(f"  归零后旋转: X=0.0° Y=0.0° Z=0.0°")
            
            # 应用坐标系转换（针对目标引擎）
            self.apply_coordinate_transform(
                model_obj,
                prefs.target_engine,
                prefs,
                file_format
            )
            
            # 导出模型
            exported_path = self.export_model(model_obj, obj_config, output_dir, file_format, prefs)
            
            # 导出后恢复模型位置、旋转和缩放（用于继续在Blender中查看）
            model_obj.location = original_position
            model_obj.scale = original_scale
            
            # 恢复旋转（根据模式）
            if model_obj.rotation_mode == 'QUATERNION' and original_rotation_quat:
                model_obj.rotation_quaternion = original_rotation_quat
            elif 'AXIS_ANGLE' in model_obj.rotation_mode and original_rotation_axis_angle:
                model_obj.rotation_axis_angle = original_rotation_axis_angle
            elif original_rotation:
                model_obj.rotation_euler = original_rotation
            
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                print(f"[批量预览+导出] 导出后恢复变换: {model_obj.name}")
                print(f"  恢复位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
                
                # 根据旋转模式显示恢复后的旋转
                if model_obj.rotation_mode == 'QUATERNION':
                    euler = model_obj.rotation_quaternion.to_euler()
                    print(f"  恢复旋转(从四元数): X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
                else:
                    print(f"  恢复旋转: X={model_obj.rotation_euler.x*180/3.14159:.1f}° Y={model_obj.rotation_euler.y*180/3.14159:.1f}° Z={model_obj.rotation_euler.z*180/3.14159:.1f}°")
                
                print(f"  恢复缩放: X={model_obj.scale.x:.3f} Y={model_obj.scale.y:.3f} Z={model_obj.scale.z:.3f}")
            
            return exported_path
            
        except Exception as e:
            self.report({'WARNING'}, f"导出模型失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
            return None
    
    def apply_coordinate_transform(self, model_obj, target_engine, prefs, export_format=None):
        """根据目标引擎应用坐标系转换"""
        try:
            if target_engine == 'NONE':
                # 不做转换
                if prefs.show_debug_info:
                    print(f"[坐标转换] 无转换模式")
                return
            
            elif target_engine == 'UE':
                # Unreal Engine转换
                # 如果导出格式是GLB/GLTF，则不在此处缩放，交给导出阶段处理
                if export_format and export_format.lower() in ['.glb', '.gltf']:
                    if prefs.show_debug_info:
                        print(f"[坐标转换] UE模式 + GLB导出：跳过场景内缩放，导出阶段再处理")
                    return
                
                # 只需要缩放模型几何体以适配UE的单位系统
                # UE使用厘米，Blender使用米，所以需要放大100倍
                UE_SCALE_FACTOR = 100.0
                
                if prefs.show_debug_info:
                    print(f"[坐标转换] UE模式：缩放系数 {UE_SCALE_FACTOR}")
                    print(f"[坐标转换] 转换前: scale={model_obj.scale}")
                
                # 应用缩放系数到scale（模型几何体放大100倍）
                current_scale = model_obj.scale.copy()
                model_obj.scale = Vector((
                    current_scale.x * UE_SCALE_FACTOR,
                    current_scale.y * UE_SCALE_FACTOR,
                    current_scale.z * UE_SCALE_FACTOR
                ))
                
                if prefs.show_debug_info:
                    print(f"[坐标转换] 转换后: scale={model_obj.scale}")

            elif target_engine == 'UNITY':
                # Unity转换（左手系，Y-up）
                if prefs.show_debug_info:
                    print(f"[坐标转换] Unity模式")
                
                # Unity使用米作为单位，但需要转换坐标轴
                # Blender: X-right, Y-forward, Z-up
                # Unity: X-right, Y-up, Z-forward
                # 这个转换在FBX导出时通过axis_forward和axis_up处理
                pass
            
            elif target_engine == 'BLENDER':
                # 保持Blender原生坐标系
                if prefs.show_debug_info:
                    print(f"[坐标转换] Blender原生坐标系")
                pass
            
            bpy.context.view_layer.update()
            
        except Exception as e:
            self.report({'WARNING'}, f"坐标系转换失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
    
    def export_model(self, model_obj, obj_config, output_dir, file_format, prefs):
        """导出模型（调用辅助函数）"""
        if file_format in ['.glb', '.gltf']:
            return export_model_glb_helper(self.report, model_obj, obj_config, output_dir, prefs)
        elif file_format == '.obj':
            return export_model_obj_helper(self.report, model_obj, obj_config, output_dir, prefs)
        else:
            # 默认导出为FBX
            return export_model_fbx_helper(self.report, model_obj, obj_config, output_dir, prefs)


# 定义类列表用于注册
classes = (
    OBJECT_OT_batch_box_preview,
    OBJECT_OT_batch_box_preview_export,
)
