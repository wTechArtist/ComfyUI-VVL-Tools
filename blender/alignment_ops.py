import bpy
import bmesh
import os
import json

from math import radians
from mathutils import Vector, Matrix

import tempfile                 # 额外导入
import mathutils                 # 额外导入
from bpy.types import Operator                 # 额外导入
from bpy.props import StringProperty                 # 额外导入

# 获取插件包名
def get_addon_name():
    return __name__.partition('.')[0]

# 数值稳定性阈值
_EPS = 1e-8

# 对齐检查的默认容差（角度，弧度）
_ALIGNMENT_TOLERANCE = 0.017453292  # 约1度

def _build_orthonormal_frame(x_dir: Vector, y_dir: Vector, z_dir: Vector):
    """由可能不正交的三个方向向量构造正交、单位、右手坐标系(x, y, z)。
    优先使用传入的x、y，使用Gram-Schmidt使之正交，然后用叉乘得到z并回正交化y，确保右手系。
    """
    # 复制并归一化，避免原向量被修改
    x = Vector(x_dir)
    y = Vector(y_dir)
    z = Vector(z_dir)

    # 如果任何向量长度过小，提供回退
    def _fallback_perp(v: Vector) -> Vector:
        # 选一个与v不共线的向量
        axis = Vector((1.0, 0.0, 0.0)) if abs(v.x) < 0.9 else Vector((0.0, 1.0, 0.0))
        perp = v.cross(axis)
        if perp.length < _EPS:
            perp = v.cross(Vector((0.0, 0.0, 1.0)))
        return perp.normalized() if perp.length > _EPS else Vector((0.0, 1.0, 0.0))

    if x.length < _EPS:
        x = Vector((1.0, 0.0, 0.0))
    else:
        x.normalize()

    # 令y去除在x方向的分量
    if y.length < _EPS:
        y = _fallback_perp(x)
    else:
        y = (y - x.dot(y) * x)
        if y.length < _EPS:
            y = _fallback_perp(x)
        else:
            y.normalize()

    # 通过叉乘得到z（保证正交和右手）
    z = x.cross(y)
    if z.length < _EPS:
        # 如果退化，再次构造一个垂直于x的y
        y = _fallback_perp(x)
        z = x.cross(y)
    z.normalize()

    # 再次用z与x确定y，减少数值误差并确保严格正交
    y = z.cross(x)
    if y.length < _EPS:
        y = _fallback_perp(x)
    else:
        y.normalize()

    return x, y, z

def check_models_alignment(ref_obj, target_obj, tolerance=_ALIGNMENT_TOLERANCE):
    """检查两个模型是否已经平行对齐
    
    Args:
        ref_obj: 参考模型
        target_obj: 目标模型
        tolerance: 角度容差（弧度）
    
    Returns:
        tuple: (is_aligned, alignment_info)
            is_aligned: 是否已对齐
            alignment_info: 对齐信息字典
    """
    try:
        # 获取两个模型的包围盒方向
        ref_dirs = get_bbox_directions_static(ref_obj)
        target_dirs = get_bbox_directions_static(target_obj)
        
        # 获取尺寸信息
        ref_dims = ref_dirs['dimensions']
        target_dims = target_dirs['dimensions']
        
        # 计算尺寸比较
        ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
        target_sizes = [target_dims.x, target_dims.y, target_dims.z]
        
        # 对尺寸进行排序，得到从小到大的轴索引
        ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
        target_sorted_indices = sorted(range(3), key=lambda i: target_sizes[i])
        
        # 检查轴的对应关系
        axis_names = ['X', 'Y', 'Z']
        ref_axes = [ref_dirs['x'], ref_dirs['y'], ref_dirs['z']]
        target_axes = [target_dirs['x'], target_dirs['y'], target_dirs['z']]
        
        # 计算对应轴之间的角度差异
        max_angle_diff = 0.0
        alignment_details = []
        
        for rank in range(3):  # 按尺寸排序检查对应轴
            ref_axis_idx = ref_sorted_indices[rank]
            target_axis_idx = target_sorted_indices[rank]
            
            ref_axis = ref_axes[ref_axis_idx]
            target_axis = target_axes[target_axis_idx]
            
            # 计算两个轴之间的角度（考虑方向可能相反）
            dot_product = ref_axis.dot(target_axis)
            # 限制dot_product在[-1, 1]范围内，避免数值误差
            dot_product = max(-1.0, min(1.0, dot_product))
            
            angle1 = abs(mathutils.Vector.angle(ref_axis, target_axis))
            angle2 = abs(mathutils.Vector.angle(ref_axis, -target_axis))
            
            # 选择较小的角度（考虑轴可能反向）
            angle = min(angle1, angle2)
            max_angle_diff = max(max_angle_diff, angle)
            
            alignment_details.append({
                'ref_axis': axis_names[ref_axis_idx],
                'target_axis': axis_names[target_axis_idx],
                'angle_diff': angle,
                'angle_deg': angle * 180 / 3.14159,
                'ref_size': ref_sizes[ref_axis_idx],
                'target_size': target_sizes[target_axis_idx]
            })
        
        is_aligned = max_angle_diff <= tolerance
        
        alignment_info = {
            'is_aligned': is_aligned,
            'max_angle_diff': max_angle_diff,
            'max_angle_diff_deg': max_angle_diff * 180 / 3.14159,
            'tolerance_deg': tolerance * 180 / 3.14159,
            'details': alignment_details
        }
        
        return is_aligned, alignment_info
        
    except Exception as e:
        return False, {'error': str(e)}

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

# 一键智能对齐操作符（自动判断并执行旋转对齐或完美重合对齐）
class OBJECT_OT_one_click_align(Operator):
    """一键完美对齐：自动完成旋转对齐+尺寸拉伸+中心对齐，使包围盒100%完美重合"""
    bl_idname = "alignment.one_click_align"
    bl_label = "一键完美对齐"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # 以当前活动对象作为目标，其它选中网格中的一个作为参考
        active_obj = context.view_layer.objects.active
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not active_obj or active_obj.type != 'MESH':
            self.report({'ERROR'}, "请将目标网格物体设为活动对象（高亮）")
            return {'CANCELLED'}

        if len(selected_objects) < 2:
            self.report({'ERROR'}, "请选中至少两个网格物体")
            return {'CANCELLED'}
        
        target_obj = active_obj
        ref_obj = next((o for o in selected_objects if o != target_obj), None)
        if ref_obj is None:
            self.report({'ERROR'}, "未找到参考模型，请至少再选中一个网格物体")
            return {'CANCELLED'}
        
        try:
            # 获取插件偏好设置
            prefs = context.preferences.addons[get_addon_name()].preferences
            
            # 检查是否已经对齐
            tolerance_deg = prefs.alignment_tolerance
            tolerance_rad = tolerance_deg * 3.14159 / 180.0
            is_aligned, alignment_info = check_models_alignment(ref_obj, target_obj, tolerance_rad)
            
            if is_aligned:
                # 已经旋转对齐，只需执行完美重合对齐
                if prefs.show_debug_info:
                    print(f"\n[一键对齐] 检测到已旋转对齐，执行完美重合对齐")
                
                return self.execute_perfect_align(context, ref_obj, target_obj, prefs)
            else:
                # 未对齐，需要执行完整流程：旋转对齐 + 完美重合对齐
                if prefs.show_debug_info:
                    print(f"\n[一键对齐] 开始完整对齐流程")
                    print(f"[一键对齐] 步骤1: 旋转对齐（角度差异: {alignment_info['max_angle_diff_deg']:.2f}°）")
                
                # 步骤1: 执行旋转对齐
                result = self.execute_rotation_align(context, ref_obj, target_obj, prefs)
                if result != {'FINISHED'}:
                    return result
                
                # 步骤2: 执行完美重合对齐
                if prefs.show_debug_info:
                    print(f"[一键对齐] 步骤2: 完美重合对齐")
                
                return self.execute_perfect_align(context, ref_obj, target_obj, prefs)
                
        except Exception as e:
            self.report({'ERROR'}, f"一键对齐失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def execute_rotation_align(self, context, ref_obj, target_obj, prefs):
        """执行旋转对齐"""
        # 备份变换信息
        if prefs.auto_backup:
            target_obj["_alignment_backup_matrix"] = [list(row) for row in target_obj.matrix_world]
            target_obj["_alignment_backup_location"] = list(target_obj.location)
            target_obj["_alignment_backup_rotation"] = list(target_obj.rotation_euler)
            target_obj["_alignment_backup_scale"] = list(target_obj.scale)
        
        # 使用包围盒方法和最薄轴算法
        ref_directions = get_bbox_directions_static(ref_obj)
        target_directions = get_bbox_directions_static(target_obj)
        
        # 获取参考物体的旋转
        ref_loc, ref_rot, ref_scale = ref_obj.matrix_world.decompose()
        
        # 获取尺寸信息用于匹配
        ref_dims = ref_directions['dimensions']
        tgt_dims = target_directions['dimensions']
        
        # 局部坐标系下的尺寸
        ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
        tgt_sizes = [tgt_dims.x, tgt_dims.y, tgt_dims.z]
        
        # 对尺寸进行排序，得到从小到大的轴索引
        ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
        tgt_sorted_indices = sorted(range(3), key=lambda i: tgt_sizes[i])
        
        # 创建轴映射
        axis_mapping = [None, None, None]
        for rank in range(3):
            tgt_axis = tgt_sorted_indices[rank]
            ref_axis = ref_sorted_indices[rank]
            axis_mapping[tgt_axis] = ref_axis
        
        # 创建置换矩阵
        from mathutils import Matrix
        perm_matrix = Matrix((
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)
        ))
        
        for i in range(3):
            perm_matrix[axis_mapping[i]][i] = 1.0
        
        # 检查行列式
        det = perm_matrix.determinant()
        if det < 0:
            last_mapping = axis_mapping[2]
            perm_matrix[last_mapping][2] *= -1
        
        # 最终旋转
        final_rotation_matrix = ref_rot.to_matrix() @ perm_matrix
        
        # 应用最终旋转
        location, old_rotation, scale = target_obj.matrix_world.decompose()
        new_rotation = final_rotation_matrix.to_quaternion()
        target_obj.matrix_world = Matrix.LocRotScale(location, new_rotation, scale)
        
        # 更新场景
        bpy.context.view_layer.update()
        
        if prefs.show_debug_info:
            print(f"[一键完美对齐 - 步骤1完成: 旋转对齐]")
            print(f"参考模型: {ref_obj.name}")
            print(f"目标模型: {target_obj.name}")
        
        # 不显示中间提示，继续执行完美重合对齐
        return {'FINISHED'}
    
    def execute_perfect_align(self, context, ref_obj, target_obj, prefs):
        """执行完美重合对齐"""
        # 备份变换信息
        if prefs.auto_backup:
            target_obj["_stretch_backup_matrix"] = [list(row) for row in target_obj.matrix_world]
            target_obj["_stretch_backup_dimensions"] = list(target_obj.dimensions)
            target_obj["_stretch_backup_scale"] = list(target_obj.scale)
        
        # 获取两个模型的尺寸信息
        ref_dirs = get_bbox_directions_static(ref_obj)
        target_dirs = get_bbox_directions_static(target_obj)
        
        ref_dims = ref_dirs['dimensions']
        target_dims = target_dirs['dimensions']
        
        # 计算尺寸排序和对应关系
        ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
        target_sizes = [target_dims.x, target_dims.y, target_dims.z]
        
        ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
        target_sorted_indices = sorted(range(3), key=lambda i: target_sizes[i])
        
        # 计算缩放因子
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
        
        # 应用缩放
        from mathutils import Vector
        current_scale = target_obj.scale.copy()
        new_scale = Vector((
            current_scale.x * scale_factors[0],
            current_scale.y * scale_factors[1],
            current_scale.z * scale_factors[2]
        ))
        
        target_obj.scale = new_scale
        bpy.context.view_layer.update()
        
        # 对齐包围盒中心点
        ref_matrix = ref_obj.matrix_world
        ref_bbox_local_center = sum((Vector(corner) for corner in ref_obj.bound_box), Vector()) / 8
        ref_bbox_world_center = ref_matrix @ ref_bbox_local_center
        
        target_matrix = target_obj.matrix_world
        target_bbox_local_center = sum((Vector(corner) for corner in target_obj.bound_box), Vector()) / 8
        target_bbox_world_center = target_matrix @ target_bbox_local_center
        
        offset = ref_bbox_world_center - target_bbox_world_center
        target_obj.location += offset
        
        bpy.context.view_layer.update()
        
        if prefs.show_debug_info:
            print(f"[一键完美对齐 - 全部完成]")
            print(f"✓ 旋转对齐完成")
            print(f"✓ 尺寸拉伸完成")
            print(f"✓ 中心对齐完成（偏移: X={offset.x:.3f} Y={offset.y:.3f} Z={offset.z:.3f}）")
            print(f"包围盒已100%完美重合")
        
        self.report({'INFO'}, f"一键完美对齐完成: {target_obj.name} ⟹ {ref_obj.name} 包围盒已100%重合")
        return {'FINISHED'}


# 快速模型对齐操作符（使用选中的物体）
class OBJECT_OT_quick_align_models(Operator):
    """快速对齐选中的两个模型（第一个作为参考，第二个作为目标）"""
    bl_idname = "alignment.quick_align_models"
    bl_label = "快速模型对齐"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # 以当前活动对象作为目标，其它选中网格中的一个作为参考
        active_obj = context.view_layer.objects.active
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not active_obj or active_obj.type != 'MESH':
            self.report({'ERROR'}, "请将目标网格物体设为活动对象（高亮）")
            return {'CANCELLED'}

        if len(selected_objects) < 2:
            self.report({'ERROR'}, "请选中至少两个网格物体")
            return {'CANCELLED'}
        
        target_obj = active_obj
        # 选择一个非活动对象作为参考
        ref_obj = next((o for o in selected_objects if o != target_obj), None)
        if ref_obj is None:
            self.report({'ERROR'}, "未找到参考模型，请至少再选中一个网格物体")
            return {'CANCELLED'}
        
        try:
            # 获取插件偏好设置
            prefs = context.preferences.addons[get_addon_name()].preferences
            
            # 检查是否已经对齐
            # 获取用户设置的容差（度转弧度）
            tolerance_deg = prefs.alignment_tolerance
            tolerance_rad = tolerance_deg * 3.14159 / 180.0
            is_aligned, alignment_info = check_models_alignment(ref_obj, target_obj, tolerance_rad)
            
            if is_aligned:
                if prefs.show_debug_info:
                    print(f"\n[对齐检查] 模型已经对齐，无需计算")
                    print(f"[对齐检查] 参考模型: {ref_obj.name}")
                    print(f"[对齐检查] 目标模型: {target_obj.name}")
                    print(f"[对齐检查] 最大角度差异: {alignment_info['max_angle_diff_deg']:.2f}°")
                    print(f"[对齐检查] 容差范围: {alignment_info['tolerance_deg']:.2f}°")
                    
                    for detail in alignment_info['details']:
                        print(f"  轴对应: {detail['ref_axis']} ↔ {detail['target_axis']}, 角度差: {detail['angle_deg']:.2f}°")
                
                self.report({'INFO'}, f"模型已经对齐（角度差异: {alignment_info['max_angle_diff_deg']:.1f}°）")
                return {'FINISHED'}
            
            if prefs.show_debug_info:
                print(f"\n[对齐检查] 模型未对齐，开始计算对齐")
                print(f"[对齐检查] 最大角度差异: {alignment_info['max_angle_diff_deg']:.2f}°")
                print(f"[对齐检查] 容差范围: {alignment_info['tolerance_deg']:.2f}°")
            
            # 备份变换信息
            if prefs.auto_backup:
                target_obj["_alignment_backup_matrix"] = [list(row) for row in target_obj.matrix_world]
                target_obj["_alignment_backup_location"] = list(target_obj.location)
                target_obj["_alignment_backup_rotation"] = list(target_obj.rotation_euler)
                target_obj["_alignment_backup_scale"] = list(target_obj.scale)
            
            # 快速对齐固定使用包围盒方法和最薄轴算法
            ref_directions = self.get_bbox_directions(ref_obj)
            target_directions = self.get_bbox_directions(target_obj)
            
            # 计算目标物体应该具有的最终旋转（最薄轴对齐）
            final_rotation = self.calculate_target_rotation(ref_obj, target_obj, ref_directions, target_directions)
            
            # 应用最终旋转，保持原有的位置和缩放
            self.apply_final_rotation(target_obj, final_rotation)
            
            # 更新场景
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                print(f"\n[快速对齐] 使用方法: 最薄轴对齐（固定算法）")
                print(f"[快速对齐] 参考模型: {ref_obj.name}")
                print(f"  - 旋转: X={ref_obj.rotation_euler.x*180/3.14159:.1f}° Y={ref_obj.rotation_euler.y*180/3.14159:.1f}° Z={ref_obj.rotation_euler.z*180/3.14159:.1f}°")
                print(f"  - 尺寸: X={ref_obj.dimensions.x:.3f} Y={ref_obj.dimensions.y:.3f} Z={ref_obj.dimensions.z:.3f}")
                print(f"[快速对齐] 目标模型(活动): {target_obj.name}")
                print(f"  - 初始旋转: X={target_obj.rotation_euler.x*180/3.14159:.1f}° Y={target_obj.rotation_euler.y*180/3.14159:.1f}° Z={target_obj.rotation_euler.z*180/3.14159:.1f}°")
                print(f"  - 尺寸: X={target_obj.dimensions.x:.3f} Y={target_obj.dimensions.y:.3f} Z={target_obj.dimensions.z:.3f}")
                # 获取新的旋转
                loc, new_rot, scale = target_obj.matrix_world.decompose()
                euler = new_rot.to_euler()
                print(f"  - 对齐后旋转: X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
            
            self.report({'INFO'}, f"快速对齐完成: {target_obj.name} → {ref_obj.name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"快速对齐失败: {str(e)}")
            return {'CANCELLED'}
    
    def get_bbox_directions(self, obj):
        """基于包围盒计算模型的主要方向（考虑当前旋转）"""
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
    
    def calculate_target_rotation(self, ref_obj, target_obj, ref_dirs, target_dirs):
        """计算目标物体应该具有的最终旋转矩阵（严格尺寸对应）"""
        # 获取参考物体的旋转
        ref_loc, ref_rot, ref_scale = ref_obj.matrix_world.decompose()
        
        # 获取尺寸信息用于匹配
        ref_dims = ref_dirs['dimensions']
        tgt_dims = target_dirs['dimensions']
        
        # 局部坐标系下的尺寸
        ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
        tgt_sizes = [tgt_dims.x, tgt_dims.y, tgt_dims.z]
        
        # 对尺寸进行排序，得到从小到大的轴索引
        ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])  # [最小轴, 中轴, 最大轴]
        tgt_sorted_indices = sorted(range(3), key=lambda i: tgt_sizes[i])
        
        # 创建轴映射：目标的第i个轴应该映射到参考的第j个轴
        axis_mapping = [None, None, None]
        for rank in range(3):  # rank: 0=最小, 1=中等, 2=最大
            tgt_axis = tgt_sorted_indices[rank]
            ref_axis = ref_sorted_indices[rank]
            axis_mapping[tgt_axis] = ref_axis
        
        # 调试信息
        prefs = bpy.context.preferences.addons[get_addon_name()].preferences
        if prefs.show_debug_info:
            print(f"  [目标旋转计算 - 严格尺寸对应]")
            print(f"  参考模型尺寸: X={ref_dims.x:.3f} Y={ref_dims.y:.3f} Z={ref_dims.z:.3f}")
            print(f"  目标模型尺寸: X={tgt_dims.x:.3f} Y={tgt_dims.y:.3f} Z={tgt_dims.z:.3f}")
            print(f"  参考轴排序: {['XYZ'[i] for i in ref_sorted_indices]} (小→大)")
            print(f"  目标轴排序: {['XYZ'[i] for i in tgt_sorted_indices]} (小→大)")
            print(f"  轴映射: X→{'XYZ'[axis_mapping[0]]}, Y→{'XYZ'[axis_mapping[1]]}, Z→{'XYZ'[axis_mapping[2]]}")
        
        # 基于轴映射构建旋转矩阵
        # 需要找到一个旋转，使得目标的每个轴都映射到对应的参考轴
        
        # 首先创建一个置换矩阵
        perm_matrix = Matrix((
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)
        ))
        
        # 设置置换：目标轴i应该去到参考轴axis_mapping[i]的位置
        for i in range(3):
            perm_matrix[axis_mapping[i]][i] = 1.0
        
        # 检查行列式，确保是旋转而非镜像
        det = perm_matrix.determinant()
        if det < 0:
            # 如果是负的，需要翻转一个轴
            # 找到一个可以翻转的轴（通常选择最后一个）
            last_mapping = axis_mapping[2]
            perm_matrix[last_mapping][2] *= -1
        
        # 最终旋转 = 参考旋转 × 置换旋转
        final_rotation_matrix = ref_rot.to_matrix() @ perm_matrix
        
        if prefs.show_debug_info:
            euler = final_rotation_matrix.to_euler()
            print(f"  置换矩阵: \n{perm_matrix}")
            print(f"  计算的最终旋转: X={euler.x*180/3.14159:.1f}° Y={euler.y*180/3.14159:.1f}° Z={euler.z*180/3.14159:.1f}°")
        
        return final_rotation_matrix.to_4x4()
    
    def apply_final_rotation(self, obj, rotation_matrix):
        """应用最终旋转，保持原有的位置和缩放"""
        # 分解当前变换矩阵
        location, old_rotation, scale = obj.matrix_world.decompose()
        
        # 提取3x3旋转矩阵部分
        rotation_3x3 = rotation_matrix.to_3x3()
        
        # 设置新的旋转
        new_rotation = rotation_3x3.to_quaternion()
        
        # 重新构建变换矩阵，保持原有位置和缩放
        obj.matrix_world = Matrix.LocRotScale(location, new_rotation, scale)


# 强制拉伸对齐操作符
class OBJECT_OT_force_stretch_align(Operator):
    """强制拉伸目标模型并对齐包围盒中心，使其包围盒与参考模型完美重合"""
    bl_idname = "alignment.force_stretch_align"
    bl_label = "完美重合对齐"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        """只有在选中两个网格物体时才启用"""
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected_meshes) >= 2
    
    def execute(self, context):
        # 获取选中的物体
        active_obj = context.view_layer.objects.active
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if not active_obj or active_obj.type != 'MESH':
            self.report({'ERROR'}, "请将目标网格物体设为活动对象（高亮）")
            return {'CANCELLED'}

        if len(selected_objects) < 2:
            self.report({'ERROR'}, "请选中至少两个网格物体")
            return {'CANCELLED'}
        
        target_obj = active_obj
        ref_obj = next((o for o in selected_objects if o != target_obj), None)
        if ref_obj is None:
            self.report({'ERROR'}, "未找到参考模型，请至少再选中一个网格物体")
            return {'CANCELLED'}
        
        try:
            # 获取插件偏好设置
            prefs = context.preferences.addons[get_addon_name()].preferences
            
            # 检查是否已经旋转对齐
            tolerance_deg = prefs.alignment_tolerance
            tolerance_rad = tolerance_deg * 3.14159 / 180.0
            is_aligned, alignment_info = check_models_alignment(ref_obj, target_obj, tolerance_rad)
            
            if not is_aligned:
                self.report({'ERROR'}, f"模型尚未旋转对齐（角度差异: {alignment_info['max_angle_diff_deg']:.1f}°），请先执行旋转对齐")
                return {'CANCELLED'}
            
            # 备份变换信息
            if prefs.auto_backup:
                target_obj["_stretch_backup_matrix"] = [list(row) for row in target_obj.matrix_world]
                target_obj["_stretch_backup_dimensions"] = list(target_obj.dimensions)
                target_obj["_stretch_backup_scale"] = list(target_obj.scale)
            
            # 获取两个模型的尺寸信息
            ref_dirs = get_bbox_directions_static(ref_obj)
            target_dirs = get_bbox_directions_static(target_obj)
            
            ref_dims = ref_dirs['dimensions']
            target_dims = target_dirs['dimensions']
            
            # 计算尺寸排序和对应关系
            ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]
            target_sizes = [target_dims.x, target_dims.y, target_dims.z]
            
            ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])
            target_sorted_indices = sorted(range(3), key=lambda i: target_sizes[i])
            
            # 计算缩放因子（按尺寸对应关系）
            scale_factors = [1.0, 1.0, 1.0]
            for rank in range(3):
                ref_axis_idx = ref_sorted_indices[rank]
                target_axis_idx = target_sorted_indices[rank]
                
                ref_size = ref_sizes[ref_axis_idx]
                target_size = target_sizes[target_axis_idx]
                
                if target_size > 1e-6:  # 避免除零
                    scale_factors[target_axis_idx] = ref_size / target_size
                else:
                    scale_factors[target_axis_idx] = 1.0
            
            # 应用缩放
            current_scale = target_obj.scale.copy()
            new_scale = Vector((
                current_scale.x * scale_factors[0],
                current_scale.y * scale_factors[1],
                current_scale.z * scale_factors[2]
            ))
            
            target_obj.scale = new_scale
            
            # 更新场景以获取新的包围盒尺寸
            bpy.context.view_layer.update()
            
            # 计算包围盒中心点并对齐（实现完美重合）
            # 获取参考模型的包围盒中心（世界坐标）
            ref_bbox_center = target_obj.matrix_world @ Vector(ref_obj.bound_box[0])
            for i in range(1, 8):
                ref_bbox_center += target_obj.matrix_world @ Vector(ref_obj.bound_box[i])
            ref_bbox_center /= 8.0
            
            # 实际上应该用参考模型的包围盒中心
            ref_matrix = ref_obj.matrix_world
            ref_bbox_local_center = sum((Vector(corner) for corner in ref_obj.bound_box), Vector()) / 8
            ref_bbox_world_center = ref_matrix @ ref_bbox_local_center
            
            # 获取目标模型的包围盒中心（世界坐标）
            target_matrix = target_obj.matrix_world
            target_bbox_local_center = sum((Vector(corner) for corner in target_obj.bound_box), Vector()) / 8
            target_bbox_world_center = target_matrix @ target_bbox_local_center
            
            # 计算需要移动的偏移量
            offset = ref_bbox_world_center - target_bbox_world_center
            
            # 应用位置偏移，使包围盒中心完全重合
            target_obj.location += offset
            
            # 最终更新场景
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                print(f"\n[强制拉伸对齐 - 完美重合]")
                print(f"参考模型: {ref_obj.name}")
                print(f"  - 尺寸: X={ref_dims.x:.3f} Y={ref_dims.y:.3f} Z={ref_dims.z:.3f}")
                print(f"  - 包围盒中心: X={ref_bbox_world_center.x:.3f} Y={ref_bbox_world_center.y:.3f} Z={ref_bbox_world_center.z:.3f}")
                print(f"目标模型: {target_obj.name}")
                print(f"  - 原始尺寸: X={target_dims.x:.3f} Y={target_dims.y:.3f} Z={target_dims.z:.3f}")
                print(f"  - 原始缩放: X={current_scale.x:.3f} Y={current_scale.y:.3f} Z={current_scale.z:.3f}")
                print(f"  - 缩放因子: X={scale_factors[0]:.3f} Y={scale_factors[1]:.3f} Z={scale_factors[2]:.3f}")
                print(f"  - 新缩放: X={new_scale.x:.3f} Y={new_scale.y:.3f} Z={new_scale.z:.3f}")
                print(f"  - 新尺寸: X={target_obj.dimensions.x:.3f} Y={target_obj.dimensions.y:.3f} Z={target_obj.dimensions.z:.3f}")
                print(f"  - 位置偏移: X={offset.x:.3f} Y={offset.y:.3f} Z={offset.z:.3f}")
                print(f"  - 新包围盒中心: X={target_bbox_world_center.x+offset.x:.3f} Y={target_bbox_world_center.y+offset.y:.3f} Z={target_bbox_world_center.z+offset.z:.3f}")
            
            self.report({'INFO'}, f"完美重合对齐完成: {target_obj.name} → {ref_obj.name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"强制拉伸对齐失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


# 恢复对齐前状态的操作符
class OBJECT_OT_restore_alignment(Operator):
    """恢复目标模型到对齐前的状态"""
    bl_idname = "alignment.restore_alignment"
    bl_label = "恢复对齐前状态"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        restored_count = 0
        
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                try:
                    # 恢复旋转对齐备份
                    if "_alignment_backup_matrix" in obj:
                        backup_matrix = obj["_alignment_backup_matrix"]
                        obj.matrix_world = Matrix(backup_matrix)
                        
                        # 清理备份数据
                        del obj["_alignment_backup_matrix"]
                        if "_alignment_backup_location" in obj:
                            del obj["_alignment_backup_location"]
                        if "_alignment_backup_rotation" in obj:
                            del obj["_alignment_backup_rotation"]
                        if "_alignment_backup_scale" in obj:
                            del obj["_alignment_backup_scale"]
                        
                        restored_count += 1
                    
                    # 恢复拉伸对齐备份
                    elif "_stretch_backup_matrix" in obj:
                        backup_matrix = obj["_stretch_backup_matrix"]
                        obj.matrix_world = Matrix(backup_matrix)
                        
                        # 清理备份数据
                        del obj["_stretch_backup_matrix"]
                        if "_stretch_backup_dimensions" in obj:
                            del obj["_stretch_backup_dimensions"]
                        if "_stretch_backup_scale" in obj:
                            del obj["_stretch_backup_scale"]
                        
                        restored_count += 1
                    
                except Exception as e:
                    self.report({'WARNING'}, f"恢复 {obj.name} 失败: {str(e)}")
        
        if restored_count > 0:
            bpy.context.view_layer.update()
            self.report({'INFO'}, f"成功恢复 {restored_count} 个物体的对齐前状态")
        else:
            self.report({'INFO'}, "没有找到可恢复的备份数据")
        
        return {'FINISHED'}


# 批量对齐操作符
class OBJECT_OT_batch_align(Operator):
    """批量下载、导入和对齐模型"""
    bl_idname = "alignment.batch_align"
    bl_label = "一键批量完美对齐"
    bl_options = {'REGISTER', 'UNDO'}
    
    json_config: StringProperty(
        name="JSON配置",
        description="批量对齐的JSON配置",
        default=""
    )
    
    def execute(self, context):
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
            
            # 认证信息（写死在代码中）
            auth_token = None  # 如果需要额外的token，在这里设置
            
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
                        print(f"[批量对齐] 创建输出目录: {output_dir}")
                except Exception as e:
                    self.report({'ERROR'}, f"创建输出目录失败: {str(e)}")
                    return {'CANCELLED'}
            
            self.report({'INFO'}, f"开始批量处理 {len(objects)} 个对象...")
            
            success_count = 0
            failed_count = 0
            processed_objects = []  # 存储处理后的对象信息
            model_objects = []  # 存储模型对象引用
            
            for idx, obj_config in enumerate(objects):
                try:
                    self.report({'INFO'}, f"处理第 {idx+1}/{len(objects)} 个对象: {obj_config.get('name', '未命名')}")
                    
                    # 下载并导入模型（返回模型对象和文件格式）
                    model_obj, file_format = self.download_and_import_model(obj_config, auth_token, prefs)
                    if not model_obj:
                        failed_count += 1
                        continue
                    
                    # 设置模型位置为原点
                    # position值在UE中使用，Blender处理时统一在原点(0,0,0)
                    model_obj.location = Vector((0, 0, 0))
                    
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 模型位置设置为原点: {model_obj.location}")
                    
                    # 创建Box作为参考
                    ref_box = self.create_reference_box(obj_config, prefs)
                    if not ref_box:
                        failed_count += 1
                        continue
                    
                    # 执行对齐（传入obj_config以应用JSON中的旋转）
                    self.align_model_to_box(model_obj, ref_box, obj_config, context, prefs)
                    
                    # 删除参考Box
                    bpy.data.objects.remove(ref_box, do_unlink=True)
                    
                    # 使用用户设置的导出格式（而不是从URL检测的格式）
                    export_format = '.glb' if prefs.export_format == 'GLB' else '.fbx'
                    
                    # 存储模型对象、配置和导出格式
                    model_objects.append((model_obj, obj_config, export_format, idx))
                    success_count += 1
                    
                except Exception as e:
                    self.report({'WARNING'}, f"处理对象 {idx+1} 失败: {str(e)}")
                    failed_count += 1
                    if prefs.show_debug_info:
                        import traceback
                        traceback.print_exc()
            
            # 导出所有成功处理的模型
            format_name = 'GLB' if prefs.export_format == 'GLB' else 'FBX'
            self.report({'INFO'}, f"开始导出 {len(model_objects)} 个模型（格式：{format_name}）...")
            
            for model_obj, obj_config, file_format, idx in model_objects:
                try:
                    # 应用坐标系转换（针对目标引擎）
                    # 只转换scale，location始终保持(0,0,0)
                    self.apply_coordinate_transform(
                        model_obj,
                        prefs.target_engine,
                        prefs,
                        file_format
                    )
                    
                    # 确保位置为原点（应该已经是0了）
                    model_obj.location = Vector((0, 0, 0))
                    bpy.context.view_layer.update()
                    
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 导出位置: {model_obj.location} [原点]")
                    
                    # 根据输入格式导出模型
                    exported_path = self.export_model(model_obj, obj_config, output_dir, file_format, prefs)
                    
                    if exported_path:
                        # 创建新的对象配置，rotation设为0
                        new_obj_config = obj_config.copy()
                        new_obj_config['rotation'] = [0, 0, 0]
                        
                        # position和scale保持原始值不变
                        # 因为：
                        # 1. position在UE中使用，Blender不管
                        # 2. scale在UE中使用，模型几何体已经被缩放
                        # 3. 所有值在输出时与输入保持一致
                        
                        if prefs.show_debug_info:
                            print(f"[批量对齐] 输出JSON:")
                            print(f"  position: {new_obj_config.get('position')} (保持原始)")
                            print(f"  rotation: [0, 0, 0] (已烘焙)")
                            print(f"  scale: {new_obj_config.get('scale')} (保持原始)")
                        
                        # 根据输入JSON的URL字段格式，更新对应的URL字段
                        # 优先级：3d_url > output.url > url
                        if '3d_url' in obj_config:
                            new_obj_config['3d_url'] = exported_path
                            if prefs.show_debug_info:
                                print(f"[批量对齐] 更新字段: 3d_url")
                        elif 'output' in obj_config and 'url' in obj_config['output']:
                            if 'output' not in new_obj_config:
                                new_obj_config['output'] = {}
                            new_obj_config['output']['url'] = exported_path
                            if prefs.show_debug_info:
                                print(f"[批量对齐] 更新字段: output.url")
                        else:
                            # 兜底：如果都没有，使用output.url
                            if 'output' not in new_obj_config:
                                new_obj_config['output'] = {}
                            new_obj_config['output']['url'] = exported_path
                            if prefs.show_debug_info:
                                print(f"[批量对齐] 更新字段: output.url (默认)")
                        
                        processed_objects.append(new_obj_config)
                        
                        if prefs.show_debug_info:
                            print(f"[批量对齐] 导出成功: {exported_path}")
                    
                except Exception as e:
                    self.report({'WARNING'}, f"导出模型 {obj_config.get('name', '未命名')} 失败: {str(e)}")
                    if prefs.show_debug_info:
                        import traceback
                        traceback.print_exc()
            
            # 生成输出JSON
            output_config = config.copy()
            output_config['objects'] = processed_objects
            
            # 保存JSON到文件
            json_output_path = os.path.join(output_dir, "output_config.json")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_config, f, indent=2, ensure_ascii=False)
                
                # 同时保存到插件设置中
                prefs.batch_output_json = json.dumps(output_config, indent=2, ensure_ascii=False)
                
                self.report({'INFO'}, f"输出JSON已保存: {json_output_path}")
                
                if prefs.show_debug_info:
                    print(f"[批量对齐] 输出JSON:\n{json.dumps(output_config, indent=2, ensure_ascii=False)}")
                
            except Exception as e:
                self.report({'WARNING'}, f"保存JSON失败: {str(e)}")
            
            self.report({'INFO'}, f"批量处理完成: 成功 {success_count} 个, 失败 {failed_count} 个, 导出 {len(processed_objects)} 个")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"批量对齐失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
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
        """根据格式导出模型"""
        if file_format in ['.glb', '.gltf']:
            return self.export_model_glb(model_obj, obj_config, output_dir, prefs)
        elif file_format == '.obj':
            return self.export_model_obj(model_obj, obj_config, output_dir, prefs)
        else:
            # 默认导出为FBX
            return self.export_model_fbx(model_obj, obj_config, output_dir, prefs)
    
    def export_model_glb(self, model_obj, obj_config, output_dir, prefs):
        """导出模型为GLB格式，贴图嵌入"""
        try:
            model_name = obj_config.get('name', 'model')
            # 清理文件名，移除不合法字符
            import re
            safe_name = re.sub(r'[<>:\"/\\|?*]', '_', model_name)
            output_filename = f"{safe_name}.glb"
            output_path = os.path.join(output_dir, output_filename)
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 准备导出GLB: {model_name} -> {output_path}")
            
            # 取消所有选择
            bpy.ops.object.select_all(action='DESELECT')
            
            # 选择要导出的模型（包括子对象）
            model_obj.select_set(True)
            
            # 同时选择所有子对象
            for child in model_obj.children_recursive:
                child.select_set(True)
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 选中对象: {model_obj.name} 及其 {len(model_obj.children_recursive)} 个子对象")
            
            # 导出为GLB格式（使用Blender 4.4兼容的最小参数集）
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                use_selection=True,
                export_texcoords=True,
                export_normals=True,
                export_materials='EXPORT',
                export_image_format='AUTO',
            )
            
            if prefs.show_debug_info:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[批量对齐] GLB导出完成: {output_path} ({file_size_mb:.2f} MB)")
            
            return output_path
        
        except Exception as e:
            self.report({'WARNING'}, f"导出GLB失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
            return None
    
    def export_model_fbx(self, model_obj, obj_config, output_dir, prefs):
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
            
            # 导出为FBX格式
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                use_selection=True,  # 只导出选中的对象
                apply_scale_options='FBX_SCALE_ALL',  # 应用缩放
                axis_forward='-Z',  # 前向轴
                axis_up='Y',  # 上向轴
                bake_space_transform=True,  # 烘焙空间变换（包括旋转）
                object_types={'MESH', 'EMPTY'},  # 只导出网格和空对象
                use_mesh_modifiers=True,  # 应用修改器
                mesh_smooth_type='OFF',  # 平滑类型
                use_tspace=True,  # 使用切线空间
                embed_textures=True,  # 嵌入贴图（关键！）
                path_mode='COPY',  # 复制贴图
                batch_mode='OFF',  # 不使用批量模式
            )
            
            if prefs.show_debug_info:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[批量对齐] 导出完成: {output_path} ({file_size_mb:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            self.report({'WARNING'}, f"导出FBX失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
            return None
    
    def export_model_obj(self, model_obj, obj_config, output_dir, prefs):
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
                path_mode='COPY',  # 复制贴图
            )
            
            if prefs.show_debug_info:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[批量对齐] OBJ导出完成: {output_path} ({file_size_mb:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            self.report({'WARNING'}, f"导出OBJ失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
            return None
    
    def download_and_import_model(self, obj_config, auth_token, prefs):
        """下载并导入模型（自动识别格式）"""
        try:
            # 兼容多种URL字段格式
            # 1. 优先尝试 3d_url
            # 2. 其次尝试 output.url
            # 3. 最后尝试 url
            url = obj_config.get('3d_url') or \
                  obj_config.get('output', {}).get('url') or \
                  obj_config.get('url', '')
            
            if not url:
                self.report({'WARNING'}, f"对象缺少URL: {obj_config.get('name')}")
                return None, None
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 检测到URL字段，开始下载: {url}")
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 下载模型: {url}")
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            filename = url.split('/')[-1].split('?')[0]  # 移除URL参数
            if not filename:
                filename = "model.fbx"
            
            # 检测文件格式
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                file_ext = '.fbx'  # 默认FBX
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 检测到文件格式: {file_ext}")
            
            temp_path = os.path.join(temp_dir, filename)
            
            # 使用requests库下载（更好的兼容性）
            try:
                import requests
                import warnings
                try:
                    from requests.packages.urllib3.exceptions import InsecureRequestWarning
                    warnings.simplefilter('ignore', InsecureRequestWarning)
                except:
                    pass
                
                if prefs.show_debug_info:
                    print(f"[批量对齐] 使用requests库下载...")
                
                # 设置请求头
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': '*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
                }
                
                # 检测dreammaker域名，添加特定认证头
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                if 'dreammaker.netease.com' in parsed_url.netloc:
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 检测到dreammaker域名，添加认证头...")
                    headers['X-Auth-User'] = 'blender-alignment-tool'
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 已添加X-Auth-User认证头")
                
                # 如果有自定义token，添加到Authorization头
                if auth_token:
                    headers['Authorization'] = f'Bearer {auth_token}'
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 已添加Authorization头")
                
                if prefs.show_debug_info:
                    print(f"[批量对齐] 下载到: {temp_path}")
                
                # 执行下载
                response = requests.get(url, headers=headers, timeout=300, verify=False)
                
                if response.status_code == 200:
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_size_mb = len(response.content) / (1024 * 1024)
                    if prefs.show_debug_info:
                        print(f"[批量对齐] 下载成功: {file_size_mb:.2f} MB")
                else:
                    raise Exception(f"HTTP错误 {response.status_code}: {response.reason}")
                
            except ImportError:
                self.report({'ERROR'}, "需要安装requests库: pip install requests")
                return None, None
            
            # 根据文件格式导入模型
            if prefs.show_debug_info:
                print(f"[批量对齐] 开始导入{file_ext}文件...")
            
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
                # 尝试作为FBX导入
                if prefs.show_debug_info:
                    print(f"[批量对齐] 未知格式{file_ext}，尝试作为FBX导入...")
                bpy.ops.import_scene.fbx(filepath=temp_path)
                imported_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
            
            if imported_obj:
                imported_obj.name = obj_config.get('name', filename)
                if prefs.show_debug_info:
                    print(f"[批量对齐] 导入成功: {imported_obj.name}, 格式: {file_ext}")
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            
            # 返回导入的对象和文件格式
            return imported_obj, file_ext
            
        except Exception as e:
            self.report({'WARNING'}, f"下载或导入模型失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
            return None, None
    
    def create_reference_box(self, obj_config, prefs):
        """根据配置创建参考Box"""
        try:
            name = obj_config.get('name', 'ReferenceBox')
            rotation = obj_config.get('rotation', [0, 0, 0])
            scale = obj_config.get('scale', [1, 1, 1])
            
            # position不需要使用，Box统一在原点(0,0,0)创建
            # JSON中的position是在UE中设置的，Blender处理时不管
            
            # 创建单位立方体（size=1），这样scale直接对应实际尺寸
            # 例如：scale=[4.0, 0.9, 0.75] → Box尺寸为 4m × 0.9m × 0.75m
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(0, 0, 0)
            )
            box = bpy.context.active_object
            box.name = f"{name}_RefBox"
            
            # 设置旋转（度转弧度）
            box.rotation_euler = [r * 3.14159 / 180 for r in rotation]
            
            # 设置缩放（直接对应JSON中的scale，即实际尺寸）
            box.scale = scale
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 创建参考Box: {box.name}")
                print(f"  位置: (0, 0, 0) [原点]")
                print(f"  旋转: {rotation}°")
                print(f"  缩放: {scale}")
                print(f"  实际尺寸: {scale[0]}m × {scale[1]}m × {scale[2]}m")
            
            return box
            
        except Exception as e:
            self.report({'WARNING'}, f"创建参考Box失败: {str(e)}")
            return None
    
    def align_model_to_box(self, model_obj, ref_box, obj_config, context, prefs):
        """将模型对齐到参考Box"""
        try:
            # 取消所有选择
            bpy.ops.object.select_all(action='DESELECT')
            
            # 选择模型和Box
            model_obj.select_set(True)
            ref_box.select_set(True)
            
            # 设置模型为活动对象
            context.view_layer.objects.active = model_obj
            
            # ⭐ 新步骤：先将JSON中的旋转应用到模型上
            # 这样可以避免自动旋转算法可能导致的反转问题
            rotation = obj_config.get('rotation', [0, 0, 0])
            model_obj.rotation_euler = [r * 3.14159 / 180 for r in rotation]
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 应用JSON旋转到模型: {rotation}°")
                print(f"[批量对齐] 模型当前旋转: X={model_obj.rotation_euler.x*180/3.14159:.1f}° Y={model_obj.rotation_euler.y*180/3.14159:.1f}° Z={model_obj.rotation_euler.z*180/3.14159:.1f}°")
            
            # 检查是否已经对齐
            tolerance_deg = prefs.alignment_tolerance
            tolerance_rad = tolerance_deg * 3.14159 / 180.0
            is_aligned, alignment_info = check_models_alignment(ref_box, model_obj, tolerance_rad)
            
            if prefs.show_debug_info:
                if is_aligned:
                    print(f"[批量对齐] ✓ 应用JSON旋转后已对齐（角度差: {alignment_info['max_angle_diff_deg']:.2f}°），跳过旋转计算")
                else:
                    print(f"[批量对齐] ✗ 应用JSON旋转后仍未对齐（角度差: {alignment_info['max_angle_diff_deg']:.2f}°），执行自动旋转对齐")
            
            if not is_aligned:
                # 执行旋转对齐
                self.execute_rotation_align_batch(ref_box, model_obj, prefs)
            
            # 执行完美重合对齐
            self.execute_perfect_align_batch(ref_box, model_obj, prefs)
            
            if prefs.show_debug_info:
                print(f"[批量对齐] 对齐完成: {model_obj.name} → {ref_box.name}")
            
        except Exception as e:
            self.report({'WARNING'}, f"对齐失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                traceback.print_exc()
    
    def execute_rotation_align_batch(self, ref_obj, target_obj, prefs):
        """批量模式的旋转对齐"""
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
    
    def execute_perfect_align_batch(self, ref_obj, target_obj, prefs):
        """批量模式的完美重合对齐"""
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


# 定义类列表用于注册
classes = (
    OBJECT_OT_one_click_align,
    OBJECT_OT_quick_align_models,
    OBJECT_OT_force_stretch_align,
    OBJECT_OT_restore_alignment,
    OBJECT_OT_batch_align,
)