"""
VVL 3D Model Smart Scaler
智能3D模型缩放处理器，基于包围盒和目标尺寸进行精确缩放
"""
import os
import sys
import json
import tempfile
import subprocess
import shutil
import math
import urllib.request
import urllib.parse
import hashlib
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import folder_paths

# 智能缩放算法类
@dataclass
class Vector3:
    """3D向量类"""
    x: float
    y: float  
    z: float
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
    
    def max_component(self) -> float:
        return max(self.x, self.y, self.z)
    
    def min_component(self) -> float:
        return min(self.x, self.y, self.z)
    
    def volume(self) -> float:
        """计算体积（假设这是包围盒尺寸）"""
        return self.x * self.y * self.z
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    bounding_box_size: Vector3
    target_scale: Vector3
    target_volume: Optional[float] = None

@dataclass 
class ScalingConfig:
    """缩放配置参数"""
    force_exact_alignment: bool = True
    standard_size: float = 100.0
    scale_range_min: float = 0.1
    scale_range_max: float = 10.0

class SmartScaler:
    """智能模型缩放处理器"""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
    
    def calculate_volume_scale(self, model: ModelInfo) -> float:
        """基于目标体积计算等比缩放因子"""
        if not model.target_volume or model.target_volume <= 0:
            print(f"警告: {model.name} 目标体积无效，使用默认缩放1.0")
            return 1.0
            
        # 计算当前体积 
        current_volume = model.bounding_box_size.volume()
        if current_volume <= 1.0:  # 防止除以接近零的值
            print(f"警告: {model.name} 当前体积太小({current_volume})，使用默认缩放1.0")
            return 1.0
            
        # 计算体积缩放因子：立方根(目标体积/当前体积)
        desired_scale = math.pow(model.target_volume / current_volume, 1.0/3.0)
        
        # 应用缩放限制
        clamped_scale = max(self.config.scale_range_min, 
                           min(self.config.scale_range_max, desired_scale))
        
        print(f"{model.name} 体积缩放分析:")
        print(f"  当前体积: {current_volume:.2f}")
        print(f"  目标体积: {model.target_volume:.2f}")  
        print(f"  计算缩放: {desired_scale:.3f}")
        print(f"  限制后缩放: {clamped_scale:.3f}")
        
        return clamped_scale
    
    def calculate_smart_scale(self, model: ModelInfo) -> Vector3:
        """基于包围盒和目标尺寸计算智能缩放"""
        current_size = model.bounding_box_size
        target_scale = model.target_scale
        
        # 第一步：等比缩放
        max_current_dimension = current_size.max_component()
        target_scale_factor = target_scale.max_component()
        target_max_dimension = self.config.standard_size * target_scale_factor
        
        # 计算等比缩放因子
        uniform_scale = target_max_dimension / max_current_dimension
        
        print(f"{model.name} 智能缩放第一步（等比缩放）:")
        print(f"  当前尺寸: ({current_size.x:.1f}, {current_size.y:.1f}, {current_size.z:.1f})")
        print(f"  最大维度: {max_current_dimension:.1f}")
        print(f"  目标缩放: ({target_scale.x:.1f}, {target_scale.y:.1f}, {target_scale.z:.1f})")
        print(f"  目标最大维度: {target_max_dimension:.1f}")
        print(f"  等比缩放因子: {uniform_scale:.3f}")
        
        # 第二步：智能轴向匹配（如果启用）
        if self.config.force_exact_alignment:
            return self._calculate_axis_aligned_scale(model, uniform_scale)
        else:
            return Vector3(uniform_scale, uniform_scale, uniform_scale)
    
    def _calculate_axis_aligned_scale(self, model: ModelInfo, uniform_scale: float) -> Vector3:
        """计算智能轴向匹配缩放"""
        current_size = model.bounding_box_size
        target_scale = model.target_scale
        
        # 计算等比缩放后的理论尺寸
        scaled_size = current_size * uniform_scale
        
        target_scales_sorted = sorted([target_scale.x, target_scale.y, target_scale.z])
        target_sizes = [
            self.config.standard_size * target_scales_sorted[0],
            self.config.standard_size * target_scales_sorted[1],
            self.config.standard_size * target_scales_sorted[2]
        ]
        
        smart_target_size = self._map_dimensions_to_targets(scaled_size, target_sizes)
        fine_tune_ratio = smart_target_size / scaled_size
        final_scale = Vector3(
            uniform_scale * fine_tune_ratio.x,
            uniform_scale * fine_tune_ratio.y, 
            uniform_scale * fine_tune_ratio.z
        )
        
        print(f"{model.name} 智能缩放第二步（轴向匹配）:")
        print(f"  等比缩放后尺寸: ({scaled_size.x:.1f}, {scaled_size.y:.1f}, {scaled_size.z:.1f})")
        print(f"  排序后目标尺寸: [{target_sizes[0]:.1f}, {target_sizes[1]:.1f}, {target_sizes[2]:.1f}]")
        print(f"  智能匹配目标: ({smart_target_size.x:.1f}, {smart_target_size.y:.1f}, {smart_target_size.z:.1f})")
        print(f"  微调比例: ({fine_tune_ratio.x:.3f}, {fine_tune_ratio.y:.3f}, {fine_tune_ratio.z:.3f})")
        print(f"  最终缩放: ({final_scale.x:.3f}, {final_scale.y:.3f}, {final_scale.z:.3f})")
        
        return final_scale
    
    def _map_dimensions_to_targets(self, scaled_size: Vector3, target_sizes: List[float]) -> Vector3:
        """将模型尺寸按大小顺序映射到目标尺寸"""
        size_with_axis = [
            (scaled_size.x, 0), (scaled_size.y, 1), (scaled_size.z, 2)
        ]
        size_with_axis.sort(key=lambda x: x[0])
        
        smart_target = [0.0, 0.0, 0.0]
        for i, (_, axis_index) in enumerate(size_with_axis):
            smart_target[axis_index] = target_sizes[i]
            
        return Vector3(smart_target[0], smart_target[1], smart_target[2])

class BlenderSmartModelScaler:
    """
    VVL智能3D模型缩放器
    基于包围盒和目标尺寸进行精确智能缩放
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {}),
                "target_size_x": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 10000.0}),
                "target_size_y": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 10000.0}),
                "target_size_z": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 10000.0}),
                "force_exact_alignment": ("BOOLEAN", {"default": True, "tooltip": "强制精确对齐：启用时允许各轴独立缩放以匹配目标尺寸（可能造成拉伸变形），禁用时保持模型原始比例进行等比缩放"}),
                "blender_path": ("STRING", {"default": "blender", "tooltip": "Blender 可执行文件路径，默认使用系统PATH中的 'blender' 命令，也可指定完整路径。Windows示例: 'C:/Program Files/Blender Foundation/Blender 4.0/blender.exe'，Linux示例: '/usr/bin/blender' 或 '/opt/blender/blender'"}),
                "output_name": ("STRING", {"default": "scaled.fbx", "tooltip": "输出文件名，支持 .fbx、.obj、.glb 等格式，文件将保存到 ComfyUI 输出目录的 3d 子文件夹中"}),
            },
            "optional": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "可选：3D模型链接或本地路径，支持 .fbx、.glb、.gltf、.obj 格式。\n• 如果是URL（http://、https://等），会自动下载到 downloads/3d_models 文件夹\n• 如果是本地路径（相对或绝对路径），会直接使用该文件\n• 支持 file:// 协议的本地文件链接\n• 如果提供此参数，mesh_path 将被忽略"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "bbox_json", "scale_info")
    FUNCTION = "process"
    CATEGORY = "VVL/3D"

    BLENDER_SCRIPT = r"""
import bpy, sys, json, math
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--")+1:]
in_path, out_path, sx, sy, sz, bbox_path, scale_info_path = argv
sx, sy, sz = float(sx), float(sy), float(sz)

print(f"[Blender] 开始处理模型: {in_path}")
print(f"[Blender] 应用缩放: sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}")
print(f"[Blender] 缩放将烘焙到顶点")

bpy.ops.wm.read_factory_settings(use_empty=True)

lower = in_path.lower()
print(f"[Blender] 导入文件类型: {lower}")
try:
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=in_path, global_scale=1.0, axis_forward='-Z', axis_up='Y',
                                use_image_search=True, use_custom_props=True)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=in_path, import_pack_images=True)
    print(f"[Blender] 导入成功")
except Exception as e:
    print(f"[Blender] 导入警告: {str(e)}")
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=in_path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=in_path)
    print(f"[Blender] 使用基本导入模式成功")

# 修复GLB导入后的纹理路径问题（确保FBX导出时纹理正确）
if lower.endswith((".glb", ".gltf")):
    print(f"[Blender] 修复GLB纹理路径以确保FBX导出兼容性...")
    image_counter = 0
    for img in bpy.data.images:
        if img.source == 'FILE' and not img.filepath:
            # 为没有filepath的图像设置一个虚假的路径
            fake_path = f"texture_{image_counter:03d}.png"
            img.filepath = fake_path
            print(f"[Blender] 修复图像路径: {img.name} -> {fake_path}")
            image_counter += 1
        elif img.packed_file and not img.filepath:
            # 对于packed文件，也需要设置路径
            fake_path = f"packed_texture_{image_counter:03d}.png"
            img.filepath = fake_path
            print(f"[Blender] 修复packed图像路径: {img.name} -> {fake_path}")
            image_counter += 1
    print(f"[Blender] 纹理路径修复完成，处理了 {image_counter} 个图像")

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
print(f"[Blender] 找到 {len(meshes)} 个网格对象")

original_gmin = Vector(( math.inf,  math.inf,  math.inf))
original_gmax = Vector((-math.inf, -math.inf, -math.inf))
for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        original_gmin.x = min(original_gmin.x, wpt.x)
        original_gmin.y = min(original_gmin.y, wpt.y)
        original_gmin.z = min(original_gmin.z, wpt.z)
        original_gmax.x = max(original_gmax.x, wpt.x)
        original_gmax.y = max(original_gmax.y, wpt.y)
        original_gmax.z = max(original_gmax.z, wpt.z)

original_size = [original_gmax[i] - original_gmin[i] for i in range(3)]
print(f"[Blender] 原始包围盒尺寸: ({original_size[0]:.2f}, {original_size[1]:.2f}, {original_size[2]:.2f})")

for o in meshes:
    o.select_set(True)
    original_scale = o.scale.copy()
    o.scale = (o.scale[0]*sx, o.scale[1]*sy, o.scale[2]*sz)
    print(f"[Blender] 对象 '{o.name}' 缩放: {original_scale} -> {o.scale}")

# 烘焙缩放到顶点
bpy.context.view_layer.objects.active = meshes[0] if meshes else None
if meshes:
    print(f"[Blender] 烘焙缩放到顶点...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# 计算缩放后的全局 AABB
gmin = Vector(( math.inf,  math.inf,  math.inf))
gmax = Vector((-math.inf, -math.inf, -math.inf))
for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        gmin.x = min(gmin.x, wpt.x)
        gmin.y = min(gmin.y, wpt.y)
        gmin.z = min(gmin.z, wpt.z)
        gmax.x = max(gmax.x, wpt.x)
        gmax.y = max(gmax.y, wpt.y)
        gmax.z = max(gmax.z, wpt.z)

final_size = [gmax[i] - gmin[i] for i in range(3)]
print(f"[Blender] 缩放后包围盒尺寸: ({final_size[0]:.2f}, {final_size[1]:.2f}, {final_size[2]:.2f})")

# 准备输出数据（先初始化占位，材质统计稍后填充）
bbox = {"min":[gmin.x,gmin.y,gmin.z], "max":[gmax.x,gmax.y,gmax.z], "size": final_size}
material_count = 0
texture_count = 0
scale_info = {
    "applied_scale": [sx, sy, sz],
    "original_size": original_size,
    "final_size": final_size,
    "size_change_ratio": [final_size[i]/original_size[i] if original_size[i] > 0 else 1.0 for i in range(3)],
    "mesh_count": len(meshes),
    "scale_applied_to_vertices": True,
    "material_count": material_count,
    "texture_count": texture_count,
    "materials_preserved": False
}

# 导出模型（保留所有材质和贴图信息）
print(f"[Blender] 导出到: {out_path}")
print(f"[Blender] 保留材质和贴图数据...")

# 检查并记录材质信息
material_count = 0
texture_count = 0
try:
    for obj in meshes:
        if obj.data.materials:
            material_count += len(obj.data.materials)
            for mat in obj.data.materials:
                if mat and hasattr(mat, 'use_nodes') and mat.use_nodes and hasattr(mat, 'node_tree'):
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and hasattr(node, 'image') and node.image:
                            texture_count += 1
                            print(f"[Blender] 发现贴图: {node.image.name}")
except Exception as e:
    print(f"[Blender] 材质检查警告: {str(e)}")

print(f"[Blender] 共有 {material_count} 个材质, {texture_count} 个贴图")

# 同步材质统计到 scale_info（以免后续导出失败导致文件未写出）
scale_info["material_count"] = material_count
scale_info["texture_count"] = texture_count
scale_info["materials_preserved"] = (texture_count > 0 or material_count > 0)

# 先写入 bbox 与 scale_info，避免后续导出失败导致文件缺失
with open(bbox_path, "w", encoding="utf-8") as f:
    json.dump(bbox, f)

with open(scale_info_path, "w", encoding="utf-8") as f:
    json.dump(scale_info, f)

# 根据输出文件格式选择导出器
out_path_lower = out_path.lower()
if out_path_lower.endswith(('.glb', '.gltf')):
    # GLB/GLTF导出
    try:
        print(f"[Blender] 尝试GLB/GLTF完整导出...")
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format='GLB' if out_path_lower.endswith('.glb') else 'GLTF_SEPARATE',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_colors=True,
            export_cameras=False,
            export_extras=True,
            export_yup=True
        )
        print(f"[Blender] GLB/GLTF完整导出成功")
    except Exception as e:
        print(f"[Blender] GLB/GLTF完整导出失败，尝试基本导出: {str(e)}")
        try:
            bpy.ops.export_scene.gltf(
                filepath=out_path,
                export_format='GLB' if out_path_lower.endswith('.glb') else 'GLTF_SEPARATE'
            )
            print(f"[Blender] GLB/GLTF基本导出成功")
        except Exception as e2:
            print(f"[Blender] GLB/GLTF基本导出也失败: {str(e2)}")
            raise e2
else:
    # FBX导出（带错误处理，针对GLB转换优化）
    try:
        print(f"[Blender] 尝试FBX完整导出...")
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            axis_forward='-Z', 
            axis_up='Y',
            path_mode='COPY',           # 复制贴图文件
            embed_textures=True,        # 嵌入贴图
            use_custom_props=True,      # 保留自定义属性
            use_mesh_modifiers=True,    # 应用修改器
            use_armature_deform_only=False,  # 包含所有骨骼
            add_leaf_bones=False,       # 不添加叶子骨骼
            primary_bone_axis='Y',      # 骨骼轴向
            secondary_bone_axis='X',    # 次要轴向
            use_metadata=True,          # 包含元数据
            global_scale=1.0
        )
        print(f"[Blender] FBX完整导出成功")
    except Exception as e:
        print(f"[Blender] FBX完整导出失败，尝试基本导出: {str(e)}")
        try:
            bpy.ops.export_scene.fbx(
                filepath=out_path,
                use_selection=False,
                axis_forward='-Z', 
                axis_up='Y',
                path_mode='COPY',      # 确保基本导出也复制纹理
                embed_textures=True    # 确保基本导出也嵌入纹理
            )
            print(f"[Blender] FBX基本导出成功")
        except Exception as e2:
            print(f"[Blender] FBX基本导出也失败: {str(e2)}")
            raise e2

# 保存结果
with open(bbox_path, "w", encoding="utf-8") as f:
    json.dump(bbox, f)

with open(scale_info_path, "w", encoding="utf-8") as f:
    json.dump(scale_info, f)

print(f"[Blender] 处理完成！")
"""

    def _ensure_blender(self, blender_path: str):
        if shutil.which(blender_path) is None:
            raise Exception(f"找不到 Blender 可执行文件：{blender_path}。请将 Blender 加入 PATH，或在本节点里填写绝对路径。")

    def _generate_unique_filename(self, base_dir: str, url: str) -> str:
        """生成唯一的文件名，避免文件覆盖"""
        parsed_url = urllib.parse.urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        
        # 如果URL没有文件名，尝试从Content-Disposition获取
        if not original_filename or '.' not in original_filename:
            original_filename = "model.fbx"  # 默认文件名
        
        # 确保文件扩展名正确
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext not in ('.fbx', '.glb', '.gltf'):
            # 尝试从URL的query参数或headers中推断格式
            if 'fbx' in url.lower():
                file_ext = '.fbx'
            elif 'glb' in url.lower():
                file_ext = '.glb'
            elif 'gltf' in url.lower():
                file_ext = '.gltf'
            else:
                file_ext = '.fbx'  # 默认为FBX
        
        base_name = os.path.splitext(original_filename)[0]
        
        # 生成时间戳和URL哈希来确保唯一性
        timestamp = str(int(time.time()))
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        unique_name = f"{base_name}_{timestamp}_{url_hash}{file_ext}"
        
        return os.path.join(base_dir, unique_name)

    def _is_url(self, path_or_url: str) -> bool:
        """判断输入是URL还是本地路径"""
        if not path_or_url or not isinstance(path_or_url, str):
            return False
            
        path_or_url = path_or_url.strip()
        
        # 明确的URL协议
        if path_or_url.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
            return True
            
        # 包含://但不是文件路径的其他协议
        if '://' in path_or_url and not path_or_url.startswith('file://'):
            return True
            
        return False
    
    def _handle_local_path(self, file_path: str) -> str:
        """处理本地路径，验证存在性并返回规范化的绝对路径"""
        if not file_path or not isinstance(file_path, str):
            raise Exception("无效的文件路径")
            
        file_path = file_path.strip()
        
        # 处理file://协议
        if file_path.startswith('file://'):
            file_path = file_path[7:]  # 移除file://前缀
            # Windows系统下可能有额外的斜杠
            if os.name == 'nt' and file_path.startswith('/'):
                file_path = file_path[1:]
        
        # 规范化路径
        file_path = os.path.normpath(file_path)
        
        # 转换为绝对路径
        if not os.path.isabs(file_path):
            # 相对路径，相对于当前工作目录
            file_path = os.path.abspath(file_path)
        
        # 验证文件存在
        if not os.path.exists(file_path):
            raise Exception(f"本地文件不存在: {file_path}")
        
        # 验证是文件而不是目录
        if not os.path.isfile(file_path):
            raise Exception(f"路径不是文件: {file_path}")
        
        # 验证文件格式
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ('.fbx', '.glb', '.gltf', '.obj'):
            print(f"[Node] 警告: 文件扩展名 '{file_ext}' 可能不受支持，支持的格式: .fbx, .glb, .gltf, .obj")
        
        print(f"[Node] 使用本地文件: {file_path}")
        return file_path

    def _download_model(self, url_or_path: str) -> str:
        """从URL下载3D模型文件或处理本地文件路径"""
        print(f"[Node] 处理模型输入: {url_or_path}")
        
        # 判断是URL还是本地路径
        if self._is_url(url_or_path):
            print(f"[Node] 检测到URL，开始下载...")
            return self._download_from_url(url_or_path)
        else:
            print(f"[Node] 检测到本地路径，验证文件...")
            return self._handle_local_path(url_or_path)
    
    def _download_from_url(self, url: str) -> str:
        """从URL下载3D模型文件"""
        print(f"[Node] 从URL下载模型: {url}")
        
        # 创建下载目录
        download_dir = os.path.join(folder_paths.get_output_directory(), "downloads", "3d_models")
        os.makedirs(download_dir, exist_ok=True)
        
        # 生成唯一的文件路径
        file_path = self._generate_unique_filename(download_dir, url)
        
        try:
            import requests
            import warnings
            from requests.packages.urllib3.exceptions import InsecureRequestWarning
            warnings.simplefilter('ignore', InsecureRequestWarning)
            
            print(f"[Node] 使用requests库下载...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            }
            
            # 检查是否为需要认证的dreammaker域名
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            if 'dreammaker.netease.com' in parsed_url.netloc:
                print(f"[Node] 检测到dreammaker域名，添加认证头...")
                headers['X-Auth-User'] = 'comfyui-dm-user'
                print(f"[Node] 已添加认证头")
            
            print(f"[Node] 下载到: {file_path}")
            response = requests.get(url, headers=headers, timeout=300, verify=False)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[Node] 下载成功: {len(response.content) / (1024*1024):.2f} MB")
                
                # 验证文件格式
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ('.fbx', '.glb', '.gltf', '.obj'):
                    print(f"[Node] 警告: 文件扩展名 '{file_ext}' 可能不受支持，支持的格式: .fbx, .glb, .gltf, .obj")
                
                return file_path
            else:
                raise Exception(f"HTTP错误 {response.status_code}: {response.reason}")
                
        except ImportError:
            raise Exception("需要安装requests库: pip install requests")
        except Exception as e:
            # 清理可能的不完整文件
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise Exception(f"下载失败: {str(e)}")

    def _get_model_bbox(self, mesh_path: str, blender_path: str) -> Vector3:
        """获取模型的初始包围盒尺寸"""
        print(f"[Node] 获取模型初始包围盒: {mesh_path}")
        
        initial_script = r"""
import bpy, sys, json, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--")+1:]
in_path, bbox_path = argv

bpy.ops.wm.read_factory_settings(use_empty=True)

lower = in_path.lower()
if lower.endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=in_path, global_scale=1.0, axis_forward='-Z', axis_up='Y')
elif lower.endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=in_path)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
gmin = Vector((math.inf, math.inf, math.inf))
gmax = Vector((-math.inf, -math.inf, -math.inf))

for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        gmin.x, gmin.y, gmin.z = min(gmin.x, wpt.x), min(gmin.y, wpt.y), min(gmin.z, wpt.z)
        gmax.x, gmax.y, gmax.z = max(gmax.x, wpt.x), max(gmax.y, wpt.y), max(gmax.z, wpt.z)

size = [gmax[i] - gmin[i] for i in range(3)]
with open(bbox_path, "w", encoding="utf-8") as f:
    json.dump({"size": size}, f)
"""
        
        with tempfile.TemporaryDirectory() as td:
            script_path = os.path.join(td, "get_bbox.py")
            bbox_path = os.path.join(td, "initial_bbox.json")
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(initial_script)
            
            cmd = [blender_path, "-b", "-noaudio", "--python", script_path, "--", mesh_path, bbox_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if proc.returncode != 0:
                raise Exception(f"获取包围盒失败：\n{proc.stderr}")
            
            with open(bbox_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                size = data["size"]
                return Vector3(size[0], size[1], size[2])

    def process(self, mesh_path: str, target_size_x: float, target_size_y: float, target_size_z: float, 
                force_exact_alignment: bool, blender_path: str, output_name: str, model_url: str = ""):
        
        print(f"\n=== VVL智能模型缩放器 开始处理 ===")
        
        # 处理模型输入：优先使用URL下载
        actual_mesh_path = mesh_path
        if model_url and model_url.strip():
            print(f"[Node] 检测到模型URL，将下载: {model_url}")
            try:
                actual_mesh_path = self._download_model(model_url.strip())
                print(f"[Node] URL下载成功，使用下载的模型: {actual_mesh_path}")
            except Exception as e:
                raise Exception(f"模型下载失败: {str(e)}")
        else:
            print(f"[Node] 使用本地模型: {mesh_path}")
            if not mesh_path or not mesh_path.strip():
                raise Exception("请提供 mesh_path 或 model_url 参数")
            actual_mesh_path = mesh_path
        
        print(f"[Node] 最终使用的模型: {actual_mesh_path}")
        print(f"[Node] 目标尺寸: ({target_size_x:.1f}, {target_size_y:.1f}, {target_size_z:.1f})")
        print(f"[Node] 强制精确对齐: {force_exact_alignment}")
        print(f"[Node] 缩放将烘焙到顶点")
        
        # 验证模型文件
        if not os.path.exists(actual_mesh_path):
            raise Exception(f"模型文件不存在: {actual_mesh_path}")
        
        ext = os.path.splitext(actual_mesh_path)[1].lower()
        if ext not in (".fbx", ".glb", ".gltf"):
            raise Exception("仅支持 .fbx / .glb / .gltf 输入。")

        self._ensure_blender(blender_path)

        # 获取模型初始包围盒
        print(f"[Node] 获取模型初始包围盒...")
        initial_bbox = self._get_model_bbox(actual_mesh_path, blender_path)
        print(f"[Node] 初始包围盒尺寸: ({initial_bbox.x:.2f}, {initial_bbox.y:.2f}, {initial_bbox.z:.2f})")
        
        # 计算智能缩放因子
        print(f"[Node] 计算智能缩放因子...")
        model_name = os.path.splitext(os.path.basename(actual_mesh_path))[0]
        
        model_info = ModelInfo(
            name=model_name,
            bounding_box_size=initial_bbox,
            target_scale=Vector3(target_size_x, target_size_y, target_size_z)
        )
        
        scaling_config = ScalingConfig(
            force_exact_alignment=force_exact_alignment,
            standard_size=1.0,
            scale_range_min=0.001,
            scale_range_max=1000.0
        )
        
        scaler = SmartScaler(scaling_config)
        final_scale = scaler.calculate_smart_scale(model_info)
        
        print(f"[Node] 计算得到的最终缩放: ({final_scale.x:.3f}, {final_scale.y:.3f}, {final_scale.z:.3f})")
        
        # 应用缩放并生成模型
        print(f"[Node] 应用缩放并生成模型...")
        out_dir = os.path.join(folder_paths.get_output_directory(), "3d")
        os.makedirs(out_dir, exist_ok=True)
        
        # 处理输出路径，确保目录存在
        out_path = os.path.join(out_dir, output_name)
        out_path = os.path.normpath(out_path)  # 规范化路径，统一使用系统的路径分隔符
        
        # 确保输出目录存在
        out_dir_for_file = os.path.dirname(out_path)
        os.makedirs(out_dir_for_file, exist_ok=True)

        # 写入临时的 Blender 脚本与输出路径
        with tempfile.TemporaryDirectory() as td:
            script_path = os.path.join(td, "blender_scale_bbox.py")
            bbox_path = os.path.join(td, "bbox.json")
            scale_info_path = os.path.join(td, "scale_info.json")
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(self.BLENDER_SCRIPT)

            cmd = [
                blender_path, "-b", "-noaudio",
                "--python", script_path, "--",
                actual_mesh_path, out_path,
                str(final_scale.x), str(final_scale.y), str(final_scale.z),
                bbox_path, scale_info_path
            ]
            
            print(f"[Node] 执行Blender命令...")
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # 打印Blender输出以便调试
            if proc.stdout:
                print(f"[Node] Blender输出:\n{proc.stdout}")
            if proc.stderr:
                print(f"[Node] Blender错误:\n{proc.stderr}")
            
            if proc.returncode != 0:
                raise Exception(f"Blender 执行失败 (返回码: {proc.returncode})：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            # 检查文件是否存在
            if not os.path.exists(bbox_path):
                raise Exception(f"Blender没有生成bbox文件: {bbox_path}")
            if not os.path.exists(scale_info_path):
                raise Exception(f"Blender没有生成scale_info文件: {scale_info_path}")

            # 读取结果
            with open(bbox_path, "r", encoding="utf-8") as f:
                bbox_json = f.read()
                
            with open(scale_info_path, "r", encoding="utf-8") as f:
                scale_info_data = json.load(f)
            
            # 添加算法信息到scale_info
            scale_info_data["algorithm_info"] = {
                "method": "smart_scaling",
                "target_size": [target_size_x, target_size_y, target_size_z],
                "force_exact_alignment": force_exact_alignment,
                "initial_bbox": [initial_bbox.x, initial_bbox.y, initial_bbox.z],
                "calculated_scale": [final_scale.x, final_scale.y, final_scale.z]
            }
            
            # 添加输入源信息
            is_url_input = bool(model_url and model_url.strip())
            final_input = model_url.strip() if is_url_input else mesh_path
            scale_info_data["input_info"] = {
                "source_type": "url_download" if is_url_input else "local_file",
                "original_mesh_path": mesh_path,
                "actual_mesh_path": actual_mesh_path,
                "model_url": model_url.strip() if model_url else None,
                "is_url": is_url_input,
                "final_input": final_input
            }
            
            # 打印材质信息
            print(f"[Node] 材质信息:")
            print(f"  - 材质数量: {scale_info_data.get('material_count', 0)}")
            print(f"  - 贴图数量: {scale_info_data.get('texture_count', 0)}")
            print(f"  - 材质保留: {scale_info_data.get('materials_preserved', False)}")
            
            scale_info_json = json.dumps(scale_info_data, indent=2, ensure_ascii=False)
        
        print(f"[Node] 处理完成！输出文件: {out_path}")
        print(f"=== VVL智能模型缩放器 结束 ===\n")
        
        return (out_path, bbox_json, scale_info_json)


class BlenderSmartModelScalerBatch:
    """
    VVL智能3D模型批量缩放器
    从JSON输入中批量处理多个3D模型，支持多线程并行处理
    
    输入：包含objects数组的JSON，每个object需要包含name、scale和3d_url字段
    输出：保持原始JSON结构，更新3d_url为处理后的本地文件路径，更新rotation为计算得到的对齐旋转值
    
    3d_url字段支持：
    • HTTP/HTTPS URL：自动下载模型文件
    • 本地绝对路径：如 C:/models/model.fbx 或 /home/user/models/model.glb
    • 本地相对路径：如 ./models/model.fbx 或 ../resources/model.glb
    • file:// 协议：如 file:///C:/models/model.fbx
    
    特性：
    - 保留JSON中的所有原始字段和结构
    - 更新成功处理的对象的3d_url为本地文件路径
    - 自动计算模型对齐旋转，并更新rotation字段为[x, y, z]度数
    - 处理失败的对象保持原始字段不变
    - 支持任意字段（如camera、subject、task_id、position等）
    - 每个对象输出到独立子目录（batch_3d/obj_XXX/），避免贴图文件冲突
    - 默认使用4个并行线程以平衡处理速度和稳定性
    - 完全保留模型的材质和贴图信息
    - 使用最薄轴对齐算法，将模型的三个维度按大小匹配到参考box的对应维度
    
    对齐算法说明：
    - 参考模型：在Blender中创建一个标准立方体(size=2)，应用JSON中的scale和rotation值
    - 例如：scale=[1,2,3], rotation=[90,-65,0] → 创建的Box会同时具有缩放和旋转
    - Blender会自动计算出旋转后的dimensions（考虑了旋转对包围盒的影响）
    - 目标模型：输入的3D模型
    - 算法：
      1. 将参考Box和目标模型的dimensions从小到大排序
      2. 建立轴映射关系（最小→最小，中等→中等，最大→最大）
      3. 检查是否已经对齐（轴映射为恒等映射X→X, Y→Y, Z→Z）
      4. 如果已对齐，保留原始rotation值；否则计算所需的旋转
    - 结果：
      - 已对齐的模型：完全保留原始rotation字段不变（无论是什么值或是否存在）
      - 需要对齐的模型：rotation字段更新为计算得到的旋转角度[x°, y°, z°]
    
    输出目录结构：
    ComfyUI/output/3d/batch_3d/
    ├── obj_000/
    │   ├── 000_模型名称.fbx
    │   └── textures/
    ├── obj_001/
    │   ├── 001_模型名称.fbx
    │   └── textures/
    """
    
    # 包含对齐功能的 Blender 脚本
    BLENDER_SCRIPT_WITH_ALIGNMENT = r"""
import bpy, sys, json, math
from mathutils import Vector, Matrix

# 数值稳定性阈值
_EPS = 1e-8

argv = sys.argv[sys.argv.index("--")+1:]
in_path, out_path, sx, sy, sz, ref_scale_x, ref_scale_y, ref_scale_z, ref_rot_x, ref_rot_y, ref_rot_z, bbox_path, scale_info_path, alignment_path = argv
sx, sy, sz = float(sx), float(sy), float(sz)
ref_scale_x, ref_scale_y, ref_scale_z = float(ref_scale_x), float(ref_scale_y), float(ref_scale_z)  # 参考box的scale值
ref_rot_x, ref_rot_y, ref_rot_z = float(ref_rot_x), float(ref_rot_y), float(ref_rot_z)  # 参考box的rotation值（度）

print(f"[Blender] 开始处理模型: {in_path}")
print(f"[Blender] 应用缩放: sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}")
print(f"[Blender] 参考Box的Scale: ({ref_scale_x:.1f}, {ref_scale_y:.1f}, {ref_scale_z:.1f})")
print(f"[Blender] 参考Box的Rotation: ({ref_rot_x:.1f}°, {ref_rot_y:.1f}°, {ref_rot_z:.1f}°)")

bpy.ops.wm.read_factory_settings(use_empty=True)

# 导入模型
lower = in_path.lower()
print(f"[Blender] 导入文件类型: {lower}")
try:
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=in_path, global_scale=1.0, axis_forward='-Z', axis_up='Y',
                                use_image_search=True, use_custom_props=True)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=in_path, import_pack_images=True)
    print(f"[Blender] 导入成功")
except Exception as e:
    print(f"[Blender] 导入警告: {str(e)}")
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=in_path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=in_path)
    print(f"[Blender] 使用基本导入模式成功")

# 修复GLB导入后的纹理路径问题
if lower.endswith((".glb", ".gltf")):
    print(f"[Blender] 修复GLB纹理路径...")
    image_counter = 0
    for img in bpy.data.images:
        if img.source == 'FILE' and not img.filepath:
            fake_path = f"texture_{image_counter:03d}.png"
            img.filepath = fake_path
            image_counter += 1
        elif img.packed_file and not img.filepath:
            fake_path = f"packed_texture_{image_counter:03d}.png"
            img.filepath = fake_path
            image_counter += 1
    print(f"[Blender] 纹理路径修复完成，处理了 {image_counter} 个图像")

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
print(f"[Blender] 找到 {len(meshes)} 个网格对象")

# 计算原始包围盒
original_gmin = Vector(( math.inf,  math.inf,  math.inf))
original_gmax = Vector((-math.inf, -math.inf, -math.inf))
for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        original_gmin.x = min(original_gmin.x, wpt.x)
        original_gmin.y = min(original_gmin.y, wpt.y)
        original_gmin.z = min(original_gmin.z, wpt.z)
        original_gmax.x = max(original_gmax.x, wpt.x)
        original_gmax.y = max(original_gmax.y, wpt.y)
        original_gmax.z = max(original_gmax.z, wpt.z)

original_size = [original_gmax[i] - original_gmin[i] for i in range(3)]
print(f"[Blender] 原始包围盒尺寸: ({original_size[0]:.2f}, {original_size[1]:.2f}, {original_size[2]:.2f})")

# ===== 对齐计算开始 =====
print(f"\n[Blender] 开始计算对齐旋转...")

# 创建参考 Box
print(f"[Blender] 创建参考 Box...")
# 创建默认大小的立方体
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
ref_box = bpy.context.active_object
ref_box.name = "Reference_Box"

# 直接应用 scale 值（与JSON中的scale字段对应）
ref_box.scale = (ref_scale_x, ref_scale_y, ref_scale_z)

# 应用rotation值（需要转换为弧度）
ref_box.rotation_euler = (math.radians(ref_rot_x), math.radians(ref_rot_y), math.radians(ref_rot_z))

bpy.context.view_layer.update()

# 获取参考 Box 的实际 dimensions
ref_dims = ref_box.dimensions.copy()
ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]

print(f"[Blender] 参考 Box 创建完成:")
print(f"  - Scale: ({ref_scale_x}, {ref_scale_y}, {ref_scale_z})")
print(f"  - Rotation: ({ref_rot_x}°, {ref_rot_y}°, {ref_rot_z}°)")
print(f"  - 实际 Dimensions: ({ref_sizes[0]:.2f}, {ref_sizes[1]:.2f}, {ref_sizes[2]:.2f})")

# 获取参考 Box 的旋转矩阵（在删除之前）
ref_loc, ref_rot, ref_scale = ref_box.matrix_world.decompose()
ref_rot_matrix = ref_rot.to_matrix()

# 获取目标模型的尺寸（当前模型）
tgt_sizes = original_size

# 删除参考 Box（我们已经获取了需要的信息）
bpy.data.objects.remove(ref_box, do_unlink=True)

# 对尺寸进行排序，得到从小到大的轴索引
ref_sorted_indices = sorted(range(3), key=lambda i: ref_sizes[i])  # [最小轴, 中轴, 最大轴]
tgt_sorted_indices = sorted(range(3), key=lambda i: tgt_sizes[i])

# 创建轴映射：目标的第i个轴应该映射到参考的第j个轴
axis_mapping = [None, None, None]
for rank in range(3):  # rank: 0=最小, 1=中等, 2=最大
    tgt_axis = tgt_sorted_indices[rank]
    ref_axis = ref_sorted_indices[rank]
    axis_mapping[tgt_axis] = ref_axis

print(f"  参考box尺寸: X={ref_sizes[0]:.1f} Y={ref_sizes[1]:.1f} Z={ref_sizes[2]:.1f}")
print(f"  目标模型尺寸: X={tgt_sizes[0]:.1f} Y={tgt_sizes[1]:.1f} Z={tgt_sizes[2]:.1f}")
print(f"  参考轴排序: {['XYZ'[i] for i in ref_sorted_indices]} (小→大)")
print(f"  目标轴排序: {['XYZ'[i] for i in tgt_sorted_indices]} (小→大)")
print(f"  轴映射: X→{'XYZ'[axis_mapping[0]]}, Y→{'XYZ'[axis_mapping[1]]}, Z→{'XYZ'[axis_mapping[2]]}")

# 检查是否已经对齐（恒等映射）
is_already_aligned = (axis_mapping[0] == 0 and axis_mapping[1] == 1 and axis_mapping[2] == 2)
if is_already_aligned:
    print(f"  [对齐检查] 模型已经与参考box平行对齐，无需旋转！")

# 如果已经对齐，直接设置为无旋转
if is_already_aligned:
    rotation_degrees = [0.0, 0.0, 0.0]
    rotation_radians = [0.0, 0.0, 0.0]
    print(f"  计算的旋转角度: X=0.0° Y=0.0° Z=0.0° (保持原始方向)")
else:
    # 基于轴映射构建旋转矩阵
    # 创建一个置换矩阵
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
        last_mapping = axis_mapping[2]
        perm_matrix[last_mapping][2] *= -1

    print(f"  置换矩阵: \n{perm_matrix}")

    # 最终旋转 = 参考旋转 × 置换旋转
    # ref_rot_matrix 已经在前面获取了（包含了参考Box的旋转）
    final_rotation_matrix = ref_rot_matrix @ perm_matrix

    # 转换为欧拉角（度）
    euler = final_rotation_matrix.to_euler()
    rotation_degrees = [math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)]
    rotation_radians = [euler.x, euler.y, euler.z]

    print(f"  计算的旋转角度: X={rotation_degrees[0]:.1f}° Y={rotation_degrees[1]:.1f}° Z={rotation_degrees[2]:.1f}°")

# 保存对齐信息
alignment_info = {
    "rotation_degrees": rotation_degrees,
    "rotation_radians": rotation_radians,
    "axis_mapping": {
        "X": "XYZ"[axis_mapping[0]],
        "Y": "XYZ"[axis_mapping[1]],
        "Z": "XYZ"[axis_mapping[2]]
    },
    "ref_box": {
        "scale": [ref_scale_x, ref_scale_y, ref_scale_z],
        "rotation": [ref_rot_x, ref_rot_y, ref_rot_z],
        "dimensions": ref_sizes
    },
    "model_sizes": tgt_sizes,
    "is_already_aligned": is_already_aligned
}

# 只有在需要旋转时才添加置换矩阵信息
if not is_already_aligned:
    alignment_info["permutation_matrix"] = [[perm_matrix[i][j] for j in range(3)] for i in range(3)]

# ===== 对齐计算结束 =====

# 应用缩放（但不应用旋转）
for o in meshes:
    o.select_set(True)
    original_scale = o.scale.copy()
    o.scale = (o.scale[0]*sx, o.scale[1]*sy, o.scale[2]*sz)
    print(f"[Blender] 对象 '{o.name}' 缩放: {original_scale} -> {o.scale}")

# 烘焙缩放到顶点
bpy.context.view_layer.objects.active = meshes[0] if meshes else None
if meshes:
    print(f"[Blender] 烘焙缩放到顶点...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# 计算缩放后的全局 AABB
gmin = Vector(( math.inf,  math.inf,  math.inf))
gmax = Vector((-math.inf, -math.inf, -math.inf))
for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        gmin.x = min(gmin.x, wpt.x)
        gmin.y = min(gmin.y, wpt.y)
        gmin.z = min(gmin.z, wpt.z)
        gmax.x = max(gmax.x, wpt.x)
        gmax.y = max(gmax.y, wpt.y)
        gmax.z = max(gmax.z, wpt.z)

final_size = [gmax[i] - gmin[i] for i in range(3)]
print(f"[Blender] 缩放后包围盒尺寸: ({final_size[0]:.2f}, {final_size[1]:.2f}, {final_size[2]:.2f})")

# 准备输出数据
bbox = {"min":[gmin.x,gmin.y,gmin.z], "max":[gmax.x,gmax.y,gmax.z], "size": final_size}
material_count = 0
texture_count = 0

# 检查材质信息
try:
    for obj in meshes:
        if obj.data.materials:
            material_count += len(obj.data.materials)
            for mat in obj.data.materials:
                if mat and hasattr(mat, 'use_nodes') and mat.use_nodes and hasattr(mat, 'node_tree'):
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and hasattr(node, 'image') and node.image:
                            texture_count += 1
except Exception as e:
    print(f"[Blender] 材质检查警告: {str(e)}")

scale_info = {
    "applied_scale": [sx, sy, sz],
    "original_size": original_size,
    "final_size": final_size,
    "size_change_ratio": [final_size[i]/original_size[i] if original_size[i] > 0 else 1.0 for i in range(3)],
    "mesh_count": len(meshes),
    "scale_applied_to_vertices": True,
    "material_count": material_count,
    "texture_count": texture_count,
    "materials_preserved": (texture_count > 0 or material_count > 0)
}

# 保存临时结果
with open(bbox_path, "w", encoding="utf-8") as f:
    json.dump(bbox, f)

with open(scale_info_path, "w", encoding="utf-8") as f:
    json.dump(scale_info, f)

with open(alignment_path, "w", encoding="utf-8") as f:
    json.dump(alignment_info, f)

# 导出模型
print(f"[Blender] 导出到: {out_path}")
out_path_lower = out_path.lower()
if out_path_lower.endswith(('.glb', '.gltf')):
    try:
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format='GLB' if out_path_lower.endswith('.glb') else 'GLTF_SEPARATE',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_colors=True,
            export_cameras=False,
            export_extras=True,
            export_yup=True
        )
        print(f"[Blender] GLB/GLTF导出成功")
    except Exception as e:
        print(f"[Blender] GLB/GLTF导出失败: {str(e)}")
        bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB' if out_path_lower.endswith('.glb') else 'GLTF_SEPARATE')
else:
    try:
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            axis_forward='-Z', 
            axis_up='Y',
            path_mode='COPY',
            embed_textures=True,
            use_custom_props=True,
            use_mesh_modifiers=True,
            use_armature_deform_only=False,
            add_leaf_bones=False,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            use_metadata=True,
            global_scale=1.0
        )
        print(f"[Blender] FBX导出成功")
    except Exception as e:
        print(f"[Blender] FBX导出失败: {str(e)}")
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            axis_forward='-Z', 
            axis_up='Y',
            path_mode='COPY',
            embed_textures=True
        )

print(f"[Blender] 处理完成！")
"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": "", "tooltip": "包含objects数组的JSON数据，每个object包含name、scale和3d_url字段\n• 3d_url支持HTTP/HTTPS链接和本地文件路径\n• 本地路径示例：C:/models/model.fbx, ./models/model.glb, file:///path/to/model.fbx"}),
                "force_exact_alignment": ("BOOLEAN", {"default": True, "tooltip": "强制精确对齐：启用时允许各轴独立缩放以匹配目标尺寸（可能造成拉伸变形），禁用时保持模型原始比例进行等比缩放"}),
                "blender_path": ("STRING", {"default": "blender", "tooltip": "Blender 可执行文件路径，默认使用系统PATH中的 'blender' 命令，也可指定完整路径"}),
            },
            "optional": {
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "最大并行处理线程数。每个模型有独立输出目录，理论上支持高并发。建议根据CPU核心数和内存大小调整"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_json",)
    FUNCTION = "process_batch"
    CATEGORY = "VVL/3D"

    def __init__(self):
        # 创建单个模型处理器实例，复用其方法
        self.single_processor = BlenderSmartModelScaler()
        
    def _process_with_alignment(self, mesh_path: str, ref_scale_x: float, ref_scale_y: float, 
                              ref_scale_z: float, ref_rot_x: float, ref_rot_y: float, ref_rot_z: float,
                              sx: float, sy: float, sz: float,
                              blender_path: str, output_path: str):
        """使用带对齐功能的Blender脚本处理模型"""
        
        with tempfile.TemporaryDirectory() as td:
            script_path = os.path.join(td, "blender_scale_align.py")
            bbox_path = os.path.join(td, "bbox.json")
            scale_info_path = os.path.join(td, "scale_info.json")
            alignment_path = os.path.join(td, "alignment.json")
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(self.BLENDER_SCRIPT_WITH_ALIGNMENT)
            
            cmd = [
                blender_path, "-b", "-noaudio",
                "--python", script_path, "--",
                mesh_path, output_path,
                str(sx), str(sy), str(sz),
                str(ref_scale_x), str(ref_scale_y), str(ref_scale_z),
                str(ref_rot_x), str(ref_rot_y), str(ref_rot_z),
                bbox_path, scale_info_path, alignment_path
            ]
            
            print(f"[Batch] 执行带对齐功能的Blender命令...")
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if proc.stdout:
                print(f"[Batch] Blender输出:\n{proc.stdout}")
            if proc.stderr:
                print(f"[Batch] Blender错误:\n{proc.stderr}")
            
            if proc.returncode != 0:
                raise Exception(f"Blender 执行失败 (返回码: {proc.returncode})")
            
            # 读取结果
            with open(bbox_path, "r", encoding="utf-8") as f:
                bbox_data = json.load(f)
                
            with open(scale_info_path, "r", encoding="utf-8") as f:
                scale_info_data = json.load(f)
                
            with open(alignment_path, "r", encoding="utf-8") as f:
                alignment_data = json.load(f)
            
            return bbox_data, scale_info_data, alignment_data
        
    def _extract_file_extension(self, url: str) -> str:
        """从URL中提取文件扩展名"""
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        
        # 获取文件名
        filename = os.path.basename(path)
        
        # 提取扩展名
        ext = os.path.splitext(filename)[1].lower()
        
        # 确保扩展名有效
        if ext not in ['.fbx', '.glb', '.gltf', '.obj']:
            # 尝试从URL推断
            if 'fbx' in url.lower():
                ext = '.fbx'
            elif 'glb' in url.lower():
                ext = '.glb'
            elif 'gltf' in url.lower():
                ext = '.gltf'
            else:
                ext = '.glb'  # 默认使用GLB
                
        return ext
    
    def _process_single_object(self, obj_data: dict, index: int, force_exact_alignment: bool, 
                              blender_path: str) -> dict:
        """处理单个对象"""
        result = {
            "index": index,
            "name": obj_data.get("name", f"object_{index}"),
            "success": False,
            "error": None,
            "output_path": None,
            "bbox": None,
            "scale_info": None,
            "alignment_info": None,
            "rotation": None
        }
        
        try:
            # 获取必要的字段
            name = obj_data.get("name", f"object_{index}")
            scale = obj_data.get("scale", [1.0, 1.0, 1.0])
            rotation = obj_data.get("rotation", [0.0, 0.0, 0.0])  # 获取原始rotation
            url = obj_data.get("3d_url", "")
            
            if not url or url.strip() == "" or "None" in url:
                print(f"[Batch] 对象 [{index}] '{name}' 的3d_url无效，跳过处理: {url}")
                result["error"] = f"3d_url无效: {url}"
                return result
                
            if not isinstance(scale, list) or len(scale) < 3:
                raise Exception(f"对象 '{name}' 的scale字段格式错误，期望[x, y, z]")
                
            if not isinstance(rotation, list) or len(rotation) < 3:
                print(f"[Batch] 对象 '{name}' 的rotation字段格式不正确或缺失，使用默认值[0, 0, 0]")
                rotation = [0.0, 0.0, 0.0]
            
            # 提取文件扩展名
            file_ext = self._extract_file_extension(url)
            
            # 生成输出文件名（使用索引确保唯一性）
            safe_name = re.sub(r'[^\w\-_\. ]', '_', name)  # 清理文件名中的特殊字符
            output_filename = f"{index:03d}_{safe_name}{file_ext}"
            
            # 设置目标尺寸（scale值乘以100作为基准尺寸）
            target_size_x = float(scale[0])
            target_size_y = float(scale[1])
            target_size_z = float(scale[2])
            
            print(f"\n[Batch] 处理对象 [{index}] '{name}':")
            print(f"  - URL: {url}")
            print(f"  - 目标缩放: {scale}")
            print(f"  - 原始旋转: {rotation}")
            print(f"  - 输出文件: {output_filename}")
            print(f"  - 文件格式: {file_ext}")
            
            # 调用处理器（现在使用带对齐功能的版本）
            try:
                # 为每个处理生成唯一的输出子目录，避免贴图文件冲突
                unique_subdir = os.path.join("batch_3d", f"obj_{index:03d}")
                output_filename_with_subdir = os.path.join(unique_subdir, output_filename)
                
                # 处理模型输入（URL下载或本地文件）
                print(f"[Batch] 处理模型: {url}")
                downloaded_path = self.single_processor._download_model(url)
                
                if not downloaded_path or not os.path.exists(downloaded_path):
                    raise Exception(f"模型下载失败或文件不存在: {downloaded_path}")
                
                # 获取模型初始包围盒
                print(f"[Batch] 获取模型初始包围盒...")
                initial_bbox = self.single_processor._get_model_bbox(downloaded_path, blender_path)
                
                if not initial_bbox:
                    raise Exception(f"获取模型包围盒失败")
                
                # 计算智能缩放因子
                model_info = ModelInfo(
                    name=name,
                    bounding_box_size=initial_bbox,
                    target_scale=Vector3(target_size_x, target_size_y, target_size_z)
                )
                
                scaling_config = ScalingConfig(
                    force_exact_alignment=force_exact_alignment,
                    standard_size=1.0,
                    scale_range_min=0.001,
                    scale_range_max=1000.0
                )
                
                scaler = SmartScaler(scaling_config)
                final_scale = scaler.calculate_smart_scale(model_info)
                
                # 准备输出路径
                out_dir = os.path.join(folder_paths.get_output_directory(), "3d")
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, output_filename_with_subdir)
                output_path = os.path.normpath(output_path)
                out_dir_for_file = os.path.dirname(output_path)
                os.makedirs(out_dir_for_file, exist_ok=True)
                
                # 使用带对齐功能的处理
                # 传递原始的scale和rotation值
                bbox_data, scale_info_data, alignment_data = self._process_with_alignment(
                    downloaded_path,
                    float(scale[0]), float(scale[1]), float(scale[2]),  # 原始scale值
                    float(rotation[0]), float(rotation[1]), float(rotation[2]),  # 原始rotation值
                    final_scale.x, final_scale.y, final_scale.z,
                    blender_path,
                    output_path
                )
                
                if not bbox_data or not scale_info_data or not alignment_data:
                    raise Exception(f"Blender处理返回了空数据")
                
                # 添加算法信息到scale_info
                scale_info_data["algorithm_info"] = {
                    "method": "smart_scaling_with_alignment",
                    "target_size": [target_size_x, target_size_y, target_size_z],
                    "force_exact_alignment": force_exact_alignment,
                    "initial_bbox": [initial_bbox.x, initial_bbox.y, initial_bbox.z],
                    "calculated_scale": [final_scale.x, final_scale.y, final_scale.z]
                }
                
                # 添加输入源信息
                is_url_input = self.single_processor._is_url(url)
                scale_info_data["input_info"] = {
                    "source_type": "url_download" if is_url_input else "local_file",
                    "original_input": url,
                    "resolved_path": downloaded_path,
                    "is_url": is_url_input
                }
                
                result["success"] = True
                result["output_path"] = output_path
                result["bbox"] = bbox_data
                result["scale_info"] = scale_info_data
                result["alignment_info"] = alignment_data
                result["rotation"] = alignment_data["rotation_degrees"]
                
                # 打印输出路径信息
                print(f"[Batch] 对象 [{index}] 处理成功")
                print(f"  - 输出路径: {output_path}")
                print(f"  - 文件存在: {os.path.exists(output_path)}")
                
                # 显示对齐状态
                if alignment_data.get('is_already_aligned', False):
                    print(f"  - 对齐状态: 已经与参考box平行对齐，保持原始方向")
                    print(f"  - 旋转角度: X=0.0° Y=0.0° Z=0.0°")
                else:
                    print(f"  - 对齐状态: 需要旋转以对齐参考box")
                    print(f"  - 计算的旋转: X={alignment_data['rotation_degrees'][0]:.1f}° Y={alignment_data['rotation_degrees'][1]:.1f}° Z={alignment_data['rotation_degrees'][2]:.1f}°")
                
                # 打印贴图处理信息
                if scale_info_data.get("texture_count", 0) > 0:
                    print(f"  - 贴图数量: {scale_info_data['texture_count']}")
                
            except Exception as e:
                result["error"] = str(e)
                print(f"[Batch] 对象 [{index}] '{name}' 处理失败: {str(e)}")
                
        except Exception as e:
            result["error"] = str(e)
            print(f"[Batch] 对象 [{index}] 处理失败: {str(e)}")
            
        return result
    
    def process_batch(self, json_text: str, force_exact_alignment: bool, blender_path: str, 
                     max_workers: int = 4, **kwargs):
        """批量处理JSON中的所有3D模型"""
        
        print(f"\n=== VVL智能模型批量缩放器 开始处理 ===")
        print(f"[Batch] 最大并行线程数: {max_workers}")
        
        try:
            # 解析JSON输入
            data = json.loads(json_text)
            
            # 检查objects字段
            if 'objects' not in data or not isinstance(data['objects'], list):
                raise Exception("JSON必须包含'objects'数组")
                
            objects = data['objects']
            if not objects:
                raise Exception("objects数组为空")
                
            print(f"[Batch] 找到 {len(objects)} 个待处理对象")
            
            # 创建基础输出目录
            base_output_dir = os.path.join(folder_paths.get_output_directory(), "3d")
            os.makedirs(base_output_dir, exist_ok=True)
            print(f"[Batch] 基础输出目录: {base_output_dir}")
            print(f"[Batch] 每个对象将创建独立子目录以避免贴图冲突")
            
            # 验证Blender路径
            self.single_processor._ensure_blender(blender_path)
            
            # 使用线程池进行并行处理
            import concurrent.futures
            
            results = []
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {}
                for index, obj in enumerate(objects):
                    future = executor.submit(
                        self._process_single_object,
                        obj,
                        index,
                        force_exact_alignment,
                        blender_path
                    )
                    future_to_index[future] = index
                
                # 获取结果
                for future in concurrent.futures.as_completed(future_to_index):
                    result = future.result()
                    results.append(result)
                    
                    # 打印进度
                    completed = len(results)
                    print(f"[Batch] 进度: {completed}/{len(objects)} ({completed/len(objects)*100:.1f}%)")
            
            # 按索引排序结果
            results.sort(key=lambda x: x['index'])
            
            # 统计处理结果
            success_count = sum(1 for r in results if r['success'])
            failed_count = len(results) - success_count
            aligned_count = sum(1 for r in results if r['success'] and r.get('alignment_info', {}).get('is_already_aligned', False))
            rotated_count = success_count - aligned_count
            elapsed_time = time.time() - start_time
            
            # 深拷贝原始数据，保持结构不变
            import copy
            output_data = copy.deepcopy(data)
            
            # 更新每个成功处理的对象的3d_url和rotation字段
            for result in results:
                if result['success'] and result['output_path']:
                    index = result['index']
                    if 0 <= index < len(output_data['objects']):
                        # 更新3d_url字段
                        output_data['objects'][index]['3d_url'] = result['output_path']
                        
                        # 更新rotation字段
                        if result.get('alignment_info') and result.get('rotation'):
                            is_aligned = result['alignment_info'].get('is_already_aligned', False)
                            if is_aligned:
                                # 如果已经对齐，完全保留原始rotation字段（不做任何修改）
                                original_rotation = output_data['objects'][index].get('rotation', 'undefined')
                                print(f"[Batch] 对象 [{index}] 已对齐，保留原始rotation: {original_rotation}")
                            else:
                                # 需要旋转对齐，使用计算得到的rotation
                                output_data['objects'][index]['rotation'] = result['rotation']
                                print(f"[Batch] 对象 [{index}] 更新rotation: {result['rotation']}")
                        
                        # 可选：添加处理信息到对象（如果需要的话）
                        # output_data['objects'][index]['_processing_info'] = {
                        #     'bbox': result['bbox'],
                        #     'scale_info': result['scale_info'],
                        #     'alignment_info': result['alignment_info']
                        # }
            
            # 不添加任何额外字段，保持原始JSON结构
            
            result_json = json.dumps(output_data, ensure_ascii=False, indent=2)
            
            # 打印处理摘要
            print(f"\n[Batch] 处理完成:")
            print(f"  - 总计: {len(objects)} 个对象")
            print(f"  - 成功: {success_count} 个")
            print(f"    - 已对齐（保持原方向）: {aligned_count} 个")
            print(f"    - 需要旋转对齐: {rotated_count} 个")
            print(f"  - 跳过/失败: {failed_count} 个")
            print(f"  - 耗时: {elapsed_time:.2f} 秒")
            print(f"  - 平均: {elapsed_time/len(objects):.2f} 秒/对象")
            
            # 打印跳过和失败的对象信息
            failed_objects = [r for r in results if not r['success']]
            if failed_objects:
                print(f"\n[Batch] 跳过/失败的对象:")
                for fail in failed_objects:
                    if "3d_url无效" in fail.get('error', ''):
                        print(f"  - [{fail['index']}] {fail['name']}: 跳过（{fail['error']}）")
                    else:
                        print(f"  - [{fail['index']}] {fail['name']}: 失败（{fail['error']}）")
            
            print(f"=== VVL智能模型批量缩放器 结束 ===\n")
            
            return (result_json,)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            print(f"[Batch] 错误: {error_msg}")
            error_result = {"error": error_msg}
            return (json.dumps(error_result, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"批量处理失败: {str(e)}"
            print(f"[Batch] 错误: {error_msg}")
            error_result = {"error": error_msg}
            return (json.dumps(error_result, ensure_ascii=False),)
