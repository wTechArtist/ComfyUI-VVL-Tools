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
                "model_url": ("STRING", {"forceInput": True, "tooltip": "可选：3D模型下载链接，支持 .fbx、.glb、.gltf 格式。如果提供此参数，mesh_path 将被忽略。模型会下载到 ComfyUI 输出目录的 downloads/3d_models 文件夹中"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "bbox_json", "scale_info")
    FUNCTION = "process"
    CATEGORY = "VVL/3D"

    BLENDER_SCRIPT = r"""
import bpy, sys, json, math
from mathutils import Vector

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

    def _download_model(self, url: str) -> str:
        """从URL下载3D模型文件"""
        print(f"[Node] 开始下载模型: {url}")
        
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
                if file_ext not in ('.fbx', '.glb', '.gltf'):
                    print(f"[Node] 警告: 文件扩展名 '{file_ext}' 可能不受支持")
                
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
        out_path = os.path.join(out_dir, output_name)

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
            scale_info_data["input_info"] = {
                "source_type": "url_download" if (model_url and model_url.strip()) else "local_file",
                "original_mesh_path": mesh_path,
                "actual_mesh_path": actual_mesh_path,
                "model_url": model_url.strip() if model_url else None
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
