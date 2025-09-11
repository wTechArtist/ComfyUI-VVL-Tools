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

# 导出 FBX（保留所有材质和贴图信息）
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

# 导出FBX（带错误处理）
try:
    print(f"[Blender] 尝试完整导出...")
    bpy.ops.export_scene.fbx(
        filepath=out_path,
        use_selection=False,
        axis_forward='-Z', 
        axis_up='Y',
        path_mode='COPY',      # 复制贴图文件
        embed_textures=True,   # 嵌入贴图
        use_custom_props=True, # 保留自定义属性
        global_scale=1.0
    )
    print(f"[Blender] 完整导出成功")
except Exception as e:
    print(f"[Blender] 完整导出失败，尝试基本导出: {str(e)}")
    try:
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            axis_forward='-Z', 
            axis_up='Y'
        )
        print(f"[Blender] 基本导出成功")
    except Exception as e2:
        print(f"[Blender] 基本导出也失败: {str(e2)}")
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
                force_exact_alignment: bool, blender_path: str, output_name: str):
        
        print(f"\n=== VVL智能模型缩放器 开始处理 ===")
        print(f"[Node] 输入模型: {mesh_path}")
        print(f"[Node] 目标尺寸: ({target_size_x:.1f}, {target_size_y:.1f}, {target_size_z:.1f})")
        print(f"[Node] 强制精确对齐: {force_exact_alignment}")
        print(f"[Node] 缩放将烘焙到顶点")
        
        ext = os.path.splitext(mesh_path)[1].lower()
        if ext not in (".fbx", ".glb", ".gltf"):
            raise Exception("仅支持 .fbx / .glb / .gltf 输入。")

        self._ensure_blender(blender_path)

        # 获取模型初始包围盒
        print(f"[Node] 获取模型初始包围盒...")
        initial_bbox = self._get_model_bbox(mesh_path, blender_path)
        print(f"[Node] 初始包围盒尺寸: ({initial_bbox.x:.2f}, {initial_bbox.y:.2f}, {initial_bbox.z:.2f})")
        
        # 计算智能缩放因子
        print(f"[Node] 计算智能缩放因子...")
        model_name = os.path.splitext(os.path.basename(mesh_path))[0]
        
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
                mesh_path, out_path,
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
            
            # 打印材质信息
            print(f"[Node] 材质信息:")
            print(f"  - 材质数量: {scale_info_data.get('material_count', 0)}")
            print(f"  - 贴图数量: {scale_info_data.get('texture_count', 0)}")
            print(f"  - 材质保留: {scale_info_data.get('materials_preserved', False)}")
            
            scale_info_json = json.dumps(scale_info_data, indent=2, ensure_ascii=False)
        
        print(f"[Node] 处理完成！输出文件: {out_path}")
        print(f"=== VVL智能模型缩放器 结束 ===\n")
        
        return (out_path, bbox_json, scale_info_json)

# 节点映射在 node_mappings.py 中统一管理