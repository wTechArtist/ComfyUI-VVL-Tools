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

# BlenderSmartModelScaler 作为辅助类，不注册为节点
class BlenderSmartModelScaler:
    """
    VVL智能3D模型缩放器辅助类
    用于 BlenderSmartModelScalerBatch 和 JsonRotationApplier 的内部辅助功能
"""

    def _ensure_blender(self, blender_path: str):
        if shutil.which(blender_path) is None:
            raise Exception(f"找不到 Blender 可执行文件：{blender_path}。请将 Blender 加入 PATH，或在本节点里填写绝对路径。")

    def _generate_unique_filename(self, base_dir: str, url: str) -> str:
        """生成唯一的文件名，避免文件覆盖"""
        parsed_url = urllib.parse.urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        
        if not original_filename or '.' not in original_filename:
            original_filename = "model.fbx"
        
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext not in ('.fbx', '.glb', '.gltf'):
            if 'fbx' in url.lower():
                file_ext = '.fbx'
            elif 'glb' in url.lower():
                file_ext = '.glb'
            elif 'gltf' in url.lower():
                file_ext = '.gltf'
            else:
                file_ext = '.fbx'
        
        base_name = os.path.splitext(original_filename)[0]
        timestamp = str(int(time.time()))
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        unique_name = f"{base_name}_{timestamp}_{url_hash}{file_ext}"
        
        return os.path.join(base_dir, unique_name)

    def _is_url(self, path_or_url: str) -> bool:
        """判断输入是URL还是本地路径"""
        if not path_or_url or not isinstance(path_or_url, str):
            return False
            
        path_or_url = path_or_url.strip()
        
        if path_or_url.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
            return True
            
        if '://' in path_or_url and not path_or_url.startswith('file://'):
            return True
            
        return False
    
    def _handle_local_path(self, file_path: str) -> str:
        """处理本地路径，验证存在性并返回规范化的绝对路径"""
        if not file_path or not isinstance(file_path, str):
            raise Exception("无效的文件路径")
            
        file_path = file_path.strip()
        
        if file_path.startswith('file://'):
            file_path = file_path[7:]
            if os.name == 'nt' and file_path.startswith('/'):
                file_path = file_path[1:]
        
        file_path = os.path.normpath(file_path)
        
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            raise Exception(f"本地文件不存在: {file_path}")
        
        if not os.path.isfile(file_path):
            raise Exception(f"路径不是文件: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ('.fbx', '.glb', '.gltf', '.obj'):
            print(f"[Node] 警告: 文件扩展名 '{file_ext}' 可能不受支持，支持的格式: .fbx, .glb, .gltf, .obj")
        
        print(f"[Node] 使用本地文件: {file_path}")
        return file_path

    def _download_model(self, url_or_path: str) -> str:
        """从URL下载3D模型文件或处理本地文件路径"""
        print(f"[Node] 处理模型输入: {url_or_path}")
        
        if self._is_url(url_or_path):
            print(f"[Node] 检测到URL，开始下载...")
            return self._download_from_url(url_or_path)
        else:
            print(f"[Node] 检测到本地路径，验证文件...")
            return self._handle_local_path(url_or_path)
    
    def _download_from_url(self, url: str) -> str:
        """从URL下载3D模型文件"""
        print(f"[Node] 从URL下载模型: {url}")
        
        download_dir = os.path.join(folder_paths.get_output_directory(), "downloads", "3d_models")
        os.makedirs(download_dir, exist_ok=True)
        
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
                
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ('.fbx', '.glb', '.gltf', '.obj'):
                    print(f"[Node] 警告: 文件扩展名 '{file_ext}' 可能不受支持，支持的格式: .fbx, .glb, .gltf, .obj")
                
                return file_path
            else:
                raise Exception(f"HTTP错误 {response.status_code}: {response.reason}")
                
        except ImportError:
            raise Exception("需要安装requests库: pip install requests")
        except Exception as e:
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


class ModelTransformParameters:
    """
    模型变换参数节点
    
    只负责设置和传递变换参数，不进行实际的模型处理：
    • 输出旋转偏移量（度）
    • 输出缩放乘数
    • 🆕 控制旋转和缩放的应用方式（模型/JSON/两者）
    • 可以连接到 BlenderSmartModelScalerBatch 节点
    • 轻量级节点，无需 Blender
    
    🎯 功能：
    • 设置 X/Y/Z 轴的旋转偏移量
    • 设置 X/Y/Z 轴的缩放乘数
    • 选择应用模式（仅模型/仅JSON/两者）
    • 输出变换参数供 BlenderSmartModelScalerBatch 使用
    
    📝 使用方式：
    1. 设置所需的旋转和缩放参数
    2. 选择应用模式（仅模型/仅JSON/两者）
    3. 将输出连接到 BlenderSmartModelScalerBatch 节点
    4. BlenderSmartModelScalerBatch 会根据应用模式处理这些变换
    
    ⚡ 执行顺序（在 BlenderSmartModelScalerBatch 中）：
    1. 应用额外旋转（根据rotation_apply_mode）
    2. 应用额外缩放（根据scale_apply_mode）
    3. 叠加变换到JSON数据（根据rotation_apply_mode和scale_apply_mode）
    4. 计算包围盒
    5. 基于更新后的JSON计算对齐和智能缩放
    6. 应用智能缩放和对齐旋转
    
    ⚙️ 参数范围：
    • 旋转：-360° 到 +360°，步进 0.1°
    • 缩放：0.001 到 1000，步进 0.001
    • 应用模式：仅模型本身/仅叠加JSON/两者都应用
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rotation_x_offset": ("FLOAT", {
                    "default": 0.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "X轴旋转偏移量（度）\n正值：绕X轴正方向旋转\n负值：绕X轴负方向旋转"
                }),
                "rotation_y_offset": ("FLOAT", {
                    "default": 0.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "Y轴旋转偏移量（度）\n正值：绕Y轴正方向旋转\n负值：绕Y轴负方向旋转"
                }),
                "rotation_z_offset": ("FLOAT", {
                    "default": 0.0, 
                    "min": -360.0, 
                    "max": 360.0, 
                    "step": 0.1,
                    "tooltip": "Z轴旋转偏移量（度）\n正值：绕Z轴正方向旋转\n负值：绕Z轴负方向旋转"
                }),
                "rotation_apply_mode": (["应用在模型本身+叠加在JSON上", "仅应用在模型本身", "仅叠加在JSON上"], {
                    "default": "仅叠加在JSON上",
                    "tooltip": "旋转应用模式：\n• 应用在模型本身+叠加在JSON上：旋转会烘焙到模型顶点，同时更新JSON的rotation字段\n• 仅应用在模型本身：旋转只烘焙到模型顶点，不修改JSON的rotation字段\n• 仅叠加在JSON上：旋转只更新JSON的rotation字段，不应用到模型顶点"
                }),
                "scale_x_multiplier": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "X轴缩放乘数\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍"
                }),
                "scale_y_multiplier": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "Y轴缩放乘数\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍"
                }),
                "scale_z_multiplier": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.001, 
                    "max": 1000.0, 
                    "step": 0.001,
                    "tooltip": "Z轴缩放乘数\n1.0：保持原尺寸\n0.5：缩小一半\n2.0：放大一倍"
                }),
                "scale_apply_mode": (["应用在模型本身+叠加在JSON上", "仅应用在模型本身", "仅叠加在JSON上"], {
                    "default": "仅叠加在JSON上",
                    "tooltip": "缩放应用模式：\n• 应用在模型本身+叠加在JSON上：缩放会烘焙到模型顶点，同时更新JSON的scale字段\n• 仅应用在模型本身：缩放只烘焙到模型顶点，不修改JSON的scale字段\n• 仅叠加在JSON上：缩放只更新JSON的scale字段，不应用到模型顶点"
                }),
            },
        }
    
    RETURN_TYPES = ("TRANSFORM_PARAMS",)
    RETURN_NAMES = ("transform_params",)
    FUNCTION = "output_transform"
    CATEGORY = "VVL/3D"
    
    def output_transform(self, rotation_x_offset, rotation_y_offset, rotation_z_offset, rotation_apply_mode,
                        scale_x_multiplier, scale_y_multiplier, scale_z_multiplier, scale_apply_mode, **kwargs):
        """输出变换参数（旋转+缩放+应用模式）"""
        print(f"[Transform] 旋转参数: X={rotation_x_offset}°, Y={rotation_y_offset}°, Z={rotation_z_offset}°")
        print(f"[Transform] 旋转应用模式: {rotation_apply_mode}")
        print(f"[Transform] 缩放参数: X={scale_x_multiplier}, Y={scale_y_multiplier}, Z={scale_z_multiplier}")
        print(f"[Transform] 缩放应用模式: {scale_apply_mode}")
        
        # 返回一个包含旋转、缩放和应用模式的字典
        params = {
            'rotation': (rotation_x_offset, rotation_y_offset, rotation_z_offset),
            'rotation_apply_mode': rotation_apply_mode,
            'scale': (scale_x_multiplier, scale_y_multiplier, scale_z_multiplier),
            'scale_apply_mode': scale_apply_mode
        }
        return (params,)


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
    
    ⚡ 处理流程（每个模型）：
    1. 导入原始模型
    2. 应用额外旋转（ModelTransformParameters，如果连接）
    3. 应用额外缩放（ModelTransformParameters，如果连接）
    4. 叠加变换到JSON（将额外旋转叠加到rotation字段，额外缩放乘以scale字段）
    5. 计算包围盒
    6. 基于更新后的JSON创建参考Box并计算对齐旋转
    7. 应用智能缩放（基于JSON的scale值）
    8. 导出处理后的模型
    
    📝 重要说明：
    • 对齐旋转**只用于计算**，不应用到模型
    • 计算出的旋转角度会更新到输出JSON的rotation字段
    • 模型在导出时**不包含对齐旋转**，只包含智能缩放
    • 这样做是为了在外部系统中通过rotation字段控制模型朝向
    
    💡 提示：
    • transform_params（可选）：从 ModelTransformParameters 节点连接
    • 额外变换在最开始应用到模型，然后叠加到JSON数据中
    • 后续的对齐计算基于更新后的JSON数据进行
    • 与v1版本向后兼容
    """
    
    # 包含对齐功能的 Blender 脚本
    BLENDER_SCRIPT_WITH_ALIGNMENT = r"""
import bpy, sys, json, math
from mathutils import Vector, Matrix

# 数值稳定性阈值
_EPS = 1e-8
_SCALE_STANDARD = 1.0


def _calculate_smart_scale(current_size, target_scale, force_exact_alignment, standard_size=_SCALE_STANDARD):
    # 计算智能缩放系数，模拟 Python 端 SmartScaler 行为
    debug = {}

    if not current_size or max(current_size) < _EPS:
        debug["note"] = "current size too small; fallback to 1"
        return (1.0, 1.0, 1.0), debug

    if not target_scale or max(target_scale) < _EPS:
        debug["note"] = "target scale too small; fallback to uniform 1"
        return (1.0, 1.0, 1.0), debug

    current_max = max(current_size)
    target_max = standard_size * max(target_scale)
    uniform_scale = target_max / current_max if current_max > _EPS else 1.0

    debug["current_size"] = list(current_size)
    debug["target_scale"] = list(target_scale)
    debug["uniform_scale"] = uniform_scale
    debug["current_max"] = current_max
    debug["target_max"] = target_max

    if not force_exact_alignment:
        debug["mode"] = "uniform"
        return (uniform_scale, uniform_scale, uniform_scale), debug

    # 先按等比缩放
    scaled_size = [dim * uniform_scale for dim in current_size]
    debug["scaled_size"] = scaled_size

    # 目标尺寸按从小到大排序
    target_scales_sorted = sorted(target_scale)
    target_sizes_sorted = [standard_size * s for s in target_scales_sorted]
    debug["target_sizes_sorted"] = target_sizes_sorted

    # 将模型的轴按缩放后的尺寸排序
    size_with_axis = sorted([(scaled_size[0], 0), (scaled_size[1], 1), (scaled_size[2], 2)], key=lambda item: item[0])
    debug["size_with_axis"] = size_with_axis

    # 将排序后的目标尺寸映射回原始轴
    smart_target = [0.0, 0.0, 0.0]
    for rank, (_, axis_index) in enumerate(size_with_axis):
        smart_target[axis_index] = target_sizes_sorted[rank]

    debug["smart_target"] = smart_target

    fine_tune_ratio = []
    final_scale = []
    for axis in range(3):
        axis_scaled = scaled_size[axis]
        if abs(axis_scaled) < _EPS:
            ratio = 1.0
        else:
            ratio = smart_target[axis] / axis_scaled
        fine_tune_ratio.append(ratio)
        final_scale.append(uniform_scale * ratio)

    debug["fine_tune_ratio"] = fine_tune_ratio
    debug["mode"] = "axis_aligned"

    return tuple(final_scale), debug

argv = sys.argv[sys.argv.index("--")+1:]
(
    in_path,
    out_path,
    ref_scale_x,
    ref_scale_y,
    ref_scale_z,
    ref_rot_x,
    ref_rot_y,
    ref_rot_z,
    additional_rot_x,
    additional_rot_y,
    additional_rot_z,
    additional_scale_x,
    additional_scale_y,
    additional_scale_z,
    rotation_apply_mode,
    scale_apply_mode,
    force_exact_alignment_flag,
    bbox_path,
    scale_info_path,
    alignment_path,
) = argv
ref_scale_x, ref_scale_y, ref_scale_z = float(ref_scale_x), float(ref_scale_y), float(ref_scale_z)  # 参考box的scale值
ref_rot_x, ref_rot_y, ref_rot_z = float(ref_rot_x), float(ref_rot_y), float(ref_rot_z)  # 参考box的rotation值（度）
additional_rot_x, additional_rot_y, additional_rot_z = float(additional_rot_x), float(additional_rot_y), float(additional_rot_z)  # 额外旋转偏移（度）
additional_scale_x, additional_scale_y, additional_scale_z = float(additional_scale_x), float(additional_scale_y), float(additional_scale_z)  # 额外缩放乘数
# rotation_apply_mode: 'both', 'model_only', 'json_only'
# scale_apply_mode: 'both', 'model_only', 'json_only'
force_exact_alignment = force_exact_alignment_flag.lower() == "true"

print(f"[Blender] 开始处理模型: {in_path}")
# 智能缩放相关变量稍后计算
json_target_scale = (ref_scale_x, ref_scale_y, ref_scale_z)
print(f"[Blender] JSON目标缩放: ({json_target_scale[0]:.3f}, {json_target_scale[1]:.3f}, {json_target_scale[2]:.3f})")
print(f"[Blender] 参考Box的Scale: ({ref_scale_x:.1f}, {ref_scale_y:.1f}, {ref_scale_z:.1f})")
print(f"[Blender] 参考Box的Rotation: ({ref_rot_x:.1f}°, {ref_rot_y:.1f}°, {ref_rot_z:.1f}°)")
print(f"[Blender] 额外旋转偏移: X={additional_rot_x:.1f}°, Y={additional_rot_y:.1f}°, Z={additional_rot_z:.1f}°")
print(f"[Blender] 旋转应用模式: {rotation_apply_mode}")
print(f"[Blender] 额外缩放乘数: X={additional_scale_x:.3f}, Y={additional_scale_y:.3f}, Z={additional_scale_z:.3f}")
print(f"[Blender] 缩放应用模式: {scale_apply_mode}")

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

# ===== 步骤1: 应用 ModelTransformParameters 的变换（最先执行） =====
# 应用额外的旋转偏移（根据rotation_apply_mode判断是否应用到模型）
should_apply_rotation_to_model = (rotation_apply_mode in ['both', 'model_only'])
if meshes and should_apply_rotation_to_model and (abs(additional_rot_x) > 0.01 or abs(additional_rot_y) > 0.01 or abs(additional_rot_z) > 0.01):
    print(f"\n[Blender] [步骤1a] 应用额外旋转到模型（ModelTransformParameters）...")
    print(f"[Blender] 应用模式: {rotation_apply_mode}")
    additional_rotation_radians = (math.radians(additional_rot_x), math.radians(additional_rot_y), math.radians(additional_rot_z))
    
    for o in meshes:
        o.select_set(True)
        o.rotation_euler.x += additional_rotation_radians[0]
        o.rotation_euler.y += additional_rotation_radians[1]
        o.rotation_euler.z += additional_rotation_radians[2]
        print(f"[Blender] 对象 '{o.name}' 旋转: X={math.degrees(o.rotation_euler.x):.1f}°, Y={math.degrees(o.rotation_euler.y):.1f}°, Z={math.degrees(o.rotation_euler.z):.1f}°")
    
    # 烘焙旋转到顶点
    bpy.context.view_layer.objects.active = meshes[0] if meshes else None
    print(f"[Blender] 烘焙旋转到顶点...")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
elif meshes and not should_apply_rotation_to_model and (abs(additional_rot_x) > 0.01 or abs(additional_rot_y) > 0.01 or abs(additional_rot_z) > 0.01):
    print(f"\n[Blender] [步骤1a] 跳过应用额外旋转到模型（rotation_apply_mode={rotation_apply_mode}）")

# 应用额外的缩放乘数（根据scale_apply_mode判断是否应用到模型）
should_apply_scale_to_model = (scale_apply_mode in ['both', 'model_only'])
if meshes and should_apply_scale_to_model and (abs(additional_scale_x - 1.0) > 0.001 or abs(additional_scale_y - 1.0) > 0.001 or abs(additional_scale_z - 1.0) > 0.001):
    print(f"\n[Blender] [步骤1b] 应用额外缩放到模型（ModelTransformParameters）...")
    print(f"[Blender] 应用模式: {scale_apply_mode}")
    
    for o in meshes:
        o.select_set(True)
        original_scale = o.scale.copy()
        o.scale = (o.scale[0] * additional_scale_x, o.scale[1] * additional_scale_y, o.scale[2] * additional_scale_z)
        print(f"[Blender] 对象 '{o.name}' 额外缩放: {original_scale} -> {o.scale}")
    
    # 烘焙缩放到顶点
    bpy.context.view_layer.objects.active = meshes[0] if meshes else None
    print(f"[Blender] 烘焙额外缩放到顶点...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
elif meshes and not should_apply_scale_to_model and (abs(additional_scale_x - 1.0) > 0.001 or abs(additional_scale_y - 1.0) > 0.001 or abs(additional_scale_z - 1.0) > 0.001):
    print(f"\n[Blender] [步骤1b] 跳过应用额外缩放到模型（scale_apply_mode={scale_apply_mode}）")

# ===== 步骤2: 计算应用额外变换后的包围盒 =====
print(f"\n[Blender] [步骤2] 计算应用额外变换后的包围盒...")
transformed_gmin = Vector(( math.inf,  math.inf,  math.inf))
transformed_gmax = Vector((-math.inf, -math.inf, -math.inf))
for o in meshes:
    for corner in o.bound_box:
        wpt = o.matrix_world @ Vector(corner)
        transformed_gmin.x = min(transformed_gmin.x, wpt.x)
        transformed_gmin.y = min(transformed_gmin.y, wpt.y)
        transformed_gmin.z = min(transformed_gmin.z, wpt.z)
        transformed_gmax.x = max(transformed_gmax.x, wpt.x)
        transformed_gmax.y = max(transformed_gmax.y, wpt.y)
        transformed_gmax.z = max(transformed_gmax.z, wpt.z)

transformed_size = [transformed_gmax[i] - transformed_gmin[i] for i in range(3)]
print(f"[Blender] 变换后包围盒尺寸: ({transformed_size[0]:.2f}, {transformed_size[1]:.2f}, {transformed_size[2]:.2f})")

# ===== 步骤3: 将额外变换叠加回JSON数据（根据应用模式） =====
print(f"\n[Blender] [步骤3] 叠加额外变换到JSON数据...")

# 判断是否需要叠加旋转到JSON
should_apply_rotation_to_json = (rotation_apply_mode in ['both', 'json_only'])
if should_apply_rotation_to_json:
    # 将额外旋转叠加到 JSON 的 rotation 字段
    updated_ref_rot_x = ref_rot_x + additional_rot_x
    updated_ref_rot_y = ref_rot_y + additional_rot_y
    updated_ref_rot_z = ref_rot_z + additional_rot_z
    print(f"[Blender] 叠加旋转到JSON（rotation_apply_mode={rotation_apply_mode}）:")
    print(f"  原始 rotation: X={ref_rot_x:.1f}°, Y={ref_rot_y:.1f}°, Z={ref_rot_z:.1f}°")
    print(f"  额外 rotation: X={additional_rot_x:.1f}°, Y={additional_rot_y:.1f}°, Z={additional_rot_z:.1f}°")
    print(f"  更新后 rotation: X={updated_ref_rot_x:.1f}°, Y={updated_ref_rot_y:.1f}°, Z={updated_ref_rot_z:.1f}°")
else:
    # 不叠加旋转到JSON，保持原值
    updated_ref_rot_x = ref_rot_x
    updated_ref_rot_y = ref_rot_y
    updated_ref_rot_z = ref_rot_z
    print(f"[Blender] 跳过叠加旋转到JSON（rotation_apply_mode={rotation_apply_mode}）")
    print(f"  保持原始 rotation: X={ref_rot_x:.1f}°, Y={ref_rot_y:.1f}°, Z={ref_rot_z:.1f}°")

# 判断是否需要叠加缩放到JSON
should_apply_scale_to_json = (scale_apply_mode in ['both', 'json_only'])
if should_apply_scale_to_json:
    # 将额外缩放乘以 JSON 的 scale 字段
    updated_ref_scale_x = ref_scale_x * additional_scale_x
    updated_ref_scale_y = ref_scale_y * additional_scale_y
    updated_ref_scale_z = ref_scale_z * additional_scale_z
    print(f"[Blender] 叠加缩放到JSON（scale_apply_mode={scale_apply_mode}）:")
    print(f"  原始 scale: X={ref_scale_x:.3f}, Y={ref_scale_y:.3f}, Z={ref_scale_z:.3f}")
    print(f"  额外 scale: X={additional_scale_x:.3f}, Y={additional_scale_y:.3f}, Z={additional_scale_z:.3f}")
    print(f"  更新后 scale: X={updated_ref_scale_x:.3f}, Y={updated_ref_scale_y:.3f}, Z={updated_ref_scale_z:.3f}")
else:
    # 不叠加缩放到JSON，保持原值
    updated_ref_scale_x = ref_scale_x
    updated_ref_scale_y = ref_scale_y
    updated_ref_scale_z = ref_scale_z
    print(f"[Blender] 跳过叠加缩放到JSON（scale_apply_mode={scale_apply_mode}）")
    print(f"  保持原始 scale: X={ref_scale_x:.3f}, Y={ref_scale_y:.3f}, Z={ref_scale_z:.3f}")

# ===== 步骤4: 对齐计算和智能缩放（基于更新后的JSON数据） =====
print(f"\n[Blender] [步骤4] 开始计算对齐旋转（基于更新后的JSON）...")

# 创建参考 Box
print(f"[Blender] 创建参考 Box...")
# 创建默认大小的立方体
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
ref_box = bpy.context.active_object
ref_box.name = "Reference_Box"

# 使用更新后的 scale 值
ref_box.scale = (updated_ref_scale_x, updated_ref_scale_y, updated_ref_scale_z)

# 使用更新后的 rotation 值（需要转换为弧度）
ref_box.rotation_euler = (math.radians(updated_ref_rot_x), math.radians(updated_ref_rot_y), math.radians(updated_ref_rot_z))

bpy.context.view_layer.update()

# 获取参考 Box 的实际 dimensions
ref_dims = ref_box.dimensions.copy()
ref_sizes = [ref_dims.x, ref_dims.y, ref_dims.z]

print(f"[Blender] 参考 Box 创建完成:")
print(f"  - Scale(更新后): ({updated_ref_scale_x:.3f}, {updated_ref_scale_y:.3f}, {updated_ref_scale_z:.3f})")
print(f"  - Rotation(更新后): ({updated_ref_rot_x:.1f}°, {updated_ref_rot_y:.1f}°, {updated_ref_rot_z:.1f}°)")
print(f"  - 实际 Dimensions: ({ref_sizes[0]:.2f}, {ref_sizes[1]:.2f}, {ref_sizes[2]:.2f})")

# 获取参考 Box 的旋转矩阵（在删除之前）
ref_loc, ref_rot, ref_scale = ref_box.matrix_world.decompose()
ref_rot_matrix = ref_rot.to_matrix()

# 获取目标模型的尺寸（当前模型，已包含额外变换）
tgt_sizes = transformed_size

# ===== 步骤4b: 基于参考Box的dimensions计算智能缩放系数 =====
print(f"\n[Blender] [步骤4b] 计算智能缩放系数...")
# 使用参考Box的dimensions作为目标尺寸（已经考虑了旋转的影响）
target_dimensions_tuple = (ref_sizes[0] / 2.0, ref_sizes[1] / 2.0, ref_sizes[2] / 2.0)  # 除以2是因为默认cube的size=2
print(f"[Blender] 目标尺寸 (考虑旋转后): ({target_dimensions_tuple[0]:.3f}, {target_dimensions_tuple[1]:.3f}, {target_dimensions_tuple[2]:.3f})")

smart_scale, smart_debug = _calculate_smart_scale(transformed_size, target_dimensions_tuple, force_exact_alignment, standard_size=1.0)
print(f"[Blender] 智能缩放结果: ({smart_scale[0]:.3f}, {smart_scale[1]:.3f}, {smart_scale[2]:.3f})")
print(f"[Blender] 智能缩放模式: {smart_debug.get('mode', 'unknown')}")
if smart_debug.get('mode') == 'axis_aligned':
    print(f"[Blender] 缩放细节:")
    print(f"  - 当前尺寸: {smart_debug.get('current_size', [])}")
    print(f"  - 等比缩放后: {smart_debug.get('scaled_size', [])}")
    print(f"  - 目标匹配: {smart_debug.get('smart_target', [])}")
    print(f"  - 微调比例: {smart_debug.get('fine_tune_ratio', [])}")

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
    final_rotation_matrix = Matrix.Identity(3)  # 单位矩阵，表示无旋转
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
        "original_scale": [ref_scale_x, ref_scale_y, ref_scale_z],
        "original_rotation": [ref_rot_x, ref_rot_y, ref_rot_z],
        "updated_scale": [updated_ref_scale_x, updated_ref_scale_y, updated_ref_scale_z],
        "updated_rotation": [updated_ref_rot_x, updated_ref_rot_y, updated_ref_rot_z],
        "dimensions": ref_sizes
    },
    "model_sizes": transformed_size,
    "is_already_aligned": is_already_aligned,
    "transform_applied": {
        "additional_rotation": [additional_rot_x, additional_rot_y, additional_rot_z],
        "additional_scale": [additional_scale_x, additional_scale_y, additional_scale_z],
        "rotation_apply_mode": rotation_apply_mode,
        "scale_apply_mode": scale_apply_mode,
        "rotation_applied_to_model": should_apply_rotation_to_model,
        "rotation_applied_to_json": should_apply_rotation_to_json,
        "scale_applied_to_model": should_apply_scale_to_model,
        "scale_applied_to_json": should_apply_scale_to_json
    }
}

# 只有在需要旋转时才添加置换矩阵信息；同时在已对齐时清晰记录"保持更新后的JSON旋转"
if not is_already_aligned:
    alignment_info["permutation_matrix"] = [[perm_matrix[i][j] for j in range(3)] for i in range(3)]
else:
    alignment_info["note"] = "model parallel to ref box; keep updated JSON rotation"

# ===== 对齐计算结束 =====

# 应用智能缩放
print(f"\n[Blender] 应用智能缩放...")
for o in meshes:
    o.select_set(True)
    original_scale = o.scale.copy()
    o.scale = (o.scale[0]*smart_scale[0], o.scale[1]*smart_scale[1], o.scale[2]*smart_scale[2])
    print(f"[Blender] 对象 '{o.name}' 缩放: {original_scale} -> {o.scale}")

# 烘焙缩放到顶点
bpy.context.view_layer.objects.active = meshes[0] if meshes else None
if meshes:
    print(f"[Blender] 烘焙智能缩放到顶点...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# 注意：对齐旋转不应用到模型，只用于计算并更新JSON中的rotation字段
# 这与v1版本的行为保持一致
if is_already_aligned:
    print(f"\n[Blender] 模型已对齐，rotation字段将保持不变")
else:
    print(f"\n[Blender] 模型需要旋转对齐，计算的rotation角度将更新到JSON")
    print(f"  - 计算的rotation: X={rotation_degrees[0]:.1f}°, Y={rotation_degrees[1]:.1f}°, Z={rotation_degrees[2]:.1f}°")
    print(f"  - 注意：旋转角度仅用于更新JSON，不应用到Blender模型")

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
    "applied_scale": list(smart_scale),
    "original_size": transformed_size,
    "final_size": final_size,
    "size_change_ratio": [final_size[i]/transformed_size[i] if transformed_size[i] > 0 else 1.0 for i in range(3)],
    "mesh_count": len(meshes),
    "scale_applied_to_vertices": True,
    "material_count": material_count,
    "texture_count": texture_count,
    "materials_preserved": (texture_count > 0 or material_count > 0),
    "additional_rotation_applied": [additional_rot_x, additional_rot_y, additional_rot_z],
    "rotation_applied_to_vertices": should_apply_rotation_to_model and (abs(additional_rot_x) > 0.01 or abs(additional_rot_y) > 0.01 or abs(additional_rot_z) > 0.01),
    "additional_scale_applied": [additional_scale_x, additional_scale_y, additional_scale_z],
    "additional_scale_applied_to_vertices": should_apply_scale_to_model and (abs(additional_scale_x - 1.0) > 0.001 or abs(additional_scale_y - 1.0) > 0.001 or abs(additional_scale_z - 1.0) > 0.001),
    "rotation_apply_mode": rotation_apply_mode,
    "scale_apply_mode": scale_apply_mode
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
                "transform_params": ("TRANSFORM_PARAMS", {"tooltip": "变换参数\n从 ModelTransformParameters 节点连接\n包含旋转偏移、缩放乘数与应用模式\n最先对模型应用（步骤1a/1b），并根据应用模式叠加到JSON（步骤3）"}),
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
                              blender_path: str, output_path: str,
                              additional_rot_x: float = 0.0, additional_rot_y: float = 0.0, additional_rot_z: float = 0.0,
                              additional_scale_x: float = 1.0, additional_scale_y: float = 1.0, additional_scale_z: float = 1.0,
                              rotation_apply_mode: str = "both", scale_apply_mode: str = "both",
                              force_exact_alignment: bool = True):
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
                str(ref_scale_x), str(ref_scale_y), str(ref_scale_z),
                str(ref_rot_x), str(ref_rot_y), str(ref_rot_z),
                str(additional_rot_x), str(additional_rot_y), str(additional_rot_z),
                str(additional_scale_x), str(additional_scale_y), str(additional_scale_z),
                rotation_apply_mode, scale_apply_mode,
                "true" if force_exact_alignment else "false",
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
    
    def _round_and_clean_rotation(self, degrees_list, decimals: int = 3):
        """将旋转角度列表四舍五入到指定小数位，并将-0.0规范为0.0"""
        if not isinstance(degrees_list, (list, tuple)):
            return degrees_list
        cleaned = []
        eps = 0.5 * (10 ** (-decimals))
        for value in degrees_list:
            try:
                r = round(float(value), decimals)
                if abs(r) < eps:
                    r = 0.0
                cleaned.append(r)
            except Exception:
                cleaned.append(value)
        return cleaned
    
    def _process_single_object(self, obj_data: dict, index: int, force_exact_alignment: bool, 
                              blender_path: str, rotation_x_offset: float = 0.0, rotation_y_offset: float = 0.0, 
                              rotation_z_offset: float = 0.0, scale_x_multiplier: float = 1.0, 
                              scale_y_multiplier: float = 1.0, scale_z_multiplier: float = 1.0,
                              rotation_apply_mode: str = "both", scale_apply_mode: str = "both") -> dict:
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
                
                # 准备输出路径
                out_dir = os.path.join(folder_paths.get_output_directory(), "3d")
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, output_filename_with_subdir)
                output_path = os.path.normpath(output_path)
                out_dir_for_file = os.path.dirname(output_path)
                os.makedirs(out_dir_for_file, exist_ok=True)
                
                # 使用带对齐功能的处理
                # 传递原始的scale和rotation值，以及额外的旋转和缩放参数
                bbox_data, scale_info_data, alignment_data = self._process_with_alignment(
                    downloaded_path,
                    float(scale[0]), float(scale[1]), float(scale[2]),  # 原始scale值
                    float(rotation[0]), float(rotation[1]), float(rotation[2]),  # 原始rotation值
                    blender_path,
                    output_path,
                    rotation_x_offset, rotation_y_offset, rotation_z_offset,  # 额外旋转偏移
                    scale_x_multiplier, scale_y_multiplier, scale_z_multiplier,  # 额外缩放乘数
                    rotation_apply_mode, scale_apply_mode,  # 应用模式
                    force_exact_alignment
                )
                
                if not bbox_data or not scale_info_data or not alignment_data:
                    raise Exception(f"Blender处理返回了空数据")
                
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
                # 规范化旋转角度，避免微小浮点误差与-0.0
                result["rotation"] = self._round_and_clean_rotation(alignment_data.get("rotation_degrees", []), decimals=3)
                
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
                     max_workers: int = 4, transform_params=None, **kwargs):
        """批量处理JSON中的所有3D模型"""
        
        # 解析变换参数
        if transform_params is not None and isinstance(transform_params, dict):
            # 解析旋转参数
            rotation = transform_params.get('rotation', (0.0, 0.0, 0.0))
            if isinstance(rotation, (tuple, list)) and len(rotation) >= 3:
                rotation_x_offset = float(rotation[0])
                rotation_y_offset = float(rotation[1])
                rotation_z_offset = float(rotation[2])
            else:
                rotation_x_offset = 0.0
                rotation_y_offset = 0.0
                rotation_z_offset = 0.0
            
            # 解析缩放参数
            scale = transform_params.get('scale', (1.0, 1.0, 1.0))
            if isinstance(scale, (tuple, list)) and len(scale) >= 3:
                scale_x_multiplier = float(scale[0])
                scale_y_multiplier = float(scale[1])
                scale_z_multiplier = float(scale[2])
            else:
                scale_x_multiplier = 1.0
                scale_y_multiplier = 1.0
                scale_z_multiplier = 1.0
            
            # 解析应用模式参数
            rotation_apply_mode_str = transform_params.get('rotation_apply_mode', '应用在模型本身+叠加在JSON上')
            scale_apply_mode_str = transform_params.get('scale_apply_mode', '应用在模型本身+叠加在JSON上')
            
            # 转换为英文标识符
            if rotation_apply_mode_str == '仅应用在模型本身':
                rotation_apply_mode = 'model_only'
            elif rotation_apply_mode_str == '仅叠加在JSON上':
                rotation_apply_mode = 'json_only'
            else:  # '应用在模型本身+叠加在JSON上'
                rotation_apply_mode = 'both'
            
            if scale_apply_mode_str == '仅应用在模型本身':
                scale_apply_mode = 'model_only'
            elif scale_apply_mode_str == '仅叠加在JSON上':
                scale_apply_mode = 'json_only'
            else:  # '应用在模型本身+叠加在JSON上'
                scale_apply_mode = 'both'
        else:
            # 默认值
            rotation_x_offset = 0.0
            rotation_y_offset = 0.0
            rotation_z_offset = 0.0
            scale_x_multiplier = 1.0
            scale_y_multiplier = 1.0
            scale_z_multiplier = 1.0
            rotation_apply_mode = 'both'
            scale_apply_mode = 'both'
        
        print(f"\n=== VVL智能模型批量缩放器 开始处理 ===")
        print(f"[Batch] 最大并行线程数: {max_workers}")
        if rotation_x_offset != 0.0 or rotation_y_offset != 0.0 or rotation_z_offset != 0.0:
            print(f"[Batch] 额外旋转偏移: X={rotation_x_offset}°, Y={rotation_y_offset}°, Z={rotation_z_offset}°")
            print(f"[Batch] 旋转应用模式: {rotation_apply_mode}")
        if scale_x_multiplier != 1.0 or scale_y_multiplier != 1.0 or scale_z_multiplier != 1.0:
            print(f"[Batch] 额外缩放乘数: X={scale_x_multiplier}, Y={scale_y_multiplier}, Z={scale_z_multiplier}")
            print(f"[Batch] 缩放应用模式: {scale_apply_mode}")
        
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
                        blender_path,
                        rotation_x_offset,
                        rotation_y_offset,
                        rotation_z_offset,
                        scale_x_multiplier,
                        scale_y_multiplier,
                        scale_z_multiplier,
                        rotation_apply_mode,
                        scale_apply_mode
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
                        if result.get('alignment_info'):
                            alignment_info = result['alignment_info']
                            apply_mode = alignment_info.get('transform_applied', {}).get('rotation_apply_mode', 'both')
                            updated_rotation = alignment_info.get('ref_box', {}).get('updated_rotation')
                            is_aligned = alignment_info.get('is_already_aligned', False)
                            
                            # 判断是否有额外的旋转偏移（非零）
                            additional_rotation = alignment_info.get('transform_applied', {}).get('additional_rotation', [0, 0, 0])
                            has_additional_rotation = any(abs(r) > 0.01 for r in additional_rotation)
                            
                            # 处理不同的应用模式
                            if apply_mode == 'model_only' and has_additional_rotation:
                                # model_only模式：额外旋转已经应用到模型
                                # 计算出的rotation是基于已旋转的模型，需要调整
                                if result.get('rotation') and not is_aligned:
                                    calculated_rotation = result['rotation']
                                    # 对于Z轴旋转-90°的情况，需要调整计算结果
                                    # 因为模型已经旋转了，所以最终JSON应该反映这个差异
                                    adjusted_rotation = list(calculated_rotation)
                                    # 特殊处理：当模型被旋转-90°后，某些计算的旋转需要调整
                                    if abs(additional_rotation[2] + 90) < 0.01:  # Z轴旋转了-90°
                                        # 根据实际情况调整旋转值
                                        if abs(calculated_rotation[0] - 180) < 0.01 and abs(calculated_rotation[2] - 90) < 0.01:
                                            # [180, 0, 90] -> [0, 0, 90]
                                            adjusted_rotation = [0.0, 0.0, 90.0]
                                    cleaned = self._round_and_clean_rotation(adjusted_rotation, decimals=3)
                                    output_data['objects'][index]['rotation'] = cleaned
                                    print(f"[Batch] 对象 [{index}] 调整后的rotation (model_only模式): {cleaned}")
                                elif is_aligned:
                                    # 已对齐，保持原始rotation
                                    original_rotation = output_data['objects'][index].get('rotation', [0, 0, 0])
                                    print(f"[Batch] 对象 [{index}] 已对齐，保留原始rotation: {original_rotation}")
                            elif apply_mode in ('json_only', 'both') and updated_rotation is not None and has_additional_rotation:
                                # 当应用模式包含JSON更新且有额外旋转时，写回更新后的JSON rotation
                                cleaned = self._round_and_clean_rotation(updated_rotation, decimals=3)
                                output_data['objects'][index]['rotation'] = cleaned
                                print(f"[Batch] 对象 [{index}] 写回JSON更新后的rotation: {cleaned} (mode={apply_mode})")
                            else:
                                # 默认逻辑：已对齐保留原始rotation；否则写入计算得到的rotation
                                if result.get('rotation'):
                                    if is_aligned:
                                        original_rotation = output_data['objects'][index].get('rotation', 'undefined')
                                        print(f"[Batch] 对象 [{index}] 已对齐，保留原始rotation: {original_rotation}")
                                    else:
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
