"""
VVL Blender Model Aligner Node
批量模型对齐节点 - 调用Blender插件实现模型自动对齐和导出
"""
import os
import sys
import json
import subprocess
import tempfile
import folder_paths
from typing import Tuple

class BlenderModelAligner:
    """
    Blender模型批量对齐节点
    通过调用Blender执行对齐脚本，实现模型的旋转对齐、拉伸对齐和导出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_config": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "输入JSON配置，包含objects数组。每个对象需要包含：\n"
                               "• name: 对象名称\n"
                               "• 3d_url/url: 模型文件URL（支持GLB/FBX/OBJ格式）\n"
                               "• rotation: [x,y,z] 初始旋转角度（度数）\n"
                               "• scale: [x,y,z] 目标包围盒尺寸（米）\n"
                               "• position: [x,y,z] 位置信息（保留用于UE，对齐时不使用）\n\n"
                               "处理逻辑：\n"
                               "1. 下载模型并导入到Blender\n"
                               "2. 根据scale创建参考Box\n"
                               "3. 应用rotation到模型\n"
                               "4. 旋转对齐：使模型轴向与Box对齐\n"
                               "5. 完美重合：拉伸模型使包围盒与Box完全重合\n"
                               "6. 导出：rotation重置为[0,0,0]，旋转已烘焙到模型"
                }),
                "export_format": (["GLB", "FBX"], {
                    "default": "GLB",
                    "tooltip": "导出模型格式：\n\n"
                               "GLB（推荐）：\n"
                               "• 单文件包含所有资源（模型+贴图）\n"
                               "• Web友好，适合游戏引擎\n"
                               "• UE模式下自动处理单位转换，无需手动缩放\n"
                               "• 文件体积较小，传输快速\n\n"
                               "FBX：\n"
                               "• 工业标准格式，兼容性好\n"
                               "• 支持嵌入贴图（embed_textures=True）\n"
                               "• UE模式下会应用100倍缩放（米→厘米）\n"
                               "• 适合需要在DCC软件中进一步编辑的场景\n\n"
                               "注意：导出格式决定坐标转换策略"
                }),
                "blender_path": ("STRING", {
                    "default": "blender",
                    "multiline": False,
                    "tooltip": "Blender可执行文件路径：\n\n"
                               "Windows示例：\n"
                               "• E:\\Software\\Blender\\4.4.3\\blender.exe（完整路径）\n"
                               "• blender（如果已添加到PATH环境变量）\n\n"
                               "Linux示例：\n"
                               "• /usr/bin/blender（系统安装）\n"
                               "• /home/user/blender-4.4.3/blender（用户目录）\n"
                               "• blender（如果在PATH中）\n\n"
                               "macOS示例：\n"
                               "• /Applications/Blender.app/Contents/MacOS/Blender\n\n"
                               "要求：\n"
                               "• Blender 2.8或更高版本\n"
                               "• 需要有执行权限（Linux/macOS: chmod +x）\n"
                               "• 建议安装requests库以支持模型下载"
                }),
                "target_engine": (["UE", "UNITY", "BLENDER", "NONE"], {
                    "default": "UE",
                    "tooltip": "目标引擎坐标系和单位转换：\n\n"
                               "UE (Unreal Engine)：\n"
                               "• 单位：厘米（GLB自动转换，FBX手动缩放100倍）\n"
                               "• 坐标系：左手系（X前 Y右 Z上）\n"
                               "• GLB导出时跳过场景缩放（导出器自动处理）\n"
                               "• FBX导出时应用100倍缩放到模型scale\n\n"
                               "UNITY：\n"
                               "• 单位：米（保持Blender原始单位）\n"
                               "• 坐标系：左手系（X右 Y上 Z前）\n"
                               "• FBX导出时转换为Unity坐标系（axis_forward/up）\n"
                               "• 不应用额外缩放\n\n"
                               "BLENDER：\n"
                               "• 单位：米\n"
                               "• 坐标系：右手系（X右 Y前 Z上）\n"
                               "• 保持Blender原生坐标系，适合继续编辑\n\n"
                               "NONE：\n"
                               "• 不做任何坐标转换\n"
                               "• 直接导出对齐后的模型"
                }),
                "processing_mode": (["STREAM", "BATCH"], {
                    "default": "STREAM",
                    "tooltip": "批量处理模式：\n\n"
                               "STREAM（流式处理）- 推荐：\n"
                               "• 逐个处理并导出后删除模型\n"
                               "• 内存占用小，稳定性高\n"
                               "• 适合大量模型处理\n"
                               "• 处理过程中只保留当前模型在内存中\n\n"
                               "BATCH（一次性处理）：\n"
                               "• 全部对齐后统一导出\n"
                               "• 速度稍快但内存占用大\n"
                               "• 适合少量模型处理\n"
                               "• 所有模型会同时加载到Blender场景中"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_result",)
    FUNCTION = "align_models"
    CATEGORY = "VVL/Blender"
    
    def align_models(self, json_config: str, export_format: str, blender_path: str, target_engine: str, processing_mode: str) -> Tuple[str]:
        """
        执行批量模型对齐
        
        Args:
            json_config: JSON配置文本
            export_format: 导出格式 (GLB/FBX)
            blender_path: Blender可执行文件路径
            target_engine: 目标引擎 (UE/UNITY/BLENDER/NONE)
            processing_mode: 处理模式 (STREAM/BATCH)
            
        Returns:
            处理后的JSON文本
        """
        try:
            print("\n" + "="*60)
            print("VVL Blender Model Aligner - 开始处理")
            print("="*60)
            
            # 1. 验证JSON配置
            try:
                config_data = json.loads(json_config)
                objects = config_data.get('objects', [])
                if not objects:
                    raise ValueError("JSON配置中没有objects数组")
                print(f"[Aligner] 解析JSON成功，共 {len(objects)} 个对象")
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析失败: {str(e)}"
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            # 2. 验证Blender路径
            if not self._validate_blender_path(blender_path):
                error_msg = f"Blender路径无效或无法执行: {blender_path}"
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            # 3. 创建输出目录
            output_base_dir = folder_paths.get_output_directory()
            output_dir = os.path.join(output_base_dir, "model_alignment")
            os.makedirs(output_dir, exist_ok=True)
            print(f"[Aligner] 输出目录: {output_dir}")
            
            # 4. 创建临时目录存放JSON
            temp_dir = tempfile.mkdtemp(prefix="blender_aligner_")
            input_json_path = os.path.join(temp_dir, "input_config.json")
            output_json_path = os.path.join(temp_dir, "output_config.json")
            
            # 写入输入JSON
            with open(input_json_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"[Aligner] 输入JSON: {input_json_path}")
            print(f"[Aligner] 输出JSON: {output_json_path}")
            
            # 5. 获取对齐脚本路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alignment_script = os.path.join(script_dir, "alignment_script.py")
            alignment_ops_script = os.path.join(script_dir, "alignment_ops.py")
            
            if not os.path.exists(alignment_script):
                error_msg = f"对齐脚本不存在: {alignment_script}"
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            # 检查 alignment_ops.py 是否存在（必需依赖）
            if not os.path.exists(alignment_ops_script):
                error_msg = f"核心函数库不存在: {alignment_ops_script}"
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            print(f"[Aligner] 对齐脚本: {alignment_script}")
            print(f"[Aligner] 核心函数库: {alignment_ops_script}")
            
            # 6. 构建Blender命令
            blender_cmd = [
                blender_path,
                "--background",
                "--python", alignment_script,
                "--",
                input_json_path,
                output_dir,
                export_format,
                target_engine,
                processing_mode,
                output_json_path
            ]
            
            print(f"[Aligner] 执行命令: {' '.join(blender_cmd)}")
            print(f"[Aligner] 参数:")
            print(f"  - 导出格式: {export_format}")
            print(f"  - 目标引擎: {target_engine}")
            print(f"  - 处理模式: {processing_mode}")
            print(f"  - 输出目录: {output_dir}")
            
            # 7. 执行Blender
            print("\n[Aligner] 开始执行Blender对齐...")
            result = subprocess.run(
                blender_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',  # 明确指定 UTF-8 编码
                errors='replace',  # 遇到无法解码的字符时替换
                timeout=3600  # 1小时超时
            )
            
            # 打印Blender输出
            if result.stdout:
                print("\n[Blender输出]:")
                print(result.stdout)
            
            if result.stderr:
                print("\n[Blender错误]:")
                print(result.stderr)
            
            # 8. 检查执行结果
            if result.returncode != 0:
                error_msg = f"Blender执行失败，返回码: {result.returncode}"
                if result.stderr:
                    error_msg += f"\n错误信息: {result.stderr[-500:]}"  # 只取最后500字符
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            # 9. 读取输出JSON
            if not os.path.exists(output_json_path):
                error_msg = "未生成输出JSON文件"
                print(f"[Aligner] 错误: {error_msg}")
                return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
            with open(output_json_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            # 10. 清理临时文件
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"[Aligner] 已清理临时目录: {temp_dir}")
            except:
                pass
            
            # 11. 返回结果
            result_json = json.dumps(output_data, ensure_ascii=False, indent=2)
            
            print("\n" + "="*60)
            print("VVL Blender Model Aligner - 处理完成")
            print(f"成功处理 {len(output_data.get('objects', []))} 个对象")
            print("="*60 + "\n")
            
            return (result_json,)
            
        except subprocess.TimeoutExpired:
            error_msg = "Blender执行超时（超过1小时）"
            print(f"[Aligner] 错误: {error_msg}")
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
            
        except Exception as e:
            error_msg = f"批量对齐失败: {str(e)}"
            print(f"[Aligner] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return (json.dumps({"error": error_msg}, ensure_ascii=False),)
    
    def _validate_blender_path(self, blender_path: str) -> bool:
        """验证Blender路径是否有效"""
        try:
            # 尝试执行 blender --version
            result = subprocess.run(
                [blender_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and "Blender" in result.stdout:
                print(f"[Aligner] Blender版本: {result.stdout.split()[1] if len(result.stdout.split()) > 1 else 'Unknown'}")
                return True
            return False
            
        except Exception as e:
            print(f"[Aligner] Blender路径验证失败: {str(e)}")
            return False


# 导出节点类
NODE_CLASS_MAPPINGS = {
    "BlenderModelAligner": BlenderModelAligner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderModelAligner": "VVL Blender Model Aligner"
}

