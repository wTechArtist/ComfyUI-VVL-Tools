"""
Blender批量对齐脚本
被ComfyUI节点调用，在Blender后台执行模型对齐和导出
【完全复用操作符逻辑】与Blender UI的"批量Box对齐预览+导出"按钮效果100%一致
"""
import bpy
import sys
import os
import json
import tempfile
from datetime import datetime
from mathutils import Vector

# ==================== 日志系统 ====================

class DualLogger:
    """双路日志输出：同时输出到控制台和文件"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_path = log_file_path
        self.original_stdout = None
        self.original_stderr = None
    
    def log(self, message):
        """同时输出到控制台和文件"""
        # 直接写入，避免调用print（会导致重复）
        if self.original_stdout:
            self.original_stdout.write(message + '\n')  # 输出到原始控制台
            self.original_stdout.flush()
        self.log_file.write(message + '\n')  # 写入文件
        self.log_file.flush()  # 立即刷新，确保实时写入
    
    def write(self, message):
        """实现write方法，用于重定向stdout"""
        # 同时输出到原始stdout和文件（包括换行符）
        if self.original_stdout:
            self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """实现flush方法"""
        if self.original_stdout:
            self.original_stdout.flush()
        self.log_file.flush()
    
    def redirect_output(self):
        """重定向标准输出到日志文件"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    
    def restore_output(self):
        """恢复标准输出"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
    
    def close(self):
        """关闭日志文件"""
        self.restore_output()
        self.log_file.close()

# 全局日志对象
logger = None

# 动态加载 alignment_ops 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
alignment_ops_path = os.path.join(script_dir, "alignment_ops.py")

# 导入 alignment_ops 模块（必需依赖）
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("alignment_ops", alignment_ops_path)
    alignment_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alignment_ops)
    
    # 初始化时使用print，logger还未创建
    print(f"[Blender脚本] ✓ 成功加载 alignment_ops 模块")
except Exception as e:
    print(f"[Blender脚本] ✗ 错误: 无法加载 alignment_ops.py")
    print(f"[Blender脚本] 详情: {str(e)}")
    print(f"[Blender脚本] 路径: {alignment_ops_path}")
    sys.exit(1)


# ==================== 模拟对象 ====================

class MockPrefs:
    """模拟 Blender 插件偏好设置对象"""
    def __init__(self, target_engine='NONE', export_format='GLB', show_debug_info=True):
        self.target_engine = target_engine
        self.export_format = export_format
        self.show_debug_info = show_debug_info


def mock_report(level, message):
    """模拟 Blender 操作符的 report 函数"""
    level_str = list(level)[0] if level else 'INFO'
    log_msg = f"[{level_str}] {message}"
    if logger:
        logger.log(log_msg)
    else:
        print(log_msg)


def execute_perfect_align_with_logging(ref_obj, target_obj, prefs, log_func):
    """
    完美重合对齐（带详细日志输出）
    【完全复用Blender插件版本的世界AABB算法】与model_alignment_tool完全一致
    
    Args:
        ref_obj: 参考Box对象
        target_obj: 目标模型对象
        prefs: 偏好设置
        log_func: 日志函数（传入logger.log或print）
    """
    # 【直接调用alignment_ops的函数，确保算法100%一致】
    # 不重复实现，避免差异
    alignment_ops.execute_perfect_align_batch_helper(ref_obj, target_obj, prefs)


class MockOperator:
    """模拟操作符类，复用OBJECT_OT_batch_box_preview_export的核心方法"""
    
    def __init__(self, prefs):
        self.prefs = prefs
    
    def report(self, level, message):
        """模拟report方法"""
        mock_report(level, message)
    
    def apply_coordinate_transform(self, model_obj, target_engine, prefs, export_format=None):
        """
        【完全复用操作符的坐标转换逻辑】
        与OBJECT_OT_batch_box_preview_export.apply_coordinate_transform完全一致
        """
        try:
            if target_engine == 'NONE':
                # 不做转换
                if prefs.show_debug_info:
                    logger.log(f"[坐标转换] 无转换模式") if logger else print(f"[坐标转换] 无转换模式")
                return
            
            elif target_engine == 'UE':
                # Unreal Engine转换
                # 如果导出格式是GLB/GLTF，则不在此处缩放，交给导出阶段处理
                if export_format and export_format.lower() in ['.glb', '.gltf']:
                    if prefs.show_debug_info:
                        logger.log(f"[坐标转换] UE模式 + GLB导出：跳过场景内缩放，导出阶段再处理") if logger else print(f"[坐标转换] UE模式 + GLB导出：跳过场景内缩放，导出阶段再处理")
                    return
                
                # 只需要缩放模型几何体以适配UE的单位系统
                # UE使用厘米，Blender使用米，所以需要放大100倍
                UE_SCALE_FACTOR = 100.0
                
                if prefs.show_debug_info:
                    logger.log(f"[坐标转换] UE模式：缩放系数 {UE_SCALE_FACTOR}") if logger else print(f"[坐标转换] UE模式：缩放系数 {UE_SCALE_FACTOR}")
                    logger.log(f"[坐标转换] 转换前: scale={model_obj.scale}") if logger else print(f"[坐标转换] 转换前: scale={model_obj.scale}")
                
                # 应用缩放系数到scale（模型几何体放大100倍）
                current_scale = model_obj.scale.copy()
                model_obj.scale = Vector((
                    current_scale.x * UE_SCALE_FACTOR,
                    current_scale.y * UE_SCALE_FACTOR,
                    current_scale.z * UE_SCALE_FACTOR
                ))
                
                if prefs.show_debug_info:
                    logger.log(f"[坐标转换] 转换后: scale={model_obj.scale}") if logger else print(f"[坐标转换] 转换后: scale={model_obj.scale}")

            elif target_engine == 'UNITY':
                # Unity转换（左手系，Y-up）
                if prefs.show_debug_info:
                    logger.log(f"[坐标转换] Unity模式") if logger else print(f"[坐标转换] Unity模式")
                
                # Unity使用米作为单位，但需要转换坐标轴
                # Blender: X-right, Y-forward, Z-up
                # Unity: X-right, Y-up, Z-forward
                # 这个转换在FBX导出时通过axis_forward和axis_up处理
                pass
            
            elif target_engine == 'BLENDER':
                # 保持Blender原生坐标系
                if prefs.show_debug_info:
                    logger.log(f"[坐标转换] Blender原生坐标系") if logger else print(f"[坐标转换] Blender原生坐标系")
                pass
            
            bpy.context.view_layer.update()
            
        except Exception as e:
            self.report({'WARNING'}, f"坐标系转换失败: {str(e)}")
            if prefs.show_debug_info:
                import traceback
                if logger:
                    logger.log(traceback.format_exc())
                else:
                    traceback.print_exc()
    
    def export_model(self, model_obj, obj_config, output_dir, file_format, prefs):
        """
        【完全复用操作符的导出逻辑】
        与OBJECT_OT_batch_box_preview_export.export_model完全一致
        """
        if file_format in ['.glb', '.gltf']:
            return alignment_ops.export_model_glb_helper(self.report, model_obj, obj_config, output_dir, prefs)
        elif file_format == '.obj':
            return alignment_ops.export_model_obj_helper(self.report, model_obj, obj_config, output_dir, prefs)
        else:
            # 默认导出为FBX
            return alignment_ops.export_model_fbx_helper(self.report, model_obj, obj_config, output_dir, prefs)


# ==================== 主函数 ====================

def main():
    """
    主处理函数 - 【完全复用操作符逻辑】
    与OBJECT_OT_batch_box_preview_export.execute()流程100%一致
    """
    global logger
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    if len(argv) < 5:
        print("错误: 参数不足")
        print("用法: blender --background --python alignment_script.py -- <input_json> <output_dir> <export_format> <target_engine> <output_json>")
        sys.exit(1)
    
    input_json_path = argv[0]
    output_dir = argv[1]
    export_format = argv[2]  # 'GLB' 或 'FBX'
    target_engine = argv[3]  # 'UE', 'UNITY', 'BLENDER', 'NONE'
    output_json_path = argv[4]
    
    # 创建日志文件（基于时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"blender_alignment_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    # 初始化日志系统
    logger = DualLogger(log_path)
    
    # 【关键】重定向所有print输出到日志文件（捕获alignment_ops的print）
    logger.redirect_output()
    
    logger.log("\n" + "="*80)
    logger.log("[Blender对齐脚本] 开始处理（完全复用操作符逻辑）")
    logger.log("="*80)
    logger.log(f"  输入JSON: {input_json_path}")
    logger.log(f"  输出目录: {output_dir}")
    logger.log(f"  导出格式: {export_format}")
    logger.log(f"  目标引擎: {target_engine}")
    logger.log(f"  日志文件: {log_path}")
    logger.log(f"  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"  重定向输出: 已启用（捕获所有print）")
    
    # 读取输入JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.log(f"[错误] 无法读取输入JSON: {str(e)}")
        logger.close()
        sys.exit(1)
    
    objects = config.get('objects', [])
    logger.log(f"  共 {len(objects)} 个对象")
    logger.log("="*80 + "\n")
    
    # 创建模拟对象（强制开启调试信息）
    prefs = MockPrefs(
        target_engine=target_engine,
        export_format=export_format,
        show_debug_info=True  # 强制开启，确保所有详细日志都输出
    )
    
    # 【验证】确认调试模式已开启
    logger.log(f"[调试模式] show_debug_info = {prefs.show_debug_info}")
    
    # 创建模拟操作符
    mock_operator = MockOperator(prefs)
    
    # 清空场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 认证信息
    auth_token = None
    
    # 【完全复用操作符的处理流程】
    success_count = 0
    failed_count = 0
    processed_objects = []
    model_box_pairs = []  # 存储(模型, BOX, 配置, 导出格式)
    
    # 阶段1：导入和对齐（与操作符完全一致）
    for idx, obj_config in enumerate(objects):
        try:
            obj_name = obj_config.get('name', f'object_{idx}')
            logger.log(f"\n[{idx+1}/{len(objects)}] 处理对象: {obj_name}")
            
            # 1. 下载并导入模型
            logger.log(f"  → 下载并导入模型")
            if prefs.show_debug_info:
                url = obj_config.get('3d_url') or obj_config.get('output', {}).get('url') or obj_config.get('url', '')
                logger.log(f"[批量预览] 下载模型: {url}")
            
            model_obj, file_format = alignment_ops.download_and_import_model_helper(
                obj_config=obj_config,
                auth_token=auth_token,
                prefs=prefs,
                report_func=mock_report
            )
            
            if not model_obj:
                failed_count += 1
                logger.log(f"  ✗ 失败: 无法导入模型")
                continue
            
            # 记录导入成功
            if prefs.show_debug_info:
                logger.log(f"[批量预览] 导入成功: {model_obj.name}")
            
            # 2. 应用JSON中的rotation和position到模型
            logger.log(f"  → 应用变换（rotation + position）")
            if prefs.show_debug_info:
                rotation = obj_config.get('rotation', [0, 0, 0])
                position = obj_config.get('position', [0, 0, 0])
                logger.log(f"[批量预览] 应用变换到模型:")
                logger.log(f"  rotation: {rotation}°")
                logger.log(f"  position: {position}")
            
            alignment_ops.apply_transform_to_model_helper(
                model_obj=model_obj,
                obj_config=obj_config,
                prefs=prefs,
                report_func=mock_report
            )
            
            # 3. 创建带position的参考Box
            logger.log(f"  → 创建参考Box（带position）")
            if prefs.show_debug_info:
                name = obj_config.get('name', 'ReferenceBox')
                rotation = obj_config.get('rotation', [0, 0, 0])
                scale = obj_config.get('scale', [1, 1, 1])
                position = obj_config.get('position', [0, 0, 0])
                if prefs.target_engine == 'UE':
                    position_display = [p * 0.01 for p in position]
                else:
                    position_display = position
                logger.log(f"[批量预览] 创建参考Box: box_{name}")
                logger.log(f"  位置: {position_display}")
                logger.log(f"  旋转: {rotation}°")
                logger.log(f"  缩放: {scale}")
            
            ref_box = alignment_ops.create_reference_box_with_position_helper(
                obj_config=obj_config,
                prefs=prefs,
                report_func=mock_report
            )
            
            if not ref_box:
                failed_count += 1
                logger.log(f"  ✗ 失败: 无法创建参考Box")
                bpy.data.objects.remove(model_obj, do_unlink=True)
                continue
            
            # 4. 执行完美重合对齐（使用带日志的版本）
            logger.log(f"  → 执行完美重合对齐")
            
            # 【使用带日志的对齐函数】完整输出所有对齐细节
            execute_perfect_align_with_logging(
                ref_obj=ref_box,
                target_obj=model_obj,
                prefs=prefs,
                log_func=logger.log  # 传入logger的log方法
            )
            
            # 【关键修复】强制更新视图层，确保对齐后的变换生效
            bpy.context.view_layer.update()
            
            # 【验证】记录对齐后的最终状态
            logger.log(f"[验证] 对齐后最终状态:")
            logger.log(f"  位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
            logger.log(f"  缩放: X={model_obj.scale.x:.3f} Y={model_obj.scale.y:.3f} Z={model_obj.scale.z:.3f}")
            logger.log(f"  尺寸: X={model_obj.dimensions.x:.3f} Y={model_obj.dimensions.y:.3f} Z={model_obj.dimensions.z:.3f}")
            
            # 5. 【关键】删除参考Box（后台模式不需要预览）
            # 注意：操作符中保留Box是为了预览，但后台脚本不需要
            logger.log(f"  → 删除参考Box（后台模式）")
            bpy.data.objects.remove(ref_box, do_unlink=True)
            bpy.context.view_layer.update()  # 删除后更新
            
            # 6. 使用用户设置的导出格式
            file_ext = '.glb' if export_format == 'GLB' else '.fbx'
            
            # 存储模型、配置和导出格式（与操作符一致）
            model_box_pairs.append((model_obj, None, obj_config, file_ext))  # ref_box已删除，设为None
            success_count += 1
            logger.log(f"  ✓ 对齐成功")
            
        except Exception as e:
            logger.log(f"  ✗ 错误: {str(e)}")
            failed_count += 1
            import traceback
            logger.log(traceback.format_exc())
    
    # 阶段2：导出（与操作符完全一致）
    format_name = 'GLB' if export_format == 'GLB' else 'FBX'
    logger.log(f"\n" + "="*80)
    logger.log(f"[导出阶段] 开始导出 {len(model_box_pairs)} 个模型（格式：{format_name}）")
    logger.log("="*80)
    
    for idx, (model_obj, ref_box, obj_config, file_format) in enumerate(model_box_pairs):
        try:
            obj_name = obj_config.get('name', f'object_{idx}')
            logger.log(f"\n[{idx+1}/{len(model_box_pairs)}] 导出: {obj_name}")
            
            # 【完全复用操作符的导出流程】
            
            # 保存模型当前变换（对齐后的位置和缩放）
            original_position = model_obj.location.copy()
            original_scale = model_obj.scale.copy()
            
            # 【调试】记录保存的原始值
            if prefs.show_debug_info:
                logger.log(f"[调试] 保存的原始变换（导出前）:")
                logger.log(f"  original_position: X={original_position.x:.3f} Y={original_position.y:.3f} Z={original_position.z:.3f}")
                logger.log(f"  original_scale: X={original_scale.x:.3f} Y={original_scale.y:.3f} Z={original_scale.z:.3f}")
            
            # 关键：导出前将模型position归零
            logger.log(f"[批量预览+导出] 导出前归零position: {model_obj.name}")
            if prefs.show_debug_info:
                logger.log(f"  当前位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
            
            model_obj.location = Vector((0, 0, 0))
            bpy.context.view_layer.update()
            
            if prefs.show_debug_info:
                logger.log(f"    归零后位置: X={model_obj.location.x:.3f} Y={model_obj.location.y:.3f} Z={model_obj.location.z:.3f}")
            
            # 应用坐标系转换（使用模拟操作符的方法，与原操作符完全一致）
            logger.log(f"  → 应用坐标系转换")
            mock_operator.apply_coordinate_transform(
                model_obj,
                prefs.target_engine,
                prefs,
                file_format
            )
            
            # 导出模型（使用模拟操作符的方法，与原操作符完全一致）
            logger.log(f"  → 导出模型")
            
            # 直接调用导出辅助函数，确保日志完整
            if file_format in ['.glb', '.gltf']:
                exported_path = alignment_ops.export_model_glb_helper(
                    mock_report,
                    model_obj,
                    obj_config,
                    output_dir,
                    prefs
                )
            elif file_format == '.obj':
                exported_path = alignment_ops.export_model_obj_helper(
                    mock_report,
                    model_obj,
                    obj_config,
                    output_dir,
                    prefs
                )
            else:
                exported_path = alignment_ops.export_model_fbx_helper(
                    mock_report,
                    model_obj,
                    obj_config,
                    output_dir,
                    prefs
                )
            
            # 【关键】导出后不需要恢复位置和缩放（后台模式下直接删除对象）
            # 注意：操作符中恢复是为了保留场景预览，但后台脚本不需要
            # 但为了调试，记录恢复信息
            if prefs.show_debug_info and exported_path:
                logger.log(f"[批量预览+导出] 导出后恢复变换: {model_obj.name}")
                logger.log(f"  恢复位置: X={original_position.x:.3f} Y={original_position.y:.3f} Z={original_position.z:.3f}")
                logger.log(f"  恢复缩放: X={original_scale.x:.3f} Y={original_scale.y:.3f} Z={original_scale.z:.3f}")
            
            if exported_path:
                # 创建新的对象配置，rotation设为0，position保持原始值
                new_obj_config = obj_config.copy()
                new_obj_config['rotation'] = [0, 0, 0]
                # position保持原始值，不修改（与操作符一致）
                
                if prefs.show_debug_info:
                    logger.log(f"[批量预览+导出] 输出JSON:")
                    logger.log(f"  position: {new_obj_config.get('position')} (保持原始)")
                    logger.log(f"  rotation: [0, 0, 0] (已烘焙)")
                    logger.log(f"  scale: {new_obj_config.get('scale')} (保持原始)")
                
                # 更新URL字段（与操作符一致）
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
                logger.log(f"[批量预览+导出] 导出成功: {exported_path}")
            else:
                logger.log(f"  ✗ 导出失败")
            
            # 删除模型对象（后台模式不需要保留）
            bpy.data.objects.remove(model_obj, do_unlink=True)
            
        except Exception as e:
            logger.log(f"  ✗ 导出错误: {str(e)}")
            import traceback
            logger.log(traceback.format_exc())
    
    # 生成输出JSON（与操作符一致）
    output_config = config.copy()
    output_config['objects'] = processed_objects
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_config, f, indent=2, ensure_ascii=False)
        
        logger.log(f"\n" + "="*80)
        logger.log(f"[Blender对齐脚本] 处理完成")
        logger.log(f"  成功: {success_count} 个")
        logger.log(f"  失败: {failed_count} 个")
        logger.log(f"  导出: {len(processed_objects)} 个")
        logger.log(f"  输出JSON: {output_json_path}")
        logger.log(f"  日志文件: {log_path}")
        logger.log(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("="*80 + "\n")
        
        if prefs.show_debug_info:
            logger.log("[输出JSON内容]:")
            logger.log(json.dumps(output_config, indent=2, ensure_ascii=False))
        
        # 关闭日志文件
        logger.close()
        
        # 最后输出日志位置（不写入日志文件）
        print(f"\n✅ 处理完成！详细日志已保存到: {log_path}")
            
    except Exception as e:
        logger.log(f"[错误] 无法写入输出JSON: {str(e)}")
        logger.close()
        sys.exit(1)


# 执行主函数
if __name__ == "__main__":
    main()
