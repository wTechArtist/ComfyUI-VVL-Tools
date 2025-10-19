"""
Blender批量对齐脚本
被ComfyUI节点调用，在Blender后台执行模型对齐和导出
【直接调用操作符】与Blender UI的"批量Box对齐预览+导出"按钮100%一致
"""
import bpy
import sys
import os
import json
from datetime import datetime

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
            self.original_stdout.write(message + '\n')
            self.original_stdout.flush()
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
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

# ==================== 常量定义 ====================

# 插件名称（与 Blender UI 插件保持一致）
ADDON_NAME = "model_alignment_tool"
FALLBACK_ADDON_NAME = "alignment_ops"  # 兼容旧版本

# ==================== 全局变量 ====================

# 全局日志对象
logger = None

# ==================== 动态加载模块 ====================

# 动态加载 alignment_ops 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
alignment_ops_path = os.path.join(script_dir, "alignment_ops.py")

# 导入 alignment_ops 模块（包含操作符类）
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("alignment_ops", alignment_ops_path)
    alignment_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alignment_ops)

    # 强制覆盖插件名称函数，保持与Blender插件一致
    alignment_ops.get_addon_name = lambda: ADDON_NAME
    alignment_ops.ADDON_NAME = ADDON_NAME

    # 提供外部偏好设置占位，便于回退
    alignment_ops._external_addon_prefs = None
    
    print(f"[Blender脚本] ✓ 成功加载 alignment_ops 模块")
except Exception as e:
    print(f"[Blender脚本] ✗ 错误: 无法加载 alignment_ops.py")
    print(f"[Blender脚本] 详情: {str(e)}")
    print(f"[Blender脚本] 路径: {alignment_ops_path}")
    sys.exit(1)


# ==================== 模拟插件环境 ====================

class FakeAddonPreferences:
    """
    模拟插件偏好设置对象（对应 Blender UI 中的 AddonPreferences）
    
    这个类模拟了 model_alignment_tool 插件的偏好设置，包含：
    - target_engine: 目标引擎（UE/Unity/Blender/None）
    - export_format: 导出格式（GLB/FBX）
    - batch_output_dir: 批量导出的输出目录
    - batch_json_config: 输入的 JSON 配置（批量对齐的场景数据）
    - batch_output_json: 输出的 JSON 配置（处理后的场景数据）
    - show_debug_info: 是否显示调试信息（始终为 True，确保完整日志）
    """
    def __init__(self, target_engine, export_format, output_dir):
        self.target_engine = target_engine
        self.export_format = export_format
        self.batch_output_dir = output_dir
        self.batch_json_config = ""
        self.batch_output_json = ""
        self.show_debug_info = True  # 始终开启调试信息

class FakeAddon:
    """
    模拟插件对象（对应 context.preferences.addons[插件名]）
    
    在 Blender 真实环境中，插件通过 bpy.utils.register_class() 注册后，
    可以通过 context.preferences.addons[插件名] 访问。
    这个类模拟了这个结构，提供 .preferences 属性。
    """
    def __init__(self, preferences):
        self.preferences = preferences

class FakePreferences:
    """
    模拟 Blender 偏好设置对象（对应 context.preferences）
    
    在 Blender 真实环境中，所有已安装的插件都注册在 context.preferences.addons 中。
    这个类模拟了这个结构，同时注册两个插件名以兼容不同版本：
    - model_alignment_tool: 标准插件名（与 Blender UI 一致）
    - alignment_ops: 回退插件名（兼容旧版本或内部调用）
    """
    def __init__(self, addon_prefs):
        addon_key = alignment_ops.get_addon_name()
        fake_addon = FakeAddon(addon_prefs)

        self.addons = {
            addon_key: fake_addon
        }

        # 兼容：部分逻辑仍然可能使用原模块名
        if addon_key != FALLBACK_ADDON_NAME:
            self.addons[FALLBACK_ADDON_NAME] = fake_addon

class FakeContext:
    """
    模拟 Blender 上下文对象（对应 bpy.context）
    
    只模拟必要的属性：
    - view_layer: 使用真实的 bpy.context.view_layer（场景渲染层）
    - preferences: 使用模拟的 FakePreferences（插件偏好设置）
    
    其他属性（如 scene, window_manager 等）通过 context.temp_override 从真实上下文获取。
    """
    def __init__(self, addon_prefs):
        self.view_layer = bpy.context.view_layer
        self.preferences = FakePreferences(addon_prefs)


def _wrap_alignment_operator_execute():
    """
    为操作符添加包装，确保在无真实插件注册时也能正常运行
    
    工作原理：
    1. 首先尝试使用 Blender 真实的插件偏好设置（正常插件运行模式）
    2. 如果真实偏好设置不存在（headless 脚本模式），则自动回退到模拟对象
    3. 通过 _ContextProxy 代理上下文，将模拟的 preferences 注入到操作符中
    
    这样做的好处：
    - ComfyUI 后台脚本可以直接调用操作符，无需手动模拟所有逻辑
    - 与 Blender UI 插件 100% 共享相同的代码和执行流程
    - 后续修改 alignment_ops.py 会自动同步到 ComfyUI
    """
    # 防止重复包装
    if getattr(alignment_ops, "_execute_wrapped", False):
        return

    def _make_wrapper(original_execute):
        """创建操作符 execute 方法的包装器"""
        def _wrapped(self, context):
            addon_key = alignment_ops.get_addon_name()

            # 尝试使用真实的插件偏好设置
            try:
                addons = getattr(context.preferences, "addons", {})
                if isinstance(addons, dict):
                    # 字典类型（模拟环境）
                    if addon_key in addons or FALLBACK_ADDON_NAME in addons:
                        return original_execute(self, context)
                else:
                    # bpy_prop_collection 类型（真实 Blender 环境）
                    # 尝试访问，不存在会抛异常
                    addons[addon_key]
                    return original_execute(self, context)
            except Exception:
                # 真实偏好设置不存在，继续使用模拟对象
                pass

            # 回退方案：使用外部注入的模拟偏好设置
            fake_prefs_obj = getattr(alignment_ops, "_external_addon_prefs", None)
            if not fake_prefs_obj:
                raise KeyError(
                    f"Addon preferences '{addon_key}' not found and no fake prefs provided. "
                    f"Please set alignment_ops._external_addon_prefs before calling the operator."
                )

            # 创建模拟的 preferences
            fake_preferences = FakePreferences(fake_prefs_obj)

            # 创建上下文代理，将模拟的 preferences 注入
            class _ContextProxy:
                """代理 Blender 上下文对象，仅替换 preferences 属性"""
                def __init__(self, base_context, fake_preferences):
                    self._base_context = base_context
                    self._fake_preferences = fake_preferences

                def __getattr__(self, item):
                    if item == "preferences":
                        return self._fake_preferences
                    # 其他属性从真实上下文获取
                    return getattr(self._base_context, item)

            proxy_context = _ContextProxy(context, fake_preferences)
            return original_execute(self, proxy_context)

        return _wrapped

    # 包装两个批量对齐操作符
    alignment_ops.OBJECT_OT_batch_box_preview.execute = _make_wrapper(
        alignment_ops.OBJECT_OT_batch_box_preview.execute
    )
    alignment_ops.OBJECT_OT_batch_box_preview_export.execute = _make_wrapper(
        alignment_ops.OBJECT_OT_batch_box_preview_export.execute
    )

    # 标记已包装，防止重复
    alignment_ops._execute_wrapped = True


_wrap_alignment_operator_execute()


# ==================== 主函数 ====================

def main():
    """
    主处理函数 - 【直接调用操作符】
    100%复用OBJECT_OT_batch_box_preview_export的完整逻辑
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
    
    # 【关键】重定向所有print输出到日志文件
    logger.redirect_output()
    
    logger.log("\n" + "="*80)
    logger.log("[Blender对齐脚本] 开始处理（直接调用操作符）")
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
            json_text = json.dumps(config, ensure_ascii=False)
    except Exception as e:
        logger.log(f"[错误] 无法读取输入JSON: {str(e)}")
        logger.close()
        sys.exit(1)
    
    objects = config.get('objects', [])
    logger.log(f"  共 {len(objects)} 个对象")
    logger.log("="*80 + "\n")
    
    # 【关键】创建假的插件偏好设置
    addon_prefs = FakeAddonPreferences(
        target_engine=target_engine,
        export_format=export_format,
        output_dir=output_dir
    )
    
    # 设置JSON配置（通过偏好设置传递给操作符）
    addon_prefs.batch_json_config = json_text
    
    logger.log(f"[初始化] 创建模拟插件环境")
    logger.log(f"  target_engine: {addon_prefs.target_engine}")
    logger.log(f"  export_format: {addon_prefs.export_format}")
    logger.log(f"  batch_output_dir: {addon_prefs.batch_output_dir}")
    logger.log(f"  show_debug_info: {addon_prefs.show_debug_info}\n")
    
    # 创建假的context对象
    fake_context = FakeContext(addon_prefs)
    
    # 【关键】将模拟偏好设置注入到alignment_ops，供操作符包装回退使用
    alignment_ops._external_addon_prefs = addon_prefs

    # 清空场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 【关键】注册操作符
    try:
        logger.log(f"[注册操作符] 开始注册 OBJECT_OT_batch_box_preview_export...")
        
        # 注册操作符类
        bpy.utils.register_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        
        logger.log(f"[注册操作符] ✓ 成功注册")
        logger.log(f"  bl_idname: alignment.batch_box_preview_export")
        logger.log(f"  可通过 bpy.ops.alignment.batch_box_preview_export() 调用\n")
        
    except Exception as e:
        logger.log(f"[错误] 注册操作符失败: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.close()
        sys.exit(1)
    
    # 【关键】通过 context override 调用操作符
    try:
        logger.log(f"[调用操作符] 通过 context override 调用 bpy.ops.alignment.batch_box_preview_export()...")
        logger.log(f"="*80 + "\n")

        # 构造override上下文（仅覆盖preferences，其他使用真实上下文）
        override_kwargs = {
            "view_layer": bpy.context.view_layer,
            "scene": bpy.context.scene,
            "preferences": fake_context.preferences,
        }

        # window_manager在后台模式可能不存在，存在则一并覆盖
        if hasattr(bpy.context, "window_manager") and bpy.context.window_manager:
            override_kwargs["window_manager"] = bpy.context.window_manager

        logger.log(f"  override keys: {', '.join(sorted(override_kwargs.keys()))}")

        # 使用临时上下文调用操作符（与Blender UI点击按钮完全相同）
        with bpy.context.temp_override(**override_kwargs):
            result = bpy.ops.alignment.batch_box_preview_export()
        
        logger.log(f"\n" + "="*80)
        if result == {'FINISHED'}:
            logger.log(f"[调用操作符] ✓ 操作符执行成功 (返回 FINISHED)")
        elif result == {'CANCELLED'}:
            logger.log(f"[调用操作符] ✗ 操作符执行取消 (返回 CANCELLED)")
        else:
            logger.log(f"[调用操作符] ⚠ 操作符返回: {result}")
        logger.log(f"="*80 + "\n")
        
    except Exception as e:
        logger.log(f"\n[错误] 操作符执行失败: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        
        # 注销操作符
        try:
            bpy.utils.unregister_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        except:
            pass
        
        logger.close()
        sys.exit(1)
    finally:
        # 清理外部偏好设置引用
        alignment_ops._external_addon_prefs = None
    
    # 【关键】获取操作符的输出JSON
    output_json_text = addon_prefs.batch_output_json
    
    if not output_json_text:
        logger.log(f"[错误] 操作符没有生成输出JSON")
        logger.log(f"  可能原因: 所有对象处理失败，或操作符内部错误")
        
        # 注销操作符
        try:
            bpy.utils.unregister_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        except:
            pass
        
        logger.close()
        sys.exit(1)
    
    # 解析输出JSON
    try:
        output_config = json.loads(output_json_text)
    except Exception as e:
        logger.log(f"[错误] 无法解析输出JSON: {str(e)}")
        
        # 注销操作符
        try:
            bpy.utils.unregister_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        except:
            pass
        
        logger.close()
        sys.exit(1)
    
    # 保存输出JSON到文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_config, f, indent=2, ensure_ascii=False)
        
        logger.log(f"[保存JSON] ✓ 输出JSON已保存")
        logger.log(f"  路径: {output_json_path}")
        logger.log(f"  对象数量: {len(output_config.get('objects', []))}\n")
        
    except Exception as e:
        logger.log(f"[错误] 无法写入输出JSON: {str(e)}")
        
        # 注销操作符
        try:
            bpy.utils.unregister_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        except:
            pass
        
        logger.close()
        sys.exit(1)
    
    # 注销操作符
    try:
        bpy.utils.unregister_class(alignment_ops.OBJECT_OT_batch_box_preview_export)
        logger.log(f"[注销操作符] ✓ 成功注销\n")
    except Exception as e:
        logger.log(f"[警告] 注销操作符失败: {str(e)}\n")
    
    # 最终总结
    logger.log(f"="*80)
    logger.log(f"[Blender对齐脚本] 处理完成")
    logger.log(f"  成功处理: {len(output_config.get('objects', []))} 个对象")
    logger.log(f"  输出JSON: {output_json_path}")
    logger.log(f"  日志文件: {log_path}")
    logger.log(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"="*80 + "\n")
    
    # 关闭日志文件
    logger.close()
    
    # 最后输出日志位置（不写入日志文件）
    print(f"\n✅ 处理完成！详细日志已保存到: {log_path}")


# 执行主函数
if __name__ == "__main__":
    main()
