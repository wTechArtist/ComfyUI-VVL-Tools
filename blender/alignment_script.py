"""
Blender批量对齐脚本
被ComfyUI节点调用，在Blender后台执行模型对齐和导出
完全调用 alignment_ops.py 中的现有功能，不重新实现
【与批量预览+导出功能一致】使用position，输出JSON保持原始position
"""
import bpy
import sys
import os
import json
import tempfile

# 动态加载 alignment_ops 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
alignment_ops_path = os.path.join(script_dir, "alignment_ops.py")

# 导入 alignment_ops 模块（必需依赖）
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("alignment_ops", alignment_ops_path)
    alignment_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alignment_ops)
    
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
    print(f"[{level_str}] {message}")


# ==================== 主函数 ====================

def main():
    """主处理函数 - 完全调用 alignment_ops.py 的功能"""
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
    
    print(f"\n[Blender对齐脚本] 开始处理")
    print(f"  输入JSON: {input_json_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  导出格式: {export_format}")
    print(f"  目标引擎: {target_engine}")
    
    # 读取输入JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[错误] 无法读取输入JSON: {str(e)}")
        sys.exit(1)
    
    objects = config.get('objects', [])
    print(f"  共 {len(objects)} 个对象\n")
    
    # 创建模拟对象
    prefs = MockPrefs(
        target_engine=target_engine,
        export_format=export_format,
        show_debug_info=True
    )
    
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
            
            # 1. 下载并导入模型（完全调用 alignment_ops.py）
            model_obj, file_format = alignment_ops.download_and_import_model_helper(
                obj_config=obj_config,
                auth_token=None,
                prefs=prefs,
                report_func=mock_report
            )
            
            if not model_obj:
                print(f"  ✗ 失败: 无法导入模型")
                failed_count += 1
                continue
            
            # 2. 应用JSON中的rotation和position到模型（完全调用 alignment_ops.py）
            # 【与批量预览+导出一致】使用原始position和rotation
            from mathutils import Vector
            alignment_ops.apply_transform_to_model_helper(
                model_obj=model_obj,
                obj_config=obj_config,
                prefs=prefs,
                report_func=mock_report
            )
            
            # 3. 创建带position的参考Box（完全调用 alignment_ops.py）
            # 【与批量预览+导出一致】使用原始position
            ref_box = alignment_ops.create_reference_box_with_position_helper(
                obj_config=obj_config,
                prefs=prefs,
                report_func=mock_report
            )
            
            if not ref_box:
                print(f"  ✗ 失败: 无法创建参考Box")
                bpy.data.objects.remove(model_obj, do_unlink=True)
                failed_count += 1
                continue
            
            # 4. 执行完美重合对齐（完全调用 alignment_ops.py）
            # 注意：使用 execute_perfect_align_batch_helper 而不是 align_model_to_box_preview_helper
            # 因为后者包含 bpy.ops 操作符，在后台模式下可能崩溃
            print(f"  → 对齐到参考Box")
            alignment_ops.execute_perfect_align_batch_helper(
                ref_obj=ref_box,
                target_obj=model_obj,
                prefs=prefs
            )
            
            # 5. 删除参考Box（独立脚本不需要预览，所以删除）
            bpy.data.objects.remove(ref_box, do_unlink=True)
            
            # 6. 导出前将模型position归零
            # 【与批量预览+导出一致】导出时position归零
            print(f"  → 导出前归零position")
            model_obj.location = Vector((0, 0, 0))
            bpy.context.view_layer.update()
            
            # 7. 应用坐标系转换（完全调用 alignment_ops.py）
            print(f"  → 应用坐标系转换")
            export_ext = export_format.lower() if export_format else 'glb'
            alignment_ops.apply_coordinate_transform_helper(
                model_obj=model_obj,
                target_engine=target_engine,
                export_format=export_ext,
                prefs=prefs
            )
            
            # 8. 导出模型（完全调用 alignment_ops.py）
            print(f"  → 导出模型")
            
            if export_format == 'GLB':
                exported_path = alignment_ops.export_model_glb_helper(
                    report_func=mock_report,
                    model_obj=model_obj,
                    obj_config=obj_config,
                    output_dir=output_dir,
                    prefs=prefs
                )
            elif export_format == 'FBX':
                exported_path = alignment_ops.export_model_fbx_helper(
                    report_func=mock_report,
                    model_obj=model_obj,
                    obj_config=obj_config,
                    output_dir=output_dir,
                    prefs=prefs
                )
            elif export_format == 'OBJ':
                exported_path = alignment_ops.export_model_obj_helper(
                    report_func=mock_report,
                    model_obj=model_obj,
                    obj_config=obj_config,
                    output_dir=output_dir,
                    prefs=prefs
                )
            else:
                # 默认使用GLB
                exported_path = alignment_ops.export_model_glb_helper(
                    report_func=mock_report,
                    model_obj=model_obj,
                    obj_config=obj_config,
                    output_dir=output_dir,
                    prefs=prefs
                )
            
            if not exported_path:
                print(f"  ✗ 失败: 导出失败")
                bpy.data.objects.remove(model_obj, do_unlink=True)
                failed_count += 1
                continue
            
            # 9. 创建输出配置
            # 【与批量预览+导出一致】rotation归零，position保持原始值
            new_obj_config = obj_config.copy()
            new_obj_config['rotation'] = [0, 0, 0]  # 旋转已烘焙到模型
            # position保持原始值，不修改（与批量预览+导出一致）
            
            print(f"  → 输出配置:")
            print(f"    position: {new_obj_config.get('position')} (保持原始)")
            print(f"    rotation: [0, 0, 0] (已烘焙)")
            print(f"    scale: {new_obj_config.get('scale')} (保持原始)")
            
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
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[错误] 无法写入输出JSON: {str(e)}")
        sys.exit(1)
    
    print(f"\n[Blender对齐脚本] 处理完成")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {failed_count} 个")
    print(f"  输出: {output_json_path}")


# 执行主函数
if __name__ == "__main__":
    main()
