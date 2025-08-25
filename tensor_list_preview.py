"""
Tensor List Preview Node for ComfyUI-VVL-Tools

This module provides a node to preview tensor lists with special format:
- Input: List of tensors with shape [1, height, width, 3]  
- Output: Preview images using ComfyUI's preview mechanism

Author: VVL Tools
"""

import torch
import numpy as np
import os
import json
import random
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args


class TensorListPreview:
    """
    预览特殊格式的tensor列表节点
    
    处理格式为 [tensor([1, H, W, 3]), tensor([1, H, W, 3]), ...] 的tensor列表
    将其转换为标准图像格式并显示预览
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_tensor_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_list": ("*", {"tooltip": "包含多个tensor的列表，每个tensor格式为[1, height, width, 3]"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_tensor_list"
    OUTPUT_NODE = True
    CATEGORY = "VVL-Tools/image"
    DESCRIPTION = "预览特殊格式的tensor列表，每个tensor应为[1, height, width, 3]格式"

    def preview_tensor_list(self, tensor_list, prompt=None, extra_pnginfo=None):
        """
        预览tensor列表
        
        Args:
            tensor_list: 包含tensor的列表，每个tensor格式为[1, H, W, 3]
            prompt: 提示信息
            extra_pnginfo: 额外的PNG信息
            
        Returns:
            dict: 包含UI预览信息的字典
        """
        if not isinstance(tensor_list, list):
            raise ValueError(f"输入必须是列表格式，当前类型: {type(tensor_list)}")
        
        if len(tensor_list) == 0:
            raise ValueError("tensor列表不能为空")
        
        # 创建保存目录
        full_output_folder = self.output_dir
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)
        
        results = []
        counter = 0
        
        for i, tensor in enumerate(tensor_list):
            try:
                # 检查tensor格式
                if not isinstance(tensor, torch.Tensor):
                    print(f"警告: 第{i}个元素不是tensor，类型: {type(tensor)}")
                    continue
                    
                # 打印tensor信息用于调试
                print(f"处理第{i}个tensor，形状: {tensor.shape}")
                
                # 处理不同可能的tensor格式
                if len(tensor.shape) == 4:
                    # 格式 [1, H, W, 3] 或 [batch, H, W, 3]
                    if tensor.shape[0] == 1:
                        # 移除第一个维度: [1, H, W, 3] -> [H, W, 3]
                        image_tensor = tensor.squeeze(0)
                    else:
                        # 取第一个batch: [B, H, W, 3] -> [H, W, 3]
                        image_tensor = tensor[0]
                elif len(tensor.shape) == 3:
                    # 格式已经是 [H, W, 3]
                    image_tensor = tensor
                else:
                    print(f"警告: 第{i}个tensor形状不支持: {tensor.shape}")
                    continue
                
                # 确保数据在正确的设备和数据类型
                image_tensor = image_tensor.cpu().float()
                
                # 确保数值范围在[0, 1]
                if image_tensor.max() > 1.0 or image_tensor.min() < 0.0:
                    print(f"警告: 第{i}个tensor数值范围异常: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
                    # 将数值夹到[0, 1]范围
                    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
                
                # 转换为numpy并缩放到[0, 255]
                image_np = (image_tensor.numpy() * 255.0).astype(np.uint8)
                
                # 创建PIL图像
                img = Image.fromarray(image_np)
                
                # 准备元数据
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    
                    # 添加tensor信息到元数据
                    metadata.add_text("tensor_info", f"Original shape: {tensor.shape}, Index: {i}")
                
                # 生成文件名
                filename = f"tensor_preview_{self.prefix_append}_{counter:05d}.png"
                filepath = os.path.join(full_output_folder, filename)
                
                # 保存图片
                img.save(filepath, pnginfo=metadata, compress_level=self.compress_level)
                
                # 添加到结果列表
                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                })
                
                counter += 1
                print(f"成功处理第{i}个tensor，保存为: {filename}")
                
            except Exception as e:
                print(f"处理第{i}个tensor时发生错误: {str(e)}")
                continue
        
        if len(results) == 0:
            raise ValueError("没有成功处理任何tensor，请检查输入数据格式")
        
        print(f"总共成功处理了 {len(results)} 个tensor")
        return {"ui": {"images": results}}


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "TensorListPreview": TensorListPreview,
}

# 节点显示名称映射  
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorListPreview": "VVL Tensor List Preview",
}