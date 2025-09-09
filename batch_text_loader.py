"""
VVL Batch Text Loader Node
参考 WAS Load Image Batch 节点，实现批量加载TXT文件的功能
"""

import os
import glob
import random
import hashlib
import logging

logger = logging.getLogger(__name__)

# 支持的文本文件扩展名
ALLOWED_TEXT_EXT = ('.txt', '.text', '.md', '.log', '.csv', '.json', '.yaml', '.yml')

class VVL_Load_Text_Batch:
    """
    批量加载文本文件节点
    支持单个文件、递增模式、随机模式
    """
    def __init__(self):
        # 简化存储，不依赖外部数据库
        self.counters = {}
        self.paths = {}
        self.patterns = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_file", "incremental_file", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'TextBatch001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*.txt', "multiline": False}),
                "encoding": (["utf-8", "gbk", "gb2312", "utf-16", "ascii"],),
            },
            "optional": {
                "filename_only": (["false", "true"],),
                "skip_empty": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LIST", "INT")
    RETURN_NAMES = ("text_content", "filename", "all_lines", "total_files")
    FUNCTION = "load_batch_texts"

    CATEGORY = "VVL-Tools/IO"

    def load_batch_texts(self, path, pattern='*.txt', index=0, mode="single_file", 
                        seed=0, label='TextBatch001', encoding='utf-8', 
                        filename_only='false', skip_empty='true'):
        
        # 检查路径是否存在
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return ("", "", [], 0)
        
        if not os.path.isdir(path):
            logger.error(f"Path is not a directory: {path}")
            return ("", "", [], 0)

        # 创建文件加载器
        fl = self.BatchTextLoader(path, label, pattern, encoding, skip_empty == 'true')
        text_paths = fl.text_paths
        
        if not text_paths:
            logger.warning(f"No text files found in {path} with pattern {pattern}")
            return ("", "", [], 0)

        text_content = ""
        filename = ""
        
        # 根据模式选择文件
        if mode == 'single_file':
            text_content, filename = fl.get_text_by_id(index)
            if text_content is None:
                logger.error(f"No valid text file found for index {index}")
                return ("", "", [], len(text_paths))
        elif mode == 'incremental_file':
            text_content, filename = fl.get_next_text()
            if text_content is None:
                logger.error("No valid text file found for next ID")
                return ("", "", [], len(text_paths))
        else:  # random mode
            random.seed(seed)
            newindex = int(random.random() * len(text_paths))
            text_content, filename = fl.get_text_by_id(newindex)
            if text_content is None:
                logger.error("No valid text file found for random selection")
                return ("", "", [], len(text_paths))

        # 处理文件名选项
        if filename_only == "true":
            filename = os.path.splitext(filename)[0]

        # 将文本内容按行分割成列表
        all_lines = text_content.splitlines() if text_content else []

        return (text_content, filename, all_lines, len(text_paths))

    class BatchTextLoader:
        """内部文本文件加载器类"""
        def __init__(self, directory_path, label, pattern, encoding='utf-8', skip_empty=True):
            self.text_paths = []
            self.encoding = encoding
            self.skip_empty = skip_empty
            self.label = label
            self.load_texts(directory_path, pattern)
            self.text_paths.sort()
            
            # 简化的状态管理（内存中）
            if not hasattr(VVL_Load_Text_Batch, '_state'):
                VVL_Load_Text_Batch._state = {}
            
            state_key = f"{label}_{directory_path}_{pattern}"
            if state_key not in VVL_Load_Text_Batch._state:
                VVL_Load_Text_Batch._state[state_key] = {'index': 0}
            
            self.state = VVL_Load_Text_Batch._state[state_key]
            self.index = self.state['index']

        def load_texts(self, directory_path, pattern):
            """加载匹配模式的文本文件"""
            try:
                search_pattern = os.path.join(directory_path, pattern)
                for file_path in glob.glob(search_pattern, recursive=True):
                    if os.path.isfile(file_path) and file_path.lower().endswith(ALLOWED_TEXT_EXT):
                        abs_file_path = os.path.abspath(file_path)
                        self.text_paths.append(abs_file_path)
                logger.info(f"Found {len(self.text_paths)} text files with pattern {pattern}")
            except Exception as e:
                logger.error(f"Error loading text files: {e}")

        def get_text_by_id(self, text_id):
            """根据索引获取文本内容"""
            if text_id < 0 or text_id >= len(self.text_paths):
                logger.error(f"Invalid text file index: {text_id}")
                return None, None
            
            return self._read_text_file(self.text_paths[text_id])

        def get_next_text(self):
            """获取下一个文本文件"""
            if not self.text_paths:
                return None, None
                
            if self.index >= len(self.text_paths):
                self.index = 0
            
            text_path = self.text_paths[self.index]
            result = self._read_text_file(text_path)
            
            self.index += 1
            if self.index >= len(self.text_paths):
                self.index = 0
                
            # 更新状态
            self.state['index'] = self.index
            logger.info(f"Loaded text file {self.index}/{len(self.text_paths)}: {os.path.basename(text_path)}")
            
            return result

        def _read_text_file(self, file_path):
            """读取单个文本文件"""
            try:
                with open(file_path, 'r', encoding=self.encoding, errors='replace') as f:
                    content = f.read()
                
                # 跳过空文件选项
                if self.skip_empty and not content.strip():
                    logger.warning(f"Skipping empty file: {os.path.basename(file_path)}")
                    return "", os.path.basename(file_path)
                
                filename = os.path.basename(file_path)
                return content, filename
                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return f"Error reading file: {e}", os.path.basename(file_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """检测文件变化"""
        if kwargs['mode'] != 'single_file':
            return float("NaN")
        
        try:
            path = kwargs.get('path', '')
            pattern = kwargs.get('pattern', '*.txt')
            
            if not os.path.exists(path):
                return False
                
            # 生成目录内容的哈希值
            search_pattern = os.path.join(path, pattern)
            files = glob.glob(search_pattern, recursive=True)
            files = [f for f in files if os.path.isfile(f) and f.lower().endswith(ALLOWED_TEXT_EXT)]
            files.sort()
            
            # 基于文件列表和修改时间生成哈希
            content_hash = hashlib.md5()
            for file_path in files:
                try:
                    stat = os.stat(file_path)
                    content_hash.update(f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode())
                except:
                    continue
                    
            return content_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Error in IS_CHANGED: {e}")
            return False

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VVL_Load_Text_Batch": VVL_Load_Text_Batch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_Load_Text_Batch": "VVL Load Text Batch",
}