"""
Enhanced Lambert 3D Renderer ComfyUI Node
A custom node for rendering 3D boxes with enhanced Lambert lighting
"""

import numpy as np
from PIL import Image, ImageDraw
import math
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
except ImportError:
    # Fallback for older ComfyUI versions
    class ComfyNodeABC:
        pass
    def IO():
        pass
    def InputTypeDict():
        pass


class EnhancedLambert3DRenderer:
    """Core 3D rendering engine"""
    
    def __init__(self, width=800, height=600, background_color="#FFFFFF", box_color="#C8C8C8"):
        """Initialize enhanced 3D renderer"""
        self.width = width
        self.height = height
        self.fov = 60
        self.near_clip = 0.1
        self.far_clip = 100.0
        
        # Light settings
        self.light_direction = np.array([1, 1, 1])  # Light direction
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
        self.ambient_light = 0.2  # Ambient light
        self.diffuse_light = 1.0  # Diffuse light
        
        # Colors
        self.background_color = self._parse_color(background_color)
        self.base_color = self._parse_color(box_color)
        
    def _parse_color(self, color_str):
        """将十六进制颜色字符串转换为RGB数组 (0-1范围)"""
        if color_str.startswith('#'):
            color_str = color_str[1:]
        
        try:
            r = int(color_str[0:2], 16) / 255.0
            g = int(color_str[2:4], 16) / 255.0
            b = int(color_str[4:6], 16) / 255.0
            return np.array([r, g, b])
        except (ValueError, IndexError):
            # 如果解析失败，返回默认颜色
            return np.array([0.8, 0.8, 0.8])
        
    def calculate_camera_distance(self, scale):
        """根据包围球与垂直视场计算相机距离，使画面更充满。

        公式：d = r / tan(fov/2) * margin，其中 r 为包围球半径。
        该方法比先前的比例法更稳健，并在默认尺寸下约为 5m。
        """
        w, h, d = scale
        # 包围球半径
        radius = 0.5 * math.sqrt(w * w + h * h + d * d)
        fov_rad = math.radians(self.fov)
        base_distance = radius / max(1e-6, math.tan(fov_rad / 2.0))
        margin = 1.2  # 少量边距，避免充满导致裁剪
        return base_distance * margin
        
    def create_perspective_matrix(self):
        """Create perspective projection matrix"""
        aspect_ratio = self.width / self.height
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        return np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far_clip + self.near_clip) / (self.near_clip - self.far_clip), 
             (2 * self.far_clip * self.near_clip) / (self.near_clip - self.far_clip)],
            [0, 0, -1, 0]
        ])
    
    def create_view_matrix(self, camera_pos, target_pos, up_vector):
        """Create view transformation matrix"""
        forward = np.array(target_pos) - np.array(camera_pos)
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array(up_vector))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        return np.array([
            [right[0], right[1], right[2], -np.dot(right, camera_pos)],
            [up[0], up[1], up[2], -np.dot(up, camera_pos)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, camera_pos)],
            [0, 0, 0, 1]
        ])
    
    def generate_box_geometry(self, scale):
        """Generate box geometry using standard axes: X=width, Y=height, Z=length"""
        # 约定：scale = [width (X), height (Y), length (Z)]
        w, h, l = scale
        
        vertices = np.array([
            [-w/2, -h/2, -l/2],  # 0: left-bottom-back
            [w/2, -h/2, -l/2],   # 1: right-bottom-back
            [w/2, h/2, -l/2],    # 2: right-top-back
            [-w/2, h/2, -l/2],   # 3: left-top-back
            [-w/2, -h/2, l/2],   # 4: left-bottom-front
            [w/2, -h/2, l/2],    # 5: right-bottom-front
            [w/2, h/2, l/2],     # 6: right-top-front
            [-w/2, h/2, l/2]     # 7: left-top-front
        ])
        
        # Define 6 faces with color info based on base color
        base = self.base_color
        faces = [
            {'vertices': [4, 5, 6, 7], 'normal': np.array([0, 0, 1]), 'name': 'Front', 'base_color': base},
            {'vertices': [1, 0, 3, 2], 'normal': np.array([0, 0, -1]), 'name': 'Back', 'base_color': base * 0.85},
            {'vertices': [5, 1, 2, 6], 'normal': np.array([1, 0, 0]), 'name': 'Right', 'base_color': base * 0.9},
            {'vertices': [0, 4, 7, 3], 'normal': np.array([-1, 0, 0]), 'name': 'Left', 'base_color': base * 0.9},
            {'vertices': [3, 7, 6, 2], 'normal': np.array([0, 1, 0]), 'name': 'Top', 'base_color': base * 1.05},
            {'vertices': [0, 1, 5, 4], 'normal': np.array([0, -1, 0]), 'name': 'Bottom', 'base_color': base * 0.8}
        ]
        
        return vertices, faces
    
    def project_vertices(self, vertices, camera_pos, target_pos):
        """Project vertices to screen coordinates"""
        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        
        # 使用Y轴为上方向，保证"height"控制垂直尺寸
        view_matrix = self.create_view_matrix(camera_pos, target_pos, [0, 1, 0])
        projection_matrix = self.create_perspective_matrix()
        
        vertices_view = (view_matrix @ vertices_homogeneous.T).T
        vertices_proj = (projection_matrix @ vertices_view.T).T
        
        screen_vertices = []
        depths = []
        
        for vertex in vertices_proj:
            if abs(vertex[3]) > 1e-6:
                x_ndc = vertex[0] / vertex[3]
                y_ndc = vertex[1] / vertex[3]
                z_ndc = vertex[2] / vertex[3]
                
                screen_x = (x_ndc + 1) * self.width / 2
                screen_y = (1 - y_ndc) * self.height / 2
                
                screen_vertices.append([screen_x, screen_y])
                depths.append(z_ndc)
            else:
                screen_vertices.append([self.width/2, self.height/2])
                depths.append(1.0)
        
        return np.array(screen_vertices), np.array(depths)
    
    def calculate_lambert_lighting(self, normal, light_direction):
        """Calculate Lambert lighting intensity"""
        dot_product = np.dot(normal, -light_direction)
        diffuse = max(0, dot_product)
        
        brightness = self.ambient_light + self.diffuse_light * diffuse
        return min(1.0, brightness)
    
    def is_face_visible(self, face_vertices, face_normal, camera_pos):
        """Backface culling"""
        face_center = np.mean(face_vertices, axis=0)
        view_direction = np.array(camera_pos) - face_center
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        return np.dot(face_normal, view_direction) > 0
    
    def render_3d_model(self, scale, camera_angle=45, camera_elevation=25, camera_distance=0.0, original_distance=None, output_depth=False, depth_gamma=1.0):
        """Render enhanced 3D model and return PIL Image
        
        If output_depth is True, returns a white-near/black-far depth visualization
        computed with a simple Z-buffer rasterizer.
        """
        # Generate geometry
        vertices, faces = self.generate_box_geometry(scale)
        
        # Calculate camera position
        if camera_distance <= 0:
            camera_distance = self.calculate_camera_distance(scale)
        else:
            # Warn if distance is too close for the object size
            min_recommended_distance = max(scale) * 1.5
            if camera_distance < min_recommended_distance:
                print(f"Warning: Camera distance {camera_distance:.1f}m might be too close for box size. Recommend >= {min_recommended_distance:.1f}m")
        angle_rad = math.radians(camera_angle)
        elevation_rad = math.radians(camera_elevation)
        
        camera_pos = [
            camera_distance * math.cos(elevation_rad) * math.cos(angle_rad),
            camera_distance * math.cos(elevation_rad) * math.sin(angle_rad),
            camera_distance * math.sin(elevation_rad)
        ]
        
        target_pos = [0, 0, 0]
        
        # Project vertices
        screen_vertices, depths = self.project_vertices(vertices, camera_pos, target_pos)

        # 如果只输出深度图，使用简单Z-Buffer进行三角形光栅化
        if output_depth:
            # 归一化到[0,1]：近=0，远=1；随后可反相用于白近黑远
            depths_norm = (depths + 1.0) * 0.5
            depth_buffer = np.ones((self.height, self.width), dtype=np.float32)

            def rasterize_triangle(p0, p1, p2, z0, z1, z2):
                min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
                max_x = min(self.width - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
                min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
                max_y = min(self.height - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))

                denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
                if abs(denom) < 1e-8:
                    return

                for y in range(min_y, max_y + 1):
                    py = y + 0.5
                    for x in range(min_x, max_x + 1):
                        px = x + 0.5
                        w1 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
                        w2 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
                        w3 = 1.0 - w1 - w2
                        if w1 >= 0 and w2 >= 0 and w3 >= 0:
                            z = w1 * z0 + w2 * z1 + w3 * z2
                            if z < depth_buffer[y, x]:
                                depth_buffer[y, x] = z

            # 只对可见面进行栅格化
            for face in [f for f in self.generate_box_geometry(scale)[1]]:
                face_vertex_indices = face['vertices']
                face_world_vertices = vertices[face_vertex_indices]
                if self.is_face_visible(face_world_vertices, face['normal'], camera_pos):
                    v = screen_vertices[face_vertex_indices]
                    z = depths_norm[face_vertex_indices]
                    # 将四边形拆成两个三角形
                    rasterize_triangle(v[0], v[1], v[2], z[0], z[1], z[2])
                    rasterize_triangle(v[0], v[2], v[3], z[0], z[2], z[3])

            # 自动增强对比：仅以对象像素的深度范围做归一化
            mask = depth_buffer < 0.9999
            if np.any(mask):
                dmin = float(np.min(depth_buffer[mask]))
                dmax = float(np.max(depth_buffer[mask]))
                if dmax - dmin > 1e-8:
                    depth_norm = (depth_buffer - dmin) / (dmax - dmin)
                else:
                    depth_norm = np.zeros_like(depth_buffer)
                # 背景设为最远(1.0)，反相后为黑
                depth_norm[~mask] = 1.0
            else:
                depth_norm = np.ones_like(depth_buffer)

            # 白近黑远 + Gamma 调整对比
            gamma = max(0.01, float(depth_gamma))
            depth_vis = 1.0 - np.power(np.clip(depth_norm, 0.0, 1.0), gamma)
            depth_img = (np.clip(depth_vis, 0.0, 1.0) * 255).astype(np.uint8)
            rgb = np.dstack([depth_img, depth_img, depth_img])
            return Image.fromarray(rgb, mode='RGB')

        # 彩色渲染路径
        # Create image with custom background color
        bg_rgb = tuple(int(c * 255) for c in self.background_color)
        img = Image.new('RGB', (self.width, self.height), bg_rgb)
        draw = ImageDraw.Draw(img)
        
        # Process visible faces
        face_data = []
        
        for face in faces:
            face_vertex_indices = face['vertices']
            face_world_vertices = vertices[face_vertex_indices]
            
            # Backface culling
            if self.is_face_visible(face_world_vertices, face['normal'], camera_pos):
                # Calculate face depth
                face_depths = depths[face_vertex_indices]
                avg_depth = np.mean(face_depths)
                
                # Calculate lighting
                brightness = self.calculate_lambert_lighting(face['normal'], self.light_direction)
                
                # Calculate face color
                color = face['base_color'] * brightness
                color = (color * 255).astype(int)
                color = tuple(np.clip(color, 0, 255))
                
                face_screen_vertices = screen_vertices[face_vertex_indices]
                
                face_data.append({
                    'depth': avg_depth,
                    'vertices': face_screen_vertices,
                    'color': color,
                    'name': face['name'],
                    'brightness': brightness
                })
        
        # Sort by depth (far to near)
        face_data.sort(key=lambda x: x['depth'], reverse=True)
        
        # Render each face
        for face_info in face_data:
            vertices_2d = face_info['vertices']
            color = face_info['color']
            
            polygon_points = [(int(v[0]), int(v[1])) for v in vertices_2d]
            
            if len(polygon_points) >= 3:
                try:
                    draw.polygon(polygon_points, fill=color, outline='black', width=1)
                except Exception:
                    pass  # Skip problematic polygons
        
        # Add info text
        # 显示为 L×W×H，保持与输入名一致：length=scale[2], width=scale[0], height=scale[1]
        draw.text((10, 10), f"Enhanced Lambert 3D - L:{scale[2]:.1f}m  W:{scale[0]:.1f}m  H:{scale[1]:.1f}m", fill='black')
        
        # Show "auto" for automatic distance, actual distance for manual
        if original_distance is None:
            original_distance = camera_distance
            
        distance_text = "auto" if original_distance <= 0 else f"{original_distance:.1f}m"
        draw.text((10, 30), f"Camera: {camera_angle}°/{camera_elevation}° - {distance_text}", fill='black')
        
        return img


class Enhanced3DRenderer(ComfyNodeABC):
    """ComfyUI增强版3D盒子渲染节点，带兰伯特光照"""
    
    DESCRIPTION = "使用增强版兰伯特光照模型渲染3D盒子，创造逼真的阴影和深度感知效果。"
    CATEGORY = "image/3d"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "3D盒子的宽度 Width(X)，单位米"
                }),
                "height": ("FLOAT", {
                    "default": 4.0, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "3D盒子的高度 Height(Y)，单位米"
                }),
                "length": ("FLOAT", {
                    "default": 1.5, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "3D盒子的长度 Length(Z)，单位米"
                }),
                "camera_angle": ("FLOAT", {
                    "default": 30.0, 
                    "min": 0.0, 
                    "max": 360.0, 
                    "step": 1.0,
                    "tooltip": "摄像机围绕盒子的旋转角度"
                }),
                "camera_elevation": ("FLOAT", {
                    "default": 30.0, 
                    "min": -90.0, 
                    "max": 90.0, 
                    "step": 1.0,
                    "tooltip": "摄像机的俯仰角度"
                }),
                "image_width": ("INT", {
                    "default": 800, 
                    "min": 64, 
                    "max": 2048, 
                    "step": 8,
                    "tooltip": "输出图像的宽度，单位像素"
                }),
                "image_height": ("INT", {
                    "default": 800, 
                    "min": 64, 
                    "max": 2048, 
                    "step": 8,
                    "tooltip": "输出图像的高度，单位像素"
                }),
                "camera_distance": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 50.0, 
                    "step": 0.5,
                    "tooltip": "摄像机距离，单位米 (0 = 自动计算，推荐5-20米)"
                }),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "背景颜色，十六进制格式如 #FFFFFF"
                }),
                "box_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "盒子主要颜色，十六进制格式如 #C8C8C8"
                }),
                "output_depth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "勾选后输出白近黑远的深度图"
                }),
                "depth_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "深度对比度(Gamma)，<1增强近景，>1增强远景"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render_3d_box"
    
    def render_3d_box(self, width, height, length, camera_angle, camera_elevation, image_width, image_height, camera_distance, background_color, box_color, output_depth=False, depth_gamma=1.0):
        """Render the 3D box and return as ComfyUI image format"""
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ComfyUI nodes. Please install torch.")
        
        # Create renderer with specified dimensions and colors
        renderer = EnhancedLambert3DRenderer(width=image_width, height=image_height, 
                                           background_color=background_color, box_color=box_color)
        
        # Render the 3D model（scale: width=X, height=Y, length=Z）
        scale = [width, height, length]
        pil_image = renderer.render_3d_model(
            scale,
            camera_angle,
            camera_elevation,
            camera_distance,
            original_distance=camera_distance,
            output_depth=output_depth,
            depth_gamma=depth_gamma,
        )
        
        # Convert PIL image to torch tensor format expected by ComfyUI
        # ComfyUI expects images in format: [batch, height, width, channels] with values 0-1
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Add batch dimension
        image_tensor = torch.from_numpy(image_array)[None, ...]
        
        return (image_tensor,)
    

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Enhanced3DRenderer": Enhanced3DRenderer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhanced3DRenderer": "VVL 3D Box Renderer"
}
