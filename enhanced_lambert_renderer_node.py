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
    
    def __init__(self, width=800, height=600):
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
        
        # Material color
        self.base_color = np.array([0.9, 0.9, 0.9])  # Base color
        
    def calculate_camera_distance(self, scale):
        """Calculate camera distance based on box size with adaptive scaling"""
        max_dimension = max(scale)
        
        # Use adaptive scaling for better viewing experience
        if max_dimension <= 2.0:
            # For small objects (≤2m): use 2.5x multiplier
            return max_dimension * 2.5
        elif max_dimension <= 5.0:
            # For medium objects (2-5m): use 2.2x multiplier  
            return max_dimension * 2.2
        elif max_dimension <= 8.0:
            # For large objects (5-8m): use 1.8x multiplier + small offset
            return max_dimension * 1.8 + 2.0
        else:
            # For very large objects (>8m): use 1.6x multiplier + larger offset
            return max_dimension * 1.6 + 4.0
        
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
        """Generate box geometry"""
        w, h, d = scale
        
        vertices = np.array([
            [-w/2, -h/2, -d/2],  # 0: left-bottom-back
            [w/2, -h/2, -d/2],   # 1: right-bottom-back
            [w/2, h/2, -d/2],    # 2: right-top-back
            [-w/2, h/2, -d/2],   # 3: left-top-back
            [-w/2, -h/2, d/2],   # 4: left-bottom-front
            [w/2, -h/2, d/2],    # 5: right-bottom-front
            [w/2, h/2, d/2],     # 6: right-top-front
            [-w/2, h/2, d/2]     # 7: left-top-front
        ])
        
        # Define 6 faces with color info
        faces = [
            {'vertices': [4, 5, 6, 7], 'normal': np.array([0, 0, 1]), 'name': 'Front', 'base_color': np.array([0.9, 0.9, 0.9])},
            {'vertices': [1, 0, 3, 2], 'normal': np.array([0, 0, -1]), 'name': 'Back', 'base_color': np.array([0.8, 0.8, 0.8])},
            {'vertices': [5, 1, 2, 6], 'normal': np.array([1, 0, 0]), 'name': 'Right', 'base_color': np.array([0.85, 0.85, 0.85])},
            {'vertices': [0, 4, 7, 3], 'normal': np.array([-1, 0, 0]), 'name': 'Left', 'base_color': np.array([0.85, 0.85, 0.85])},
            {'vertices': [3, 7, 6, 2], 'normal': np.array([0, 1, 0]), 'name': 'Top', 'base_color': np.array([0.95, 0.95, 0.95])},
            {'vertices': [0, 1, 5, 4], 'normal': np.array([0, -1, 0]), 'name': 'Bottom', 'base_color': np.array([0.75, 0.75, 0.75])}
        ]
        
        return vertices, faces
    
    def project_vertices(self, vertices, camera_pos, target_pos):
        """Project vertices to screen coordinates"""
        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        
        view_matrix = self.create_view_matrix(camera_pos, target_pos, [0, 0, 1])
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
    
    def render_3d_model(self, scale, camera_angle=45, camera_elevation=25, camera_distance=0.0, original_distance=None):
        """Render enhanced 3D model and return PIL Image"""
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
        
        # Create image
        img = Image.new('RGB', (self.width, self.height), 'white')
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
        draw.text((10, 10), f"Enhanced Lambert 3D - {scale[0]:.1f}m×{scale[1]:.1f}m×{scale[2]:.1f}m", fill='black')
        
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
                    "tooltip": "3D盒子的宽度，单位米"
                }),
                "height": ("FLOAT", {
                    "default": 4.0, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "3D盒子的高度，单位米"
                }),
                "depth": ("FLOAT", {
                    "default": 1.5, 
                    "min": 0.1, 
                    "max": 10, 
                    "step": 0.1,
                    "tooltip": "3D盒子的深度，单位米"
                }),
                "camera_angle": ("FLOAT", {
                    "default": 45.0, 
                    "min": 0.0, 
                    "max": 360.0, 
                    "step": 1.0,
                    "tooltip": "摄像机围绕盒子的旋转角度"
                }),
                "camera_elevation": ("FLOAT", {
                    "default": 25.0, 
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
                    "default": 600, 
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
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render_3d_box"
    
    def render_3d_box(self, width, height, depth, camera_angle, camera_elevation, image_width, image_height, camera_distance):
        """Render the 3D box and return as ComfyUI image format"""
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ComfyUI nodes. Please install torch.")
        
        # Create renderer with specified dimensions
        renderer = EnhancedLambert3DRenderer(width=image_width, height=image_height)
        
        # Render the 3D model
        scale = [width, height, depth]
        pil_image = renderer.render_3d_model(scale, camera_angle, camera_elevation, camera_distance, original_distance=camera_distance)
        
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
