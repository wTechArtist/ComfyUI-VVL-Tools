import json
from typing import List, Dict, Any, Tuple

class CollisionError(Exception):
    """自定义异常，用于报告对象之间的碰撞。"""
    pass

class RoomGenerator:
    """
    根据用户定义的内部尺寸和结构厚度，生成符合规范的房间JSON对象。

    该类负责将易于理解的房间尺寸（米）转换为下游AI能理解的、
    严格结构化的机器语言（JSON，厘米单位），并确保生成的所有对象
    在物理上不发生碰撞。
    """

    def __init__(
        self,
        length_m: float,
        width_m: float,
        height_m: float,
        wall_thickness_cm: float = 20.0,
        floor_thickness_cm: float | None = None,
        ceiling_thickness_cm: float | None = None,
        safety_margin_cm: float = 0.0,
        scene_style: str = "modern style"
    ):
        """
        初始化RoomGenerator。

        Args:
            length_m (float): 房间内部长度（X轴），单位：米。
            width_m (float): 房间内部宽度（Y轴），单位：米。
            height_m (float): 房间内部高度（Z轴），单位：米。
            wall_thickness_cm (float): 墙体厚度，单位：厘米。
            floor_thickness_cm (float): 地板厚度，单位：厘米。
            ceiling_thickness_cm (float): 天花板厚度，单位：厘米。
            safety_margin_cm (float): 对象间的最小安全间距，单位：厘米。
            scene_style (str): 场景风格描述。
        """
        if not all(d > 0 for d in [length_m, width_m, height_m, wall_thickness_cm]):
            raise ValueError("房间尺寸和墙体厚度必须为正数。")

        # 如果地板或天花板厚度未指定，则默认为墙体厚度
        _floor_thickness_cm = floor_thickness_cm if floor_thickness_cm is not None else wall_thickness_cm
        _ceiling_thickness_cm = ceiling_thickness_cm if ceiling_thickness_cm is not None else wall_thickness_cm

        if not all(d >= 0 for d in [_floor_thickness_cm, _ceiling_thickness_cm]):
            raise ValueError("地板和天花板厚度必须为非负数。")

        self.length_cm = self._meters_to_cm(length_m)
        self.width_cm = self._meters_to_cm(width_m)
        self.height_cm = self._meters_to_cm(height_m)
        self.wall_thickness_cm = wall_thickness_cm
        self.floor_thickness_cm = _floor_thickness_cm
        self.ceiling_thickness_cm = _ceiling_thickness_cm
        self.safety_margin_cm = safety_margin_cm
        self.scene_style = scene_style
        
        self.scene_size = self._calculate_scene_size()

    def _meters_to_cm(self, value_m: float) -> float:
        """将米转换为厘米。"""
        return value_m * 100

    def _calculate_scene_size(self) -> List[float]:
        """计算并返回场景的外部总尺寸（包含安全边距）。"""
        return [
            self.length_cm + 2 * self.wall_thickness_cm,
            self.width_cm + 2 * self.wall_thickness_cm,
            self.height_cm + self.floor_thickness_cm + self.ceiling_thickness_cm
        ]

    def _create_object(
        self, name: str, position: List[float], rotation: List[float], scale: List[float]
    ) -> Dict[str, Any]:
        """创建一个基础的3D对象字典，格式与input.json对齐。"""
        return {
            "name": name,
            "position": position,
            "rotation": rotation,
            "scale": scale,
            "3d_url": ""
        }

    def _create_floor(self) -> Dict[str, Any]:
        """生成地板对象。"""
        name = "floor"
        scale = [self.scene_size[0], self.scene_size[1], self.floor_thickness_cm]
        # 地板中心点位于Z轴原点下方
        position = [0, 0, -self.floor_thickness_cm / 2]
        rotation = [0, 0, 0]
        return self._create_object(name, position, rotation, scale)

    def _create_ceiling(self) -> Dict[str, Any]:
        """生成天花板对象。"""
        name = "ceiling"
        scale = [self.scene_size[0], self.scene_size[1], self.ceiling_thickness_cm]
        # 天花板中心点位于房间内部高度之上
        position = [0, 0, self.height_cm + self.ceiling_thickness_cm / 2]
        rotation = [0, 0, 0]
        return self._create_object(name, position, rotation, scale)
        
    def _create_walls(self) -> List[Dict[str, Any]]:
        """
        生成四面墙体对象（包裹式布局，与intput.json对齐）。
        采用包裹式设计：左右墙体（Y方向）作为主结构，其长度等于房间外部长度，
        完全包裹前后墙体（X方向）。
        """
        walls = []
        wall_z_position = self.height_cm / 2

        # 前后墙（-X, +X）：宽度等于房间内部宽度
        back_front_wall_scale = [self.wall_thickness_cm, self.width_cm, self.height_cm]
        
        # 后墙 (-X)
        back_wall_pos = [-(self.length_cm / 2) - (self.wall_thickness_cm / 2), 0, wall_z_position]
        walls.append(self._create_object("back wall (at -X)", back_wall_pos, [0, 0, 0], back_front_wall_scale))
        
        # 前墙 (+X)  
        front_wall_pos = [(self.length_cm / 2) + (self.wall_thickness_cm / 2), 0, wall_z_position]
        walls.append(self._create_object("front wall (at +X)", front_wall_pos, [0, 0, 180], back_front_wall_scale))

        # 左右墙（-Y, +Y）：长度等于房间外部长度，以包裹前后墙
        left_right_wall_scale = [self.wall_thickness_cm, self.length_cm + 2 * self.wall_thickness_cm, self.height_cm]

        # 左墙 (-Y)
        left_wall_pos = [0, -(self.width_cm / 2) - (self.wall_thickness_cm / 2), wall_z_position]
        walls.append(self._create_object("left wall (at -Y)", left_wall_pos, [0, 0, 90], left_right_wall_scale))

        # 右墙 (+Y)
        right_wall_pos = [0, (self.width_cm / 2) + (self.wall_thickness_cm / 2), wall_z_position]
        walls.append(self._create_object("right wall (at +Y)", right_wall_pos, [0, 0, -90], left_right_wall_scale))
        
        return walls


    def _create_camera(self, back_wall_pos: List[float]) -> Dict[str, Any]:
        """
        根据后墙位置生成默认的摄像机设置。
        摄像机位于后墙中心后方200cm处，朝向+X轴。
        """
        camera_pos = [back_wall_pos[0] + 200, 0, back_wall_pos[2]]
        return {
            "position": camera_pos,
            "rotation": [0, 0, 0],  # UE默认朝向+X
            "fov": 60
        }

    def _get_object_aabb(self, obj: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
        """
        计算给定对象的轴对齐包围盒 (AABB)。
        特别处理Z轴旋转90度的对象，此时其scale.x和scale.y需交换。
        """
        pos = obj['position']
        scale = obj['scale']
        rot_z = obj['rotation'][2]

        s_x, s_y = scale[0], scale[1]
        if rot_z == 90 or rot_z == -90:
            s_x, s_y = s_y, s_x  # 交换X和Y的尺寸

        min_x = pos[0] - s_x / 2
        max_x = pos[0] + s_x / 2
        min_y = pos[1] - s_y / 2
        max_y = pos[1] + s_y / 2
        min_z = pos[2] - scale[2] / 2
        max_z = pos[2] + scale[2] / 2
        
        return min_x, max_x, min_y, max_y, min_z, max_z

    def _check_collision(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """检查两个对象是否发生碰撞（考虑安全间距）。"""
        min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = self._get_object_aabb(obj1)
        min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = self._get_object_aabb(obj2)
        
        margin = self.safety_margin_cm

        # 如果在任何一个轴上没有重叠（且满足安全间距），则不碰撞
        # 注意：当margin=0时，边界相切（max_x1 == min_x2）不算碰撞
        if (max_x1 <= min_x2 + margin or min_x1 >= max_x2 - margin or
            max_y1 <= min_y2 + margin or min_y1 >= max_y2 - margin or
            max_z1 <= min_z2 + margin or min_z1 >= max_z2 - margin):
            return False
        
        # 否则，视为碰撞
        return True

    def _validate_objects(self, objects: List[Dict[str, Any]]):
        """
        遍历所有对象对，执行碰撞检测。
        如果检测到任何碰撞，则抛出CollisionError。
        """
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self._check_collision(objects[i], objects[j]):
                    raise CollisionError(f"碰撞检测失败: 对象 {i} 和 {j} 发生碰撞。")

    def generate_room_json(self) -> Dict[str, Any]:
        """
        主入口方法。
        生成所有房间结构对象，执行验证，并返回最终的JSON兼容字典。
        """
        scene = {
            "size": self.scene_size,
            "units": "cm",
            "coordinate_system": "left_hand_ue",
            "notes": self._generate_notes()
        }
        
        objects = []
        objects.append(self._create_floor())
        objects.append(self._create_ceiling())
        walls = self._create_walls()
        objects.extend(walls)

        # 在输出前执行严格的内部验证
        self._validate_objects(objects)
        
        # 从墙体列表中找到后墙的位置 (总是在索引2)
        back_wall_position = walls[0]["position"]
        camera = self._create_camera(back_wall_position)
        
        return {"scene": scene, "camera": camera, "subject": self.scene_style, "objects": objects}

    def _generate_notes(self) -> str:
        """
        生成场景的notes字段内容，与room_10x8x3.5.json保持一致。
        """
        notes_parts = [
            f"--length {self._cm_to_meters(self.length_cm)}",
            f"--width {self._cm_to_meters(self.width_cm)}",
            f"--height {self._cm_to_meters(self.height_cm)}"
        ]
        if self.wall_thickness_cm != 20.0:
            notes_parts.append(f"--wall_thickness {self.wall_thickness_cm}")
        if self.floor_thickness_cm != self.wall_thickness_cm or self.floor_thickness_cm != 20.0:
            notes_parts.append(f"--floor_thickness {self.floor_thickness_cm}")
        if self.ceiling_thickness_cm != self.wall_thickness_cm or self.ceiling_thickness_cm != 20.0:
            notes_parts.append(f"--ceiling_thickness {self.ceiling_thickness_cm}")
        if self.safety_margin_cm != 0.0:
            notes_parts.append(f"--safety_margin {self.safety_margin_cm}")

        return " ".join(notes_parts)
    
    def _cm_to_meters(self, value_cm: float) -> float:
        """将厘米转换为米，保留一位小数。"""
        return round(value_cm / 100, 1)


