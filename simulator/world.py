import irsim
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

class IRSimulator:
    """
    irsim 仿真器封装类，提供启动世界和获取传感器/障碍物数据的接口
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: int = logging.INFO, enable_render: bool = True):
        """
        初始化 irsim 仿真器
        
        参数:
            config_path: 配置文件路径，可以为空，稍后通过 start_world 指定
            log_level: 日志级别，默认为 INFO
            enable_render: 是否启用渲染，默认为 True
        """
        self.config_path = config_path
        self.env = None
        self.robot = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.enable_render = enable_render
        
        # 传感器数据缓存
        self._latest_sensor_data = None
        
        # 配置参数
        self.world_config = {}
        self.width = 20  # 默认世界宽度
        self.height = 20  # 默认世界高度

    def _center_view(self) -> None:
        """
        自动设置视图中心，使原点位于屏幕中央
        """
        if self.env is None or self.env._env_plot is None:
            self.logger.warning("Cannot center view: environment or plot not initialized")
            return
            
        try:
            self.env._env_plot.ax.set_xlim([-self.width / 2, self.width / 2])
            self.env._env_plot.ax.set_ylim([-self.height / 2, self.height / 2])
            self.env._env_plot.ax.set_aspect("equal")
            self.logger.info(f"View centered: xlim=[{-self.width/2}, {self.width/2}], ylim=[{-self.height/2}, {self.height/2}]")
        except Exception as e:
            self.logger.error(f"Failed to center view: {e}")
    
    def _validate_sensors(self) -> None:
        """
        验证机器人上的传感器配置
        """
        if self.robot is None:
            self.logger.warning("Robot not available for sensor validation")
            return
            
        if hasattr(self.robot, 'lidar'):
            self.logger.info(f"Lidar configured: Range [{self.robot.lidar.range_min}, {self.robot.lidar.range_max}], "
                            f"Steps: {self.robot.lidar.number}")
        else:
            self.logger.warning("No Lidar found on robot!")
            
        # 可以扩展其他传感器验证
        if hasattr(self.robot, 'camera'):
            self.logger.info("Camera sensor found")
        if hasattr(self.robot, 'gps'):
            self.logger.info("GPS sensor found")

    def start_world(self, config_path: Optional[str] = None, auto_center_view: bool = True) -> bool:
        """
        启动仿真世界
        
        参数:
            config_path: 配置文件路径（如果初始化时已指定则可省略）
            auto_center_view: 是否自动将视图居中，默认 True
            
        返回:
            bool: 是否成功启动
        """
        # 使用传入的路径或初始化时的路径
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None:
            self.logger.error("No configuration path provided")
            return False
            
        self.config_path = config_path
        self.logger.info(f"Loading configuration from: {config_path}")
        
        try:
            # 创建环境
            self.env = irsim.make(str(config_path))
            
            # 获取世界配置
            self.world_config = self.env.env_config.parse.get('world', {})
            self.width = self.world_config.get('width', 20)
            self.height = self.world_config.get('height', 20)
            
            # 自动设置视图中心
            if auto_center_view:
                self._center_view()
                
            # 获取机器人对象
            self.robot = self.env.robot
            
            # 验证传感器
            self._validate_sensors()
            
            self.logger.info("Environment loaded successfully.")
            self.logger.info(f"World size: {self.width}x{self.height}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load environment: {e}")
            self.env = None
            self.robot = None
            return False
    
    def step(self, render_delay: float = 0.05) -> None:
        """
        执行一步仿真
        
        参数:
            render_delay: 渲染延迟时间（秒）
        """
        if self.env is None:
            self.logger.error("Cannot step: environment not initialized")
            return
            
        try:
            R = 5.0          # 半径（米）
            omega = 0.5      # 角速度（rad/s）
            v = omega * R    # 对应线速度
            self.env.step({'obstacle_0': [v, omega]})
            if self.enable_render:
                self.env.render(render_delay)
        except Exception as e:
            self.logger.error(f"Error during step: {e}")
    
    def done(self) -> bool:
        """检查仿真是否完成"""
        if self.env is None:
            return False
        return self.env.done()
    
    def end(self) -> None:
        """结束仿真"""
        if self.env is not None:
            try:
                self.env.end()
                self.logger.info("Simulation finished.")
            except Exception as e:
                self.logger.error(f"Error ending simulation: {e}")
    
    def reset(self) -> bool:
        """重置仿真到初始状态"""
        if self.env is None:
            self.logger.warning("Cannot reset: environment not initialized")
            return False
        try:
            self.env.reset()
            self.logger.info("Simulation reset successful.")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting simulation: {e}")
            return False

    def get_sensor_data(self) -> Optional[Dict[str, Any]]:
        """
        获取机器人传感器数据
        
        返回:
            包含以下键的字典:
                - 'robot_position': 机器人位置 [x, y]
                - 'robot_orientation': 机器人朝向 theta
                - 'lidar_ranges': 激光雷达距离数组
                - 'lidar_points': 激光雷达点云坐标 (2xN)
                - 'timestamp': 当前时间步
        """
        if self.robot is None:
            self.logger.warning("Cannot get sensor data: robot not available")
            return None
            
        sensor_data = {
            'robot_position': self.robot.state[:2].flatten() if self.robot.state.shape[0] >= 2 else np.zeros(2),
            'robot_orientation': float(self.robot.state[2, 0]) if self.robot.state.shape[0] >= 3 else 0.0,
            'lidar_ranges': None,
            'lidar_points': None,
            'timestamp': getattr(self.env, 'step_count', 0)
        }
        
        # 获取机器人速度（如果可用）
        if hasattr(self.robot, 'velocity'):
            sensor_data['robot_velocity'] = self.robot.velocity.flatten()
        
        # 获取激光雷达数据
        if hasattr(self.robot, 'lidar'):
            try:
                scan_data = self.robot.lidar.get_scan()
                sensor_data['lidar_ranges'] = scan_data.get('ranges')
                
                scan_points = self.robot.lidar.get_points()
                sensor_data['lidar_points'] = scan_points
            except Exception as e:
                self.logger.error(f"Error getting lidar data: {e}")
        
        # 缓存最新数据
        self._latest_sensor_data = sensor_data
        return sensor_data
    
    def get_latest_sensor_data(self) -> Optional[Dict[str, Any]]:
        """获取最近一次的传感器数据（不重新获取）"""
        return self._latest_sensor_data
    
    def get_world_info(self) -> Dict[str, Any]:
        """获取世界信息"""
        return {
            'width': self.width,
            'height': self.height,
            'config': self.world_config,
            'step_time': self.world_config.get('step_time', 0.1),
            'sample_time': self.world_config.get('sample_time', 0.1)
        }
    
    def get_robot_info(self) -> Dict[str, Any]:
        """获取机器人信息"""
        if self.robot is None:
            return {}
        info = {
            'type': self.robot.__class__.__name__,
            'kinematics': getattr(self.robot, 'kinematics', None),
            'shape': getattr(self.robot, 'shape', None),
            'state': self.robot.state.flatten() if hasattr(self.robot, 'state') else None
        }
        if hasattr(self.robot, 'sensors'):
            info['sensors'] = [s.__class__.__name__ for s in self.robot.sensors]
        return info

    def get_obstacles_info(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取环境中所有障碍物的信息
        
        返回:
            List[Dict]: 每个字典代表一个障碍物，包含:
                - 'id': 障碍物索引
                - 'position': 障碍物中心位置 [x, y]
                - 'orientation': 障碍物朝向 (如果是圆形可能不相关)
                - 'velocity': 速度向量 [vx, vy] (仅动态障碍物)
                - 'is_dynamic': 是否为动态障碍物
                - 'shape_type': 形状类型 ('circle', 'polygon', etc.)
                - 'geometry_data': 几何数据 (圆的 radius 或多边形的 vertices)
        """
        if self.env is None:
            self.logger.warning("Cannot get obstacle info: environment not initialized")
            return None

        obstacles_list = []
        
        # irsim 通常将障碍物存储在 env.obstacles 列表中
        # 遍历所有障碍物
        for i, obs in enumerate(self.env.obstacle_list):
            info = {
                'id': i,
                'position': obs.state[:2].flatten().tolist(),
                'velocity': None,
                'is_dynamic': False,
                'shape_type': 'unknown',
                'geometry_data': None
            }
            
            # 获取朝向
            if obs.state.shape[0] >= 3:
                info['orientation'] = float(obs.state[2, 0])
            
            # 判断是否动态：是否有运动学模型
            if hasattr(obs, 'kinematics') and obs.kinematics is not None:
                info['is_dynamic'] = True
                if hasattr(obs, 'velocity'):
                    info['velocity'] = obs.velocity.flatten().tolist()
            
            # 获取几何信息
            if hasattr(obs, 'shape'):
                shape = obs.shape
                shape_name = shape.__class__.__name__.lower() if hasattr(shape, '__class__') else 'unknown'
                info['shape_type'] = shape_name
                
                # 如果是圆形，获取半径
                if hasattr(shape, 'radius'):
                    info['geometry_data'] = {'radius': float(shape.radius)}
                # 如果是多边形，获取顶点 (局部坐标)
                elif hasattr(shape, 'vertices') and shape.vertices is not None:
                    info['geometry_data'] = {'vertices': shape.vertices.tolist()}
            
            obstacles_list.append(info)
            
        return obstacles_list

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    # 获取配置文件路径
    current_file_path = Path(__file__).resolve()
    config_path = current_file_path.parent.parent / 'config' / 'robot_move.yaml'
    
    # 创建仿真器
    simulator = IRSimulator(log_level=logging.INFO)
    
    # 启动世界
    if not simulator.start_world(str(config_path)):
        logging.error("Failed to start simulation world")
        return
    
    obstacles = simulator.get_obstacles_info()
    if obstacles:
        logging.info(f"Total obstacles found: {len(obstacles)}")
        # 打印前2个障碍物的信息作为示例
        for i, obs in enumerate(obstacles[:2]):
            logging.info(f"Obstacle {obs['id']}: Type={obs['shape_type']}, Pos={obs['position']}, Dynamic={obs['is_dynamic']}")
    
    # 获取世界信息
    world_info = simulator.get_world_info()
    logging.info(f"World info: {world_info}")
    
    # 获取机器人信息
    robot_info = simulator.get_robot_info()
    logging.info(f"Robot info: {robot_info}")
    
    # 开始仿真循环
    logging.info("Starting simulation loop (300 steps)...")
    
    # all_time = 0.0
    # sensor_count = 0
    for i in range(300):
        # start_time = time.time()
        # 执行一步
        simulator.step(0.01)
        
        # 获取传感器数据
        sensor_data = simulator.get_sensor_data()
        
        if i % 50 == 0 and sensor_data:
            logging.info(f"Step {i}:")
            logging.info(f"  Robot Position: {sensor_data['robot_position']}")
            
            # 再次打印障碍物信息以观察动态障碍物的移动
            current_obstacles = simulator.get_obstacles_info()
            if current_obstacles and len(current_obstacles) > 10:
                # 假设第11个障碍物是动态的（根据你的yaml配置，后10个是动态的）
                dyn_obs = current_obstacles[10]
                if dyn_obs['is_dynamic']:
                     logging.info(f"  Dynamic Obstacle 10 Pos: {dyn_obs['position']}")
        
        # 检查是否完成
        if simulator.done():
            logging.info("Environment signal 'done'.")
            break

        # end_time = time.time()
        # all_time += end_time - start_time
        # sensor_count += 1
        # if(all_time > 1.0):
        #     logging.info(f"Average step time over last {sensor_count} steps: {all_time/sensor_count:.4f} seconds")
        #     all_time = 0.0
        #     sensor_count = 0
    
    # 结束仿真
    simulator.end()


if __name__ == '__main__':
    main()
