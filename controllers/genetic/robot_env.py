import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor # type: ignore
import math

# --- MODIFICADO: Importa seu módulo de ações ---
import Action

# --- Constantes ---
LIDAR_SAMPLES = 64 
MAX_STEPS_PER_EPISODE = 1000

class RobotEnv(gym.Env):
    """Ambiente Gymnasium personalizado para o robô no Webots."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, robot_supervisor: Supervisor):
        super(RobotEnv, self).__init__()

        self.robot = robot_supervisor
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.robot_node = self.robot.getSelf()
        # Enable keyboard input
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)  # type: ignore
        # Toggle for printing actions
        self.print_actions = False
        self.target_node = self.robot.getFromDef("TARGET")
        if not self.target_node:
            raise ValueError("Nó 'TARGET' não encontrado no mundo Webots.")
        
        self.initial_robot_translation = self.robot_node.getPosition()
        self.initial_robot_rotation = self.robot_node.getOrientation()

        self.wheels = [self.robot.getDevice(f) for f in 
                       ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))  # type: ignore
            wheel.setVelocity(0.0)  # type: ignore

        self.lidar = self.robot.getDevice('Ibeo Lux')  # type: ignore
        self.lidar.enable(self.timestep)  # type: ignore
        self.lidar.enablePointCloud()  # type: ignore
        
        # --- MODIFICADO: O espaço de ação continua o mesmo (4 ações) ---
        # Mas agora elas serão mapeadas para as strings do seu Action.py
        self.action_space = spaces.Discrete(4) 
        self._action_to_name = {
            0: "seguir",
            1: "v_esq",
            2: "v_dir",
            3: "parar",
        }
        
        observation_size = LIDAR_SAMPLES + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)
        
        self.step_count = 0
        self.last_distance_to_target = self._get_distance_to_target()

    # O método _get_observation permanece o mesmo
    def _get_observation(self) -> np.ndarray:
        """Coleta dados dos sensores e os formata em um array numpy."""
        lidar_raw = self.lidar.getRangeImage()  # type: ignore
        indices = np.linspace(0, len(lidar_raw) - 1, LIDAR_SAMPLES, dtype=int)
        lidar_obs = np.array(lidar_raw)[indices]
        lidar_obs[lidar_obs == np.inf] = 10.0 

        robot_pos = np.array(self.robot_node.getPosition())
        target_pos = np.array(self.target_node.getPosition())
        
        vector_to_target = target_pos - robot_pos
        distance = np.linalg.norm(vector_to_target)
        
        robot_rotation = self.robot_node.getOrientation()
        forward_vector = np.dot(np.array(robot_rotation).reshape(3, 3), np.array([1, 0, 0]))
        angle_to_target = math.atan2(
            vector_to_target[1], vector_to_target[0]
        ) - math.atan2(forward_vector[1], forward_vector[0])

        if angle_to_target > np.pi: angle_to_target -= 2 * np.pi
        if angle_to_target < -np.pi: angle_to_target += 2 * np.pi
        
        target_obs = np.array([distance, angle_to_target], dtype=np.float32)
        
        return np.concatenate([lidar_obs, target_obs]).astype(np.float32)

    # O método reset permanece o mesmo
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reseta o ambiente para um novo episódio."""
        super().reset(seed=seed)
        
        self.robot.simulationReset()
        self.robot_node.getField('translation').setSFVec3f(self.initial_robot_translation)
        self.robot_node.getField('rotation').setSFRotation(self.initial_robot_rotation)
        self.robot.simulationResetPhysics()
        
        self.step_count = 0
        self.last_distance_to_target = self._get_distance_to_target()
        
        # Para as rodas usando a função do seu módulo
        Action.stopAction()
        Action.updateWheels(self.wheels, Action.velocity)
            
        self.robot.step(self.timestep)
        
        obs = self._get_observation()
        info = {}
        
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executa um passo no ambiente."""
        
        # 1. Executa a ação (agora usando seu módulo)
        self._apply_action(action)

        # 2. Simula um passo no Webots
        if self.robot.step(self.timestep) == -1:
            return self._get_observation(), 0, True, True, {}
            
        # Check for keyboard input
        key = self.keyboard.getKey()  # type: ignore
        if key == ord('H'):
            self.print_actions = not self.print_actions
            print(f"Printing actions: {'ON' if self.print_actions else 'OFF'}")
        
        # Print action if toggle is on
        if self.print_actions:
            action_name = self._action_to_name.get(action, "unknown")
            print(f"Action executed: {action_name}")
            
        self.step_count += 1
        obs = self._get_observation()
        reward, terminated = self._compute_reward_and_termination(obs)
        truncated = self.step_count >= MAX_STEPS_PER_EPISODE
        info = {}
        
        return obs, reward, terminated, truncated, info

    # --- MODIFICADO: Este método foi totalmente atualizado ---
    def _apply_action(self, action: int):
        """Mapeia a ação discreta para as funções do módulo Action.py."""
        
        # 1. Pega o nome da ação (ex: "seguir") do nosso dicionário interno
        action_name = self._action_to_name.get(action, "parar")

        # 2. Pega a função correspondente do action_map em Action.py
        action_function = Action.action_map.get(action_name)

        # 3. Executa a função da ação, que atualiza a variável global Action.velocity
        if action_function:
            action_function()
        else:
            # Por segurança, se a ação não for encontrada, para o robô.
            Action.stopAction()

        # 4. Aplica as velocidades calculadas às rodas usando a função de Action.py
        Action.updateWheels(self.wheels, Action.velocity)
            
    # O método _compute_reward_and_termination permanece o mesmo
    def _compute_reward_and_termination(self, obs: np.ndarray) -> tuple[float, bool]:
        """Calcula a recompensa e verifica se o episódio terminou."""
        terminated = False
        reward = 0.0
        lidar_obs = obs[:-2]
        dist_to_target = obs[-2]
        
        reward += (self.last_distance_to_target - dist_to_target) * 10.0 
        self.last_distance_to_target = dist_to_target
        reward -= 0.01

        if dist_to_target < 0.5:
            reward += 100.0
            terminated = True
            print("Objetivo alcançado!")
            
        if np.min(lidar_obs) < 0.2:
            reward -= 50.0
            terminated = True
            print("Colisão detectada!")

        return reward, terminated
        
    # Funções auxiliares (_get_distance_to_target) também permanecem as mesmas.
    def _get_distance_to_target(self) -> float:
        return float(np.linalg.norm(np.array(self.robot_node.getPosition()) - np.array(self.target_node.getPosition())))  # type: ignore
