# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Controlador Webots para um robô de navegação com desvio de obstáculos.
O robô utiliza APENAS dados de sensores (Lidar, Câmera) como observação.
Utiliza a API Gymnasium e Stable-Baselines3 para treinar um agente de RL.
"""

# Standard libraries
import numpy as np
import os
from pathlib import Path
import sys

# Webots libraries
from controller import Supervisor, Lidar, Camera, Motor

# RL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Local imports
sys.path.append(str(Path(__file__).parent))
try:
    import Action
except ImportError:
    sys.exit("Erro: Não foi possível encontrar o arquivo 'Action.py'.")


class NavigationEnv(Supervisor, gym.Env):
    """
    Ambiente Gymnasium para um robô que deve navegar até um alvo.
    A observação do agente é baseada exclusivamente em sensores (Lidar e Câmera).
    O Supervisor é utilizado apenas para resetar o mundo e calcular a recompensa.
    """

    def __init__(self, max_episode_steps=2000):
        super().__init__()
        self._setup_webots()
        self._setup_gymnasium(max_episode_steps)
        self._initialize_state()

    def _setup_webots(self):
        """Configura os componentes do Webots, como sensores e atuadores."""
        self.__timestep = int(self.getBasicTimeStep())

        # Configuração dos sensores
        self.lidar: Lidar = self.getDevice('Ibeo Lux')  # type: ignore
        self.lidar.enable(self.__timestep)
        self.camera: Camera = self.getDevice('camera')  # type: ignore
        self.camera.enable(self.__timestep)

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)  # type: ignore

        # Atuadores serão configurados no reset
        self.wheels = []

        # Nós do Supervisor para reset e cálculo de recompensa
        self.robot_node = self.getSelf()
        self.target_node = self.getFromDef("TARGET")
        if not self.target_node:
            raise ValueError("O nó 'TARGET' não foi encontrado.")

        self.initial_robot_translation = self.robot_node.getField('translation').getSFVec3f()
        self.initial_robot_rotation = self.robot_node.getField('rotation').getSFRotation()

    def _setup_gymnasium(self, max_episode_steps):
        """Configura os espaços de observação e ação do ambiente Gymnasium."""
        self.LIDAR_SAMPLES = 64
        self.CAMERA_WIDTH = self.camera.getWidth()
        self.CAMERA_HEIGHT = self.camera.getHeight()

        # Espaço de observação como dicionário para Lidar e Câmera
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=1, shape=(self.LIDAR_SAMPLES,), dtype=np.float32),
            'camera': spaces.Box(low=0, high=255, shape=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 3), dtype=np.uint8)
        })

        self.action_space = spaces.Discrete(4)
        self._action_to_name = {0: "seguir", 1: "v_esq", 2: "v_dir", 3: "parar"}

        self.max_episode_steps = max_episode_steps

    def _initialize_state(self):
        """Inicializa variáveis de estado do episódio."""
        self.current_step = 0
        self.last_distance_to_target = 0.0
        self.simulation_running = True

    def _get_observation(self):
        """
        Obtém a observação do ambiente usando apenas dados dos sensores.
        Nenhuma informação do Supervisor é utilizada aqui.
        """
        # Processamento dos dados do Lidar
        lidar_raw = self.lidar.getRangeImage()
        indices = np.linspace(0, len(lidar_raw) - 1, self.LIDAR_SAMPLES, dtype=int)
        lidar_obs = np.array(lidar_raw)[indices]
        lidar_obs[lidar_obs == np.inf] = self.lidar.getMaxRange()
        lidar_obs = np.clip(lidar_obs / self.lidar.getMaxRange(), 0, 1).astype(np.float32)

        # Processamento dos dados da Câmera
        camera_image = np.array(self.camera.getImageArray(), dtype=np.uint8)

        return {'lidar': lidar_obs, 'camera': camera_image}

    def _apply_action(self, action_id):
        """Aplica a ação selecionada ao robô."""
        action_name = self._action_to_name.get(action_id, "parar")
        action_function = Action.action_map.get(action_name, Action.stopAction)
        action_function()
        Action.updateWheels(self.wheels, Action.velocity)

    def reset(self, *, seed=None, options=None):
        """Reinicia o ambiente para um novo episódio."""
        super().reset(seed=seed)
        self.simulationResetPhysics()
        self.simulationReset()
        self.robot_node.getField('translation').setSFVec3f(self.initial_robot_translation)
        self.robot_node.getField('rotation').setSFRotation(self.initial_robot_rotation)
        super().step(self.__timestep)

        self.wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel: Motor = self.getDevice(name)  # type: ignore
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)

        self.current_step = 0
        self.last_distance_to_target = self._get_distance_to_target()
        self._apply_action(3)  # Ação "parar"

        return self._get_observation(), {}

    def step(self, action):
        """Executa um passo no ambiente com a ação fornecida."""
        self._apply_action(action)
        self.current_step += 1

        if super().step(self.__timestep) == -1:
            self.simulation_running = False
            dummy_obs = self.observation_space.sample()
            return dummy_obs, 0, True, True, {}

        self._handle_keyboard_input(action)

        obs = self._get_observation()
        reward, terminated, truncated = self._calculate_reward_and_status(action)
        return obs, reward, terminated, truncated, {}

    def _handle_keyboard_input(self, action):
        """Processa entrada do teclado para depuração ou interrupção."""
        key = self.keyboard.getKey()  # type: ignore
        if key == ord('H'):
            action_name = self._action_to_name.get(action, "unknown")
            print(f"Action executed: {action_name}")
        elif key == ord('C'):
            raise KeyboardInterrupt

    def _calculate_reward_and_status(self, action):
        """Calcula a recompensa e verifica se o episódio deve terminar."""
        dist_to_target = self._get_distance_to_target()
        min_lidar_dist = np.min(self.lidar.getRangeImage())

        reward = 0
        terminated = False

        if action == 3:  # Penalidade por parar
            reward -= 10.0
        reward += (self.last_distance_to_target - dist_to_target) * 10.0

        if dist_to_target < 0.5:
            reward += 200.0
            terminated = True
            print("Sucesso: Robô alcançou o alvo!")
        elif min_lidar_dist < 0.2:
            reward -= 100.0
            terminated = True
            print("Falha: Colisão detectada!")

        self.last_distance_to_target = dist_to_target
        truncated = self.current_step >= self.max_episode_steps
        if truncated and not terminated:
            print("Truncado: Tempo máximo do episódio atingido.")

        return reward, terminated, truncated

    def _get_distance_to_target(self):
        """Calcula a distância entre o robô e o alvo."""
        return np.linalg.norm(np.array(self.robot_node.getPosition()) - np.array(self.target_node.getPosition()))


def train_model(env, model_path="ppo_navigation_model.zip", device='cuda', total_timesteps=1_000_000):
    """Treina o modelo PPO com o ambiente fornecido."""
    if os.path.exists(model_path):
        print(f"Carregando modelo existente de '{model_path}'")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Nenhum modelo encontrado. Criando um novo modelo PPO com MultiInputPolicy.")
        model = PPO('MultiInputPolicy', env, n_steps=2048, verbose=1,
                    tensorboard_log="./ppo_navigation_tensorboard/", device=device)

    try:
        print("Iniciando treinamento... Pressione C para parar e salvar.")
        model.learn(total_timesteps=total_timesteps,
                    reset_num_timesteps=not os.path.exists(model_path))
        print("Treinamento concluído.")
        model.save(model_path)
    except KeyboardInterrupt:
        print("\nTreinamento interrompido. Salvando modelo...")
        model.save(model_path)
        print(f"Modelo salvo em '{model_path}'.")
    return model


def replay_simulation(env, model):
    """Executa uma simulação de replay com o modelo treinado."""
    print("\n--- Iniciando Replay com o modelo treinado ---")
    obs, info = env.reset()
    while env.simulation_running:
        action, _states = model.predict(obs, deterministic=True)
        # Convert numpy array to scalar integer if necessary
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if env.simulation_running:
                print("Episódio finalizado. Reiniciando replay...")
                obs, info = env.reset()
            else:
                break
    print("Simulação encerrada.")


def main():
    """Função principal para configurar o ambiente e executar treinamento e replay."""
    env = NavigationEnv(max_episode_steps=2048)
    # check_env(env)  # Descomente para verificar a consistência do ambiente
    model_path = "ppo_navigation_model.zip"
    model = train_model(env, model_path, 'cuda')
    replay_simulation(env, model)


if __name__ == '__main__':
    main()
