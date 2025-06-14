"""we bots robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot,Motor, Lidar,Camera, Supervisor # type: ignore
import numpy as np

from Action import *

def run(robot):
   

    if True:
        # create the Robot instance.
        timestep = int(robot.getBasicTimeStep())
        back_left_wheel = robot.getDevice('back left wheel')  # type: Motor
        back_right_wheel = robot.getDevice('back right wheel')  # type: Motor
        front_left_wheel = robot.getDevice('front left wheel')  # type: Motor
        front_right_wheel = robot.getDevice('front right wheel')  # type: Motor
        # Enable the motors
        back_left_wheel.setPosition(float('inf'))  # Set to infinity for velocity control
        back_right_wheel.setPosition(float('inf'))  # Set to infinity for velocity control
        front_left_wheel.setPosition(float('inf'))  # Set to infinity for velocity control
        front_right_wheel.setPosition(float('inf'))  # Set to infinity for velocity control
        
        back_left_wheel.setVelocity(0.0)  # Initialize velocity to 0
        back_right_wheel.setVelocity(0.0)  # Initialize velocity to 0
        front_left_wheel.setVelocity(0.0)  # Initialize velocity to 0
        front_right_wheel.setVelocity(0.0)  # Initialize velocity to 0
        
        lidar = robot.getDevice('Ibeo Lux')  # type: Lidar
        lidar.enable(timestep)  # Enable the lidar sensor
        lidar.enablePointCloud()  # Enable point cloud for lidar
        
        camera = robot.getDevice('camera')  # type: Camera
        camera.enable(timestep)  # Enable the camera sensor
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller

    while robot.step(timestep) != -1:
        front_left_wheel.setVelocity(MAX_SPEED* 0.24)
        front_right_wheel.setVelocity(MAX_SPEED * 0.25)

        lidar_data = lidar.getRangeImage()
        cameraData = camera.getImageArray()


def inferAction(dist, ang):
    """
    Controla o robô com base na distância e ângulo estimados.
    Recebe como entrada:
    - dist: distância discreta estimada ao obstáculo
    - ang: ângulo discreto estimado ao alvo
    """
    # Exemplo de controle simples baseado em distância e ângulo
    MAX_SPEED = 6.28  # velocidade máxima dos motores
    # speed = MAX_SPEED * (1 - dist / 10)  # reduz a velocidade conforme a distância aumenta
    # turn_rate = ang / 180 * np.pi  # converte ângulo para radianos
    action = "stop"

    return action

def infer(lidar_data, camera_image):
    """
    Realiza inferência usando uma RNA treinada para prever:
    - DistToObject: distância estimada ao obstáculo
    - AngToTarget: ângulo estimado ao alvo
    Recebe como entrada:
    - lidar_data: lista/array de pontos do LIDAR
    - camera_image: imagem da câmera (formato Webots)
    Retorna:
    - [dist, ang]: valores contínuos
    """
    # Exemplo usando PyTorch (adapte para seu framework/modelo)
    try:
        
        # Carregue o modelo apenas uma vez (singleton)
        if not hasattr(infer, 'model'):
            # Substitua 'model.pth' pelo caminho do seu modelo treinado
            infer.model = torch.load('model.pth', map_location='cpu')
            infer.model.eval()

        model = infer.model
        # Pré-processamento dos dados do LIDAR
        lidar_np = np.array(lidar_data, dtype=np.float32)
        # Pré-processamento da imagem da câmera
        # Exemplo: converter imagem Webots para array numpy (ajuste conforme necessário)
        img_width = 100  # ajuste para o tamanho do seu modelo
        img_height = 2

        # camera_image pode ser um buffer, converta para numpy array
        img_np = np.frombuffer(camera_image, dtype=np.uint8)
        img_np = img_np.reshape((img_height, img_width, 4))[:, :, :3]  # RGBA para RGB
        # Normalização e reshape conforme o modelo
        lidar_tensor = torch.tensor(lidar_np).unsqueeze(0)  # batch x features
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # batch x C x H x W
        # Inferência

        with torch.no_grad():
            output = model(lidar_tensor, img_tensor)
            dist = float(output[0, 0].item())
            ang = float(output[0, 1].item())

        return [dist, ang]
    
    except Exception as e:
        # Caso não haja modelo ou erro, retorna valores dummy e avisa
        print(f"[WARN] RNA não carregada ou erro na inferência: {e}")
        return [0, 0]

if __name__ == "__main__":
    run(robot = Supervisor())



