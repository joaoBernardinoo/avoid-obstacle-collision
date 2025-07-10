import torch
import torch.nn as nn

# Define as dimensões da imagem como referência
IMG_HEIGHT = 40
IMG_WIDTH = 200
IMG_CHANNELS = 4  # BGRA do Webots

class CNNNavigationModel(nn.Module):
    """
    Define a arquitetura de um modelo CNN robusto para navegação usando dados de câmera e LiDAR.

    Este modelo aprimorado calcula dinamicamente o tamanho da entrada para as camadas
    totalmente conectadas, tornando-o robusto a mudanças nas dimensões da imagem de entrada
    ou na arquitetura da CNN. Ele também refatora a ramificação da câmera para maior clareza
    e inclui inicialização de pesos Kaiming.
    """
    def __init__(self, lidar_shape_in, img_shape=(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)):
        """
        Inicializa o modelo CNN.

        Args:
            lidar_shape_in (int): O número de características de entrada para a ramificação LiDAR.
            img_shape (tuple): A forma da imagem de entrada (C, H, W).
        """
        super(CNNNavigationModel, self).__init__()

        # --- Ramificação da Câmera ---
        # Definimos a parte convolucional primeiro para calcular dinamicamente seu tamanho de saída.
        camera_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_shape[0], out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # **MELHORIA 1: Cálculo dinâmico do tamanho das características**
        # Criamos um tensor "dummy" e o passamos pelas camadas convolucionais para descobrir o tamanho achatado.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *img_shape)
            flattened_cam_size = camera_conv_layers(dummy_input).shape[1]
            print(f"Tamanho da característica da câmera achatada calculado dinamicamente: {flattened_cam_size}")


        # **MELHORIA 2: Clareza arquitetural**
        # Combinamos todo o pipeline de processamento da câmera em um único módulo sequencial.
        self.camera_branch = nn.Sequential(
            camera_conv_layers,
            nn.Linear(flattened_cam_size, 64),
            nn.ReLU()
        )

        # --- Ramificação do LiDAR ---
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_shape_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # --- Cabeça Combinada ---
        # Esta parte permanece conceitualmente a mesma.
        self.combined_head = nn.Sequential(
            nn.Linear(64 + 64, 128),  # 64 da câmera + 64 do LiDAR
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Saída de 2 valores (dist, angulo)
            nn.Tanh()          # Escala a saída para [-1, 1] para alvos normalizados
        )

        # **MELHORIA 3: Inicialização de Pesos**
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Aplica a inicialização Kaiming He às camadas Conv2d e Linear.
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, cam_input, lidar_input):
        """
        Define a passagem para a frente (forward pass) do modelo.

        Args:
            cam_input (torch.Tensor): O tensor de entrada da câmera.
                                      Forma: (N, C, H, W)
            lidar_input (torch.Tensor): O tensor de entrada do LiDAR.
                                        Forma: (N, lidar_shape_in)

        Returns:
            torch.Tensor: O tensor de saída do modelo. Forma: (N, 2)
        """
        cam_features = self.camera_branch(cam_input)
        lidar_features = self.lidar_branch(lidar_input)
        combined_features = torch.cat((cam_features, lidar_features), dim=1)
        output = self.combined_head(combined_features)
        return output

# --- Exemplo de Uso ---
if __name__ == '__main__':
    # Exemplo de dados LiDAR com 360 pontos
    LIDAR_POINTS = 360
    BATCH_SIZE = 4 # Exemplo de tamanho do lote

    # Cria uma instância do modelo
    model = CNNNavigationModel(lidar_shape_in=LIDAR_POINTS)
    print("\nArquitetura do Modelo:")
    print(model)

    # Cria tensores de entrada "dummy" para testar a passagem para a frente
    dummy_cam_input = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    dummy_lidar_input = torch.randn(BATCH_SIZE, LIDAR_POINTS)

    # Realiza uma passagem para a frente
    print(f"\nTestando com formas de entrada: Cam={dummy_cam_input.shape}, Lidar={dummy_lidar_input.shape}")
    output = model(dummy_cam_input, dummy_lidar_input)

    print(f"Forma da saída: {output.shape}") # Esperado: (BATCH_SIZE, 2)
    print(f"Valores de saída (primeira amostra):\n {output[0]}")

    # --- DEMONSTRANDO A ROBUSTEZ ---
    # Agora, vamos tentar um tamanho de imagem diferente sem alterar o código do modelo
    print("\n" + "="*50)
    print("DEMONSTRANDO A ROBUSTEZ COM NOVO TAMANHO DE IMAGEM")
    print("="*50 + "\n")
    NEW_IMG_HEIGHT = 60
    NEW_IMG_WIDTH = 240
    model_robust = CNNNavigationModel(
        lidar_shape_in=LIDAR_POINTS,
        img_shape=(IMG_CHANNELS, NEW_IMG_HEIGHT, NEW_IMG_WIDTH)
    )

    dummy_cam_input_new = torch.randn(BATCH_SIZE, IMG_CHANNELS, NEW_IMG_HEIGHT, NEW_IMG_WIDTH)
    print(f"Testando com NOVAS formas de entrada: Cam={dummy_cam_input_new.shape}, Lidar={dummy_lidar_input.shape}")
    output_new = model_robust(dummy_cam_input_new, dummy_lidar_input)
    print(f"Forma da saída (nova): {output_new.shape}") # Ainda funciona!
    print("O modelo se adaptou com sucesso ao novo tamanho da imagem sem nenhuma alteração no código.")
