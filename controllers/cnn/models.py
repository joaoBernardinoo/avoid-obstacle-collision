import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import joblib



class LidarVisionFusionNet(nn.Module):
    def __init__(self, lidar_input_features=20, num_outputs=2):
        super(LidarVisionFusionNet, self).__init__()

        # --- Vision Branch (Inalterada) ---
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.vision_transforms = weights.transforms()
        mobilenet = mobilenet_v3_small(weights=weights)
        self.vision_branch = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        vision_output_features = 576 # Saída do MobileNetV3-Small features

        # --- LiDAR Branch (Simplificada) ---
        # Reduzida para uma única camada linear para extrair features do LiDAR
        lidar_output_features = 32
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_input_features, lidar_output_features),
            nn.ReLU()
        )

        # --- Fusion Head (Muito Simplificada) ---
        # Reduzida para uma camada oculta. Sem BatchNorm para maior velocidade.
        self.fusion_head = nn.Sequential(
            nn.Linear(vision_output_features + lidar_output_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout pode ser um pouco maior em modelos menores
            nn.Linear(128, num_outputs)
        )

    def forward(self, image, lidar):
        # Processamento da visão (inalterado)
        vision_features = self.vision_branch(image)
        vision_features = self.avgpool(vision_features)
        vision_features = vision_features.view(vision_features.size(0), -1)

        # Processamento do LiDAR
        if lidar.dim() == 3 and lidar.shape[1] == 1:
            lidar = lidar.squeeze(1)
        lidar_features = self.lidar_branch(lidar)

        # Fusão e Classificação
        combined_features = torch.cat([vision_features, lidar_features], dim=1)
        output = self.fusion_head(combined_features)
        return output