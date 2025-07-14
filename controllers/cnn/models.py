import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class LidarVisionFusionNet(nn.Module):
    def __init__(self, lidar_input_features=360, num_outputs=2):
        super(LidarVisionFusionNet, self).__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.vision_transforms = weights.transforms()

        mobilenet = mobilenet_v3_small(weights=weights)
        self.vision_branch = mobilenet.features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        vision_output_features = 576

        self.lidar_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lidar_input_features, 256),
            nn.ReLU()
        )

        lidar_output_features = 256

        self.fusion_head = nn.Sequential(
            nn.Linear(vision_output_features + lidar_output_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        lidar = lidar.unsqueeze(1)
        lidar_features = self.lidar_branch(lidar)

        combined_features = torch.cat([vision_features, lidar_features], dim=1)

        output = self.fusion_head(combined_features)
        return output
