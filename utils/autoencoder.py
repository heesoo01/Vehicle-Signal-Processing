import torch
import torch.nn as nn


class PSDAutoencoder(nn.Module):
    def __init__(self, input_dim=129, latent_dim=4):
        """
        주파수(PSD) 데이터를 압축하고 복원하는 선형 오토인코더 모델
        :param input_dim: 입력 데이터의 크기 (기본값 129)
        :param latent_dim: 가장 작게 압축될 병목 구간의 크기 (기본값 4)
        """
        super(PSDAutoencoder, self).__init__()

        # 인코더 (Encoder): 129 -> 64 -> 16 -> 4 차원으로 압축
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

        # 디코더 (Decoder): 4 -> 16 -> 64 -> 129 차원으로 다시 복원
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        """데이터가 모델을 통과하는 순서"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded