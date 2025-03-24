import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectorValueModel(nn.Module):
    def __init__(self):
        super(DetectorValueModel, self).__init__()

        # CNN for board_input (3, 17, 13)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (32, 17, 13)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 17, 13)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (64, 1, 1)
        )

        # Dense for misc_input (40,)
        self.misc_fc = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU()
        )

        # Combine both → final output
        self.fc_final = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 최종 value 출력
        )

    def forward(self, board_input, misc_input):
        # board_input: (B, 3, 17, 13)
        board_feat = self.cnn(board_input)      # (B, 64, 1, 1)
        board_feat = board_feat.view(board_feat.size(0), -1)  # (B, 64)

        # misc_input: (B, 40)
        misc_feat = self.misc_fc(misc_input)    # (B, 64)

        x = torch.cat([board_feat, misc_feat], dim=1)  # (B, 128)
        value = self.fc_final(x)  # (B, 1)
        return value.squeeze(1)   # (B,) → 스칼라 형태로

class JackValueModel(nn.Module):
    def __init__(self):
        super(JackValueModel, self).__init__()

        # CNN for board_input (3, 17, 13)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (32, 17, 13)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 17, 13)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (64, 1, 1)
        )

        # Dense for misc_input (48,)
        self.misc_fc = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU()
        )

        # Combine both → final output
        self.fc_final = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 최종 value 출력
        )

    def forward(self, board_input, misc_input):
        # board_input: (B, 3, 17, 13)
        board_feat = self.cnn(board_input)      # (B, 64, 1, 1)
        board_feat = board_feat.view(board_feat.size(0), -1)  # (B, 64)

        # misc_input: (B, 48)
        misc_feat = self.misc_fc(misc_input)    # (B, 64)

        x = torch.cat([board_feat, misc_feat], dim=1)  # (B, 128)
        value = self.fc_final(x)  # (B, 1)
        return value.squeeze(1)   # (B,) → 스칼라 형태로
