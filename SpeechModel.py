import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSpeechModel(nn.Module):
    def __init__(self, num_output_classes=6):
        super(BaseSpeechModel, self).__init__()
        self.num_output_classes = num_output_classes

class FirstModel(BaseSpeechModel):
    def __init__(self, num_output_classes=6):
        super().__init__(num_output_classes)

        self.conv1 = nn.Conv1d(1, 256, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 128, kernel_size=5)
        self.dropout1 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=8)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=5)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=5)
        self.dropout3 = nn.Dropout(0.2)

        # Temporarily calculate the output shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 65)
            dummy_out = self._forward_features(dummy)
            flattened_size = dummy_out.shape[1]

        self.fc = nn.Linear(flattened_size, num_output_classes)
        self.bn5 = nn.BatchNorm1d(num_output_classes)

    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.relu(self.bn4(x))
        x = self.dropout2(x)
        x = self.conv6(x)
        x = x.flatten(start_dim=1)
        x = self.dropout3(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 193)
        x = self._forward_features(x)
        x = self.fc(x)
        x = self.bn5(x)
        return x

class SecondModel(BaseSpeechModel):
    def __init__(self, num_output_classes=6):
        super().__init__(num_output_classes)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5)
        self.batch_norm1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5)
        self.dropout1 = nn.Dropout(0.1)

        self.maxpool = nn.MaxPool1d(kernel_size=8)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(self._get_flattened_size(), num_output_classes)

    def _get_flattened_size(self):
        # Pass a dummy tensor through the layers to calculate the flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, 65)
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = self.dropout1(x)

            x = self.maxpool(x)

            x = self.conv3(x)
            x = self.batch_norm2(x)
            x = F.relu(x)
            x = self.dropout2(x)

            x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x