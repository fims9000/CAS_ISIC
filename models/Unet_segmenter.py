import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNetPlusPlusOld(nn.Module):
    """
    Упрощённая версия UNet++ из OLD, адаптированная под текущую инфраструктуру.
    ВАЖНО: возвращает logits (БЕЗ sigmoid) для совместимости с BCEWithLogits и др.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,  # игнорируется, оставлено для совместимости сигнатур
        attention_gates: bool = False,  # игнорируется
        deep_supervision: bool = False,  # игнорируется
        debug_mode: bool = False,  # игнорируется
    ):
        super().__init__()

        # Encoder
        self.conv00 = ConvBlock(input_channels, base_channels)
        self.pool0 = nn.MaxPool2d(2)

        self.conv10 = ConvBlock(base_channels, base_channels * 2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv20 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool2 = nn.MaxPool2d(2)

        self.conv30 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool3 = nn.MaxPool2d(2)

        self.conv40 = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder (Nested)
        self.up31 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv31 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up21 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv21 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up11 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv11 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up01 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv01 = ConvBlock(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Decoder (Nested)
        x31 = self.conv31(torch.cat([self.up31(x40), x30], dim=1))
        x21 = self.conv21(torch.cat([self.up21(x31), x20], dim=1))
        x11 = self.conv11(torch.cat([self.up11(x21), x10], dim=1))
        x01 = self.conv01(torch.cat([self.up01(x11), x00], dim=1))

        out = self.final(x01)
        # Возвращаем logits, без сигмоиды
        return out


