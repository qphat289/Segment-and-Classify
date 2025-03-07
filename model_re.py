class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)
        self.se = SEBlock(out_channels)  # SE block
        self.dropout = nn.Dropout2d(dropout_rate)  # Add dropout

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)  # Apply SE block
        x = self.dropout(x)  # Apply dropout after SE block
        return x

class MobileNetUNet(nn.Module):
    def __init__(self, img_ch=1, seg_ch=4, num_classes=4, dropout_rate=0.3):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        # Load pretrained MobileNetV2
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Modify first conv for grayscale input
        if img_ch == 1:
            self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder stages
        self.encoder1 = self.backbone.features[:2]    # 1/2
        self.encoder2 = self.backbone.features[2:4]   # 1/4
        self.encoder3 = self.backbone.features[4:7]   # 1/8
        self.encoder4 = self.backbone.features[7:14]  # 1/16
        self.encoder5 = self.backbone.features[14:]   # 1/32

        # Bottleneck dropout (higher rate at bottleneck)
        self.bottleneck_dropout = nn.Dropout2d(dropout_rate * 1.5)
        
        # Decoder stages with skip connections and dropout
        self.decoder4 = DecoderBlock(1280, 96, 256, dropout_rate)   # 1/16 -> 1/8
        self.decoder3 = DecoderBlock(256, 32, 128, dropout_rate)    # 1/8 -> 1/4
        self.decoder2 = DecoderBlock(128, 24, 64, dropout_rate)     # 1/4 -> 1/2
        self.decoder1 = DecoderBlock(64, 16, 32, dropout_rate)      # 1/2 -> 1/1

        # Final upsampling to match input resolution (no dropout here)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnRelu(32, 16)
        )

        # Segmentation head (minimal dropout to preserve details)
        self.seg_head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout rate here
            nn.Conv2d(16, seg_ch, kernel_size=1)
        )

        # Enhanced classification head (already has dropout)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store input size for later use
        input_size = x.shape[-2:]

        # Encoder path with skip connections
        e1 = self.encoder1(x)      # 1/2
        e2 = self.encoder2(e1)     # 1/4
        e3 = self.encoder3(e2)     # 1/8
        e4 = self.encoder4(e3)     # 1/16
        e5 = self.encoder5(e4)     # 1/32
        
        # Apply dropout at bottleneck
        e5 = self.bottleneck_dropout(e5)

        # Classification branch
        cls_output = self.cls_head(e5)

        # Decoder path with skip connections
        d4 = self.decoder4(e5, e4)  # 1/16 -> 1/8
        d3 = self.decoder3(d4, e3)  # 1/8 -> 1/4
        d2 = self.decoder2(d3, e2)  # 1/4 -> 1/2
        d1 = self.decoder1(d2, e1)  # 1/2 -> 1/1

        # Final upsampling and segmentation
        x = self.final_upsample(d1)
        seg_output = self.seg_head(x)

        # Ensure output size matches input size
        if seg_output.shape[-2:] != input_size:
            seg_output = nn.functional.interpolate(
                seg_output, 
                size=input_size, 
                mode='bilinear', 
                align_corners=True
            )

        return seg_output, cls_output