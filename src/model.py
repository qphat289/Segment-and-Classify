import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)
        self.se = SEBlock(out_channels)  # Added SE block

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)  # Apply SE block
        return x

class MobileNetUNet(nn.Module):
    def __init__(self, img_ch=1, seg_ch=4, num_classes=4):
        super().__init__()
        
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

        # Decoder stages with skip connections
        self.decoder4 = DecoderBlock(1280, 96, 256)   # 1/16 -> 1/8
        self.decoder3 = DecoderBlock(256, 32, 128)    # 1/8 -> 1/4
        self.decoder2 = DecoderBlock(128, 24, 64)     # 1/4 -> 1/2
        self.decoder1 = DecoderBlock(64, 16, 32)      # 1/2 -> 1/1

        # Final upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnRelu(32, 16)
        )

        # Segmentation head with deep supervision
        self.seg_head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, seg_ch, kernel_size=1)
        )

        # Enhanced classification head
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



def model_info(model, input_size=(1, 256, 256)):
    """
    Prints:
      - model name (type)
      - input shape
      - total / trainable parameters
      - segmentation output shape
      - classification output shape
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create a single test input (N=1)
    x = torch.randn(1, *input_size)
    with torch.no_grad():
        seg_out, cls_out = model(x)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Segmentation output shape: {seg_out.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print("------------------------------------------------")

##############################
#   EXAMPLE USAGE / DEMO
##############################
from torchinfo import summary

if __name__ == "__main__":
    mobile_net_unet = MobileNetUNet(img_ch=1, seg_ch=4, num_classes=4)

    summary(
        mobile_net_unet, 
        input_size=(1, 1, 256, 256),   # (batch_size, channels, height, width)
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=3  # how deep to display
    )


# import torch
# import torch.nn as nn
# from torch.nn import init
# from src.layers import *
# from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

# ######################################################
# ################ ResNeXt50-UNet ######################
# ######################################################

# class ResNeXtUNet(nn.Module):
#     def __init__(self, n_classes_seg, n_classes_cls):
#         super(ResNeXtUNet, self).__init__()
#         self.base_model = resnext50_32x4d(pretrained=True)

#         # Freeze pretrained layers (optional)
#         for i, param in enumerate(self.base_model.parameters()):
#             param.requires_grad = False

#         self.base_layers = list(self.base_model.children())
#         filters = [4*64, 4*128, 4*256, 4*512]
        
#         # Down (Encoder)
#         self.encoder0 = nn.Sequential(*self.base_layers[:3])
#         self.encoder1 = nn.Sequential(*self.base_layers[4])        
#         self.encoder2 = nn.Sequential(*self.base_layers[5])        
#         self.encoder3 = nn.Sequential(*self.base_layers[6])        
#         self.encoder4 = nn.Sequential(*self.base_layers[7])        
        
#         # Up (Decoder for Segmentation)
#         self.decoder4 = ResNeXt_decoder(filters[3], filters[2])
#         self.decoder3 = ResNeXt_decoder(filters[2], filters[1])        
#         self.decoder2 = ResNeXt_decoder(filters[1], filters[0])        
#         self.decoder1 = ResNeXt_decoder(filters[0], filters[0])        
        
#         # Segmentation head
#         self.seg_head = nn.Sequential(
#             ConvRelu(256, 128, 3, 1),
#             nn.Conv2d(128, n_classes_seg, 3, padding=1)
#         )
        
#         # Classification head
#         self.cls_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(2048, 512),  # 2048 is the channel size of encoder4 output
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, n_classes_cls)
#         )

#     def forward(self, x):
#         # Encoder path
#         x = self.encoder0(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
        
#         # Classification branch
#         cls_out = self.cls_head(e4)
        
#         # Decoder path (Segmentation)
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
        
#         # Segmentation output
#         seg_out = self.seg_head(d1)
        
#         return seg_out, cls_out

#     def setTrainableLayer(self, trainable_layers):
#         for name, node in self.base_model.named_children():
#             unlock = name in trainable_layers
#             for param in node.parameters():
#                 param.requires_grad = unlock



# def init_weights(net, init_type='normal', gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)
