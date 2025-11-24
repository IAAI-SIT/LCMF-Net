import torch
import torch.nn as nn
from attention.CrossAttentionSigmoid import *
from VSS2D import VSSM
from NewRI_block import IncResBlock, IncResUpBlock

class Mmodel(nn.Module):
    def __init__(self, args, in_chanel=5):
        super(Mmodel, self).__init__()
        self.in_chanel = in_chanel
        self.bottomBlock = (DownCross(2048, 1024))
        self.flair_encode = nn.ModuleList()
        self.t1_encode = nn.ModuleList()
        self.t1ce_encode = nn.ModuleList()
        self.t2_encode = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.flair_encode.append(IncResBlock(self.in_chanel, 64))
                self.t1_encode.append(IncResBlock(self.in_chanel, 64))
                self.t1ce_encode.append(IncResBlock(self.in_chanel, 64))
                self.t2_encode.append(IncResBlock(self.in_chanel, 64))
            elif i == 1:
                self.flair_encode.append(IncResBlock(64, 128))
                self.t1_encode.append(IncResBlock(64, 128))
                self.t1ce_encode.append(IncResBlock(64, 128))
                self.t2_encode.append(IncResBlock(64, 128))
            elif i == 2:
                self.flair_encode.append(IncResBlock(128, 256))
                self.t1_encode.append(IncResBlock(128, 256))
                self.t1ce_encode.append(IncResBlock(128, 256))
                self.t2_encode.append(IncResBlock(128, 256))

            elif i == 3:
                self.flair_encode.append(IncResBlock(256, 512))
                self.t1_encode.append(IncResBlock(256, 512))
                self.t1ce_encode.append(IncResBlock(256, 512))
                self.t2_encode.append(IncResBlock(256, 512))
        self.ssm = VSSM(in_chans=1024,dims=[1024])
        self.up_conv4 = IncResUpBlock(1024 + 512, 512)
        self.up_conv3 = IncResUpBlock(512 + 256, 256)
        self.up_conv2 = IncResUpBlock(256 + 128, 128)
        self.up_conv1 = IncResUpBlock(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=1)


    def forward(self, flair, t1, t1ce, t2):
        skip_feature = []
        flair_x, skip1 = self.flair_encode[0](flair)
        t1_x, skip2 = self.t1_encode[0](t1)
        t1ce_x, skip3 = self.t1ce_encode[0](t1ce)
        t2_x, skip4 = self.t2_encode[0](t2)
        fusion_skip1 = (skip1+skip2+skip3+skip4)/4.0
        future_atten = (flair_x+t1_x+t1ce_x+t2_x)/4.0
        atten_mtl = torch.sigmoid(future_atten)
        flair_x, t1_x,t1ce_x,t2_x = atten_mtl*flair_x,atten_mtl*t1_x,atten_mtl*t1ce_x,atten_mtl*t2_x
        skip_feature.append(fusion_skip1)

        flair_x, skip11 = self.flair_encode[1](flair_x)
        t1_x, skip22 = self.t1_encode[1](t1_x)
        t1ce_x, skip33 = self.t1ce_encode[1](t1ce_x)
        t2_x, skip44 = self.t2_encode[1](t2_x)
        fusion_skip2 = (skip11+skip22+skip33+skip44)/4.0
        future_atten = (flair_x+t1_x+t1ce_x+t2_x)/4.0
        atten_mtl = torch.sigmoid(future_atten)
        flair_x, t1_x,t1ce_x,t2_x = atten_mtl*flair_x,atten_mtl*t1_x,atten_mtl*t1ce_x,atten_mtl*t2_x
        skip_feature.append(fusion_skip2)

        flair_x, skip111 = self.flair_encode[2](flair_x)
        t1_x, skip222 = self.t1_encode[2](t1_x)
        t1ce_x, skip333 = self.t1ce_encode[2](t1ce_x)
        t2_x, skip444 = self.t2_encode[2](t2_x)
        fusion_skip3 = (skip111+skip222+skip333+skip444)/4.0
        future_atten = (flair_x+t1_x+t1ce_x+t2_x)/4.0
        atten_mtl = torch.sigmoid(future_atten)
        flair_x, t1_x,t1ce_x,t2_x = atten_mtl*flair_x,atten_mtl*t1_x,atten_mtl*t1ce_x,atten_mtl*t2_x
        skip_feature.append(fusion_skip3)

        flair_x, skip1111 = self.flair_encode[3](flair_x)
        t1_x, skip2222 = self.t1_encode[3](t1_x)
        t1ce_x, skip3333 = self.t1ce_encode[3](t1ce_x)
        t2_x, skip4444 = self.t2_encode[3](t2_x)
        fusion_skip4 = (skip1111+skip2222+skip3333+skip4444)/4.0
        future_atten = (flair_x+t1_x+t1ce_x+t2_x)/4.0
        atten_mtl = torch.sigmoid(future_atten)
        flair_x, t1_x,t1ce_x,t2_x = atten_mtl*flair_x,atten_mtl*t1_x,atten_mtl*t1ce_x,atten_mtl*t2_x
        skip_feature.append(fusion_skip4)
        
        x = torch.cat([flair_x, t1_x, t1ce_x, t2_x], dim=1)
        x = self.bottomBlock(x)
        # print("ssm_out",x.shape)
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        # print("ssm_out1",x.shape)
        x = self.ssm(x)
        x = F.interpolate(x, size=(10, 10), mode='bilinear', align_corners=False)
        # print("ssm_out2",x.shape)
        # for i in range(4):
        #     print("skip_shape:",skip_feature[i].shape)
        x = self.up_conv4(x, skip_feature[3])
        # x = self.up_conv4(x)
        # print("up4_out",x.shape)
        x = self.up_conv3(x, skip_feature[2])
        x = self.up_conv2(x, skip_feature[1])
        x = self.up_conv1(x, skip_feature[0])
        x = self.conv_last(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    x_tensor = torch.randn(16, 5, 160, 160).to(device)  # (batch_size, channels, height, width)
    prev_tensor = torch.randn(16, 5, 160, 160).to(device)
    next_tensor = torch.randn(16, 5, 160, 160).to(device)
    final_tensor = torch.randn(16, 5, 160, 160).to(device)

    model = Mmodel(args=None, in_chanel=5).to(device)

    output = model(x_tensor, prev_tensor, next_tensor, final_tensor)
