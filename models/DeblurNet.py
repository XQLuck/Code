from models.submodules import *
from models.FAC.kernelconv2d import KernelConv2D
from torch import nn
from correlation.correlation import FunctionCorrelation 
import models.model_map as model_map
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Self_Attn(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x




class confidence_map(nn.Module):
    def __init__(self):
        super(confidence_map, self).__init__()
        ks_2d = 5
        ch3 = 128       
        self.netFeat = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1))

        self.netMain = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=49, out_channels=32, kernel_size=1, stride=1, padding=0),
		torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0),)        
        self.flowComp = model_map.UNet(6, 4)
        self.ArbTimeFlowIntrp = model_map.UNet(20, 5)
        self.vis_map_conv_1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2)
        self.vis_map_conv_2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)


        self.net_alpha_second = torch.nn.Sequential(
            torch.nn.Conv2d(259,512, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1,inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(1024, ch3 * ks_2d ** 2, kernel_size=1, stride=1, padding=0)
        )
        self.net_beta_second = torch.nn.Sequential(
            torch.nn.Conv2d(259,512, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1,inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1,inplace=True),
            torch.nn.Conv2d(1024, ch3 * ks_2d ** 2, kernel_size=1, stride=1, padding=0)
        )# self.attention = Self_Attn()

    def forward(self, img_blur, last_img_blur, img_blur_fea, last_img_blur_fea,output_last_fea,kernel_wrap):
            
        flowOut = self.flowComp(torch.cat((last_img_blur, img_blur), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]

        tenFeaturesFirst = self.netFeat(last_img_blur_fea)
        tenFeaturesSecond = self.netFeat(img_blur_fea)

        tenCorrelation = torch.nn.functional.leaky_relu(
            input=FunctionCorrelation(tenFirst=tenFeaturesFirst, tenSecond=tenFeaturesSecond, intStride=1),
            negative_slope=0.1, inplace=True)
        cost_volume_ten = self.netMain(tenCorrelation)
        t=1/2
        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]       
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        C,H,W,G=last_img_blur.size()
        flow_wrap=model_map.backWarp(G,W,device)
        g_I0_F_t_0 = flow_wrap(last_img_blur, F_t_0)
        g_I1_F_t_1 = flow_wrap(img_blur, F_t_1)
        intrpOut = self.ArbTimeFlowIntrp(torch.cat((last_img_blur, img_blur, F_0_1, F_1_0,F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        V_t_0   = intrpOut[:, 4:5, :, :]
        V_t_0 = self.vis_map_conv_2(self.vis_map_conv_1(V_t_0))
        information = torch.cat([cost_volume_ten, output_last_fea, V_t_0], 1)      
        alpha = self.net_alpha_second(information)
        beta = self.net_beta_second(information)
        kernel_warp = kernel_wrap.mul(alpha) + beta
        return kernel_warp


class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        ks = 3
        ks_2d = 5
        ch1 = 32
        ch2 = 64
        ch3 = 128
        self.down_fea_1=nn.Conv2d(2*ch3, 2*ch3, 3, 2, 1)
        self.img_down=nn.Conv2d(3, 3, 3, 2, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fea = conv(2*ch3, ch3, kernel_size=ks, stride=1)
        self.confidencemap=confidence_map()
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)

        self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=ks_2d)
        self.kconv_deblur = KernelConv2D.KernelConv2D(kernel_size=ks_2d)


        self.upconv2_u = upconv(2*ch3, ch2)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks)

        self.img_prd = conv(ch1, 3, kernel_size=ks)

        self.kconv1_1 = conv(9, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = resnet_block(ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, kernel_size=ks)

        self.fac_warp = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.kconv4 = conv(ch3 * ks_2d ** 2, ch3, kernel_size=1)

        self.fac_deblur = nn.Sequential(
            conv(2*ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

    def forward(self, img_blur, last_img_blur, output_last_img, output_last_fea,output_last_fea_down):
        img_blur_down=self.img_down(img_blur)
        last_img_blur_down=self.img_down(last_img_blur)
        output_last_img_down=self.img_down(output_last_img)
        
        merge = torch.cat([img_blur, last_img_blur, output_last_img], 1)
        merge_down=torch.cat([img_blur_down, last_img_blur_down, output_last_img_down], 1)


        kconv1 = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge)))
        kconv2 = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1)))
        kconv3 = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2)))
        
        kconv1_down = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge_down)))
        kconv2_down = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1_down)))
        kconv3_down = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2_down)))
             
        kernel_warp = self.fac_warp(kconv3)
        kernel_warp_ori=kernel_warp
        kernel_warp_down=self.fac_warp(kconv3_down)

        conv1_d = self.conv1_1(last_img_blur)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))
        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))
        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))
        last_img_blur_fea=conv3_d
               
        conv1_d = self.conv1_1(img_blur)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))
        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))
        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))
        img_blur_fea=conv3_d
        
        
        conv1_d = self.conv1_1(last_img_blur_down)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))
        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))
        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))
        last_img_blur_fea_down=conv3_d
               
        conv1_d = self.conv1_1(img_blur_down)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))
        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))
        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))
        img_blur_fea_down=conv3_d

        
                
        if output_last_fea is None:
            output_last_fea = torch.cat([img_blur_fea, img_blur_fea],1)
        if output_last_fea_down is None:
            output_last_fea_down=torch.cat([img_blur_fea_down, img_blur_fea_down],1)
        
        
        kernel_warp=self.confidencemap(img_blur,last_img_blur,img_blur_fea,last_img_blur_fea,output_last_fea,kernel_warp)
        kernel_warp_down=self.confidencemap(img_blur_down,last_img_blur_down,img_blur_fea_down,last_img_blur_fea_down,output_last_fea_down,kernel_warp_down)
        
        
        kernel_warp=kernel_warp+self.upsample(kernel_warp_down)        
        kconv4 = self.kconv4(kernel_warp)
        kernel_deblur = self.fac_deblur(torch.cat([kconv3, kconv4],1))        
        conv3_d_k = self.kconv_deblur(img_blur_fea, kernel_deblur)

        output_last_fea = self.fea(output_last_fea)
        
        conv_a_k = self.kconv_warp(output_last_fea, kernel_warp)
        conv_a_k_1 = self.kconv_warp(output_last_fea, kernel_warp_ori)
        conv_a_k=conv_a_k+conv_a_k_1
        
        conv3 = torch.cat([conv3_d_k, conv_a_k],1)

        upconv2 = self.upconv2_1(self.upconv2_2(self.upconv2_u(conv3)))
        upconv1 = self.upconv1_1(self.upconv1_2(self.upconv1_u(upconv2)))
        output_img = self.img_prd(upconv1) + img_blur
        output_fea = conv3
        output_last_fea_down=self.down_fea_1(conv3) 
        
        return output_img,output_fea,output_last_fea_down
