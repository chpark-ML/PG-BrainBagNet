# from __future__ import division
from modules import *

class network(nn.Module):
    def __init__(self, config):
        """ init """
        in_p = 16
        f_out = [16, 32, 64, 128]
        self.widening_factor = 2
        self.inplanes = in_p * self.widening_factor
        f_out = [f_out[i] * self.widening_factor for i in range(len(f_out))]
        super(network, self).__init__()

        """ kernel, size """
        self.RF_index = st.Index_sizeOfPatch
        if self.RF_index == 0:
            self.kernel = [5, 3, 1, 1, 1, 1, 1, 1, 1, 1]
        elif self.RF_index == 1:
            self.kernel = [5, 3, 3, 1, 1, 1, 1, 1, 1, 1]
        elif self.RF_index == 2:
            self.kernel = [5, 3, 3, 1, 3, 1, 1, 1, 1, 1]
        elif self.RF_index == 3:
            self.kernel = [5, 3, 3, 1, 3, 1, 3, 1, 1, 1]
        elif self.RF_index == 4:
            self.kernel = [5, 3, 3, 1, 3, 1, 3, 1, 3, 1]

        self.padding = [2, 1,
                        self.kernel[2] // 2, self.kernel[3] // 2,
                        self.kernel[4] // 2, self.kernel[5] // 2,
                        self.kernel[6] // 2, self.kernel[7] // 2,
                        self.kernel[8] // 2, self.kernel[9] // 2]
        self.stride = [2, 2, 1, 1, 2, 1, 1, 1, 1, 1]

        """ position information """
        self.n_layers = [10]
        self.with_r = False
        self.addcoords = AddCoords_size(with_r=self.with_r)
        self.shape_of_input = np.array([st.x_size, st.y_size, st.z_size])
        self.coord_basis = self.addcoords(self.shape_of_input)
        if self.with_r == False:
            self.coord_inplane = 3
        else :
            self.coord_inplane = 4

        """ encoder """
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=self.kernel[0], stride=self.stride[0], padding=self.padding[0], padding_mode='replicate', dilation=1, groups=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(self.inplanes, affine=False)
        self.act_func = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=self.kernel[1], stride=self.stride[1], padding=self.padding[1])

        self.layer1 = Basic_residual_block_v3(inplanes=self.inplanes, planes=f_out[0], kernel_size=(self.kernel[2]), stride=1, dilation=1, padding=(self.padding[2]), flag_res=True)
        self.layer2 = Basic_residual_block_v3(inplanes=f_out[0], planes=f_out[1], kernel_size=(self.kernel[4]), stride=2, dilation=1, padding=(self.padding[4]), flag_res=True)
        self.layer3 = Basic_residual_block_v3(inplanes=f_out[1], planes=f_out[2], kernel_size=(self.kernel[6]), stride=1, dilation=1, padding=(self.padding[6]), flag_res=True)
        self.layer4 = Basic_residual_block_v3(inplanes=f_out[2], planes=f_out[3], kernel_size=(self.kernel[8]), stride=1, dilation=1, padding=(self.padding[8]), flag_res=True)
        f_out_final = f_out[-1]

        """ gate network """
        self.gateNet = GateNet(f_out_final, 16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_r=self.with_r, mode='coord')

        """" classifier """
        self.classifier = nn.Sequential(
            nn.Conv3d(f_out_final, 1, kernel_size=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_0, *args):
        with torch.no_grad():
            if self.with_r == False:
                n_coord_input_channel = 3
            else:
                n_coord_input_channel = 4

            if self.training == True :
                coord_img_1 = self.coord_basis.repeat(input_0.size(0), 1, 1, 1, 1)
                if fst.flag_random_flip == True:
                    info_flip = args[0][0]
                    coord_img_1[info_flip] = coord_img_1[info_flip].flip(-3)

                if fst.flag_random_scale == True:
                    info_scale = args[0][1]
                    for i_tmp in range(input_0.size(0)):
                        tmp = F.interpolate(coord_img_1[i_tmp].unsqueeze(0), scale_factor=tuple(info_scale[i_tmp]), mode='trilinear', align_corners=False)
                        p3d = (0, 50, 0, 50, 0, 50)
                        tmp = F.pad(tmp, p3d, 'replicate', 0)
                        coord_img_1[i_tmp] = tmp[0, :, :coord_img_1.size(-3), : coord_img_1.size(-2), :coord_img_1.size(-1)]

                if fst.flag_cropping == True:
                    crop_list = args[0][2]
                    coord_img = torch.zeros(size=input_0.size()[-3:]).cuda().float().unsqueeze(0).unsqueeze(0).repeat(
                        input_0.size(0), self.coord_basis.size(1), 1, 1, 1)

                    # pad
                    pad_size = (st.crop_pad_size)
                    tmp_coord_basis = F.pad(coord_img_1, pad_size, "replicate", 0)

                    # crop
                    for batch_i in range(crop_list.shape[0]):
                        for axis_i in range(crop_list.shape[1]):
                            for channel_i in range(n_coord_input_channel):
                                coord_img[batch_i][channel_i] = ut.crop_tensor(tmp_coord_basis[0][channel_i],
                                                                               crop_list[batch_i],
                                                                               input_0.size()[-3:])
            else:
                if fst.flag_cropping == True:
                    tmp_size_x_1 = (st.x_size - st.eval_crop_size[0]) // 2
                    tmp_size_x_2 = tmp_size_x_1 + st.eval_crop_size[0]
                    tmp_size_y_1 = (st.y_size - st.eval_crop_size[1]) // 2
                    tmp_size_y_2 = tmp_size_y_1 + st.eval_crop_size[1]
                    tmp_size_z_1 = (st.z_size - st.eval_crop_size[2]) // 2
                    tmp_size_z_2 = tmp_size_z_1 + st.eval_crop_size[2]
                    coord_img = self.coord_basis[:, :, tmp_size_x_1: tmp_size_x_2, tmp_size_y_1: tmp_size_y_2,
                                tmp_size_z_1: tmp_size_z_2].repeat(input_0.size(0), 1, 1, 1, 1)

            out_coord_img = []
            tmp_count = 0
            tmp_coord_img = coord_img
            for i in range(len(self.kernel)):
                self.torch_filter = torch.zeros(n_coord_input_channel, 1, self.kernel[i], self.kernel[i],
                                                self.kernel[i]).cuda()
                self.torch_filter[:, :, self.kernel[i] // 2, self.kernel[i] // 2, self.kernel[i] // 2] = 1
                tmp_coord_img = F.conv3d(tmp_coord_img, self.torch_filter, stride=self.stride[i],
                                         padding=self.padding[i], groups=self.torch_filter.shape[0])

                tmp_count += 1
                if tmp_count in self.n_layers:
                    out_coord_img.append(tmp_coord_img )

        """ encoder """
        x_0 = self.conv1(input_0)
        x_0 = self.norm1(x_0)
        x_0 = self.act_func(x_0)
        x_0 = self.pool(x_0)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        """ classifier """
        x_ce_1 = self.classifier(x_4)

        """ gate  """
        gate_1 = self.gateNet(None, out_coord_img[0])

        """ attention pooling """
        x_ce_1 = x_ce_1 * gate_1
        out_logit_1 = torch.sum(x_ce_1, dim=(2, 3, 4), keepdim=True) / torch.sum(gate_1, dim=(2, 3, 4), keepdim=True)
        out_logit_1 = out_logit_1.view(out_logit_1.size(0), -1)

        dict_result = {
            "logits": out_logit_1,  # batch, 2
            "Aux_logits": None,  # batch, 2
            "logitMap": None,  # batch, 2, w, h ,d
            "l1_norm": None,
            "entropy": [gate_1],
            "list_plot_1": [gate_1],
            "list_plot_2": [x_ce_1],
            "final_evidence": None,  # batch, 2, w, h, d
            "featureMaps": [],
        }
        return dict_result

def Model(config):
    model = network(config)
    return model

