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
        if config.rf == None:
            self.RF_index = st.Index_sizeOfPatch
        else:
            self.RF_index = int(config.rf)
        self.RF_size = [9, 17, 25, 41, 57]

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


        """ encoder """
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=self.kernel[0], stride=self.stride[0], padding=self.padding[0], padding_mode='replicate', dilation=1, groups=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(self.inplanes, affine=False)
        self.act_func = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=self.kernel[1], stride=self.stride[1], padding=self.padding[1])

        self.layer1 = Basic_residual_block_v3(inplanes=self.inplanes, planes=f_out[0], kernel_size=(self.kernel[2]), stride=1, dilation=(1, 1), padding=(self.padding[2]), flag_res=True)
        self.layer2 = Basic_residual_block_v3(inplanes=f_out[0], planes=f_out[1], kernel_size=(self.kernel[4]), stride=2, dilation=(1, 1), padding=(self.padding[4]), flag_res=True)
        self.layer3 = Basic_residual_block_v3(inplanes=f_out[1], planes=f_out[2], kernel_size=(self.kernel[6]), stride=1, dilation=(1, 1), padding=(self.padding[6]), flag_res=True)
        self.layer4 = Basic_residual_block_v3(inplanes=f_out[2], planes=f_out[3], kernel_size=(self.kernel[8]), stride=1, dilation=(1, 1), padding=(self.padding[8]), flag_res=True)
        f_out_final = f_out[-1]

        """" classifier """
        self.classifier = nn.Sequential(
            nn.Conv3d(f_out_final, 1, kernel_size=1, stride=1, bias=True),
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

        """ pooling """
        out_logit_1 = nn.AvgPool3d(kernel_size=x_ce_1.size()[-3:])(x_ce_1)
        out_logit_1 = out_logit_1.view(out_logit_1.size(0), -1)


        dict_result = {
            "logits": out_logit_1,  # batch, 2
            # "Aux_logits": [out_logit_1],  # batch, 2
            "Aux_logits": None,  # batch, 2
            "logitMap": None,  # batch, 2, w, h ,d
            "l1_norm": None,
            # "l1_norm": [x_ce_1],
            # "entropy": [gate_1],
            "entropy": None,
            "list_plot_1": None,
            # "list_plot_1": [gate_1],
            "list_plot_2": [x_ce_1],
            "final_evidence": None,  # batch, 2, w, h, d
            "featureMaps": [],
        }
        return dict_result


def Model(config):
    model = network(config)
    return model

