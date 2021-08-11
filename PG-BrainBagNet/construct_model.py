import setting as st
from model_arch import *

""" model """
def construct_model(config, flag_model_num = 0):
    """ construct model """
    if flag_model_num == 0:
        model_num = st.model_num_0

    if model_num == 0:
        model = BrainBagNet.Model(config).cuda()
    elif model_num == 1:
        model = FG_BrainBagNet.Model(config).cuda()
    elif model_num == 2:
        model = PG_BrainBagNet.Model(config).cuda()


    return model



