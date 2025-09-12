import argparse
import os
import numpy as np
import random
import mindspore as ms
from collections import namedtuple
from exp.exp_informer_ms import Exp_Informer
import ruamel.yaml as yaml
from mindspore.communication import get_group_size, get_rank, init
from datetime import datetime

today_str = datetime.today().strftime('%Y-%m-%d')
parser = argparse.ArgumentParser(description='Informer MindSpore')
parser.add_argument('--config_name', type=str, default='./configs/informer_GPU.yaml', help='Informer config path')
parser.add_argument('--distribute', action='store_true', default=False)
config = parser.parse_args()

def dict_to_namedtuple(dic: dict):
    return namedtuple('tuple', dic.keys())(**dic)

with open(config.config_name, 'r') as f:
    tmp = yaml.YAML(typ='rt')
    args = tmp.load(f)

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'JFNG_data_15min':{'data':'JFNG_data_15min.csv','T':'tp','M':[6,6,6],'S':[1,1,1],'MS':[6,6,1]},
}

if args['data'] in data_parser.keys():
    data_info = data_parser[args['data']]
    args['data_path'] = data_info['data']
    args['target'] = data_info['T']
    args['enc_in'], args['dec_in'], args['c_out'] = data_info[args['features']]
args['s_layers'] = [int(s_l) for s_l in args['s_layers'].replace(' ','').split(',')]

args['detail_freq'] = args['freq']
args['freq'] = args['freq'][-1:]
args['distribute'] = config.distribute

if args['distribute']:
    init()
    args['device_num'] = get_group_size()
    args['rank_id'] = get_rank()
    ms.set_auto_parallel_context(
        device_num=args['device_num'],
        parallel_mode="data_parallel",
        gradients_mean=True,
    )
else:
    args['device_num'] = None
    args['rank_id'] = None

args = dict_to_namedtuple(args)
print("DEVICE:", args.device)
ms.set_context(device_target=args.device)
ms.set_context(mode=ms.PYNATIVE_MODE)

def setup_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(args.seed)

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for i in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_{}'.format(
                args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, i, today_str)
    print("Setting:", setting)

    exp = Exp(args) # set experiments
    model = exp._get_model()
    print(model)
    if args.do_train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(model, setting)
    else:
        print(">>>>>>>loading : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(args.ckpt_path))
        ms.load_param_into_net(model, ms.load_checkpoint(args.ckpt_path))
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(model, setting)