
import onnx

import os
import argparse
from torch.autograd import Variable
import torch
import onnxsim

from crnn.crnn_mnet import CRNN_MNET
from crnn.crnn_rnet import CRNN_RNET

def parse_args():
    parser = argparse.ArgumentParser(description='CRNN')
    parser.add_argument('--network', default="mnet", type=str)
    args = parser.parse_args()
    return args

args = parse_args()


print("=====> load pytorch checkpoint...")

device = torch.device("cpu")
pt = './checkpoint/crnn_'+args.network+'_last.pt'

config = torch.load(pt)['config']

if args.network == 'mnet':
    backbone = CRNN_MNET(nChannel=3, nHeight=config.height, nClass=config.maxLabel, nHidden=config.mnet.hidden, nMultiple=config.mnet.multiple).to(device)
elif args.network == 'vgg':
    backbone = CRNN_VGG(nChannel=3, nHeight=config.height, nClass=config.maxLabel, nHidden=config.vgg.hidden).to(device)
elif args.network == 'rnet':
    backbone = CRNN_RNET(nChannel=3, nHeight=config.height, nClass=config.maxLabel, nHidden=config.rnet.hidden).to(device)
#backbone = torch.load(pt)['backbone']

backbone.load_state_dict(torch.load(pt)['data'])
backbone.to(device)
backbone.eval()
print(backbone)

onnx_url = './checkpoint/crnn_'+args.network+'.onnx'
onnx_sim = './checkpoint/crnn_'+args.network+'_sim.onnx'

print("=====> convert pytorch model to onnx...")
dummy = torch.randn(1, 3, config.height, config.width).to(device)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(backbone,
                  dummy,
                  onnx_url,
                  verbose=False,
                  opset_version=11,
                  export_params=True,
                  keep_initializers_as_inputs=True,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")
model = onnx.load(onnx_url)
onnx.checker.check_model(model)

print("====> Simplifying...")
input_shapes = {None: [1, 3, config.height, config.width]}
model_opt, check = onnxsim.simplify(onnx_url, input_shapes=input_shapes)
# print("model_opt", model_opt)
onnx.save(model_opt, onnx_sim)
print("onnx model simplify Ok!")
