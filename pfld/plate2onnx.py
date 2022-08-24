# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import onnx

import os
import argparse
from pfld.plate.pfld import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim

from config import config

parser = argparse.ArgumentParser(description='plate2onnx')
parser.add_argument('--torch_model', default="./checkpoint/snapshot/plate_mnet_last.pth.tar")
parser.add_argument('--onnx_model', default="./output/plate.onnx")
parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./output/plate-sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
pfld_backbone = PFLDInference(config.plate.multiple)
#print("plate bachbone:", pfld_backbone)
pfld_backbone.load_state_dict(checkpoint['data'])

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 3, config.plate.size, config.plate.size))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(pfld_backbone,
                  dummy_input,
                  args.onnx_model,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(args.onnx_model)
# print("model_opt", model_opt)
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")
