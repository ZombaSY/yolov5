import torch
import sys
import transformers

from torch.onnx import OperatorExportTypes
from torch.autograd import Variable

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append('/mnt/prj/users/ssy/project/model_ensemble')

from models import model_implements


model = model_implements.Unet_HH().cuda()

target_layers = [model.down4.maxpool_conv[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# model.load_state_dict(torch.load(
#     '/mnt/prj/users/ssy/project/model_ensemble/model_checkpoints/2022-11-04 005436/Unet-HH_Epoch_569_mIoU_0.7866824356528682.pt'))

# opset_version = 9

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 512, 512)).cuda()
torch.onnx.export(cam, dummy_input, 'model' + '.onnx', operator_export_type=OperatorExportTypes)
