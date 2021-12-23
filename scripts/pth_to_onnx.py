import torch
import torchreid

model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=16,
    loss='softmax',
    pretrained=False,
    use_gpu=True
)
dummy_input = torch.randn(1, 3, 256, 128)
model_state_dict = torch.load('log/osnet_x0_25_market1501_softmax/model/model.pth.tar-180', map_location=torch.device('cuda'))
for k in model_state_dict["state_dict"].keys():
    nk = k.replace('module.','')
    model.state_dict()[nk] = model_state_dict["state_dict"][k]
# model.load_state_dict(model_state_dict["state_dict"])
# print(model.keys())
# torch.save('osnet_x0_25_market1501_softmax.pth', model)
torch.onnx._export(model, dummy_input, "pix2pix.onnx", verbose=True, opset_version=11)

# import onnx