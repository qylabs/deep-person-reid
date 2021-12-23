import torch
from torchvision.models import resnet18
import torch_pruning as tp
import torchreid

#rootmanager，build构建
model = resnet18(pretrained=True).eval()
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name='mobilenetv2_x1_0_v1',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)

model.eval()

#导入现有模型
# model = torch.load(r'pre-tr-m/purned_layer1_to_6_mobilenetv2.pth')#结构导入
pretrain_dict = torch.load(r'pre-tr-m/74.4-model.pth.tar-70')#权重导入
# print(pretrain_dict)
# model_dict = model.state_dict()

pretrain_dict_ = {
   k.replace('module.', ''): v
   for k, v in pretrain_dict['state_dict'].items()
}
# model_dict.update(pretrain_dict)
model.load_state_dict(pretrain_dict_,strict=False)
# model.load_state_dict(pretrain_dict_)
# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1, 3, 128, 64))

# print(model)
# # 3. get a pruning plan from the dependency graph.

######################第一层（单层）##################
# pruning_idxs = strategy(model.conv1.conv.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9, ...]
#
# pruning_plan = DG.get_pruning_plan(model.conv1.conv, tp.prune_conv, idxs=pruning_idxs)
#
# pruning_plan.exec()
#############################第二到八层(单层)#################
# for m in model.conv7.modules():
#     if isinstance(m, torch.nn.Conv2d) :
#         global idxs
#         idxs=strategy(m.weight, amount=0.5)
#         pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs)
#         print(pruning_plan)
#         # execute the plan (prune the model)
#         pruning_plan.exec()
# ###################全剪枝######################
for m in model.conv1.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()

for m in model.conv2.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv3.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv4.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.1) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv5.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.1) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv6.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.6) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv7.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.6) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()
for m in model.conv8.modules():
    if isinstance(m, torch.nn.Conv2d):
        global indxs
        pruning_idxs = strategy(m.weight, amount=0.6) # or manually selected pruning_idxs=[2, 6, 9, ...]
        #
        pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=pruning_idxs)
        pruning_plan.exec()


#########################################
# for i in model.conv2:
#     pruning_idxs = strategy(i.conv3[0].weight, amount=0.4) # or manually selected pruning_idxs=[2, 6, 9, ...]
#     pruning_plan = DG.get_pruning_plan(i.conv3[0], tp.prune_conv, idxs=pruning_idxs)
#     pruning_plan.exec()

# fix the broken dependencies manually
# tp.prune_batchnorm(model.conv1.bn, idxs=pruning_idxs)
# tp.prune_related_conv( model.layer2[0].conv1, idxs=pruning_idxs )
# print(pruning_plan)
# 4. execute this plan (prune the model)
torch.save(model, 'pre-tr-m/purned_all_layers_3_mobilenetv2.pth') # obj (arch + weights), recommended.

from torchstat import stat
stat(model,(3,128,64))