# '''
# Much better simply load previous layers
# odel complexity: params=165,696 flops=24,555,520
# mAP: 50.6%
# CMC curve
# Rank-1  : 74.1%
# Rank-5  : 89.2%
# Rank-10 : 93.3%
# Rank-20 : 96.2%


# original model zoo results:Rank-1(mAP)=85.6 (67.3)
# '''

CONFIG=configs/im_mobilnet_cust_v1_softmax_128x64_amsgrad.yaml
PATH_TO_DATA=data
EXPORT_LOAD_WEIGHT=log/mobilenet_cust_v1_1_market1501_softmax_pretrained/model/model.pth.tar-80


python scripts/main.py \
        --config-file ${CONFIG} \
        --transforms random_flip random_erase \
        --root ${PATH_TO_DATA} \
        export.onnx True \
        export.load_weights ${EXPORT_LOAD_WEIGHT}