##Much better simply load previous layers
##Model complexity: params=165,696 flops=24,555,520
CONFIG=configs/im_mobilnet_cust_v1_softmax_128x64_amsgrad.yaml
PATH_TO_DATA=data


python scripts/main.py \
        --config-file ${CONFIG} \
        --transforms random_flip random_erase \
        --root ${PATH_TO_DATA} \
        model.load_weights checkpoint/mobilenetv2_1dot0_market.pth.tar \
        data.save_dir log/mobilenet_cust_v1_1_market1501_softmax_pretrained