
CONFIG=configs/im_mobilnet_cust_v1_softmax_128x64_amsgrad.yaml
PATH_TO_DATA=data


python scripts/main.py \
        --config-file ${CONFIG} \
        --transforms random_flip random_erase \
        --root ${PATH_TO_DATA}