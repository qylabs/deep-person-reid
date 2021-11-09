# '''
# train from scratch not good
# mAP: 34.8%
# CMC curve
# Rank-1  : 55.9%
# Rank-5  : 77.6%
# Rank-10 : 85.0%
# Rank-20 : 90.7%
# '''

# Before run export script, Be sure to change model forward 

CONFIG=configs/im_mobilnet_cust_v1_softmax_128x64_amsgrad.yaml
PATH_TO_DATA=data


python scripts/main.py \
        --config-file ${CONFIG} \
        --transforms random_flip random_erase \
        --root ${PATH_TO_DATA}