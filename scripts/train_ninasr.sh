#!/bin/bash

train_network () {
    arch=$1
    scale=$2
    echo Training network "${arch}" "x${scale}"
    echo python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --log-dir "logs_train/${arch}_x${scale}" --save-checkpoint "${arch}_x${scale}.pt" \
        --lr "${learning_rate}" --epochs "${epochs}" --lr-decay-steps $((epochs*2/3)) $((epochs*5/6)) --lr-decay-rate 3 \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4
    python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --log-dir "logs_train/${arch}_x${scale}" --save-checkpoint "${arch}_x${scale}.pt" \
        --lr "${learning_rate}" --epochs "${epochs}" --lr-decay-steps $((epochs*2/3)) $((epochs*5/6)) --lr-decay-rate 3 \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4 \
        --weight-norm
}

train_network_with_pretrained () {
    arch=$1
    scale=$2
    pretrained_scale=$3
    echo Pretraining network "${arch}" "x${scale}" from "x${pretrained_scale}"
    echo python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --save-checkpoint "${arch}_x${scale}_pre.pt" \
        --load-pretrained "${arch}_x${pretrained_scale}_model.pt" \
        --lr "${learning_rate}" --epochs 1 --freeze-backbone \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4
    python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --save-checkpoint "${arch}_x${scale}_pre.pt" \
        --load-pretrained "${arch}_x${pretrained_scale}_model.pt" \
        --lr "${learning_rate}" --epochs 1 \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4
    echo Training network "${arch}" "x${scale}"
    echo python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --log-dir "logs_train/${arch}_x${scale}" --save-checkpoint "${arch}_x${scale}.pt" \
        --load-pretrained "${arch}_x${scale}_pre_model.pt" \
        --lr "${learning_rate}" --epochs "${epochs}" --lr-decay-steps $((epochs*2/3)) $((epochs*5/6)) --lr-decay-rate 3 \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4
    python -m torchsr.train --arch "${arch}" --scale "${scale}" --tune-backend \
        --log-dir "logs_train/${arch}_x${scale}" --save-checkpoint "${arch}_x${scale}.pt" \
        --load-pretrained "${arch}_x${scale}_pre_model.pt" \
        --lr "${learning_rate}" --epochs "${epochs}" --lr-decay-steps $((epochs*2/3)) $((epochs*5/6)) --lr-decay-rate 3 \
        --patch-size-train $(( (patch_size+1) * scale)) --shave-border "${scale}" --replication-pad 4 \
        --weight-norm
}

epochs=300
patch_size=48
learning_rate=0.001

# NinaSR-B0
train_network ninasr_b0 2
train_network ninasr_b0 3 
train_network ninasr_b0 4 
train_network ninasr_b0 8 

epochs=500
patch_size=48
learning_rate=0.0003

# NinaSR-B1
train_network ninasr_b1 2
train_network_with_pretrained ninasr_b1 3 2
train_network_with_pretrained ninasr_b1 4 3
train_network_with_pretrained ninasr_b1 8 4
train_network_with_pretrained ninasr_b1 2 4

epochs=1000
patch_size=48
learning_rate=0.0001

# NinaSR-B2
train_network ninasr_b2 2
train_network_with_pretrained ninasr_b2 3 2
train_network_with_pretrained ninasr_b2 4 3
train_network_with_pretrained ninasr_b2 8 4
train_network_with_pretrained ninasr_b2 2 4
