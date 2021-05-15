#!/bin/sh
# Evaluate on luminance channels, removing the border pixel; this is the convention used by most publications

for scale in 2 3 4
do
    for dataset in div2k_bicubic
    do
        for arch in carn carn_m edsr_baseline edsr ninasr_b0 ninasr_b1 ninasr_b2 rcan rdn
        do
            echo -n "${dataset} ${arch} x${scale}: "
            python main.py --validation-only --arch $arch --scale $scale --dataset-val $dataset --chop-size 400 --download-pretrained --shave-border $scale --eval-luminance
        done
    done
done

for scale in 8
do
    for dataset in div2k_bicubic
    do
        for arch in ninasr_b0 ninasr_b1 ninasr_b2 rcan
        do
            echo -n "${dataset} ${arch} x${scale}: "
            python main.py --validation-only --arch $arch --scale $scale --dataset-val $dataset --chop-size 400 --download-pretrained --shave-border $scale --eval-luminance
        done
    done
done
