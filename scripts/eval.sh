#!/bin/sh

for scale in 2 3 4
do
    for arch in edsr_baseline edsr ninasr_b0 ninasr_b1 ninasr_b2 rcan rdn
    do
	echo -n "${arch} x${scale}: "
        python main.py --validation-only --arch $arch --scale $scale --chop-size 400 --download-pretrained
    done
done

