for arch in edsr_baseline edsr ninasr_b0 ninasr_b1 ninasr_b2 rdn rcan carn carn_m
do
	python main.py --arch $arch --scale 2 --download-pretrained --validation-only --dataset-val set5
done
