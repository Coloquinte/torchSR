
# NinaSR-B0
#   x2
python main.py --tune-backend --lr 0.001 --patch-size-train  96 --epochs 300 --lr-decay-steps 250 290 --arch ninasr_b0 --scale 2 --log-dir logs/ninasr_b0_x2 --save-checkpoint ninasr_b0_x2.pt
#   x3
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 1 --lr-decay-steps 60 90 --freeze-backbone --arch ninasr_b0 --scale 3 --save-checkpoint ninasr_b0_x3.pt --load-pretrained ninasr_b0_x2_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 100 --lr-decay-steps 60 90 --arch ninasr_b0 --scale 3 --log-dir logs/ninasr_b0_x3 --save-checkpoint ninasr_b0_x3.pt --load-pretrained ninasr_b0_x3_model.pt
#   x4
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 1 --lr-decay-steps 60 90 --freeze-backbone --arch ninasr_b0 --scale 4 --save-checkpoint ninasr_b0_x4.pt --load-pretrained ninasr_b0_x3_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 100 --lr-decay-steps 60 90 --arch ninasr_b0 --scale 4 --log-dir logs/ninasr_b0_x4 --save-checkpoint ninasr_b0_x4.pt --load-pretrained ninasr_b0_x4_model.pt
#   x8
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 1 --lr-decay-steps 60 90 --freeze-backbone --arch ninasr_b0 --scale 8 --save-checkpoint ninasr_b0_x8.pt --load-pretrained ninasr_b0_x4_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 100 --lr-decay-steps 60 90 --arch ninasr_b0 --scale 8 --log-dir logs/ninasr_b0_x8 --save-checkpoint ninasr_b0_x8.pt --load-pretrained ninasr_b0_x8_model.pt

# NinaSR-B1
#   x2
python main.py --tune-backend --lr 0.001 --patch-size-train  96 --epochs 300 --lr-decay-steps 200 290 --arch ninasr_b1 --scale 2 --log-dir logs/ninasr_b1_x2 --save-checkpoint ninasr_b1_x2.pt
#   x3
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 1 --lr-decay-steps 50 90 --freeze-backbone --arch ninasr_b1 --scale 3 --save-checkpoint ninasr_b1_x3.pt --load-pretrained ninasr_b1_x2_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 100 --lr-decay-steps 50 90 --arch ninasr_b1 --scale 3 --log-dir logs/ninasr_b1_x3 --save-checkpoint ninasr_b1_x3.pt --load-pretrained ninasr_b1_x3_model.pt
#   x4
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 1 --lr-decay-steps 50 90 --freeze-backbone --arch ninasr_b1 --scale 4 --save-checkpoint ninasr_b1_x4.pt --load-pretrained ninasr_b1_x3_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 100 --lr-decay-steps 50 90 --arch ninasr_b1 --scale 4 --log-dir logs/ninasr_b1_x4 --save-checkpoint ninasr_b1_x4.pt --load-pretrained ninasr_b1_x4_model.pt
#   x8
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 1 --lr-decay-steps 50 90 --freeze-backbone --arch ninasr_b1 --scale 8 --save-checkpoint ninasr_b1_x8.pt --load-pretrained ninasr_b1_x4_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 100 --lr-decay-steps 50 90 --arch ninasr_b1 --scale 8 --log-dir logs/ninasr_b1_x8 --save-checkpoint ninasr_b1_x8.pt --load-pretrained ninasr_b1_x8_model.pt

# NinaSR-B2
#   x2
python main.py --tune-backend --lr 0.001 --patch-size-train  96 --epochs 300 --lr-decay-steps 200 280 --arch ninasr_b2 --scale 2 --log-dir logs/ninasr_b2_x2 --save-checkpoint ninasr_b2_x2.pt
#   x3
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 1 --lr-decay-steps 50 80 --freeze-backbone --arch ninasr_b2 --scale 3 --save-checkpoint ninasr_b2_x3.pt --load-pretrained ninasr_b2_x2_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 144 --epochs 100 --lr-decay-steps 50 80 --arch ninasr_b2 --scale 3 --log-dir logs/ninasr_b2_x3 --save-checkpoint ninasr_b2_x3.pt --load-pretrained ninasr_b2_x3_model.pt
#   x4
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 1 --lr-decay-steps 50 80 --freeze-backbone --arch ninasr_b2 --scale 4 --save-checkpoint ninasr_b2_x4.pt --load-pretrained ninasr_b2_x3_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 192 --epochs 100 --lr-decay-steps 50 80 --arch ninasr_b2 --scale 4 --log-dir logs/ninasr_b2_x4 --save-checkpoint ninasr_b2_x4.pt --load-pretrained ninasr_b2_x4_model.pt
#   x8
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 1 --lr-decay-steps 50 80 --freeze-backbone --arch ninasr_b2 --scale 8 --save-checkpoint ninasr_b2_x8.pt --load-pretrained ninasr_b2_x4_model.pt
python main.py --tune-backend --lr 0.001 --patch-size-train 384 --epochs 100 --lr-decay-steps 50 80 --arch ninasr_b2 --scale 8 --log-dir logs/ninasr_b2_x8 --save-checkpoint ninasr_b2_x8.pt --load-pretrained ninasr_b2_x8_model.pt



