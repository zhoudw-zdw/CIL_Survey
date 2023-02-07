CUDA_VISIBLE_DEVICES=2 python main.py \
    -model foster \
    --dataset imagenet100 \
    -ms 4671\
    -init 10 \
    -incre 10 \
    -p fair \
    -net resnet18 \
    -d 0 \
    --skip