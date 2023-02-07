CUDA_VISIBLE_DEVICES=3 python main.py \
    -model foster \
    --dataset imagenet100 \
    -ms 2000 \
    -init 10 \
    -incre 10 \
    -net resnet18 \
    -p benchmark \
    -d 0 \
    --skip