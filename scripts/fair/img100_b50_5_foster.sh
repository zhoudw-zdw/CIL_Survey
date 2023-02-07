CUDA_VISIBLE_DEVICES=0 python main.py \
    -model foster \
    --dataset imagenet100 \
    -net resnet18 \
    -ms 4970\
    -init 50 \
    -incre 5 \
    -d 0\
    -p fair\
    --skip