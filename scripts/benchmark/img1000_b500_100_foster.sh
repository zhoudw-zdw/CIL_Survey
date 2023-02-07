python main.py \
    -model foster \
    --dataset imagenet1000 \
    -ms 20000 \
    -init 500 \
    -incre 100 \
    -net resnet18 \
    -p benchmark \
    -d 0 1 \
    --skip