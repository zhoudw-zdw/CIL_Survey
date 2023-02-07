python main.py \
    -model icarl \
    --dataset imagenet1000 \
    -ms 20000 \
    -init 100 \
    -incre 100 \
    -net resnet18 \
    -p benchmark \
    -d 1 \
    --skip