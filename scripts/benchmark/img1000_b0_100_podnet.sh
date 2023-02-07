python main.py \
    -model podnet \
    --dataset imagenet1000 \
    -ms 20000 \
    -init 100 \
    -incre 100 \
    -net cosine_resnet18 \
    -p benchmark \
    -d 3