CUDA_VISIBLE_DEVICES=0 python main.py \
    -model coil \
    --dataset imagenet1000 \
    -ms 20000 \
    -init 100 \
    -incre 100 \
    -net cosine_resnet18 \
    -p benchmark \
    -d 0