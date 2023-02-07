CUDA_VISIBLE_DEVICES=2 python main.py \
    --dataset imagenet100 \
    -model coil \
    -init 10 \
    -incre 10 \
    -net cosine_resnet18 \
    -p benchmark \
    -d 0