CUDA_VISIBLE_DEVICES=3 python main.py \
    --dataset imagenet100 \
    -model coil \
    -init 50 \
    -incre 50 \
    -net cosine_resnet18 \
    -p benchmark \
    -d 0