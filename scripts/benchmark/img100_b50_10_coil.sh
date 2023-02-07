CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --dataset imagenet100 \
    -model coil \
    -init 50 \
    -incre 10 \
    -net cosine_resnet18 \
    -p benchmark \
    -d 0 1