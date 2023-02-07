CUDA_VISIBLE_DEVICES=3 python main.py \
    --dataset imagenet100 \
    -model coil \
    -ms 4671\
    -init 10\
    -incre 10 \
    -net cosine_resnet18 \
    -d 0\
    -p fair