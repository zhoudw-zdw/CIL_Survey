CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset imagenet100 \
    -model coil \
    -ms 4970\
    -init 50\
    -incre 5 \
    -net cosine_resnet18 \
    -d 0\
    -p fair