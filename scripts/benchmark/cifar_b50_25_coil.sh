CUDA_VISIBLE_DEVICES=2 python main.py \
    -model coil \
    -init 50 \
    -incre 25 \
    -net cosine_resnet32 \
    -p benchmark \
    -d 0