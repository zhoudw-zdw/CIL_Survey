CUDA_VISIBLE_DEVICES=2 python main.py \
    -model coil \
    -init 10 \
    -incre 10 \
    -net cosine_resnet32 \
    -p benchmark \
    -d 0