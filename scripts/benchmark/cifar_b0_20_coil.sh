CUDA_VISIBLE_DEVICES=3 python main.py \
    -model coil \
    -init 20 \
    -incre 20 \
    -net cosine_resnet32 \
    -p benchmark \
    -d 0