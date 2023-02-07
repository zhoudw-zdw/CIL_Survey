CUDA_VISIBLE_DEVICES=0 python main.py \
    -model coil \
    -ms 13475\
    -init 5\
    -incre 5 \
    -net cosine_resnet32 \
    -d 0\
    -p fair