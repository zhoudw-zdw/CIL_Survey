CUDA_VISIBLE_DEVICES=3 python main.py \
    -model coil \
    -ms 7400\
    -init 10 \
    -incre 10 \
    -net cosine_resnet32 \
    -d 0\
    -p fair