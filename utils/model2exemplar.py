from .inc_net import get_convnet
from .toolkit import count_parameters
import math

# AUC, specific backbone for DER & MEMO
cifar_point2net = {
    1: 'conv2',
    2: 'resnet14_cifar',
    3: 'resnet20_cifar',
    4: 'resnet26_cifar',
    5: 'resnet32'
}

imagenet_point2net = {
    1: 'conv4',
    2: 'resnet10_imagenet',
    3: 'resnet18',
    4: 'resnet26_imagenet',
    5: 'resnet34_imagenet',
    6: 'resnet50_imagenet'
}

class model2examplar:
    def __init__(self, args) -> None:
        self.args = args
        self.verify_params()
        self.setup_base_infos()
        
    def verify_params(self):
        if self.args['dataset'] == 'cifar100':
            self.model_params = 463504 #resnet32
            self.img_size = 32 * 32 * 3
        elif self.args['dataset'] == 'imagenet100':
            self.model_params = 11176512 #resnet18
            self.img_size = 224 * 224 * 3
        else:
            raise ValueError("Dataset donot match convnet!")
        
    def setup_base_infos(self,):
        if self.args['dataset'] in ['cifar100', 'imagenet100']:
            self.total_classes = 100
        elif self.args['dataset'] == 'imagenet1000':
            self.total_classes = 1000
        else:
            raise ValueError("Dataset ERROR")
        self.task_num = (self.total_classes - self.args['init_cls']) // self.args['increment']
        
        assert isinstance(self.task_num, int)
        self.task_num += 1
        
    def fair2exemplar(self,):
        assert self.args['model_name'] != 'der', 'DER with 2000!'
        if self.args['model_name'] == 'memo':
            generalized_blocks, specialized_blocks = get_convnet(self.args['convnet_type'])
            g_params, s_params = count_parameters(generalized_blocks), count_parameters(specialized_blocks)
            total_params = g_params + s_params * self.task_num
            der_params = self.model_params * self.task_num #parameters of DER
            delta_params = der_params - total_params

        else:
            total_params = self.model_params
            delta_params = self.model_params * (self.task_num-1)

        extra_exemplars = delta_params * 4 / self.img_size
        extra_exemplars = math.floor(extra_exemplars)
        return extra_exemplars, total_params
    
    def auc2exemplar(self, point_idx:int):
        """
        - 5 points in CIFAR, 6 points in ImageNet
        
        CIFAR
        Memory Cost| 7.6      | 12.4     | 16.0     | 19.8     | 23.5     |
        ---------  | -------- | -------- | -------- | -------- | -------- |
        ConvType   | ConvNet2 | ResNet14 | ResNet20 | ResNet26 | ResNet32 |
        
        ImageNet
        Memory Cost| 329      | 493      | 755      | 872      | 1180     | 1273     |
        -----------|----------|----------|----------|----------|----------|----------|
        ConvType   | ConvNet4 | ResNet10 | ResNet18 | ResNet26 | ResNet34 | ResNet50 |
        """
        
        if self.args['dataset'] == 'cifar100':
            conv_type = cifar_point2net[point_idx]
        elif self.args['dataset'] in ['imagenet100', 'imagenet1000']:
            conv_type = imagenet_point2net[point_idx]
        else:
            raise ValueError('dataset error')
        
        der_backbone = get_convnet(conv_type)
        backbone_params = count_parameters(der_backbone)
        total_params = backbone_params * self.task_num
        
        if point_idx == 1:
            # (resnet18, 2000) & (resnet32, 2000) for single backbone methods (e.g., iCaRL)
            assert self.args['model_name'] in ['memo', 'der'], "Single backbone methods with 2000!"
            
            if self.args['model_name'] == 'memo':
                memo_conv_type = "memo_" + conv_type
                g_blocks, s_blocks = get_convnet(memo_conv_type)
                g_params, s_params = count_parameters(g_blocks), count_parameters(s_blocks)
                cur_total_params = g_params + self.task_num * s_params
                delta_params = self.model_params - cur_total_params
            elif self.args['model_name'] == 'der':
                cur_total_params = total_params
                delta_params = self.model_params - total_params
            else:
                raise ValueError('Only DER and MEMO!')
            
            assert delta_params > 0
            extra_exemplars = (delta_params * 4) / self.img_size
            extra_exemplars = math.floor(extra_exemplars)
        else:
            assert self.args['model_name'] != 'der', 'DER with 2000!'
            if self.args['model_name'] == 'memo':
                memo_conv_type = "memo_" + conv_type
                g_blocks, s_blocks = get_convnet(memo_conv_type)
                g_params, s_blocks = count_parameters(g_blocks), count_parameters(s_blocks)
                cur_total_params = g_params + self.task_num * s_blocks
                delta_params = total_params - cur_total_params
            else:
                cur_total_params = self.model_params
                delta_params = total_params - self.model_params
            
            assert delta_params > 0
            extra_exemplars = (delta_params * 4) / self.img_size
            extra_exemplars = math.floor(extra_exemplars)
        
        return extra_exemplars, cur_total_params
    
    def get_infos(self, **kwargs):
        if self.args['prefix'] == 'fair':
            extra_exempars, total_params = self.fair2exemplar()
            
        elif self.args['prefix'] == 'auc':
            extra_exempars, total_params = self.auc2exemplar(kwargs['point_idx'])
        
        total_exemplars = extra_exempars + 2000
                    
        model_cost = total_params * 4 / 1024 / 1024
        exemplar_cost = total_exemplars * self.img_size / 1024 / 1024
        
        params_M = total_params / 1e6
        
        print(f"{self.args['model_name']}, {self.args['dataset']}, {total_exemplars}, {exemplar_cost}MB, {params_M}M, {model_cost}MB, {exemplar_cost+model_cost}MB")