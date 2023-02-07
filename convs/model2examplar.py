
class model2examplar:
    def __init__(self, args) -> None:
        self.args = args
        self.verify_params()
        self.setup_base_infos()
        
    def verify_params(self):
        if self.args['prefix'] == 'fair':
            if self.args['dataset'] == 'cifar100':
                assert  'resnet32' in self.args['convnet_type']
                self.model_params = 463504
                self.img_size = 32 * 32 *3
            elif "imagenet" in self.args['dataset']:
                assert 'resnet18' in self.args['convnet_type']
            else:
                raise ValueError("Dataset donot match convnet!")
        elif self.args['prefix'] == 'auc':
            pass
        else:
            raise  ValueError("Prefix ERROR")
        
    def setup_base_infos(self,):
        if self.args['dataset'] in ['cifar100', 'imagenet100']:
            self.total_classes = 100
        elif self.args['dataset'] == 'imagenet1000':
            self.total_classes = 1000
        else:
            raise ValueError("Dataset ERROR")
        self.model2exemplar = int(self.model_params*4/self.img_size)
        
    def fair2exemplar(self,):
        task_num = (self.total_classes - self.args['init_cls']) / self.args['increment']
        assert isinstance(task_num, int)
        task_num += 1
        
        extra_examplars = self.model2exemplar * (task_num-1)
        return extra_examplars
        