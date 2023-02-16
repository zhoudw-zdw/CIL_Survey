import argparse
from utils.model2exemplar import model2examplar

# benchmark backbone
dataset2net = {
    "cifar100": 'resnet32',
    'imagenet100': 'resnet18'
}

parser = argparse.ArgumentParser(description='A example of computing the number of exemplars.')

parser.add_argument('--dataset', type=str, default="cifar100")
parser.add_argument('--memory_size','-ms',type=int, default=2000)
parser.add_argument('--init_cls', '-init', type=int, default=10)
parser.add_argument('--increment', '-incre', type=int, default=10)
parser.add_argument('--model_name','-model', type=str, default=None)
parser.add_argument('--convnet_type','-net', type=str, default='resnet32')
parser.add_argument('--prefix','-p',type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])

args = parser.parse_args()
args = vars(args)

if args['prefix'] == 'fair':
    for dataset in ['cifar100', 'imagenet100']:
        print(f">>> {dataset}-{args['init_cls']}-{args['increment']}:")
        for model_name in ['icarl', 'memo']:
            args['model_name'] = model_name
            args['dataset'] = dataset
            
            args['convnet_type'] = dataset2net[dataset]
            
            if model_name == 'memo':
                args['convnet_type'] = "memo_" + args['convnet_type']
                
            exemplar_manager = model2examplar(args)
            exemplar_manager.get_infos()

elif args['prefix'] == 'auc':
    for dataset in ['cifar100', 'imagenet100']:
        args['dataset'] = dataset
        if dataset == 'cifar100':
            args['init_cls'], args['increment'] = 10, 10
            point_list = list(range(1,6))
        elif dataset == 'imagenet100':
            args['init_cls'], args['increment'] = 50, 5
            point_list = list(range(1,7))
        else:
            raise ValueError("Dataset error!")
        
        for point_idx in point_list:
            print(f"{dataset} point_idx:{point_idx}")
            if point_idx == 1:
                for model_name in ['memo', 'der']:
                    args['model_name'] = model_name
                    exemplar_manager = model2examplar(args)
                    exemplar_manager.get_infos(point_idx=point_idx)
            else:
                for model_name in ['memo', 'icarl']:
                    args['model_name'] = model_name
                    exemplar_manager = model2examplar(args)
                    exemplar_manager.get_infos(point_idx=point_idx)
        
            