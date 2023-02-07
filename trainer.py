import copy
import datetime
import json
import logging
import os
import sys
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args['time_str'] = time_str
    
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["convnet_type"],
        args["seed"],
        init_cls,
        args["increment"],
    )
    args['exp_name'] = exp_name

    if args['debug']:
        logfilename = "logs/debug/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )
    else:
        logfilename = "logs/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )

    args['logfilename'] = logfilename

    csv_name = "{}_{}_{}_B{}_Inc{}".format( 
        args["dataset"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"],
    )
    args['csv_name'] = csv_name
    os.makedirs(logfilename, exist_ok=True)

    log_path = os.path.join(args["logfilename"], "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Time Str >>> {args['time_str']}")
    # save config
    config_filepath = os.path.join(args["logfilename"], 'configs.json')
    with open(config_filepath, "w") as fd:
            json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve, no_nme = {"top1": [], "top5": []}, {"top1": [], "top5": []}, True
    start_time = time.time()
    logging.info(f"Start time:{start_time}")
    
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        if task == data_manager.nb_tasks-1:
            cnn_accy, nme_accy = model.eval_task(save_conf=True)
            no_nme = True if nme_accy is None else False
        else:
            cnn_accy, nme_accy = model.eval_task(save_conf=False)
        model.after_task()
        
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
    
    end_time = time.time()
    logging.info(f"End Time:{end_time}")
    cost_time = end_time - start_time
    save_time(args, cost_time)
    save_results(args, cnn_curve, nme_curve, no_nme)
    if args['model_name'] not in ["podnet", "coil"]:
        save_fc(args, model)
    else:
        save_model(args, model)

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")

def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve['top5']
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve['top5']
    
    #-------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    #-------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")


    #-------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")       

        #-------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n") 
