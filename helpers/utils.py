import os
import torch
import numpy as np
import random

def _seed_everything(random_seed=1223):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def count_parameters(model):
    "Couting number of trainable params"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_trainable(model, boolean: bool = True, except_layers: list = [], device_ids: list = []):
    if boolean:
        for i, param in model.named_parameters():
            param.requires_grad = True
        if len(except_layers) > 0:  # Except some layers
            for layer in except_layers:
                assert layer is not None
                if len(device_ids) <= 1:
                    for param in getattr(model, layer).parameters():
                        param.requires_grad = False
                else:
                    for param in getattr(model.module, layer).parameters():
                        param.requires_grad = False
    else:
        #         assert len(except_layers) > 0, "Require free layer"
        for i, param in model.named_parameters():
            param.requires_grad = False
        for layer in except_layers:  # Except some layers
            assert layer is not None
            if len(device_ids) <= 1:
                for param in getattr(model, layer).parameters():
                    param.requires_grad = True
            else:
                for param in getattr(model.module, layer).parameters():
                    param.requires_grad = True

    print("Training params: ", count_parameters(model))
    return model

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def check_folder(log_dir):
    os.makedirs(log_dir, exist_ok =True)
    return log_dir       

class Logger(object):
    def __init__(self, file):
        self.open(file, 'a')
    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def log(self, message, display=True):
        if display:
            print(message)
        self.file.write(message+"\n")
        self.file.flush()
    def close(self):
        self.file.close()

class FixedLengthQueue:
    """
    Queue class with fixed length
    If the queue reach max length, and enqueue method is call, it will dequeue the first and append the item
    
    """
    def __init__(self, max_length=10):
        self.queue = []
        self.max_length = max_length

    def enqueue(self, item):
        if self.size() < self.max_length:
            self.queue.append(item)
        elif self.is_full():
            self.dequeue()
            self.queue.append(item)
        else:
            raise ValueError("Queue is over the length")

    def dequeue(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            raise ValueError("Queue is empty")

    def is_full(self):
        return len(self.queue) == self.max_length

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)
    
    def avg(self):
        if not self.is_empty():
            return np.mean(self.queue)
        else:
            return 0.0
    def std(self):
        if not self.is_empty():
            return np.std(self.queue)
        else:
            return 0.0
        
def nth_prime_number(n):
    if n==1:
        return 2
    count = 1
    num = 1
    while(count < n):
        num +=2 #optimization
        if is_prime(num):
            count +=1
    return num

def is_prime(num):
    factor = 2
    while (factor * factor <= num):
        if num % factor == 0:
             return False
        factor +=1
    return True

def save_model(best_eval, cur_eval, model, epoch, optimizer, scheduler, config, **kwargs):

    logger = kwargs.get('logger')
    if (cur_eval["auc"]-cur_eval["HTER"])>=(best_eval["best_auc"]-best_eval["best_HTER"]):
        
        best_eval["best_auc"] = cur_eval["auc"]
        best_eval["best_HTER"] = cur_eval["HTER"]
        best_eval["tpr95"] = cur_eval["tpr"]
        best_eval["best_epoch"] = epoch
        model_path = os.path.join(config.PATH.model_path, "{}_p{}_best.pth".format(config.MODEL.model_name, config.protocol))
        torch.save({
            'epoch': epoch,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler,
            'args':config,
            'eva': (cur_eval["HTER"], cur_eval["auc"])
        }, model_path)
    logger.log("[Best result]\t: epoch={}, HTER={:.4f}, AUC={:.4f}".format(best_eval["best_epoch"],  best_eval["best_HTER"], best_eval["best_auc"]))

    model_path = os.path.join(config.PATH.model_path, "{}_p{}_recent.pth".format(config.MODEL.model_name, config.protocol))
    torch.save({
        'epoch': epoch,
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler,
        'args':config,
        'eva': (cur_eval["HTER"], cur_eval["auc"])
    }, model_path)

    return True