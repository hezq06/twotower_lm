"""
Python package for NCA learning with Deepspeed
Main file for training evaluation
Developer: Harry He
2022-04/25
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time, os, pickle, sys, shutil, copy, datetime, argparse
import torch
import deepspeed

# from ncautil2.ncamath import *
from ttlm.datautil import *
from ttlm.model import CAL_LOSS

class PyTrainConfig(object):
    """
    A config class to manage parameter for PyTrain_Main
    """
    def __init__(self, para={}):

        self.run_config = para
        self.description = para["description"]
        self.ds_config = para.get("ds_config", "deepspeed_config.json")
        self.epoch = para.get("epoch", 2)
        self.profiler_mode = para.get("profiler_mode", "loss") # loss, top1, top5
        self.print_step = para.get("print_step", 200)
        self.check_point_path = para.get("check_point_path", None) # None means do not checkpoint
        self.mem_eval_mode = para.get("mem_eval_mode", None)
        self.lr_schedule_mode = para.get("lr_schedule_mode", None)  # None, step_decay, linear_decay
        self.lr_linear_decay = para.get("lr_linear_decay", 1.0)  # How much going to be decayed
        self.warm_up_steps = para.get("warm_up_steps", 10000)
        # self.step_decay_epoch = para.get("step_decay_epoch", 30)
        self.save_best_flag = para.get("save_best_flag", False)
        self.save_all_checkpoint = para.get("save_all_checkpoint", False)
        self.epoch_shift_resume = para.get("epoch_shift_resume", 0)
        self.seed = para.get("seed", 12345)
        self.master_port = para.get("master_port", 60000)

    def __repr__(self):
        print_str=""
        for arg in dir(self):
            if not arg.startswith('__'):
                print_str = print_str + str(arg)+": "+str(getattr(self,arg))+"\n"
        return print_str

class PyTrainLog(object):
    """
    A log class for handling all training logs, can output text overview, and save pickle training/evaluation history
    for after processing
    """

    def __init__(self, config, profiler):

        self.config = config
        self.profiler = profiler
        currentDT = datetime.datetime.now()
        self.result = {
            "train_hist" : [],
            "eval_hist" : [],
            "best_evalres": None
        }

        self.log=""
        self.add_log(self.config.description)
        self.add_log("Time of training starting %s. " % str(currentDT))
        self.add_log("Start training with config:")
        self.add_log(str(self.config))

        self.startt = time.time()

        self.checkpoint_pathname = None
        self.best_evalres = None

    def add_log(self, line):
        print(line)
        self.log = self.log + line + "\n"

    def log_time(self):
        endt = time.time()
        print("Time used till now:", endt - self.startt)
        self.log = self.log + "Time used till now: " + str(endt - self.startt) + "\n"

    def save(self, file):
        save_text(self.log, file)

    def profile(self, iis, loss, lr_current, output, target, datax_size_0, timeend, print_step=200, force_print=False):

        self.profiler.profile(loss, output, target, datax_size_0, timeend)

        if iis % print_step == 0:

            if lr_current is not None:
                self.add_log("Current learning rate: %s"%lr_current)
            self.profiler.print(iis)
            self.add_log(self.profiler.progress.print_str)

        if force_print:
            self.profiler.print(iis)
            self.add_log(self.profiler.progress.print_str)

        sys.stdout.flush()

    def plot_result(self, result = None, window=10):
        if result is None:
            result = self.result
        train_res = np.array(result["train_hist"][:-1])
        epoch, nperepoch = train_res.shape
        cutlen = int(np.floor(nperepoch/window)*window)
        train_res = train_res[:,:cutlen].reshape(-1, window)
        train_res = np.mean(train_res, axis=-1)
        epochx_train = np.array(range(len(train_res)))/len(train_res)*epoch
        plt.plot(epochx_train, train_res)

        eval_res = np.array(result["eval_hist"][:-1])
        epoch, nperepoch = eval_res.shape
        cutlen = int(np.floor(nperepoch/window)*window)
        eval_res = eval_res[:,:cutlen].reshape(-1, window)
        eval_res = np.mean(eval_res, axis=-1)
        epochx_eval = np.array(range(len(eval_res))) / len(eval_res) * epoch
        plt.plot(epochx_eval, eval_res)

        plt.legend(["train","eval"])
        plt.show()

class PyTrainMain(object):
    """
    A main class to run training
    """

    def __init__(self, loss_model, data_dict, config, device="cuda:0"):

        self.data_dict = data_dict
        self.device = device
        self.config = config
        self.profiler = CustomProfiler(config.batch_size, mode=config.profiler_mode)
        self.pylog = PyTrainLog(config, self.profiler)
        if self.device is not None:
            torch.cuda.set_device(self.device)

        parser = argparse.ArgumentParser()
        parser.deepspeed_config = config.ds_config
        parser.print_steps = self.config.print_step
        parser.master_port = self.config.master_port

        weight_decay=config.ds_config_dict['optimizer']['params']['weight_decay']
        try:
            params = loss_model.model.prepare_optimizer_parameters(weight_decay)
        except:
            print("No model specify optimizer parameter preparer, use default.")
            params = self._default_optimizer_parameters(loss_model)

        loss_model.loss_mode = "train"
        self.model, optimizer, _, _ = deepspeed.initialize(args=parser,
                                                             model=loss_model,
                                                             model_parameters=params)


        self.pt_model = loss_model
        self.optimizer = optimizer

        self.previous_best_evalres_path = None
        self.previous_final_eval_path = None


    def _default_optimizer_parameters(self, loss_model):

        return loss_model.parameters()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def lr_schedule(self, ii_tot, ii_epoch, tot_epoch, step_per_epoch, init_lr):

        if self.config.lr_schedule_mode == "linear_decay":
            if self.config.warm_up_steps > 0 and ii_tot < self.config.warm_up_steps:
                self.set_lr((ii_tot / self.config.warm_up_steps) * init_lr)
            else:  # lr_linear_decay from 0 to 1, indicate final decay rate, 0 is no decay
                lr_cstep = (ii_tot - self.config.warm_up_steps) / (tot_epoch * step_per_epoch - self.config.warm_up_steps)
                self.set_lr((1 - self.config.lr_linear_decay * lr_cstep) * init_lr)
        elif self.config.lr_schedule_mode == "step_decay":
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = init_lr * (0.1 ** (ii_epoch // self.config.step_decay_epoch))
            self.set_lr(lr)

    def _check_point(self, mode, ii_epoch, forward_hist):

        if mode == "train":
            self.pylog.result["train_hist"].append(forward_hist)
        elif mode == "val":
            self.pylog.result["eval_hist"].append(forward_hist)
            evalres = np.mean(forward_hist)

            ## Eval profile save checkpoint
            if self.config.check_point_path is not None:
                ### Final Eval Save
                checkpoint_pathname = self.config.check_point_path + "epoch%s" % (ii_epoch+self.config.epoch_shift_resume) + "perp%.5s." % evalres + self.device
                if self.previous_final_eval_path is not None and not self.config.save_all_checkpoint:
                    try:
                        shutil.rmtree(self.previous_final_eval_path)
                    except:
                        pass
                self.save_session(checkpoint_pathname+"_final")
                self.previous_final_eval_path = checkpoint_pathname+"_final"
                ### Best Eval Save
                if self.config.profiler_mode=="loss":
                    best_eval=evalres
                elif self.config.profiler_mode=="top1":
                    best_eval = -self.profiler.top1.avg.item()
                elif self.config.profiler_mode == "top5":
                    best_eval = -self.profiler.top5.avg.item()

                if self.config.save_best_flag:
                    checkpoint_pathname_best = checkpoint_pathname + "_best_"+"{:.2f}".format(best_eval)+"_"
                    if self.previous_best_evalres_path is None:
                        self.save_session(checkpoint_pathname_best)
                        self.previous_best_evalres_path = checkpoint_pathname_best
                        self.pylog.result["best_evalres"] = best_eval
                    elif self.pylog.result["best_evalres"] > best_eval:
                        try:
                            shutil.rmtree(self.previous_best_evalres_path )
                        except:
                            pass
                        self.save_session(checkpoint_pathname_best)
                        self.previous_best_evalres_path = checkpoint_pathname_best
                        self.pylog.result["best_evalres"] = best_eval
            else:
                pass

    def _prepare_env(self, ii_epoch, mode="train"):

        self.profiler.reset()
        self.profiler.set_prefix("Batch [%s]" % ii_epoch)
        self.profiler.update_batch(len(self.data_dict[mode]))

        if mode=="train":
            self.model.train()
            self.pt_model.loss_mode = "train"
        elif mode=="val":
            self.model.eval()
            self.pt_model.loss_mode = "eval"

    def _transfer_device(self, datax):
        try:
            datax = datax.to(self.device)
        except:
            datax = [item.to(self.device) for item in datax]
        return datax

    def _forward_model(self, dataset, ii_epoch, mode="train", eval_mem_flag=False):

        step_per_epoch = len(dataset)
        lr = self.config.ds_config_dict['optimizer']['params']['lr']
        forward_hist = []
        gradient_acc_step = self.config.ds_config_dict["gradient_acc_step"]
        self.lr_schedule(gradient_acc_step, 0, self.config.epoch, step_per_epoch, lr) ## Initialize lr

        for iis, (datax, labels) in enumerate(dataset):

            ii_tot = iis + ii_epoch * step_per_epoch
            cstep = ii_tot / (self.config.epoch * step_per_epoch)
            timeend = time.time()

            datax = self._transfer_device(datax)
            labels = self._transfer_device(labels)

            if mode == "train":

                self.model.train()
                self.pt_model.train()
                self.pt_model.loss_mode = "train"

                loss = self.model(datax, labels, schedule=cstep)
                self.model.backward(loss)
                if self.model.is_gradient_accumulation_boundary():
                    # Macro step would advance optimizer learning rate
                    self.lr_schedule(ii_tot, ii_epoch, self.config.epoch, step_per_epoch, lr)
                    self.model.step()
                else:
                    # Call DeepSpeed engine step on micro steps
                    self.model.step()

                self.pylog.profile(iis, loss, self.optimizer.param_groups[0]["lr"], self.pt_model.output, labels,
                                   self.config.batch_size, timeend, print_step=self.config.print_step)

            elif mode == "val":

                self.model.eval()
                self.pt_model.eval()
                with torch.no_grad():
                    loss = self.model(datax, labels, schedule=cstep)

                if eval_mem_flag:
                    self.pt_model.model.eval_mem(datax, labels)

                self.pylog.profile(iis, loss, None, self.pt_model.output, labels,
                                   self.config.batch_size, timeend, print_step=self.config.print_step)

            else:
                raise Exception("Unknown mode.")

            forward_hist.append(loss.item())

        if eval_mem_flag:
            try:
                self.pt_model.model.post_eval_mem()
            except Exception as e:
                print(e)

        self.pylog.profile(iis, loss, None, self.pt_model.output, labels,
                           self.config.batch_size, timeend, print_step=self.config.print_step, force_print=True)
        self._check_point(mode, ii_epoch, forward_hist)

    def _postscript(self):
        pass

    def save_session(self,file_name):

        def _save_session(model, file_name):
            try:
                os.mkdir(file_name)
            except:
                print("Warning, folder %s exist"%file_name)
            save_model(model, os.path.join(file_name, file_name + ".model"))
            session_log = dict([])
            session_log["result"] = self.pylog.result
            session_log["config"] = self.config
            # session_log["optimizer_state"] =  self.optimizer.state_dict().cpu() # Tooooo big
            save_data(session_log, os.path.join(file_name, file_name + ".data"))
            self.pylog.save(os.path.join(file_name, file_name + ".txt"))

        _save_session(self.pt_model.model, file_name)

    def run_training(self):
        self.run_training_eval("train")

    def do_eval(self, eval_mem_flag=False):
        self.run_training_eval("val", eval_mem_flag=eval_mem_flag)

    def run_training_eval(self, run_mode, eval_mem_flag=False):

        assert run_mode in ["train","val"]
        self.pylog.startt = time.time()

        # epoch, lr, optimizer_label, momentum = self.config.epoch, self.config.lr, self.config.optimizer_label, self.config.momentum
        epoch, profiler = self.config.epoch, self.profiler

        if run_mode=="val":
            epoch=1

        for ii_epoch in range(epoch):
            self.pylog.add_log("Starting epoch %s." % str(ii_epoch))
            sys.stdout.flush() # writting to profile

            ## train
            if run_mode=="train":
                self._prepare_env(ii_epoch, mode=run_mode)
                try:
                    self.data_dict[run_mode].dataset.reshuffle()
                except:
                    self.pylog.add_log("WARNING! dataset has no reshuffle method.")

                self._forward_model(self.data_dict[run_mode], ii_epoch, mode="train", eval_mem_flag=eval_mem_flag)

                self.pylog.log_time()

            ## validation
            self.pylog.add_log("Validation of epoch "+str(ii_epoch)+":")
            self.pylog.add_log("Start evaluation ..." +str(len(self.data_dict["val"])))
            self._prepare_env(ii_epoch, mode="val")
            self._forward_model(self.data_dict["val"], ii_epoch, mode="val", eval_mem_flag=eval_mem_flag)

            self.pylog.log_time()

        self._postscript()

    def plot_result(self, result=None, window=10):
        self.pylog.plot_result(result=result, window=window)

    def get_model(self):
        return self.pt_model.model

    def get_data_loader(self, mode="val"):
        return self.data_dict[mode]

    def get_data(self, mode="val"):
        return self.data_dict[mode].dataset

    def set_data_mode(self, mode="all"):
        data = self.get_data_loader()
        data.dataset.mode = mode

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.print_str = '\t'.join(entries)
        # print(self.print_str)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class CustomProfiler(object):

    def __init__(self, data_batch_len, ii_epoch=0, mode="loss"):
        """
        Code borrowed from effetient-net project
        :param data_batch_len: batch_size, like 24 for bert
        :param ii_epoch:
        """
        self.batch_time = AverageMeter('Time', ':6.3f')
        self.losses = AverageMeter('Loss', ':.4e')
        self.mode = mode
        if mode == "loss":
            self.progress = ProgressMeter(data_batch_len, self.batch_time, self.losses, prefix="Epoch: [{}]".format(ii_epoch))
        elif mode == "top1":
            self.top1 = AverageMeter('Acc@1', ':6.2f')
            self.progress = ProgressMeter(data_batch_len, self.batch_time, self.losses, self.top1, prefix="Epoch: [{}]".format(ii_epoch))
        elif mode == "top5":
            self.top1 = AverageMeter('Acc@1', ':6.2f')
            self.top5 = AverageMeter('Acc@5', ':6.2f')
            self.progress = ProgressMeter(data_batch_len, self.batch_time, self.losses, self.top1, self.top5, prefix="Epoch: [{}]".format(ii_epoch))
        else:
            raise Exception("Unknown mode ...")

    def reset(self):
        self.batch_time.reset()
        self.losses.reset()
        if self.mode == "top1":
            self.top1.reset()
        if self.mode == "top5":
            self.top1.reset()
            self.top5.reset()

    def set_prefix(self,prefix):
        self.progress.prefix=prefix

    def update_batch(self,num_batches):
        self.progress.batch_fmtstr = self.progress._get_batch_fmtstr(num_batches)

    def profile(self, loss, output, target, datax_size_0, timeend):
        # measure accuracy and record loss (pytorch imagenet example)

        self.losses.update(loss.item(), datax_size_0)
        if self.mode == "top1":
            acc1,  = self.accuracy(output, target, topk=(1, ))
            self.top1.update(acc1[0], datax_size_0)
            # self.acc1=acc1
        if self.mode == "top5":
            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            self.top1.update(acc1[0], datax_size_0)
            self.top5.update(acc5[0], datax_size_0)
            # self.acc1 = acc1
            # self.acc5 = acc5
        self.batch_time.update(time.time() - timeend)

    def print(self, iis):
        self.progress.print(iis)

    def accuracy(self, output, target, topk=(1,)):
        """ Computes the accuracy over the k top predictions for the specified values of k
        from pytorch imagenet example"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

def nca_prepare_training(config):
    """
    Main entrance for training initialization
    :param config: a dict for everything
    "model": class of model for training
    "dataset": class of dataset for training/eval
    :return:
    """
    # Setting all the seeds so that the task is random but same accross processes

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    ## Prepare Model
    model_path = config.get("pretrained_model", None)
    ModelType = config["model_type"]
    model = ModelType(config)
    # cuda_device = os.getenv('CUDA_VISIBLE_DEVICES')
    rank = os.getenv('LOCAL_RANK', '0')
    DEVICE = "cuda:" + rank
    if model_path is not None:
        model_temp = load_model(model_path, map_location="cpu")
        copy_model_state(model, model_temp)
    loss = CAL_LOSS(model)

    ## Read Deepspeed config
    ptconfig = PyTrainConfig(config)

    ds_config = load_data(ptconfig.ds_config,"json")
    batch_size = ds_config['train_micro_batch_size_per_gpu']
    ptconfig.ds_config_dict = ds_config
    ptconfig.batch_size = batch_size
    assert batch_size == config["batch_size"]
    ## Prepare Data
    DatasetType = config["dataset_type"]
    dataloader = DatasetType(config["data_train"]).get_dataloader(batch_size)
    dataloader_val = DatasetType(config["data_val"]).get_dataloader(batch_size)
    data_dict = {"train": dataloader, "val": dataloader_val}

    ## Prepare pytrain
    ptM = PyTrainMain(loss, data_dict, ptconfig, device=DEVICE)
    return ptM

def resume_training(folder_path, device="cuda:0", config_extra = None):
    """
    Resume the ptM of a folder
    :param folder_path:
    :return:
    """
    files = os.listdir(folder_path)
    for item in files:
        if item.endswith("data"):
            data = load_data(os.path.join(folder_path,item))
            break
    for item in files:
        if item.endswith("model"):
            model=os.path.join(folder_path, item)
            break
    config = data["config"].run_config
    config["pretrained_model"] = model
    config["device"]=device
    if config_extra is not None:
        config.update(config_extra)
    ptM = nca_prepare_training(config)
    # try:
    #     ptM.optimizer.load_state_dict(data["optimizer_state"])
    # except:
    #     print("Skip optimizer state load.")
    return ptM
