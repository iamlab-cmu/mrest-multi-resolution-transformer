# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import copy
import pickle
import os
import csv
import wandb

from pathlib import Path
from typing import Any, Callable, Optional, Dict, Mapping
from heapq import heappush, heappop
from omegaconf import DictConfig, ListConfig, MissingMandatoryValue
from collections import OrderedDict


def convert_config_dict_to_dict(config: DictConfig) -> OrderedDict:
    cfg_dict = OrderedDict()
    for k in config.keys():
        try: 
            v = config[k]
        except MissingMandatoryValue:
            print(f'Did not find any values for key: {k}')
            continue

        if isinstance(v, ListConfig):
            cfg_dict[k] = [val for val in v]
        elif isinstance(v, DictConfig):
            cfg_dict[k] = convert_config_dict_to_dict(v)
        else:
            cfg_dict[k] = copy.deepcopy(v)

    return cfg_dict


class TopKLogger:
    def __init__(self, k: int):
        self.max_to_keep = k
        self.checkpoint_queue = []
    
    def push(self, ckpt: str, success: float):
        # NOTE: We have a min heap
        if len(self.checkpoint_queue) < self.max_to_keep:
            heappush(self.checkpoint_queue, (success, ckpt))
            return True
        else:
            curr_min_success, _ = self.checkpoint_queue[0]
            if curr_min_success < success:
                heappop(self.checkpoint_queue)
                heappush(self.checkpoint_queue, (success, ckpt))
                return True
            else:
                return False


class DataLog:

    STEP_SUFFIX = '_step'

    def __init__(self):
        self.log = {}
        self.max_len = 0

    def init_wb(self, cfg, unique_kwargs: Optional[Mapping[str, Any]] = None):
        print(cfg.keys())
        run = wandb.init(project=cfg.wandb.project, 
                         entity=cfg.wandb.entity, 
                         group=cfg.wandb.group,
                         name=f'{cfg.wandb.name}_{cfg.job_name}_{cfg.tag}')
        train_proj_path = (Path(__file__) / '../../').resolve()
        wandb.run.log_code(str(train_proj_path))

        # Save mdetr kwargs using mdetr namespace
        mdetr_cfg = {}
        for k, v in cfg.mdetr.items():
            mdetr_cfg['mdetr_' + k] = v
        lr_params = {
             'optimizer_name': cfg.bc_kwargs.optimizer.name,
        }
        if 'param_groups' in cfg.bc_kwargs.optimizer:
            lr_params['param_groups_use'] = cfg.bc_kwargs.optimizer.param_groups.use
            for k, v in cfg.bc_kwargs.optimizer.param_groups.items():
                if (isinstance(v, dict) or isinstance(v, DictConfig)) and 'lr' in v:
                    lr_params['param_groups_' + k + '_lr'] = v['lr']

        fullcfg = {**cfg, **cfg.env_kwargs, **cfg.bc_kwargs, **lr_params, **mdetr_cfg}
        if unique_kwargs:
            fullcfg.update(unique_kwargs)
        wandb.config.update(fullcfg)
        train_proj_path = (Path(__file__) / '../../').resolve()
        wandb.run.log_code(str(train_proj_path))

        return run
    
    def init_eval_wb(self, project, entity, run_id, cfg, update_cfg: bool = True):
        wandb.init(project=project, entity=entity, id=run_id, resume='must')
        if update_cfg:
            fullcfg = {**cfg, **cfg.env_kwargs, **cfg.bc_kwargs}
            wandb.config.update(fullcfg, allow_val_change=True)

    def log_kv(self, key, value):
        # logs the (key, value) pair

        # TODO: This implementation is error-prone:
        # it would be NOT aligned if some keys are missing during one iteration.
        if key not in self.log:
            self.log[key] = []
            self.log[key + DataLog.STEP_SUFFIX] = []
        self.log[key].append(value)
        if len(self.log[key + DataLog.STEP_SUFFIX]) == 0:
            self.log[key + DataLog.STEP_SUFFIX].append(1)
        else:
            self.log[key + DataLog.STEP_SUFFIX].append(self.log[key + DataLog.STEP_SUFFIX][-1] + 1)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def log_loss(self, loss, optim_logs: Optional[Mapping[str, float]] = None):
        log_dict = {'train_loss': loss}
        if optim_logs:
            log_dict.update(optim_logs)
        wandb.log(log_dict)

    def save_wb(self, step, logs: Optional[Dict] = None, filter_step_key: bool = False):
        '''
        Save logs to wandb:

        step: Integer step used as overall step in wandb.
        logs: Optional[dict]. Log dictionary to save.
        filter_step_key: If True will remove any `_step` keys since they are always
            set to 1 for now.
        '''
        if not logs:
            logs = self.get_current_log()

        if filter_step_key:
            logs = { k: v for k, v in logs.items() 
                if not k.endswith(DataLog.STEP_SUFFIX)
                }
        wandb.log(logs, step=step)


    def save_log(self, save_path):
        # TODO: Validate all lengths are the same.
        pickle.dump(self.log, open(save_path + '/log.pickle', 'wb'))
        with open(save_path + '/log.csv', 'w') as csv_file:
            fieldnames = list(self.log.keys())
            if 'iteration' not in fieldnames:
                fieldnames = ['iteration'] + fieldnames

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {'iteration': row}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            # TODO: this is very error-prone (alignment is not guaranteed)
            row_dict[key] = self.log[key][-1]
        return row_dict

    def shrink_to(self, num_entries):
        for key in self.log.keys():
            self.log[key] = self.log[key][:num_entries]

        self.max_len = num_entries
        assert min([len(series) for series in self.log.values()]) == \
            max([len(series) for series in self.log.values()])

    def read_log(self, log_path):
        assert log_path.endswith('log.csv')

        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row, row_dict in enumerate(listr):
                for key in keys:
                    try:
                        data[key].append(eval(row_dict[key]))
                    except:
                        print("ERROR on reading key {}: {}".format(key, row_dict[key]))

                if 'iteration' in data and data['iteration'][-1] != row:
                    raise RuntimeError("Iteration %d mismatch -- possibly corrupted logfile?" % row)

        self.log = data
        self.max_len = max(len(v) for k, v in self.log.items())
        print("Log read from {}: had {} entries".format(log_path, self.max_len))
    
    def replace_values_with_key(self, log_key_prefix: str, reduce_func: Callable, new_log_key_prefix: str):
        '''Replace keys with new values as obtained from reduce_func.'''
        reduced_values = {}
        for log_key, log_val in self.log.items():
            if log_key.startswith(log_key_prefix) and not log_key.endswith(DataLog.STEP_SUFFIX):
                reduced_val = reduce_func(log_val)
                reduced_key = log_key.replace(log_key_prefix, new_log_key_prefix)
                reduced_values[reduced_key] = reduced_val

        return reduced_values


