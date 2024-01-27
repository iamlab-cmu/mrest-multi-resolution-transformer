import torch

from typing import Any, Mapping, Optional, Tuple

from timm.scheduler.scheduler_factory import create_scheduler


def check_config_flag(config: Optional[Mapping[str, Any]], flag: str, use_str: 'str' = 'use') -> bool:
    """Return True if flag is present and enabled in config.
    
    Args:
        config: Config dictionary. 
        flag: Flag to check.
        use_str: String to check if flag is enabled.
    Returns:
        True if flag is present and enabled in config.
    """
    if config is None or flag not in config:
        return False
    return config[flag][use_str]


def get_nonlinearity_from_str(nonlinearity: str):
    if nonlinearity == 'relu':
        return torch.nn.ReLU()
    elif nonlinearity == 'tanh':
        return torch.nn.Tanh()
    elif nonlinearity == 'gelu':
        return torch.nn.GELU()
    else:
        raise ValueError(f'Invalid nonlinearity: {nonlinearity}')

def create_optimizer_with_params(config: Mapping[str, Any], params):
    optimizer = None
    if config['name'] == 'Adam':
        adam_params = config['Adam']
        optimizer = torch.optim.Adam(params, lr=adam_params['lr'], eps=adam_params['eps'])
    elif config['name'] == 'AdamW':
        adamw_params = config[config['name']]
        optimizer = torch.optim.AdamW(params, lr=adamw_params['lr'], 
                                      eps=adamw_params['eps'],
                                      weight_decay=adamw_params['weight_decay'])
    else:
        raise ValueError(f"Invalid optimizer: {config['name']}")
    return optimizer


def create_optimizer_with_param_groups(config: Mapping[str, Any], param_groups: Mapping[str, Any]):
    from prettytable import PrettyTable
    params_table = PrettyTable()
    params_table.field_names = ["Type", "lr"]
    param_dicts = []
    for param_key, params in param_groups.items():
        if param_key not in config['param_groups']:
            raise ValueError(f'No lr specified for param group: {param_key}')
        lr = config['param_groups'][param_key]['lr']
        param_dicts.append(
            {
                'params': params,
                # timm_cosine uses lr as the learning rate key from param groups
                'lr': lr,
            }
        )
        params_table.add_row([param_key, f'{lr:.4e}'])
    print(params_table)

    if config['name'] == 'Adam':
        adam_params = config['Adam']
        optimizer = torch.optim.Adam(param_dicts, lr=adam_params['lr'], eps=adam_params['eps'])
    elif config['name'] == 'AdamW':
        adamw_params = config[config['name']]
        optimizer = torch.optim.AdamW(param_dicts, lr=adamw_params['lr'], 
                                      eps=adamw_params['eps'],
                                      weight_decay=adamw_params['weight_decay'])
    else:
        raise ValueError(f"Invalid optimizer: {config['name']}")
    return optimizer

    
def create_schedulder_with_params(config: Mapping[str, Any], optimizer) -> Tuple[Optional[Any], Mapping]:
    scheduler = None
    extra_dict = {}

    # Get scheduler params config
    scheduler_config = config[config['name']]
    if scheduler_config['use_timm']: 
        # Use timm scheduler
        total_epochs = scheduler_config['epochs']
        # Manually set the epochs correctly
        cooldown_epochs = scheduler_config.get('cooldown_epochs', 0)
        scheduler_config['epochs'] = total_epochs - cooldown_epochs
        scheduler, num_epochs = create_scheduler(scheduler_config, optimizer)

        assert num_epochs == total_epochs, (
            f"timm scheduler epochs {num_epochs} and total epochs {total_epochs} do not match.")

        extra_dict['num_epochs'] = num_epochs
        # Update scheduler in epochs
        extra_dict['t_in_epochs'] = True
        extra_dict['timm_scheduler'] = True

    elif config['name'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_config['t_max'], 
            eta_min=scheduler_config['eta_min'], 
            last_epoch=scheduler_config['last_epoch'])
        extra_dict['t_in_epochs'] = False
        extra_dict['timm_scheduler'] = False

    else:
        raise ValueError(f"Invalid optim schedulder: {config['name']}")
    
    return scheduler, extra_dict
