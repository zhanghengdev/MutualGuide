import math


def adjust_learning_rate(optimizer, base_lr, iteration, warm_iter, max_iter):
    """ warmup + cosine lr decay """
    if iteration <= warm_iter:
        lr = 1e-3 + (base_lr - 1e-3) * iteration / warm_iter
    else:
        lr = 1e-3 + (base_lr - 1e-3) * 0.5 * (1 + math.cos((iteration - warm_iter) * math.pi / (max_iter - warm_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def tencent_trick(model):
    """ no weight decay for bn and conv bias """
    (decay, no_decay) = ([], [])
    for (name, param) in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]
