import math

import numpy as np


def update_lr(optimizer, epoch, max_epochs=25, base_lr=2e-5, min_lr=1e-7,
              warmup_length=1, warmup_type='linear',
              rampdown_start=1, rampdown_stop=20, rampdown_type='cosine'):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if rampdown_stop <= 0:
        rampdown_stop = max_epochs

    # Update learning rate
    lr = base_lr
    if epoch < warmup_length:
        if warmup_type == 'linear':
            lr = lr * epoch / warmup_length
        elif warmup_type == 'exp':
            epoch = np.clip(epoch, 0.5, warmup_length)
            phase = 1.0 - epoch / warmup_length
            lr = lr * float(np.exp(-5.0 * phase * phase))
        else:
            raise NotImplementedError
    elif epoch < rampdown_start:
        lr = lr
    elif epoch < rampdown_stop:
        if rampdown_type == 'cosine':
            offset = rampdown_start
            lr = min_lr + (lr - min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - offset) / (rampdown_stop - offset)))
        elif rampdown_type.startswith('step'):
            distance, factor = rampdown_type.split('_')[1:]
            distance, factor = int(distance), float(factor)
            steps = epoch // distance
            lr = lr * (factor ** steps)
            lr = max(lr, min_lr)
        elif rampdown_type == 'linear':
            e = epoch - rampdown_start
            m = rampdown_stop - rampdown_start
            lr -= (lr - min_lr) * (e / m)
            lr = max(lr, min_lr)
        else:
            raise NotImplementedError
    else:
        lr = min_lr

    # Update optimizer
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
