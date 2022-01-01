import os
import torch


class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
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


def get_best_metric(args, default_best=0):
    best_metrc_filenm = os.path.join(args.saveloc, 'best_metric.txt')
    if os.path.exists(best_metrc_filenm):
        file1 = open(os.path.join(args.saveloc, 'best_metric.txt'), 'r')
        metric_s = file1.readline()
        config_s = file1.readline()
        file1.close()
    else:
        metric_s = ''
        config_s = ''

    if metric_s == '':
        best_metric = default_best
    else:
        best_metric = float(metric_s.split(':')[1])

    return best_metric, config_s


def save_checkpoint(state, save=True, filename='checkpoint.pth.tar'):
    if not save:
        return
    print('saving checkpoint...')
    torch.save(state, filename)


def write_best_metric(args, rewards_max, config_max):
    best_metrc_filenm = os.path.join(args.saveloc, 'best_metric.txt')
    file1 = open(best_metrc_filenm, 'w')
    file1.write(f'Best metric : {rewards_max:.3f}')
    file1.write(str(config_max))
    file1.close()

