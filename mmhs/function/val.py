from collections import namedtuple
import os
import csv
from tqdm import tqdm, trange
import numpy as np
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch, epoch_num,
                  prefix, frequent, args, config):

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net.eval()
    metrics.reset()

    image_ids = []
    predicted_probs = []
    targets = []
    cur_id = 0
    for nbatch, batch in zip(trange(len(val_loader)), val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        outputs = net(*datas)
        outputs.update({'label': label})
        metrics.update(outputs)

        val_database = val_loader.dataset.database
        bs = val_loader.batch_sampler.batch_size if val_loader.batch_sampler is not None else val_loader.batch_size
        image_ids.extend([val_database[id]['id'] for id in range(cur_id, min(cur_id + bs, len(val_database)))])
        predicted_probs.extend(outputs['label_probs'].detach().cpu().tolist())
        targets.extend(label.detach().cpu().tolist())
        cur_id += bs

    if (epoch_num + 1) % frequent == 0:
        param_name = '{}-{:04d}.model'.format(prefix, epoch_num)
        save_path = os.path.dirname(param_name)
        save_name = os.path.basename(param_name)

        predicted_probs = np.array(predicted_probs)
        targets = np.array(targets)
        result = [{'id': id, 'proba': np.round(proba, 4),
                   'label': label, 'target': target}
                  for id, proba, label, target in
                  zip(image_ids, predicted_probs, (predicted_probs > .5).astype(int), targets)]

        cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
        result_csv_path = os.path.join(save_path, '{}_mmhs_{}.csv'.format(cfg_name if save_name is None else save_name,
                                                                         config.DATASET.VAL_IMAGE_SET))
        with open(result_csv_path, 'w', newline='') as f:
            fieldnames = ['id', 'proba', 'label', 'target']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for data in result:
                writer.writerow(data)
        print('Val result csv saved to {}.'.format(result_csv_path))
