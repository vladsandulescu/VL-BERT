import torch
from .eval_metric import EvalMetric

import numpy as np
from sklearn.metrics import roc_auc_score


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['label_logits']
            label = outputs['label']
            self.sum_metric += float((logits.argmax(dim=1) == label).sum().item())
            self.num_inst += logits.shape[0]


class ClsLoss(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ClsLoss, self).__init__('ClsLoss', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            self.sum_metric += float(outputs['cls_loss'].mean().item())
            self.num_inst += 1


class AUROC(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(AUROC, self).__init__('AUROC', allreduce=allreduce, num_replicas=num_replicas)

        self.eoe_metric = True
        self.outputs = {'logits': [], 'label': []}

    def update(self, outputs):
        with torch.no_grad():
            self.outputs['logits'].extend(outputs['label_logits'].detach().cpu().tolist())
            self.outputs['label'].extend(outputs['label'].detach().cpu().numpy())

            # we calculate AUC on epoch callback on accumulated results,
            # since AUC doesn't make sense on small batches

    def update_eoe(self):
        with torch.no_grad():
            self.sum_metric = torch.tensor(roc_auc_score(self.outputs['label'],
                                                          np.array(self.outputs['logits'])[:, 1]))
            self.num_inst = torch.tensor(1.)
