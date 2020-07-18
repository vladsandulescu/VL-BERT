import os
from hm.function.test import test_net


class TestMonitor(object):
    def __init__(self, prefix, frequent, args, config):
        super(TestMonitor, self).__init__()
        self.prefix = prefix
        self.frequent = frequent
        self.args = args
        self.config = config

    def __call__(self, epoch_num, net, optimizer, writer, validation_monitor=None):
        if (epoch_num + 1) % self.frequent == 0:
            param_name = '{}-{:04d}.model'.format(self.prefix, epoch_num)
            save_to_best = False
            if validation_monitor is not None:
                if validation_monitor.best_epoch == epoch_num:
                    save_to_best = True

            test_net(args=self.args, config=self.config, ckpt_path=param_name,
                     save_path=os.path.dirname(param_name),
                     save_name=os.path.basename(param_name))

            if save_to_best:
                best_param_name = '{}-best.model'.format(self.prefix)
                test_net(args=self.args, config=self.config, ckpt_path=best_param_name,
                         save_path=os.path.dirname(best_param_name),
                         save_name=os.path.basename(best_param_name))
                print('Save new best model to {}.'.format(best_param_name))
