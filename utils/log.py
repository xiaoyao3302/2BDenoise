import datetime
import copy
import torch
import os
import pdb


class IOStream():
    """
    Logging to screen and file
    """
    def __init__(self, args):
        self.path = args.out_path + '/' + args.dataset + '/' + args.model
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if args.exp_name is None:
            timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
            self.path = self.path + '/' + timestamp
        else:
            self.path = self.path + '/' + args.exp_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.f = open(self.path + '/run.log', 'a')
        self.args = args

    def cprint(self, text):
        datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        to_print = "%s: %s" % (datetime_string, text)
        print(to_print)
        self.f.write(to_print + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def save_model(self, model, epoch, mode='mean'):
        path = self.path + '/' + mode + '_' + self.args.model + '.pt'
        best_model = copy.deepcopy(model)

        state = {
            'epoch': epoch+1,
            'model': best_model.state_dict()
        }
        torch.save(state, path)

        return best_model

    def print_loss(self, partition, epoch, print_losses):
        outstr = "%s set, epoch %d" % (partition, epoch)

        for loss, loss_val in print_losses.items():
            outstr += ", %s loss: %.4f" % (loss, loss_val)

        self.cprint(outstr)

    def print_result(self, partition, epoch, print_results):
        outstr = "%s set, epoch %d" % (partition, epoch)

        for result, result_val in print_results.items():
            outstr += ", %s: %.4f" % (result, result_val)

        self.cprint(outstr)


    # def save_conf_mat(self, conf_matrix, fname, domain_set):
    #     df = pd.DataFrame(conf_matrix, columns=list(label_to_idx.keys()), index=list(label_to_idx.keys()))
    #     fname = domain_set + "_" + fname
    #     df.to_csv(self.path + "/" + fname)

    # def print_progress(self, branch, domain_set, partition, epoch, print_losses, true=None, pred=None):
    #     outstr = "branch %s - %s - %s %d" % (branch, partition, domain_set, epoch)
    #     acc = 0
    #     if true is not None and pred is not None:
    #         acc = metrics.accuracy_score(true, pred)
    #         avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
    #         outstr += ", acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc)

    #     for loss, loss_val in print_losses.items():
    #         outstr += ", %s loss: %.4f" % (loss, loss_val)
    #     self.cprint(outstr)
    #     return acc
