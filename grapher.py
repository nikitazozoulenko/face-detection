import matplotlib.pyplot as plt
# plt.switch_backend("agg")
import numpy as np

class Grapher():
    def __init__(self):
        pass
    

    def textlog2numpy(self, path = "train_losses.txt"):
        values = []
        indices = []
        with open("savedir/logs/"+path, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                line = line.split(",")
                values += [float(line[0])]
                indices += [int(line[1])]
        return values, indices


    def values2ewma(self, losses, alpha = 0.9):
        losses_ewma = []
        ewma = losses[0]
        for loss in losses:
            ewma = alpha*ewma + (1-alpha)*loss
            losses_ewma += [ewma]
        return losses_ewma

    
    def graph(self, list_values, list_indices, list_colors, legendlabels, ylabel, xlabel, list_ewmas = None):
        if list_ewmas != None:
            for values, alpha in zip(list_values, list_ewmas):
                values[:] = self.values2ewma(values, alpha=alpha)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        for values, indices, color, legenlabel in zip(list_values, list_indices, list_colors, legendlabels):
            plt.plot(indices, values, color, label=legenlabel)
        plt.legend(loc=1)
        plt.show()
        #plt.savefig(legendlabels[0]+".png")


if __name__ == "__main__":
    grapher = Grapher()

    t_tot, t_tot_ind = grapher.textlog2numpy("train_total_losses.txt")
    t_cls, t_cls_ind = grapher.textlog2numpy("train_class_losses.txt")
    t_coord, t_coord_ind = grapher.textlog2numpy("train_coord_losses.txt")
    v_tot, v_tot_ind = grapher.textlog2numpy("val_total_losses.txt")
    v_cls, v_cls_ind = grapher.textlog2numpy("val_class_losses.txt")
    v_coord, v_coord_ind = grapher.textlog2numpy("val_coord_losses.txt")


    grapher.graph([t_tot, t_cls, t_coord],
                  [t_tot_ind, t_cls_ind, t_coord_ind],
                  ["b", "g", "r"],
                  ["train total", "train class", "train coord"],
                  "Loss", "Iterations",
                  list_ewmas = [0.9, 0.9, 0.9])
    
    grapher.graph([v_tot, v_cls, v_coord],
                  [v_tot_ind, v_cls_ind, v_coord_ind],
                  ["b", "g", "r"],
                  ["val total", "val class", "val coord"],
                  "Loss", "Iterations",
                  list_ewmas = [0.9, 0.9, 0.9])
    
    grapher.graph([t_tot, v_tot],
                  [t_tot_ind, v_tot_ind],
                  ["b", "g"],
                  ["train total", "val total"],
                  "Loss", "Iterations",
                  list_ewmas = [0.9, 0.9])

    grapher.graph([t_cls, v_cls],
                  [t_cls_ind, v_cls_ind],
                  ["b", "g"],
                  ["train class", "val class"],
                  "Loss", "Iterations",
                  list_ewmas = [0.9, 0.9])
    
    grapher.graph([t_coord, v_coord],
                  [t_coord_ind, v_coord_ind],
                  ["b", "g"],
                  ["train coord", "val coord"],
                  "Loss", "Iterations",
                  list_ewmas = [0.9, 0.9])
