class Logger():
    def __init__(self, name = "train_losses.txt"):
        self.name = name
        self.file = open("savedir/logs/"+name, "w") 

    
    def write_log(self, loss, index):
        loss = loss.data.squeeze().cpu().numpy()[0]
        self.file.write(str(loss)+","+str(index)+"\n")


    def read_log(self, path):
        return path

    
    def kill(self):
        self.file.close()


if __name__ == "__main__":
    train_losses = Logger("train_losses.txt")
    import torch
    from torch.autograd import Variable
    loss = Variable(torch.Tensor([0.49535]))
    i = 11
    train_losses.write_log(loss, i)
    train_losses.write_log(loss+1, i+1)