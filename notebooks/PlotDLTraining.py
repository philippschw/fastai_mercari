from fastai.sgdr import Callback
import matplotlib.pyplot as plt

class PlotDLTraining(Callback):
    def __init__(self, model):
        self.train_loss = []
        self.val_loss = []
        self.rmse = []
        self.m = model
        self.n_epochs = 0

    def on_epoch_end(self, metrics):
        self.n_epochs += 1
        self.train_loss.append(self.m.sched.losses[-1])
        self.val_loss.append(metrics[0][0])
        self.rmse.append(metrics[1])
        
    def plot(self):
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        ax1, ax2 = ax

        ax1.plot(list(range(self.n_epochs)), self.train_loss, label='Training Loss')
        ax1.plot(list(range(self.n_epochs)), self.val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.set_title('Training and Validation Loss Vs Epochs')

        ax2.plot(list(range(self.n_epochs)), self.rmse)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSLE')
        ax2.set_title('Root Mean Square Log Error Vs Epochs')              