import numpy as np
from matplotlib.pyplot import plot, draw, show, figure, subplot, ion, pause, ylabel, xlabel



dataFile = '/home/rjangir/fetchDemoData/plots/plotting_data_FetchPickAndPlace-v0_without_L2_Loss.npz'
plotData = np.load(dataFile)



#ion()
#show()

figure(1)
subplot(211)
ylabel('Critic Loss')
xlabel('Epoch')
plot(plotData['epoch'], plotData['critic_loss'], 'k')

subplot(212)
ylabel('Actor Loss')
xlabel('Epoch')
plot(plotData['epoch'], plotData['actor_loss'], 'r--')


figure(2)
subplot(211)
ylabel('Cloning Loss')
xlabel('Epoch')
plot(plotData['epoch'], plotData['cloning_loss'], 'r--')

subplot(212)
ylabel('mean Q value')
xlabel('Epoch')
plot(plotData['epoch'], plotData['q_value'], 'r--')

show()