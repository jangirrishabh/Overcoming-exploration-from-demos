import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 100))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    
    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)
    
        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--smooth', type=int, default=1)
args = parser.parse_args()

# Load all data.
data = {}
args.dir = '/home/rjangir/policies/naya/' #parsing arguments not working well
paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]
for curr_path in paths:
    if not os.path.isdir(curr_path):
        print("Not os path is current path")
        continue
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f) #load the parameters from the param file

    success_rate = np.array(results['test/success_rate'])
    mean_Q = np.array(results['test/mean_Q'])
    epoch = np.array(results['epoch']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    if replay_strategy == 'future':
        config = 'her'
    else:
        config = 'ddpg'
    if 'Dense' in env_id:
        config += '-dense'
    else:
        config += '-sparse'
    env_id = env_id.replace('Dense', '')

    if params['bc_loss']:
        config+='-demo'

    # Process and smooth data.
    assert success_rate.shape == epoch.shape
    x = epoch
    y = success_rate
    if args.smooth:
        x, y = smooth_reward_curve(epoch, success_rate) #smoothen the data
    assert x.shape == y.shape

    assert mean_Q.shape == epoch.shape
    z = mean_Q
    if args.smooth:
        x, z = smooth_reward_curve(epoch, mean_Q) #smoothen the data
    assert x.shape == z.shape

    if env_id not in data:
        data[env_id] = {}
    if config not in data[env_id]:
        data[env_id][config] = []
    data[env_id][config].append((x, y, z))

# Plot data.
for env_id in sorted(data.keys()): # for all the env ids
    print('exporting {}'.format(env_id))
    plt.clf()

    for config in sorted(data[env_id].keys()): #for all the configs in env ids
        xs, ys, zs = zip(*data[env_id][config])
        xs, ys, zs = pad(xs), pad(ys), pad(zs)
        assert xs.shape == ys.shape
        assert xs.shape == zs.shape

        
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
        plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
        plt.xlabel('Epoch')
        plt.ylabel('Median Success Rate')
        plt.title(env_id)
        plt.legend()
        
        #plt.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id + '-Qvalues')))
        

        plt.subplot(212)
        plt.plot(xs[0], np.nanmedian(zs, axis=0), label=config)
        plt.fill_between(xs[0], np.nanpercentile(zs, 25, axis=0), np.nanpercentile(zs, 75, axis=0), alpha=0.25)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Q value')
        plt.legend()

        plt.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.35)

        plt.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id+ '-success_rate')))

        #plt.figure(1)
        
        

        #plt.figure(2)
        


    
    


    
