import os
import sys
import time
import click
import numpy as np
import json
from mpi4py import MPI
import resource

sys.path.append('/home/rjangir/software/workSpace/Overcoming-exploration-from-demos/')

sofaEnv_path = '/home/rjangir/software/sofa/build/v17.12/bin/'

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import config
from rollout import RolloutWorker, RolloutWorkerOriginal, RolloutWorkerSofa
from util import mpi_fork

from subprocess import CalledProcessError


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]



class Service:
    def __init__(self, *args):
        self._args = args
        self._service_name = sofaEnv_path + os.path.splitext(os.path.basename(args[0]))[0]
        self._request_file_name = self._service_name + "_request"
        self._response_file_name = self._service_name + "_response"
        self._request_file = None
        self._response_file = None

    def __call__(self, action):
        json.dump(action, self._request_file)
        self._request_file.write("\n")
        self._request_file.flush()
        return json.loads(self._response_file.readline())
        #return 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def start(self):
        if not os.path.exists(self._request_file_name):
            os.mkfifo(self._request_file_name)
        if not os.path.exists(self._response_file_name):
            os.mkfifo(self._response_file_name)
        self._request_file = open(self._request_file_name, "w")
        self._response_file = open(self._response_file_name, "r")

    def end(self):
        if self._request_file is not None:
            self._request_file.close()
        if self._response_file is not None:
            self._response_file.close()
        if os.path.exists(self._request_file_name):
            os.remove(self._request_file_name)
        if os.path.exists(self._response_file_name):
            os.remove(self._response_file_name)

def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    best_success_epoch = 0

    if policy.bc_loss == 1: policy.initDemoBuffer(demo_file) #initializwe demo buffer
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            
            #episode = rollout_worker.generate_rollouts_from_demo() # uncomment to train with resets
            
            episode = rollout_worker.generate_rollouts() # uncomment to train without resets
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        logger.info("Testing")
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        

        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()


        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        #print ("TRAINIGN SUCCESS RATES ", evaluator.current_success_rate(), success_rate)
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            best_success_epoch = epoch
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        logger.info("Best success rate so far ", best_success_rate, " In epoch number ", best_success_epoch)
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, demo_file,
    override_params={}, save_policies=True
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    # Prepare params.
    

    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()



    rollout_worker_service = Service("action_reaction")
    evaluator_service = Service("action_reaction_evaluator")

    rollout_worker_service.start()
    evaluator_service.start()

    dims = config.configure_dims(params, rollout_worker_service)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'render': 0,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        #'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'render': 0,
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    

    rollout_worker = RolloutWorkerSofa(rollout_worker_service, policy, dims, logger, **rollout_params)
    #rollout_worker.seed(rank_seed)

    evaluator = RolloutWorkerSofa(evaluator_service, policy, dims, logger, **eval_params)
    #evaluator.seed(rank_seed)


    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, demo_file = demo_file)


@click.command()
#@click.option('--env', type=str, default='FetchPickAndPlace-v0', help='the name of the OpenAI Gym environment that you want to train on')
#@click.option('--env', type=str, default='GazeboWAMemptyEnv-v2', help='the name of the OpenAI Gym environment that you want to train on')
#@click.option('--env', type=str, default='GazeboWAMemptyEnv-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--env', type=str, default='SofaEnv-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='/home/rjangir/results/sofa/data_sofa_vertex_to_point', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=10000, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = '/home/rjangir/sofaDemoData/data_sofa_vertex_to_point_20.npz', help='demo data file path')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
