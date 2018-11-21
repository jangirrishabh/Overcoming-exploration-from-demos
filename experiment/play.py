import click
import numpy as np
import pickle
import sys
import json
import os
sys.path.append('/home/rjangir/software/workSpace/Overcoming-exploration-from-demos/')

from baselines import logger
from baselines.common import set_global_seeds
import config
from rollout import RolloutWorker, RolloutWorkerOriginal, RolloutWorkerSofa

sofaEnv_path = '/home/rjangir/software/sofa/build/v17.12/bin/'

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

@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=100)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    evaluator_service = Service("action_reaction_play")
    evaluator_service.start()
    dims = config.configure_dims(params, evaluator_service)

    
    
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    

    evaluator = RolloutWorkerSofa(evaluator_service, policy, dims, logger, **eval_params)
    #evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
