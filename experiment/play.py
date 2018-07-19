import click
import numpy as np
import pickle
import sys

sys.path.append('/home/rjangir/software/workSpace/Overcoming-exploration-from-demos/')

from baselines import logger
from baselines.common import set_global_seeds
import config
from rollout import RolloutWorker, RolloutWorkerOriginal


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

    dims = config.configure_dims(params)

    if params['env_name'] == 'GazeboWAMemptyEnv-v1':
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            #'render': bool(render),
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        madeEnv = config.cached_make_env(params['make_env'])
        evaluator = RolloutWorker(madeEnv, params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)
    else:
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            'render': bool(render),
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        evaluator = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)

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
