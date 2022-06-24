from ..utils import basic_transforms
from ..lambda_wrappers import observation_lambda_v0


def basic_obs_wrapper(env, module, param):
    def change_space(space):
        module.check_param(space, param)
        space = module.change_obs_space(space, param)
        return space

    def change_obs(obs, obs_space):
        return module.change_observation(obs, obs_space, param)
    return observation_lambda_v0(env, change_obs, change_space)


def flatten_v0(env):
    return basic_obs_wrapper(env, basic_transforms.flatten, True)
