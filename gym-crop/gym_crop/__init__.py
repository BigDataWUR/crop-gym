from gym.envs.registration import register

register(
    id='irrigation-v0',
    entry_point='gym_crop.envs:IrrigationEnv',
)
