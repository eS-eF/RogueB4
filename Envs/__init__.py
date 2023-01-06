from gym.envs.registration import register
 
register(
    id='myrogue-v0',
    entry_point='Envs.myenv:RogueEnv',
)