import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "LunarLander-v2"

env = gym.make(env_name)


episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        score+=reward
        
    print(f"Episode : {episode} Score : {score}")
env.close()


env = DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=200000)

model.save("final_ppo_model_lunar_lander")

m2 = PPO.load("F:/sector16/Reinforcement Learning/Lunar_lander/final_ppo_model_lunar_lander.zip")

evaluate_policy(m2, env, n_eval_episodes=10, render=True)
env.close()
