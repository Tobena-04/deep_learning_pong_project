import torch
model0 = torch.load('/Users/fardeenb/Documents/Projects/deep_learning_pong_project/rl_trainer/pong_policy_episode_0.pth')
model50 = torch.load('/Users/fardeenb/Documents/Projects/deep_learning_pong_project/rl_trainer/pong_policy_episode_50.pth')


print(f'0: {model0} ')
print(f'0: {model50}')