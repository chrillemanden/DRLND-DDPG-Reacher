import random
import numpy as np
from collections import namedtuple, deque

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Implementation imports
from model import Actor, Critic 
from utils import OUNoise, ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4 		# 
ACTOR_LR = 5e-3			# actor learning rate
CRITIC_LR = 5e-4		# critic learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def copy_weights(source_network, target_network):
	"""Copy source network weights to target"""
	for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
		target_param.data.copy_(source_param.data)


class Agent():
	#''' DDPG agent '''
	def __init__(self, state_size, action_size, num_agents, seed, actor_hidden_layers, critic_hidden_layers, use_batch_norm=False, use_noise=False):
		super(Agent, self).__init__()
		
		self.state_size = state_size
		self.action_size = action_size
		
		self.random_seed = random.seed(seed)
		
		# Actor networks
		self.actor_local = Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device)
		self.actor_target = Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)
		copy_weights(self.actor_local, self.actor_target)
		
		# Critic networks
		self.critic_local = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
		self.critic_target = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)
		copy_weights(self.critic_local, self.critic_target)
		
		# Replay Memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		
		# Noise process
		self.noise = OUNoise((num_agents, action_size), seed)
		sefl.use_noise = use_noise

		self.t_step = 0

	def step(self, states, actions, rewards, next_states, dones):
		''' Save experience in replay memory, and use random sample from buffer to learn. '''
		for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
			memory.add(state, action, reward, next_state, done)

		# update time steps
		self.t_s = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# time to learn again
			# provided that there are enough
			#if len(shared_memory.shared_buffer) > BATCH_SIZE:
			if len(memory) > BATCH_SIZE:
				#experiences = self.memory.sample()
				#experiences = shared_memory.shared_buffer.sample()
				experiences = memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, state):
		''' Returns actions for a given state as per current policy '''
		
		# Make current state into a Tensor that can be passed as input to the network
		state = torch.from_numpy(state).float().to(device)

		# Set network in evaluation mode to prevent things like dropout from happening
		self.actor_local.eval()

		# Turn off the autograd engine
		with torch.no_grad():
			# Do a forward pass through the network
			action_values = self.actor_local(state).cpu().data.numpy()

		# Put network back into training mode
		self.actor_local.train()
		
		if self.use_noise:
			action_values += self.noise.sample()

		return np.clip(action_values, -1, 1)
		
	def reset(self):
		''' Reset the noise in the OU process '''
		self.noise.reset()


	def learn(self, experiences, gamma):
		''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) '''

		states, actions, rewards, next_states, dones = experiences

		# ------------------------ Update Critic Network ------------------------ #
		next_actions = self.actor_target(next_states)
		Q_targets_prime = self.critic_target(next_states, next_actions)

		# Compute y_i
		Q_targets = rewards + (gamma * Q_targets_prime * (1 - dones))

		# Compute the critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimise the loss
		self.critic_optimizer.zero_grad() # Reset the gradients to prevent accumulation
		critic_loss.backward()            # Compute gradients
		torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()      # Update weights

		# ------------------------ Update Actor Network ------------------------- #
		# Compute the actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()

		# Minimise the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


		# ------------------------ Update Target Networks ----------------------- #
		self.soft_update(critic_local, critic_target, TAU)
		self.soft_update(actor_local, actor_target, TAU)
		
	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)