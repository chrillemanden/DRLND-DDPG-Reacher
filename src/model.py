import numpy as np

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, seed, hidden_layers = [100, 100], use_batch_norm = False):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			hidden_layers (int[]): list of the sizes of the hidden layers
			use_batch_norm (bool): to determine whether to use batch normalisation in between layers
		"""
		super(Actor, self).__init__()
		# Setting the seed for the random generator in Pytorch
		self.seed = torch.manual_seed(seed)
		
		self.using_batch_norm = use_batch_norm
		
		# If using batch normalisation set up the model with batch normalisation between layers
		if self.using_batch_norm:
			# Add the input layer to a hidden layer
			self.hidden_layers = nn.ModuleList([nn.BatchNorm1d(state_size)]) #,nn.Linear(state_size, hidden_layers[0])])
			layers = [state_size] + hidden_layers
			# Add the remaining hidden layers
			for i in range(len(layers)-1): 
				self.hidden_layers.extend([nn.Linear(layers[i], layers[i+1])])
				self.hidden_layers.extend([nn.BatchNorm1d(layers[i+1])])
				
		# Else set up a normal model with only linear layers        
		else:
			self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
			layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
			self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
			
		# Add the output layer
		self.output = nn.Linear(hidden_layers[-1], action_size)
		

	def forward(self, state):
		# Forward propagation
		x = state
		if self.using_batch_norm:
			x = self.hidden_layers[0](x) # Run through first batch norm layer
			for i in range(1, len(self.hidden_layers)-1, 2): # Pass through hidden layers
				x = F.relu(self.hidden_layers[i+1](self.hidden_layers[i](x)))
		else:
			for linear in self.hidden_layers:
				x = F.relu(linear(x))

		return F.tanh(self.output(x)) # Return the output of the output layer


class Critic(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, seed, hidden_layers = [100, 50]):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			hidden_layers (int[]): list of the sizes of the hidden layers
		"""
		
		super(Critic, self).__init__()
		
		# Must have at least two hidden layers
		if len(hidden_layers) < 2:
			hidden_layers = [100, 50]
		
		# Initialise the random seed
		self.seed = torch.manual_seed(seed)
		
		# Add first linear layer as well as the one following with the action_size added to the input so that can be concatenated in the forward function
		self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0]), nn.Linear(hidden_layers[0] + action_size, hidden_layers[1])])
		layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
		self.hidden_layers.extend([nn.Linear(h1,h2) for h1, h2 in layer_sizes])
		
		# Add the output layer
		self.output = nn.Linear(hidden_layers[-1], 1)


	def forward(self, state, action):
		# Forward propagation
		# Perform the concatenation with the action tensor in the second layer
		x = F.relu(self.hidden_layers[0](state))
		x = torch.cat((x, action), dim=1)
		
		# Iterate through the remaining layers
		for i in range(1, len(self.hidden_layers)):
			x = F.relu(self.hidden_layers[i](x))
		
		return self.output(x)
		