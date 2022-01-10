import numpy as np
from pyibl import Agent

class IBLAgent_Equal(Agent):
	def __init__(self, world, m_noise, m_decay, default_utility, IBLtype = 'equal'):
		super(IBLAgent_Equal, self).__init__("My Agent", ["action", "state_y", "state_x"], noise=m_noise, decay=m_decay, default_utility=default_utility)
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 4 
		self.action_pool = np.array([[1,0], [-1,0], [0,-1], [0,1]])
		self.last_action = 0

		self.world = world
		self.select_position()
		self.options = {}
		self.outcome_goals = np.zeros(self.world.num_goal)

	def generate_options(self):
		self.options[(self.y,self.x)] = [{"action": a, "state_y":self.y, "state_x": self.x} for a in range(self.actions)]
		
	def select_action(self): 
		if (self.y,self.x) not in self.options:
			self.generate_options()
		action_selected = self.choose(*self.options[(self.y,self.x)])
		return action_selected["action"]
	
	def add_world(self,world):
		self.world = world 

	def add_outcome(self,outcome_goals):
		self.outcome_goals = outcome_goals

	def get_position(self):
		return self.y, self.x

	def get_last_action(self):
		return self.last_action

	def move(self,init_x,init_y,max_step):
		self.last_goal_consumed = None 
		self.trajectory = []  
		self.action_history = []
		is_done = False
		self.set_position(init_x,init_y)
        
		self.trajectory.append((init_x, init_y))
		preferred_goal = np.argmax(self.outcome_goals)
		step_preferred_goal = None

		delay = {}
		count_step = 0
		for step in range(0, max_step):
			count_step += 1
			action = self.select_action()
			self.last_action = action
			new_pos = np.array([self.y, self.x]) + self.action_pool[action]

			delay[(step)] = self.respond()
			delay[(step)].update(-0.01)

			if self.world.get_walls()[new_pos[0], new_pos[1]]==1:
				new_pos = np.array([self.y, self.x])
				self.last_action = 4
				delay[(step)].update(-0.05)

			if np.sum(self.world.get_goals(),axis=0)[new_pos[0], new_pos[1]]==1:
				is_done = True
				for g in range(self.world.num_goal):
					if self.world.get_goals()[g,new_pos[0], new_pos[1]] ==1:
						self.last_goal_consumed = g
						break
			
			# delay[(step)] = self.respond()
			self.y = new_pos[0]
			self.x = new_pos[1]

			self.trajectory.append((self.x, self.y))
			self.action_history.append(self.last_action)

			if is_done == True:
				break
		if self.last_goal_consumed == preferred_goal:
			step_preferred_goal = step
		if self.last_goal_consumed != None:
			for i in range(len(self.action_history)):
					delay[(i)].update(self.outcome_goals[self.last_goal_consumed])
		
		return self.last_goal_consumed, count_step, step_preferred_goal

	def select_position(self):
		i, j = np.where(self.world.get_walls()+np.sum(self.world.get_goals(),axis=0)==0)
		empty_cells = np.random.permutation(len(i))
		self.y = i[empty_cells[0]]
		self.x = j[empty_cells[0]]
	
	def set_position(self, x, y):
		self.x = x
		self.y = y


class IBLAgent_TD(Agent):
	def __init__(self, world, m_noise, m_decay, default_utility, m_gamma, m_lr, IBLtype = 'IBLTD'):
		super(IBLAgent_TD, self).__init__("My Agent", ["action", "state_y", "state_x"], noise=m_noise, decay=m_decay, default_utility=default_utility)
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 4 
		self.action_pool = np.array([[1,0], [-1,0], [0,-1], [0,1]])
		self.last_action = 0

		self.world = world
		self.select_position()
		self.options = {}
		self.inst_history = {}
		# initial outcome for goals
		self.outcome_goals = np.zeros(self.world.num_goal)
		# self.gamma = 0.99
		self.gamma = m_gamma
		self.alpha = m_lr
		# self.alpha = 0.75
		# self.alpha = 0.1

	def generate_options(self):
		self.options[(self.y,self.x)] = [{"action": a, "state_y":self.y, "state_x": self.x} for a in range(self.actions)]
	
	def select_action(self): 
		if (self.y,self.x) not in self.options:
			self.generate_options()
		action_selected = self.choose(*self.options[(self.y,self.x)])
		return action_selected["action"]
	
	def add_world(self,world):
		self.world = world 

	def add_outcome(self,outcome_goals):
		self.outcome_goals = outcome_goals

	def get_position(self):
		return self.y, self.x

	def get_last_action(self):
		return self.last_action

	def move(self,init_x,init_y,max_step):
		self.last_goal_consumed = None 
		self.trajectory = []  
		self.action_history = []
		is_done = False
		self.set_position(init_x,init_y)
        
		self.trajectory.append((init_x, init_y))
		preferred_goal = np.argmax(self.outcome_goals)
		step_preferred_goal = None

		self.blended = []
		count_step = 0
		for step in range(1, max_step+1): 
			count_step += 1
			action = self.select_action()
			self.last_action = action
			if (self.y, self.x, action) not in self.inst_history:
				self.inst_history[(self.y, self.x, action)] = []
			
			query = self.options[(self.y,self.x)][action]
			blended_tmp = self._memory.blend("_utility", **query)

			self.inst_history[(self.y, self.x, action)].append(self.respond())

			new_pos = np.array([self.y, self.x]) + self.action_pool[action]

			if self.world.get_walls()[new_pos[0], new_pos[1]]==1:
				new_pos = np.array([self.y, self.x])
				self.last_action = 4
				self.inst_history[(self.y, self.x, action)][-1].update(-0.05)
			elif np.sum(self.world.get_goals(),axis=0)[new_pos[0], new_pos[1]]==1:
				is_done = True
				for g in range(self.world.num_goal):
					if self.world.get_goals()[g,new_pos[0], new_pos[1]] ==1:
						self.last_goal_consumed = g
						break
				outcomes =  blended_tmp + self.alpha*(self.outcome_goals[self.last_goal_consumed] - blended_tmp)
				## 
				for j in range(len(self.inst_history[(self.y, self.x, action)])):
					self.inst_history[(self.y, self.x, action)][j].update(outcomes)
				##
				# self.inst_history[(self.y, self.x, action)][-1].update(outcomes)
			else:
				if (new_pos[0],new_pos[1]) not in self.options:
					blended_max = [self.default_utility]
				else:
					blended_max = [self._memory.blend("_utility", **self.options[(new_pos[0],new_pos[1])][a]) for a in range(self.actions)]
				outcomes =  blended_tmp + self.alpha*(-0.01 + self.gamma*max(blended_max) - blended_tmp)
				for j in range(len(self.inst_history[(self.y, self.x, action)])):
					self.inst_history[(self.y, self.x, action)][j].update(outcomes)
			
			self.y = new_pos[0]
			self.x = new_pos[1]

			self.trajectory.append((self.x, self.y))
			self.action_history.append(self.last_action)

			if is_done == True:
				break
		if self.last_goal_consumed == preferred_goal:
			step_preferred_goal = step

		return self.last_goal_consumed, count_step, step_preferred_goal

	def select_position(self):
		i, j = np.where(self.world.get_walls()+np.sum(self.world.get_goals(),axis=0)==0)
		empty_cells = np.random.permutation(len(i))
		self.y = i[empty_cells[0]]
		self.x = j[empty_cells[0]]
	
	def set_position(self, x, y):
		self.x = x
		self.y = y
	

class IBLAgent_Exp(Agent):
	def __init__(self, world, m_noise, m_decay, default_utility, m_gamma, IBLtype = 'exponential'):
		super(IBLAgent_Exp, self).__init__("My Agent", ["action", "state_y", "state_x"], noise = m_noise, decay=m_decay, default_utility=default_utility)
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 4 
		self.action_pool = np.array([[1,0], [-1,0], [0,-1], [0,1]])
		self.last_action = 0

		self.world = world
		self.select_position()
		self.options = {}
		self.outcome_goals = np.zeros(self.world.num_goal)
		self.gamma = m_gamma
		self.returns = None
		

	def generate_options(self):
		self.options[(self.y,self.x)] = [{"action": a, "state_y":self.y, "state_x": self.x} for a in range(self.actions)]
		
	def select_action(self): 
		if (self.y,self.x) not in self.options:
			self.generate_options()
		action_selected = self.choose(*self.options[(self.y,self.x)])
		return action_selected["action"]
	
	def add_world(self,world):
		self.world = world 

	def add_outcome(self,outcome_goals):
		self.outcome_goals = outcome_goals

	def get_position(self):
		return self.y, self.x

	def get_last_action(self):
		return self.last_action

	def move(self,init_x,init_y,max_step):
		self.last_goal_consumed = None 
		self.trajectory = []  
		self.action_history = []
		is_done = False
		self.set_position(init_x,init_y)
        
		self.trajectory.append((init_x, init_y))
		preferred_goal = np.argmax(self.outcome_goals)
		step_preferred_goal = None

		delay = {}
		count_step = 0
		for step in range(0, max_step):
			count_step += 1
			action = self.select_action()
			self.last_action = action
			new_pos = np.array([self.y, self.x]) + self.action_pool[action]
			delay[(step)] = self.respond()
			delay[(step)].update(-0.01)
			if self.world.get_walls()[new_pos[0], new_pos[1]]==1:
				new_pos = np.array([self.y, self.x])
				self.last_action = 4
				delay[(step)].update(-0.05)

			if np.sum(self.world.get_goals(),axis=0)[new_pos[0], new_pos[1]]==1:
				is_done = True
				for g in range(self.world.num_goal):
					if self.world.get_goals()[g,new_pos[0], new_pos[1]] ==1:
						self.last_goal_consumed = g
						break
			
			# delay[(step)] = self.respond()
			self.y = new_pos[0]
			self.x = new_pos[1]

			self.trajectory.append((self.x, self.y))
			self.action_history.append(self.last_action)

			if is_done == True:
				break

		if self.last_goal_consumed == preferred_goal:
			step_preferred_goal = step

		if self.last_goal_consumed != None:
			for i in range(len(self.action_history)):
				if self.action_history[i]!=4:
					delay[(i)].update(self.outcome_goals[self.last_goal_consumed]*self.gamma**(len(self.action_history)-i-1))
		
		return self.last_goal_consumed, count_step, step_preferred_goal

	def select_position(self):
		i, j = np.where(self.world.get_walls()+np.sum(self.world.get_goals(),axis=0)==0)
		empty_cells = np.random.permutation(len(i))
		self.y = i[empty_cells[0]]
		self.x = j[empty_cells[0]]
	
	def set_position(self, x, y):
		self.x = x
		self.y = y
