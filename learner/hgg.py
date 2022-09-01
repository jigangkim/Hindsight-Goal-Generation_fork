import copy
import numpy as np
from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int

class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold
		self.goal_distance = get_goal_distance(args)

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx): # get HGG goal
		if self.args.env[:5]=='Fetch':
			return self.add_noise(self.pool[idx])
		else:
			return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad() # (\phi(s_t)t=1:T, s_0) for pool_length(=1000) trajectories
		candidate_goals = []
		candidate_edges = []
		candidate_id = []
		candidate_initial_dist = []
		candidate_desired_dist = []
		candidate_value = []

		agent = self.args.agent
		get_obs = lambda x, y: np.concatenate([x, y], axis=-1)
		obs = np.concatenate([get_obs(np.repeat(np.atleast_2d(s0),len(traj),axis=0), traj) for s0, traj in zip(achieved_pool_init_state, achieved_pool)], axis=0)
		feed_dict = {
			agent.raw_obs_ph: obs
		}
		value = agent.sess.run(agent.q_pi, feed_dict)[:,0] # V(s_0; \phi(s_t))
		value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
		achieved_value = np.split(np.squeeze(value), np.cumsum([len(traj) for traj in achieved_pool])[:-1], axis=0)

		n = 0 # node number
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)): # \phi(s_t)t=1:T for pool_length(=1000) trajectories
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)): # g* K(=50)
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)): # pool_length(=1000)
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)): # pool_length(=1000)
			initial_dists, desired_dists, values = [], [], []
			for j in range(len(desired_goals)): # K(=50)
				res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma)) # l2_norm(g* - \phi(s_t)) - 1/L*V(s_0; \phi(s_t))
				match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c # min_t{res} + c*l2_norm(\phi(s_0) - \phi(s*_0))
				match_idx = np.argmin(res)
				
				initial_dist = self.goal_distance(achieved_pool[i][0], initial_goals[j])/self.max_dis
				desired_dist = (np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1))/self.max_dis)[match_idx]
				value = (achieved_value[i]*(1-self.args.gamma))[match_idx]
				initial_dists.append(initial_dist)
				desired_dists.append(desired_dist)
				values.append(value)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx]) # \phi(s_tmin) -> g_cand
				candidate_edges.append(edge)
				candidate_id.append(j)
			candidate_initial_dist.append(initial_dists)
			candidate_desired_dist.append(desired_dists)
			candidate_value.append(values)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length

		explore_goals = [None]*self.length
		explore_goals_info = {
			'initial_dist': [None]*self.length,
			'desired_dist': [None]*self.length,
			'value': [None]*self.length
		}
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
				explore_goals_info['initial_dist'][candidate_id[i]] = np.array(candidate_initial_dist).flatten()[i]
				explore_goals_info['desired_dist'][candidate_id[i]] = np.array(candidate_desired_dist).flatten()[i]
				explore_goals_info['value'][candidate_id[i]] = np.array(candidate_value).flatten()[i]
		if None in explore_goals_info['initial_dist']:
			raise RuntimeError('Goal mismatch!')
		self.pool = np.array(explore_goals)

		self.args.logger.add_record('sampler/InitialDist/min', np.min(candidate_initial_dist))
		self.args.logger.add_record('sampler/InitialDist/mean', np.mean(candidate_initial_dist))
		self.args.logger.add_record('sampler/InitialDist/max', np.max(candidate_initial_dist))
		self.args.logger.add_record('sampler/DesiredDist/min', np.min(candidate_desired_dist))
		self.args.logger.add_record('sampler/DesiredDist/mean', np.mean(candidate_desired_dist))
		self.args.logger.add_record('sampler/DesiredDist/max', np.max(candidate_desired_dist))
		self.args.logger.add_record('sampler/Value/min', np.min(candidate_value))
		self.args.logger.add_record('sampler/Value/mean', np.mean(candidate_value))
		self.args.logger.add_record('sampler/Value/max', np.max(candidate_value))
		self.args.logger.add_record('sampler/CandidateGoals/InitialDist/min', np.min(explore_goals_info['initial_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/InitialDist/mean', np.mean(explore_goals_info['initial_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/InitialDist/max', np.max(explore_goals_info['initial_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/DesiredDist/min', np.min(explore_goals_info['desired_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/DesiredDist/mean', np.mean(explore_goals_info['desired_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/DesiredDist/max', np.max(explore_goals_info['desired_dist']))
		self.args.logger.add_record('sampler/CandidateGoals/Value/min', np.min(explore_goals_info['value']))
		self.args.logger.add_record('sampler/CandidateGoals/Value/mean', np.mean(explore_goals_info['value']))
		self.args.logger.add_record('sampler/CandidateGoals/Value/max', np.max(explore_goals_info['value']))

class HGGLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.goal_distance = get_goal_distance(args)

		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

	def learn(self, args, env, env_test, agent, buffer):
		initial_goals = []
		desired_goals = []
		for i in range(args.episodes): # sample from target distribution (K=50)
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy()) # \phi(s*_0)
			desired_goals.append(goal_d.copy()) # g*

		self.sampler.update(initial_goals, desired_goals) # weighted bipartite matching

		achieved_trajectories = []
		achieved_init_states = []
		for i in range(args.episodes): # for i = 1, 2, ..., K(=50)
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()
			goal_idx = i
			# goal_idx = np.random.randint(0, args.episodes)
			explore_goal = self.sampler.sample(goal_idx) # construct intermediate task distribution
			self.env_List[i].goal = explore_goal.copy() # critical step: hindsight goal-oriented exploration
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			for timestep in range(args.timesteps): # for t = 0, 1, ..., H-1(=49)
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			achieved_trajectories.append(np.array(trajectory))
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches): # for i = 1, 2, ..., M(=20)
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes): # reject trajectories with dist(\phi(s_0),\phi(s_T)) ~= 0
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())
		# for s0_T, s0 in zip(achieved_trajectories, achieved_init_states):
		# 	self.achieved_trajectory_pool.insert(s0_T.copy(), s0.copy())
