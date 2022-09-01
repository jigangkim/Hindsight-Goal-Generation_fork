import numpy as np
import time
from common import get_args,experiment_setup

if __name__=='__main__':
	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)

	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# HGG info
	hgg_info_keys = [
		'sampler/InitialDist/min',
		'sampler/InitialDist/mean',
		'sampler/InitialDist/max',
		'sampler/DesiredDist/min',
		'sampler/DesiredDist/mean',
		'sampler/DesiredDist/max',
		'sampler/Value/min',
		'sampler/Value/mean',
		'sampler/Value/max',
		'sampler/CandidateGoals/InitialDist/min',
		'sampler/CandidateGoals/InitialDist/mean',
		'sampler/CandidateGoals/InitialDist/max',
		'sampler/CandidateGoals/DesiredDist/min',
		'sampler/CandidateGoals/DesiredDist/mean',
		'sampler/CandidateGoals/DesiredDist/max',
		'sampler/CandidateGoals/Value/min',
		'sampler/CandidateGoals/Value/mean',
		'sampler/CandidateGoals/Value/max',
	]
	for key in hgg_info_keys:
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()

	for epoch in range(args.epochs):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()

			learner.learn(args, env, env_test, agent, buffer)
			tester.cycle_summary()

			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

		tester.epoch_summary()

	tester.final_summary()
