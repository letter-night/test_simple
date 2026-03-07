# -*- coding: utf-8 -*-
"""
[Treatment Effects with RNNs] cancer_simulation
Created on 2/4/2018 8:14 AM

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

Notes:
- Simulation time taken to be in days

@author: limsi
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import truncnorm

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL:
# https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed


def calc_volume(diameter):
	return 4 / 3 * np.pi * (diameter / 2) ** 3


def calc_diameter(volume):
	return ((volume / (4 / 3 * np.pi)) ** (1 / 3)) * 2


# Tumour constants per
TUMOUR_CELL_DENSITY = 5.8 * 10 ** 8 # cells per cm^3
TUMOUR_DEATH_THRESHOLD = calc_volume(13) # assume spherical

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist
tumour_size_distributions = {'I': (1.72, 4.70, 0.3, 5.0),
                             'II': (1.96, 1.63, 0.3, 13.0),
                             'IIIA': (1.91, 9.40, 0.3, 13.0),
                             'IIIB': (2.76, 6.87, 0.3, 13.0),
                             'IV': (3.86, 8.82, 0.3, 13.0)}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {'I': 1432,
                             "II": 128,
                             "IIIA": 1306,
                             "IIIB": 7248,
                             "IV": 12840}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions


def generate_params(num_patients, radio_coeff, window_size, lag):
	"""
	Get original patient-specific simulation parameters, and add extra ones to control confounding
	
	:param num_patients: Number of patients to simulate
	:param radio_coeff: Bias on action policy for radiotherapy assignments
	:return: dict of parameters
	"""

	basic_params = get_standard_params(num_patients)
	patient_types = basic_params['patient_types']

	D_MAX = calc_diameter(TUMOUR_DEATH_THRESHOLD)
	basic_params['radio_sigmoid_intercepts'] = np.array([D_MAX / 2.0 for _ in patient_types])

	basic_params['radio_sigmoid_betas'] = np.array([radio_coeff / D_MAX for _ in patient_types])

	basic_params['window_size'] = window_size
	basic_params['lag'] = lag

	return basic_params


def get_standard_params(num_patients): # additional params
	"""
	Simulation parameters from the Nature article + adjustments for static variables
	
	:param num_patients: Number of patients to simulate
	:return: simulation_parameters: Initial volumes + Static variables (e.g. response to treatment); randomly shuffled
	"""

	# INITIAL VOLUMES SAMPLING
	TOTAL_OBS = sum(cancer_stage_observations.values())
	cancer_stage_proportions = {k: cancer_stage_observations[k] / TOTAL_OBS for k in cancer_stage_observations}

	# remove possible entries
	possible_stages = list(tumour_size_distributions.keys())
	possible_stages.sort()

	initial_stages = np.random.choice(possible_stages, num_patients, 
								   p=[cancer_stage_proportions[k] for k in possible_stages])
	
	# Get info on patient stages and initial volumes
	output_initial_diam = []
	patient_sim_stages = []
	for stg in possible_stages:
		count = np.sum((initial_stages == stg) * 1)

		mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

		# Convert lognorm bounds in to standard normal bounds
		lower_bound = (np.log(lower_bound) - mu) / sigma
		upper_bound = (np.log(upper_bound) - mu) / sigma

		logging.info(("Simulating initial volumes for stage {} " + 
				" with norm params: mu={}, sigma={}, lb={}, ub={}").format(
					stg, mu, sigma, lower_bound, upper_bound))
		
		norm_rvs = truncnorm.rvs(lower_bound, upper_bound, size=count)
		# truncated normal for realistic clinical outcome

		initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
		output_initial_diam += list(initial_volume_by_stage)
		patient_sim_stages += [stg for i in range(count)]
	
	# STATIC VARIABLES SAMPLING
	# Fixed params
	K = calc_volume(30) # carrying capacity given in cm, so convert to volume
	ALPHA_BETA_RATIO = 10
	ALPHA_RHO_CORR = 0.87

	# Distributional parameters for dynamics
	parameter_lower_bound = 0.0
	parameter_upper_bound = np.inf
	rho_params = (7 * 10 ** -5, 7.23 * 10 ** -3)
	alpha_params = (0.0398, 0.168)

	# Get correlated simulation parameters (alpha, beta, rho) which respects bounds
	alpha_rho_cov = np.array([[alpha_params[1] ** 2, ALPHA_RHO_CORR * alpha_params[1] * rho_params[1]],
                              [ALPHA_RHO_CORR * alpha_params[1] * rho_params[1], rho_params[1] ** 2]])
	
	alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

	simulated_params = []

	while len(simulated_params) < num_patients: # Keep on simulating till we get the right number of params

		param_holder = np.random.multivariate_normal(alpha_rho_mean, alpha_rho_cov, size=num_patients)

		for i in range(param_holder.shape[0]):

			# Ensure that all params fulfill conditions
			if param_holder[i, 0] > parameter_lower_bound and param_holder[i, 1] > parameter_lower_bound:
				simulated_params.append(param_holder[i, :])
		
		logging.info("Got correlated params for {} patients".format(len(simulated_params)))
	
	# Adjustments for static variables
	possible_patient_types = [1, 2, 3]
	patient_types = np.random.choice(possible_patient_types, num_patients)
	radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])

	simulated_params = np.array(simulated_params)[:num_patients, :] 
	alpha_adjustments = alpha_params[0] * radio_mean_adjustments
	alpha = simulated_params[:, 0] + alpha_adjustments
	rho = simulated_params[:, 1]
	beta = alpha / ALPHA_BETA_RATIO

	output_holder = {'patient_types':patient_types,
				  'initial_stages': np.array(patient_sim_stages),
				  'initial_volumes': calc_volume(np.array(output_initial_diam)),
				  'alpha':alpha,
				  'rho':rho,
				  'beta':beta,
				  'K':np.array([K for _ in range(num_patients)])}
	
	# Randomise output params
	logging.info("Randomising outputs")
	idx = [i for i in range(num_patients)]
	np.random.shuffle(idx)

	output_params = {}
	for k in output_holder:
		output_params[k] = output_holder[k][idx]
	
	return output_params


def simulate_factual(simulation_params, seq_length, assigned_actions=None):
	"""
	Simulation of factual patient trajectories (for train and valiation subset)
	
	:param simulation_params: Parameters of the simulation
	:param seq_length: Maximum trajectory length
	:param assigned_actions: Fixed non-random treatment assignment policy, if None - standard biased random assignment is applied
	:return: simulated data dict
	:"""

	total_num_radio_treatments = 1

	radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)]) # Gy

	# Unpack simulation parameters
	initial_stages = simulation_params['initial_stages']
	initial_volumes = simulation_params['initial_volumes']
	alphas = simulation_params['alpha']
	rhos = simulation_params['rho']
	betas = simulation_params['beta']
	Ks = simulation_params['K']
	patient_types = simulation_params['patient_types']
	window_size = simulation_params['window_size']
	lag = simulation_params['lag']

	# Coefficients for treatment assignment probabilities
	radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
	radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

	num_patients = initial_stages.shape[0]

	# Commence Simulation
	cancer_volume = np.zeros((num_patients, seq_length))
	radio_dosage = np.zeros((num_patients, seq_length))
	radio_application_point = np.zeros((num_patients, seq_length))
	sequence_lengths = np.zeros(num_patients)
	death_flags = np.zeros((num_patients, seq_length))
	recovery_flags = np.zeros((num_patients, seq_length))
	radio_probabilities = np.zeros((num_patients, seq_length))

	noise_terms = 0.01 * np.random.randn(num_patients, seq_length) # 5% cel variability
	recovery_rvs = np.random.rand(num_patients, seq_length)

	radio_application_rvs = np.random.rand(num_patients, seq_length)

	# Run actual simulation
	for i in tqdm(range(num_patients), total=num_patients):

		noise = noise_terms[i]

		# initial values
		cancer_volume[i, 0] = initial_volumes[i]
		alpha = alphas[i]
		beta = betas[i]
		rho = rhos[i]
		K = Ks[i]

		# Setup cell volume
		b_death = False
		b_recover = False
		for t in range(1, seq_length):

			cancer_volume[i, t] = cancer_volume[i, t - 1] * \
				(1 + rho * np.log(K / cancer_volume[i, t - 1]) - 
                    (alpha * radio_dosage[i, t - 1] + beta * radio_dosage[i, t - 1] ** 2) + noise[t])
			
			# Action probabilities + death or recovery simulations
			if t >= lag:
				cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag, 0)]
			else:
				cancer_volume_used = np.zeros((1, ))
			cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
			cancer_metric_used = cancer_diameter_used

			# probabilities
			if assigned_actions is not None:
				radio_prob = assigned_actions[i, t, 1]
			else:

				radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
			radio_probabilities[i, t] = radio_prob

			# Action application
			if radio_application_rvs[i, t] < radio_prob:
				radio_application_point[i, t] = 1
				radio_dosage[i, t] = radio_amt[0]
			
			if cancer_volume[i, t] > TUMOUR_DEATH_THRESHOLD:
				cancer_volume[i, t] = TUMOUR_DEATH_THRESHOLD
				b_death = True
				break # patient death

			# recovery threshold as defined by the previous stuff
			if recovery_rvs[i, t] < np.exp(-cancer_volume[i, t] * TUMOUR_CELL_DENSITY):
				cancer_volume[i, t] = 0
				b_recover = True
				break

		# Package outputs
		sequence_lengths[i] = int(t)
		death_flags[i, t] = 1 if b_death else 0
		recovery_flags[i, t] = 1 if b_recover else 0
	
	outputs = {'cancer_volume':cancer_volume, 
			'radio_dosage': radio_dosage, 
			'radio_application': radio_application_point, 
			'radio_probabilities': radio_probabilities,
			'sequence_lengths': sequence_lengths,
			'death_flags': death_flags, 
			'recovery_flags': recovery_flags,
			'patient_types': patient_types}
	
	return outputs


def simulate_counterfactual_1_step(simulation_params, seq_length):
	"""
	Simulation of test trajectories to assess all one-step ahead counterfactuals
	:param simulation_params: Parameters of the simulation
	:param seq_length: Maximum trajectory length (number of factual time-steps)
	:return: simulated data dict with number of rows equal to num_patients * seq_length * num_treatments
	"""

	total_num_radio_treatments = 1 

	num_treatments = 2 # No treatment / Radiotherapy

	radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)]) # Gy

	# Unpack simulation parameters
	initial_stages = simulation_params['initial_stages']
	initial_volumes = simulation_params['initial_volumes']
	alphas = simulation_params['alpha']
	rhos = simulation_params['rho']
	betas = simulation_params['beta']
	Ks = simulation_params['K']
	patient_types = simulation_params['patient_types']
	window_size = simulation_params['window_size'] # controls the lookback of the treatment assignment policy
	lag = simulation_params['lag']

	# Coefficient for treatment assignment probabilities
	radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
	radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

	num_patients = initial_stages.shape[0]

	num_test_points = num_patients * seq_length * num_treatments

	# Commence Simulation
	cancer_volume = np.zeros((num_test_points, seq_length))
	radio_application_point = np.zeros((num_test_points, seq_length))
	sequence_lengths = np.zeros(num_test_points)
	patient_types_all_trajectories = np.zeros(num_test_points)

	test_idx = 0

	# Run actual simulation
	for i in tqdm(range(num_patients), total=num_patients):

		noise = 0.01 * np.random.randn(seq_length) # 5% cell variability
		recovery_rvs = np.random.rand(seq_length)

		# initial values
		factual_cancer_volume = np.zeros(seq_length)
		factual_radio_dosage = np.zeros(seq_length)
		factual_radio_application_point = np.zeros(seq_length)
		factual_radio_probabilities = np.zeros(seq_length)

		radio_application_rvs = np.random.rand(seq_length)

		factual_cancer_volume[0] = initial_volumes[i]

		alpha = alphas[i]
		beta = betas[i]
		rho = rhos[i]
		K = Ks[i]

		for t in range(0, seq_length - 1):

			# Action probabilities + death or recovery simulations
			if t >= lag:
				cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
			else:
				cancer_volume_used = np.zeros((1, ))
			cancer_diameter_used = np.array(
				[calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
			cancer_metric_used = cancer_diameter_used

			# probabilities
			radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))

			factual_radio_probabilities[t] = radio_prob
			
			# Action application
			if radio_application_rvs[t] < radio_prob:
				factual_radio_application_point[t] = 1
				factual_radio_dosage[t] = radio_amt[0]
			
			# Factual prev_treatments andn outcomes
			factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
			(1 + rho * np.log(K / factual_cancer_volume[t]) - 
				(alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])
			
			factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

			# Populate arrays
			cancer_volume[test_idx] = factual_cancer_volume
			radio_application_point[test_idx] = factual_radio_application_point
			patient_types_all_trajectories[test_idx] = patient_types[i]
			sequence_lengths[test_idx] = int(t) + 1
			test_idx = test_idx + 1

			# Counterfactual prev_treatments and outcomes
			treatment_options = [0, 1]

			for treatment_option in treatment_options:
				if factual_radio_application_point[t] == treatment_option:
					# This represents the factual treatment which was already considered
					continue
				counterfactual_radio_dosage = 0.0
				counterfactual_radio_application_point = 0

				if treatment_option == 1:
					counterfactual_radio_application_point = 1
					counterfactual_radio_dosage = radio_amt[0]
				
				counterfactual_cancer_volume = factual_cancer_volume[t] * \
				(1 + rho * np.log(K / factual_cancer_volume[t]) - (alpha * counterfactual_radio_dosage + beta * counterfactual_radio_dosage ** 2) + noise[t + 1])
				
				cancer_volume[test_idx][:t + 2] = np.append(factual_cancer_volume[:t + 1], 
												[counterfactual_cancer_volume])
				radio_application_point[test_idx][:t + 1] = np.append(factual_radio_application_point[:t], 
														  [counterfactual_radio_application_point])
				patient_types_all_trajectories[test_idx] = patient_types[i]
				sequence_lengths[test_idx] = int(t) + 1
				test_idx = test_idx + 1
			
			if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
				recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
				break
	
	outputs = {'cancer_volume': cancer_volume[:test_idx],
			'radio_application': radio_application_point[:test_idx],
			'sequence_lengths': sequence_lengths[:test_idx],
			'patient_types': patient_types_all_trajectories[:test_idx]}
	
	print('Call to simulate counterfactuals data')

	return outputs


def simulate_counterfactuals_treatment_seq(simulation_params, seq_length, projection_horizon, cf_seq_mode='sliding_treatment',
										   cf_treatment_sequence=None):
	"""
	Simulation of test trajectories to assess a subset of multiple-step ahead counterfactuals
	:param simulation_params: Parameters of the simulation
	:param seq_length: Maximum trajectory length (number of factual time-steps)
	:param cf_seq_mode: Counterfactual sequence setting: sliding_treatment / random_trajectories
	:return: simulated data dict with number of rows equal to num_patients * seq_length * 2 * projection_horizon
	"""

	if cf_seq_mode == 'sliding_treatment':
		radio_arr = np.stack([np.zeros((projection_horizon, projection_horizon), dtype=int),
                              np.eye(projection_horizon, dtype=int)], axis=-1)
		treatment_options = radio_arr
	elif cf_seq_mode == 'random_trajectories':
		treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 1))
	elif cf_seq_mode == 'fixed_treatment':
		treatment_options = []
		for i in range(projection_horizon):
			if cf_treatment_sequence[i][1] == [1.0]:
				treatment_options.append([1])
			else:
				treatment_options.append([0])
		treatment_options = np.array([treatment_options])
	else:
		raise NotImplementedError()
	
	total_num_radio_treatments = 1

	radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)]) # Gy

	# Unpack simulation parameters
	initial_stages = simulation_params['initial_stages']
	initial_volumes = simulation_params['initial_volumes']
	alphas = simulation_params['alpha']
	rhos = simulation_params['rho']
	betas = simulation_params['beta']
	Ks = simulation_params['K']
	patient_types = simulation_params['patient_types']
	window_size = simulation_params['window_size'] # controls the lookback of the treatment assignmend policy
	lag = simulation_params['lag']

	# Coefficients for treatment assignment probabilities
	radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
	radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

	num_patients = initial_stages.shape[0]

	num_test_points = len(treatment_options) * num_patients * seq_length

	# Commence Simulation
	cancer_volume = np.zeros((num_test_points, seq_length + projection_horizon))
	radio_application_point = np.zeros((num_test_points, seq_length + projection_horizon))
	sequence_lengths = np.zeros(num_test_points)
	patient_types_all_trajectories = np.zeros(num_test_points)
	patient_ids_all_trajectories = np.zeros(num_test_points)
	patient_current_t = np.zeros(num_test_points)

	test_idx = 0

	# Run actual simulation
	for i in tqdm(range(num_patients), total=num_patients):

		noise = 0.01 * np.random.randn(seq_length + projection_horizon) # 5% cell variability
		recovery_rvs = np.random.rand(seq_length)

		# initial values
		factual_cancer_volume = np.zeros(seq_length)
		factual_radio_dosage = np.zeros(seq_length)
		factual_radio_application_point = np.zeros(seq_length)
		factual_radio_probabilities = np.zeros(seq_length)

		radio_application_rvs = np.random.rand(seq_length)

		factual_cancer_volume[0] = initial_volumes[i]

		alpha = alphas[i]
		beta = betas[i]
		rho = rhos[i]
		K = Ks[i]

		for t in range(0, seq_length - 1):

			# Action probabilities + death or recovery simulations
			if t >= lag:
				cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
			else:
				cancer_volume_used = np.zeros((1,))
			cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
			cancer_metric_used = cancer_diameter_used

			# probabilities
			radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))

			factual_radio_probabilities[t] = radio_prob

			# Action application
			if radio_application_rvs[t] < radio_prob:
				factual_radio_application_point[t] = 1
				factual_radio_dosage[t] = radio_amt[0]
			
			# Factual prev_treatments and outcomes
			factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
                (1 + rho * np.log(K / factual_cancer_volume[t]) - 
                    (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])
			
			factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

			if cf_seq_mode == 'random_trajectories':
				treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 1))
			
			for treatment_option in treatment_options:

				counterfactual_cancer_volume = np.zeros(shape=(t + 1 + projection_horizon + 1))
				counterfactual_radio_application_point = np.zeros(shape=(t + 1 + projection_horizon))
				counterfactual_radio_dosage = np.zeros(shape=(t + 1 + projection_horizon))

				counterfactual_cancer_volume[:t + 2] = factual_cancer_volume[:t + 2]
				counterfactual_radio_application_point[:t + 1] = factual_radio_application_point[:t + 1]
				counterfactual_radio_dosage[:t + 1] = factual_radio_dosage[:t + 1]

				for projection_time in range(-1, projection_horizon):

					current_t = t + 1 + projection_time

					counterfactual_radio_dosage[current_t] = 0.0
					if treatment_option[projection_time, 0] == 1:
						counterfactual_radio_application_point[current_t] = 1
						counterfactual_radio_dosage[current_t] = radio_amt[0]
					
					v = max(counterfactual_cancer_volume[current_t], 0.0)
					counterfactual_cancer_volume[current_t + 1] = counterfactual_cancer_volume[current_t] *\
                        (1 + rho * np.log((K + 1e-7) / (v + 1e-7)) -
                         (alpha * counterfactual_radio_dosage[current_t] + beta * counterfactual_radio_dosage[current_t] ** 2) +
                         noise[current_t + 1])
						
				if (np.isnan(counterfactual_cancer_volume).any()):
					continue

				cancer_volume[test_idx][:t + 1 + projection_horizon + 1] = counterfactual_cancer_volume
				radio_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_radio_application_point
				patient_types_all_trajectories[test_idx] = patient_types[i]
				patient_ids_all_trajectories[test_idx] = i
				patient_current_t[test_idx] = t

				sequence_lengths[test_idx] = int(t) + projection_horizon + 1
				test_idx = test_idx + 1
			
			if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
				recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
				break
	
	outputs = {'cancer_volume': cancer_volume[:test_idx],
			'radio_application': radio_application_point[:test_idx],
			'sequence_lengths': sequence_lengths[:test_idx],
			'patient_types': patient_types_all_trajectories[:test_idx],
			'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
			'patient_current_t': patient_current_t[:test_idx]}
	
	return outputs


def get_scaling_params(sim):
	real_idx = ['cancer_volume', 'radio_dosage']

	means = {}
	stds = {}
	seq_lengths = sim['sequence_lengths']
	for k in real_idx:
		active_values = []
		for i in range(seq_lengths.shape[0]):
			end = int(seq_lengths[i])
			active_values += list(sim[k][i, :end])
		
		means[k] = np.mean(active_values)
		stds[k] = np.std(active_values)
	
	# Add means for static variables
	means['patient_types'] = np.mean(sim['patient_types'])
	stds['patient_types'] = np.std(sim['patient_types'])

	return pd.Series(means), pd.Series(stds)

