/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iostream>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
    normal_distribution<double> x_normal_distribution(x, std[0]);
    normal_distribution<double> y_normal_distribution(y, std[1]);
    normal_distribution<double> theta_normal_distribution(theta, std[2]);
	random_device device;
	default_random_engine generator(device());
	
	for (int particle_id = 0; particle_id < num_particles; particle_id++) {
		Particle new_particle = Particle();
		new_particle.id = particle_id;
		new_particle.x = x_normal_distribution(generator);
		new_particle.y = y_normal_distribution(generator);
		new_particle.theta = theta_normal_distribution(generator);
		new_particle.weight = 1;
		particles.push_back(new_particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	for (int particle_id=0; particle_id < particles.size(); particle_id++) {
		
		// TODO: Sprawdzić, czy usunięcie nawiasów pozwoli działać poprawnie
		Particle* current_particle = &particles[particle_id];
		if (fabs(yaw_rate) > 0.0001) {

			double theta_change = yaw_rate * delta_t;
			double velocity_over_yawrate = velocity/yaw_rate;

			current_particle->x = current_particle->x + 
				velocity_over_yawrate * (sin(current_particle->theta + theta_change) - sin(current_particle->theta));

			current_particle->y = current_particle->y + 
				velocity_over_yawrate * (cos(current_particle->theta) - cos(current_particle->theta + theta_change));

			current_particle->theta = current_particle->theta + theta_change;
		} else {
			current_particle->theta += yaw_rate * delta_t;
			current_particle->x = current_particle->x + velocity * cos(current_particle->theta) * delta_t;
			current_particle->y = current_particle->y + velocity * sin(current_particle->theta) * delta_t;
		}
		
		normal_distribution<double> x_normal_distribution(current_particle->x, std_pos[0]);
		normal_distribution<double> y_normal_distribution(current_particle->y, std_pos[1]);
		normal_distribution<double> theta_normal_distribution(current_particle->theta, std_pos[2]);
		random_device device;
		default_random_engine generator(device());

		current_particle->x = x_normal_distribution(generator);
		current_particle->y = y_normal_distribution(generator);
		current_particle->theta = theta_normal_distribution(generator);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int observation_nb=0; observation_nb < observations.size(); observation_nb++) {
		LandmarkObs* observation = &observations[observation_nb];
		double min_distance = dist(observation->x, observation->y, predicted[0].x, predicted[0].y);
		double closest_landmark_id = predicted[0].id;

		for (int predicted_nb = 1; predicted_nb < predicted.size(); predicted_nb++) {
			double distance = dist(observation->x, observation->y, predicted[predicted_nb].x, predicted[predicted_nb].y);
			if (distance < min_distance) {
				min_distance = distance;
				closest_landmark_id = predicted[predicted_nb].id;
			}
		}
		observation->id = closest_landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();
	for (int particle_nb = 0; particle_nb < particles.size(); particle_nb++) {
		vector<LandmarkObs> transformed_observations;
		Particle* current_particle = &particles[particle_nb];

		for (int observation_nb = 0; observation_nb < observations.size(); observation_nb++) {
			LandmarkObs vehicle_observation = observations[observation_nb];
			LandmarkObs transformed_observation;

			double translated_landmark_x = current_particle->x + vehicle_observation.x;
			double translated_landmark_y = current_particle->y + vehicle_observation.y;

			double cos_theta = cos(current_particle->theta);
			double sin_theta = sin(current_particle->theta);

			double px_sub_ox = translated_landmark_x - current_particle->x;
			double py_sub_oy = translated_landmark_y - current_particle->y;
			transformed_observation.x = cos_theta * px_sub_ox - sin_theta * py_sub_oy + current_particle->x;
			transformed_observation.y = sin_theta * px_sub_ox + cos_theta * py_sub_oy + current_particle->y;

			transformed_observations.push_back(transformed_observation);
		}
		
		vector<LandmarkObs> map_landmarks_as_observations;
		for (int landmark_nb = 0; landmark_nb < map_landmarks.landmark_list.size(); landmark_nb++) {
			LandmarkObs map_landmark_observation = LandmarkObs();

			map_landmark_observation.id = map_landmarks.landmark_list[landmark_nb].id_i;
			map_landmark_observation.x = map_landmarks.landmark_list[landmark_nb].x_f;
			map_landmark_observation.y = map_landmarks.landmark_list[landmark_nb].y_f;

			map_landmarks_as_observations.push_back(map_landmark_observation);	
		}

		dataAssociation(map_landmarks_as_observations, transformed_observations);

		double bivariate_normal_denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
		double x_term_denominator = 2 * std_landmark[0] * std_landmark[0];
		double y_term_denominator = 2 * std_landmark[1] * std_landmark[1];
		
		double new_particle_weight = 1;
		for (int transformed_observation_id = 0; transformed_observation_id < transformed_observations.size(); transformed_observation_id++) {
			LandmarkObs transformed_observation = transformed_observations[transformed_observation_id];
	
			double mu_x = map_landmarks.landmark_list[transformed_observation.id - 1].x_f;
			double mu_y = map_landmarks.landmark_list[transformed_observation.id - 1].y_f;
			int id = map_landmarks.landmark_list[transformed_observation.id - 1].id_i;
			
			new_particle_weight *= 
				exp(-( 
					(pow(transformed_observation.x - mu_x, 2) / x_term_denominator) + 
					(pow(transformed_observation.y - mu_y, 2) / y_term_denominator) 
				)) /
				bivariate_normal_denominator;
		}
		current_particle->weight = new_particle_weight;
		weights.push_back(new_particle_weight);
	}

	double particle_weight_sum = 0;
	for (int i = 0; i < particles.size(); i++) {
		particle_weight_sum += particles[i].weight;
	}
	for (int i = 0; i < particles.size(); i++) {
		particles[i].weight = particles[i].weight / particle_weight_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device device;
	default_random_engine generator(device());
    discrete_distribution<> distribution(weights.begin(), weights.end());
	
	vector<Particle> resampled_particles;
	for(int i=0; i<num_particles; i++) {
		int sampled_particle_index = distribution(generator);
		resampled_particles.push_back(particles[sampled_particle_index]);
	}
	particles = resampled_particles;
}

void ParticleFilter::write(string filename) {
	// You don't need to modify this file.
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
