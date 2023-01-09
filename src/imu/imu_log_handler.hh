// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2021-2022 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// ---------------------------------------------------------------------

#ifndef __IMU_LOG_HANDLER_HH__
#define __IMU_LOG_HANDLER_HH__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <fstream> 
#include <string>
#include <sstream> // sstream usage
#include <algorithm> //Not part of string, use "algorithms" replace()
#include <iostream>
#include <vector>
//#include <cmath>
#include <math.h>
#include <ctime>        // std::time
#include <assert.h>


class imu_log_handler
{
	public:
		imu_log_handler();
		~imu_log_handler();
		void load_imu_log(const char *imulogfilename);
		void load_px4_log(const char *imulogfilename);
		void randomize_training_dataset();
		void create_split_and_create_random_dataset(float test_split_coeff);
		
		void create_report(const char *reportfilename);
		void create_mag_mah_report(const char *reportfilename);
		void create_loss_report(const char *reportfilename);
		void create_time_report(const char *reportfilename);
		
		std::vector<float> get_sensors(uint32_t it);
		
		std::vector<std::vector<std::vector<float>>> get_input_test(uint32_t nb_timestep);
		std::vector<std::vector<std::vector<float>>> get_target_test(uint32_t nb_timestep);
		std::vector<std::vector<std::vector<float>>> get_target_phi_test(uint32_t nb_timestep);
		std::vector<std::vector<std::vector<float>>> get_input_rdm_train(uint32_t nb_timestep);
		std::vector<std::vector<std::vector<float>>> get_target_rdm_train(uint32_t nb_timestep);
		std::vector<std::vector<std::vector<float>>> get_target_phi_rdm_train(uint32_t nb_timestep);

		uint32_t get_max_train_samples(){return _max_train_samples;};
		uint32_t get_max_test_samples(){return _max_test_samples;};
		uint32_t get_nb_elem(){return _nb_elem;};
		uint32_t get_training_samples_size();
		
		float get_first_phi(){return _phi[0];};
		float get_first_theta(){return _theta[0];};
		float get_first_psi(){return _psi[0];};
		
		// Handles for outputs logs
		void add_one_epoch_for_lstm_sgd_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_lstm_sgd_phi(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_lstm_sgd_all_tdim(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_gru_sgd_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_star_sgd_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_mgu_sgd_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_lstm_adam_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_gru_adam_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_star_adam_all(std::vector<std::vector<float>> nn_output_epoch);
		void add_one_epoch_for_mgu_adam_all(std::vector<std::vector<float>> nn_output_epoch);
		
		// Handles for error logs
		void add_epoch_train_error_for_lstm_sgd_all(float loss) {_loss_train_sgd_lstm_all.push_back(loss);};
		void add_epoch_train_error_for_lstm_sgd_all_tdim(float loss) {_loss_train_sgd_lstm_all_tdim.push_back(loss);};
		void add_epoch_train_error_for_gru_sgd_all(float loss) {_loss_train_sgd_gru_all.push_back(loss);};
		void add_epoch_train_error_for_star_sgd_all(float loss) {_loss_train_sgd_star_all.push_back(loss);};
		void add_epoch_train_error_for_mgu_sgd_all(float loss) {_loss_train_sgd_mgu_all.push_back(loss);};
		void add_epoch_train_error_for_lstm_adam_all(float loss) {_loss_train_adam_lstm_all.push_back(loss);};
		void add_epoch_train_error_for_gru_adam_all(float loss) {_loss_train_adam_gru_all.push_back(loss);};
		void add_epoch_train_error_for_star_adam_all(float loss) {_loss_train_adam_star_all.push_back(loss);};
		void add_epoch_train_error_for_mgu_adam_all(float loss) {_loss_train_adam_mgu_all.push_back(loss);};
		void add_epoch_train_error_for_lstm_adamax_all(float loss) {_loss_train_adamax_lstm_all.push_back(loss);};
		void add_epoch_train_error_for_gru_adamax_all(float loss) {_loss_train_adamax_gru_all.push_back(loss);};
		void add_epoch_train_error_for_star_adamax_all(float loss) {_loss_train_adamax_star_all.push_back(loss);};
		void add_epoch_train_error_for_mgu_adamax_all(float loss) {_loss_train_adamax_mgu_all.push_back(loss);};
		void add_epoch_test_error_for_lstm_sgd_all(float loss) {_loss_test_sgd_lstm_all.push_back(loss);};
		void add_epoch_test_error_for_lstm_sgd_all_tdim(float loss) {_loss_test_sgd_lstm_all_tdim.push_back(loss);};
		void add_epoch_test_error_for_gru_sgd_all(float loss) {_loss_test_sgd_gru_all.push_back(loss);};
		void add_epoch_test_error_for_star_sgd_all(float loss) {_loss_test_sgd_star_all.push_back(loss);};
		void add_epoch_test_error_for_mgu_sgd_all(float loss) {_loss_test_sgd_mgu_all.push_back(loss);};
		void add_epoch_test_error_for_lstm_adam_all(float loss) {_loss_test_adam_lstm_all.push_back(loss);};
		void add_epoch_test_error_for_gru_adam_all(float loss) {_loss_test_adam_gru_all.push_back(loss);};
		void add_epoch_test_error_for_star_adam_all(float loss) {_loss_test_adam_star_all.push_back(loss);};
		void add_epoch_test_error_for_mgu_adam_all(float loss) {_loss_test_adam_mgu_all.push_back(loss);};
		void add_epoch_test_error_for_lstm_adamax_all(float loss) {_loss_test_adamax_lstm_all.push_back(loss);};
		void add_epoch_test_error_for_gru_adamax_all(float loss) {_loss_test_adamax_gru_all.push_back(loss);};
		void add_epoch_test_error_for_star_adamax_all(float loss) {_loss_test_adamax_star_all.push_back(loss);};
		void add_epoch_test_error_for_mgu_adamax_all(float loss) {_loss_test_adamax_mgu_all.push_back(loss);};
		
		// Add timings info
		void add_epoch_train_time_for_lstm_sgd_all(std::vector<float> times) {_time_train_sgd_lstm_all.push_back(times);};
		void add_epoch_train_time_for_gru_sgd_all(std::vector<float> times) {_time_train_sgd_gru_all.push_back(times);};
		void add_epoch_train_time_for_star_sgd_all(std::vector<float> times) {_time_train_sgd_star_all.push_back(times);};
		void add_epoch_train_time_for_mgu_sgd_all(std::vector<float> times) {_time_train_sgd_mgu_all.push_back(times);};
		void add_epoch_train_time_for_lstm_adam_all(std::vector<float> times) {_time_train_adam_lstm_all.push_back(times);};
		void add_epoch_train_time_for_gru_adam_all(std::vector<float> times) {_time_train_adam_gru_all.push_back(times);};
		void add_epoch_train_time_for_star_adam_all(std::vector<float> times) {_time_train_adam_star_all.push_back(times);};
		void add_epoch_train_time_for_mgu_adam_all(std::vector<float> times) {_time_train_adam_mgu_all.push_back(times);};
		void add_epoch_train_time_for_lstm_adamax_all(std::vector<float> times) {_time_train_adamax_lstm_all.push_back(times);};
		void add_epoch_train_time_for_gru_adamax_all(std::vector<float> times) {_time_train_adamax_gru_all.push_back(times);};
		void add_epoch_train_time_for_star_adamax_all(std::vector<float> times) {_time_train_adamax_star_all.push_back(times);};
		void add_epoch_train_time_for_mgu_adamax_all(std::vector<float> times) {_time_train_adamax_mgu_all.push_back(times);};
		void add_epoch_test_time_for_lstm_sgd_all(std::vector<float> times) {_time_test_sgd_lstm_all.push_back(times);};
		void add_epoch_test_time_for_gru_sgd_all(std::vector<float> times) {_time_test_sgd_gru_all.push_back(times);};
		void add_epoch_test_time_for_star_sgd_all(std::vector<float> times) {_time_test_sgd_star_all.push_back(times);};
		void add_epoch_test_time_for_mgu_sgd_all(std::vector<float> times) {_time_test_sgd_mgu_all.push_back(times);};
		void add_epoch_test_time_for_lstm_adam_all(std::vector<float> times) {_time_test_adam_lstm_all.push_back(times);};
		void add_epoch_test_time_for_gru_adam_all(std::vector<float> times) {_time_test_adam_gru_all.push_back(times);};
		void add_epoch_test_time_for_star_adam_all(std::vector<float> times) {_time_test_adam_star_all.push_back(times);};
		void add_epoch_test_time_for_mgu_adam_all(std::vector<float> times) {_time_test_adam_mgu_all.push_back(times);};
		void add_epoch_test_time_for_lstm_adamax_all(std::vector<float> times) {_time_test_adamax_lstm_all.push_back(times);};
		void add_epoch_test_time_for_gru_adamax_all(std::vector<float> times) {_time_test_adamax_gru_all.push_back(times);};
		void add_epoch_test_time_for_star_adamax_all(std::vector<float> times) {_time_test_adamax_star_all.push_back(times);};
		void add_epoch_test_time_for_mgu_adamax_all(std::vector<float> times) {_time_test_adamax_mgu_all.push_back(times);};
		
		void add_phi_mah(int iter, float val);
		void add_theta_mah(int iter, float val);
		void add_psi_mah(int iter, float val);
		void add_phi_mah2(int iter, float val);
		void add_theta_mah2(int iter, float val);
		void add_psi_mah2(int iter, float val);
		
		void add_phi_mag(int iter, float val);
		void add_theta_mag(int iter, float val);
		void add_psi_mag(int iter, float val);
		void add_phi_mag2(int iter, float val);
		void add_theta_mag2(int iter, float val);
		void add_psi_mag2(int iter, float val);

	private: 
		uint32_t _nb_elem;
		std::vector<uint32_t> _it_vector;
		std::vector<uint32_t> _random_it_vector_train;
		std::vector<uint32_t> _it_vector_test;
		uint32_t _max_train_samples;
		uint32_t _max_test_samples;
		uint32_t _test_start_index;
		
		// inputs from log
		std::vector<float> _mag_x;
		std::vector<float> _mag_y;
		std::vector<float> _mag_z;
		std::vector<float> _acc_x;
		std::vector<float> _acc_y;
		std::vector<float> _acc_z;
		std::vector<float> _gyr_x;
		std::vector<float> _gyr_y;
		std::vector<float> _gyr_z;
		// labels from log
		std::vector<float> _phi;
		std::vector<float> _theta;
		std::vector<float> _psi;
		
		// Mahony/Magdwick no Magnetometers
		std::vector<float> _phi_mah;
		std::vector<float> _theta_mah;
		std::vector<float> _psi_mah;	
		std::vector<float> _phi_mag;
		std::vector<float> _theta_mag;
		std::vector<float> _psi_mag;
		std::vector<float> _phi_mah2;
		std::vector<float> _theta_mah2;
		std::vector<float> _psi_mah2;	
		std::vector<float> _phi_mag2;
		std::vector<float> _theta_mag2;
		std::vector<float> _psi_mag2;
		
		// logs outputs
		// SGD
		std::vector<std::vector<std::vector<float>>> _lstm_sgd_all;
		std::vector<std::vector<std::vector<float>>> _lstm_sgd_all_tdim;
		std::vector<std::vector<std::vector<float>>> _lstm_sgd_phi;
		std::vector<std::vector<std::vector<float>>> _gru_sgd_all;
		std::vector<std::vector<std::vector<float>>> _star_sgd_all;
		std::vector<std::vector<std::vector<float>>> _mgu_sgd_all;
		// ADAM
		std::vector<std::vector<std::vector<float>>> _lstm_adam_all;
		std::vector<std::vector<std::vector<float>>> _gru_adam_all;
		std::vector<std::vector<std::vector<float>>> _star_adam_all;
		std::vector<std::vector<std::vector<float>>> _mgu_adam_all;
		// ADAMAX
		std::vector<std::vector<std::vector<float>>> _lstm_adamax_all;
		std::vector<std::vector<std::vector<float>>> _gru_adamax_all;
		std::vector<std::vector<std::vector<float>>> _star_adamax_all;
		std::vector<std::vector<std::vector<float>>> _mgu_adamax_all;

		// logs losses
		// SGD
		std::vector<float> _loss_train_sgd_lstm_all;
		std::vector<float> _loss_train_sgd_lstm_all_tdim;
		std::vector<float> _loss_train_sgd_lstm_phi;
		std::vector<float> _loss_train_sgd_gru_all;
		std::vector<float> _loss_train_sgd_star_all;
		std::vector<float> _loss_train_sgd_mgu_all;
		std::vector<float> _loss_test_sgd_lstm_all;
		std::vector<float> _loss_test_sgd_lstm_all_tdim;
		std::vector<float> _loss_test_sgd_gru_all;
		std::vector<float> _loss_test_sgd_star_all;
		std::vector<float> _loss_test_sgd_mgu_all;
		// ADAM
		std::vector<float> _loss_train_adam_lstm_all;
		std::vector<float> _loss_train_adam_gru_all;
		std::vector<float> _loss_train_adam_star_all;
		std::vector<float> _loss_train_adam_mgu_all;
		std::vector<float> _loss_test_adam_lstm_all;
		std::vector<float> _loss_test_adam_gru_all;
		std::vector<float> _loss_test_adam_star_all;
		std::vector<float> _loss_test_adam_mgu_all;
		// ADAMAX
		std::vector<float> _loss_train_adamax_lstm_all;
		std::vector<float> _loss_train_adamax_gru_all;
		std::vector<float> _loss_train_adamax_star_all;
		std::vector<float> _loss_train_adamax_mgu_all;
		std::vector<float> _loss_test_adamax_lstm_all;
		std::vector<float> _loss_test_adamax_gru_all;
		std::vector<float> _loss_test_adamax_star_all;
		std::vector<float> _loss_test_adamax_mgu_all;
		
		// log timings
		std::vector<std::vector<float>> _time_train_sgd_lstm_all;
		std::vector<std::vector<float>> _time_train_sgd_gru_all;
		std::vector<std::vector<float>> _time_train_sgd_star_all;
		std::vector<std::vector<float>> _time_train_sgd_mgu_all;
		std::vector<std::vector<float>> _time_test_sgd_lstm_all;
		std::vector<std::vector<float>> _time_test_sgd_gru_all;
		std::vector<std::vector<float>> _time_test_sgd_star_all;
		std::vector<std::vector<float>> _time_test_sgd_mgu_all;
		// ADAM
		std::vector<std::vector<float>> _time_train_adam_lstm_all;
		std::vector<std::vector<float>> _time_train_adam_gru_all;
		std::vector<std::vector<float>> _time_train_adam_star_all;
		std::vector<std::vector<float>> _time_train_adam_mgu_all;
		std::vector<std::vector<float>> _time_test_adam_lstm_all;
		std::vector<std::vector<float>> _time_test_adam_gru_all;
		std::vector<std::vector<float>> _time_test_adam_star_all;
		std::vector<std::vector<float>> _time_test_adam_mgu_all;
		// ADAMAX
		std::vector<std::vector<float>> _time_train_adamax_lstm_all;
		std::vector<std::vector<float>> _time_train_adamax_gru_all;
		std::vector<std::vector<float>> _time_train_adamax_star_all;
		std::vector<std::vector<float>> _time_train_adamax_mgu_all;
		std::vector<std::vector<float>> _time_test_adamax_lstm_all;
		std::vector<std::vector<float>> _time_test_adamax_gru_all;
		std::vector<std::vector<float>> _time_test_adamax_star_all;
		std::vector<std::vector<float>> _time_test_adamax_mgu_all;
		
		// secure end of log timesteps...to avoid segfault...
		uint32_t _end_of_file_timesteps;
};

#endif 



