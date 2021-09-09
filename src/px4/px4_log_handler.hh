// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2021 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// goncalo.fontes-neves@student.isae-supaero.fr
// ---------------------------------------------------------------------

#ifndef __PX4_LOG_HANDLER_HH__
#define __PX4_LOG_HANDLER_HH__

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


class px4_log_handler
{
	public:
		px4_log_handler();
		~px4_log_handler();
		void load_log(const char *px4logfilename);
		void load_log11(const char *px4logfilename);
		void update_adds();
		void print_adds();
		void normalize();
		void randomize();
		void create_random_dataset(uint32_t sample_nb, uint32_t nb_timestep);
		
		void create_report(const char *reportfilename);
		
		std::vector<std::vector<std::vector<float>>> get_input_rdm_test(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size);
		std::vector<std::vector<std::vector<float>>> get_target_rdm_test(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size);
		std::vector<std::vector<std::vector<float>>> get_input_rdm_train(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size);
		std::vector<std::vector<std::vector<float>>> get_target_rdm_train(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size);
		std::vector<std::vector<float>> get_input_for_inference(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size);
		std::vector<std::vector<float>> get_target_for_inference(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size);
		std::vector<std::vector<float>> get_input_test(uint32_t batch, uint32_t nb_elem);
		std::vector<std::vector<float>> get_target_test(uint32_t batch, uint32_t nb_elem);

		uint32_t get_max_train_samples(){return _max_train_samples;};
		uint32_t get_max_test_samples(){return _max_test_samples;};
		uint32_t get_nb_test_batches(){return _random_it_vector_test.size();};
		uint32_t get_nb_elem(){return _nb_elem;};

	private: 
		uint32_t _nb_elem;
		std::vector<uint32_t> _it_vector;
		std::vector<uint32_t> _random_it_vector_train;
		std::vector<uint32_t> _random_it_vector_test;
		uint32_t _max_train_samples;
		uint32_t _max_test_samples;
		
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
		std::vector<float> _pitch;
		std::vector<float> _roll;
		std::vector<float> _yaw;
		
		//normalized inputs
		std::vector<float> _mag_x_stdscore_norm;
		std::vector<float> _mag_y_stdscore_norm;
		std::vector<float> _mag_z_stdscore_norm;
		std::vector<float> _acc_x_stdscore_norm;
		std::vector<float> _acc_y_stdscore_norm;
		std::vector<float> _acc_z_stdscore_norm;
		std::vector<float> _gyr_x_stdscore_norm;
		std::vector<float> _gyr_y_stdscore_norm;
		std::vector<float> _gyr_z_stdscore_norm;
		
		std::vector<float> _input0_rdm;
		std::vector<float> _input1_rdm;
		std::vector<float> _input2_rdm;
		std::vector<float> _input3_rdm;
		std::vector<float> _input4_rdm;
		std::vector<float> _input5_rdm;
		std::vector<float> _input6_rdm;
		std::vector<float> _input7_rdm;
		std::vector<float> _input8_rdm;
		
		std::vector<float> _target0_rdm;
		std::vector<float> _target1_rdm;
		std::vector<float> _target2_rdm;
		
		std::vector<float> _mag_x_stdscore;
		std::vector<float> _mag_y_stdscore;
		std::vector<float> _mag_z_stdscore;
		std::vector<float> _acc_x_stdscore;
		std::vector<float> _acc_y_stdscore;
		std::vector<float> _acc_z_stdscore;
		std::vector<float> _gyr_x_stdscore;
		std::vector<float> _gyr_y_stdscore;
		std::vector<float> _gyr_z_stdscore;
		
		std::vector<float>  _mag_x_adds_stdscore;
		std::vector<float>  _mag_y_adds_stdscore;
		std::vector<float>  _mag_z_adds_stdscore;
		std::vector<float>  _acc_x_adds_stdscore;
		std::vector<float>  _acc_y_adds_stdscore;
		std::vector<float>  _acc_z_adds_stdscore;
		std::vector<float>  _gyr_x_adds_stdscore;
		std::vector<float>  _gyr_y_adds_stdscore;
		std::vector<float>  _gyr_z_adds_stdscore;
		std::vector<float>  _pitch_adds_stdscore;
		std::vector<float>  _roll_adds_stdscore;
		std::vector<float>  _yaw_adds_stdscore;
		
		std::vector<float> _mag_x_norm;
		std::vector<float> _mag_y_norm;
		std::vector<float> _mag_z_norm;
		std::vector<float> _acc_x_norm;
		std::vector<float> _acc_y_norm;
		std::vector<float> _acc_z_norm;
		std::vector<float> _gyr_x_norm;
		std::vector<float> _gyr_y_norm;
		std::vector<float> _gyr_z_norm;
		
		// normalized from output
		std::vector<float> _pitch_norm;
		std::vector<float> _roll_norm;
		std::vector<float> _yaw_norm;
		std::vector<float> _pitch_stdscore;
		std::vector<float> _roll_stdscore;
		std::vector<float> _yaw_stdscore;
		std::vector<float> _pitch_stdscore_norm;
		std::vector<float> _roll_stdscore_norm;
		std::vector<float> _yaw_stdscore_norm;
		
		// normalized from output
		std::vector<float> _pitch_lstm;
		std::vector<float> _roll_lstm;
		std::vector<float> _yaw_lstm;
		
		// additionals (min, max, mean, std)
		std::vector<float>  _mag_x_adds;
		std::vector<float>  _mag_y_adds;
		std::vector<float>  _mag_z_adds;
		std::vector<float>  _acc_x_adds;
		std::vector<float>  _acc_y_adds;
		std::vector<float>  _acc_z_adds;
		std::vector<float>  _gyr_x_adds;
		std::vector<float>  _gyr_y_adds;
		std::vector<float>  _gyr_z_adds;
		std::vector<float>  _pitch_adds;
		std::vector<float>  _roll_adds;
		std::vector<float>  _yaw_adds;
};

#endif 



