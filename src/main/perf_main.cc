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

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <chrono>
#include <sys/stat.h>


// Local includes
#include "px4_log_handler.hh"
#include <lstm.hh>
#include <gru.hh>
#include <random>

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_int_distribution<int> dist (0,1);


int main(int argc, char *argv[])
{	
	
	srand((int)time(NULL));
	static std::vector<px4_log_handler> my_log_handlers(11);
	
	static px4_log_handler my_log_handler_16;
	static px4_log_handler my_log_handler_32;
	static px4_log_handler my_log_handler_64;
	
	uint32_t epoch_max = 50;
	uint32_t nb_timestep_1 = 32;
	uint32_t sample_nb = 200;
	uint32_t max_samples = 100;
	
	//static lstm my_lstm_sgd_16_9_3;
	//static lstm my_lstm_adam_16_9_3;
	//static lstm my_lstm_adamax_16_9_3;
	//float min_fp_lstm_sgd_16_9_3 = 10.0, max_fp_lstm_sgd_16_9_3 =  0.0, mean_fp_lstm_sgd_16_9_3 =  0.0, sum_fp_lstm_sgd_16_9_3 =  0.0;
	//float min_bp_lstm_sgd_16_9_3 = 10.0, max_bp_lstm_sgd_16_9_3 =  0.0, mean_bp_lstm_sgd_16_9_3 =  0.0, sum_bp_lstm_sgd_16_9_3 =  0.0;
	//float min_fp_lstm_adam_16_9_3 = 10.0, max_fp_lstm_adam_16_9_3 =  0.0, mean_fp_lstm_adam_16_9_3 =  0.0, sum_fp_lstm_adam_16_9_3 =  0.0;
	//float min_bp_lstm_adam_16_9_3 = 10.0, max_bp_lstm_adam_16_9_3 =  0.0, mean_bp_lstm_adam_16_9_3 =  0.0, sum_bp_lstm_adam_16_9_3 =  0.0;
	//float min_fp_lstm_adamax_16_9_3 = 10.0, max_fp_lstm_adamax_16_9_3 =  0.0, mean_fp_lstm_adamax_16_9_3 =  0.0, sum_fp_lstm_adamax_16_9_3 =  0.0;
	//float min_bp_lstm_adamax_16_9_3 = 10.0, max_bp_lstm_adamax_16_9_3 =  0.0, mean_bp_lstm_adamax_16_9_3 =  0.0, sum_bp_lstm_adamax_16_9_3 =  0.0;
	
	//static gru my_gru_sgd_16_9_3;
	//static gru my_gru_adam_16_9_3;
	//static gru my_gru_adamax_16_9_3;
	//float min_fp_gru_sgd_16_9_3 = 10.0, max_fp_gru_sgd_16_9_3 =  0.0, mean_fp_gru_sgd_16_9_3 =  0.0, sum_fp_gru_sgd_16_9_3 =  0.0;
	//float min_bp_gru_sgd_16_9_3 = 10.0, max_bp_gru_sgd_16_9_3 =  0.0, mean_bp_gru_sgd_16_9_3 =  0.0, sum_bp_gru_sgd_16_9_3 =  0.0;
	//float min_fp_gru_adam_16_9_3 = 10.0, max_fp_gru_adam_16_9_3 =  0.0, mean_fp_gru_adam_16_9_3 =  0.0, sum_fp_gru_adam_16_9_3 =  0.0;
	//float min_bp_gru_adam_16_9_3 = 10.0, max_bp_gru_adam_16_9_3 =  0.0, mean_bp_gru_adam_16_9_3 =  0.0, sum_bp_gru_adam_16_9_3 =  0.0;
	//float min_fp_gru_adamax_16_9_3 = 10.0, max_fp_gru_adamax_16_9_3 =  0.0, mean_fp_gru_adamax_16_9_3 =  0.0, sum_fp_gru_adamax_16_9_3 =  0.0;
	//float min_bp_gru_adamax_16_9_3 = 10.0, max_bp_gru_adamax_16_9_3 =  0.0, mean_bp_gru_adamax_16_9_3 =  0.0, sum_bp_gru_adamax_16_9_3 =  0.0;
	
	static lstm my_lstm_sgd_32_9_3;
	static lstm my_lstm_adam_32_9_3;
	static lstm my_lstm_adamax_32_9_3;
	float min_fp_lstm_sgd_32_9_3 = 10.0, max_fp_lstm_sgd_32_9_3 =  0.0, mean_fp_lstm_sgd_32_9_3 =  0.0, sum_fp_lstm_sgd_32_9_3 =  0.0;
	float min_bp_lstm_sgd_32_9_3 = 10.0, max_bp_lstm_sgd_32_9_3 =  0.0, mean_bp_lstm_sgd_32_9_3 =  0.0, sum_bp_lstm_sgd_32_9_3 =  0.0;
	float min_fp_lstm_adam_32_9_3 = 10.0, max_fp_lstm_adam_32_9_3 =  0.0, mean_fp_lstm_adam_32_9_3 =  0.0, sum_fp_lstm_adam_32_9_3 =  0.0;
	float min_bp_lstm_adam_32_9_3 = 10.0, max_bp_lstm_adam_32_9_3 =  0.0, mean_bp_lstm_adam_32_9_3 =  0.0, sum_bp_lstm_adam_32_9_3 =  0.0;
	float min_fp_lstm_adamax_32_9_3 = 10.0, max_fp_lstm_adamax_32_9_3 =  0.0, mean_fp_lstm_adamax_32_9_3 =  0.0, sum_fp_lstm_adamax_32_9_3 =  0.0;
	float min_bp_lstm_adamax_32_9_3 = 10.0, max_bp_lstm_adamax_32_9_3 =  0.0, mean_bp_lstm_adamax_32_9_3 =  0.0, sum_bp_lstm_adamax_32_9_3 =  0.0;
	
	static gru my_gru_sgd_32_9_3;
	static gru my_gru_adam_32_9_3;
	static gru my_gru_adamax_32_9_3;
	float min_fp_gru_sgd_32_9_3 = 10.0, max_fp_gru_sgd_32_9_3 =  0.0, mean_fp_gru_sgd_32_9_3 =  0.0, sum_fp_gru_sgd_32_9_3 =  0.0;
	float min_bp_gru_sgd_32_9_3 = 10.0, max_bp_gru_sgd_32_9_3 =  0.0, mean_bp_gru_sgd_32_9_3 =  0.0, sum_bp_gru_sgd_32_9_3 =  0.0;
	float min_fp_gru_adam_32_9_3 = 10.0, max_fp_gru_adam_32_9_3 =  0.0, mean_fp_gru_adam_32_9_3 =  0.0, sum_fp_gru_adam_32_9_3 =  0.0;
	float min_bp_gru_adam_32_9_3 = 10.0, max_bp_gru_adam_32_9_3 =  0.0, mean_bp_gru_adam_32_9_3 =  0.0, sum_bp_gru_adam_32_9_3 =  0.0;
	float min_fp_gru_adamax_32_9_3 = 10.0, max_fp_gru_adamax_32_9_3 =  0.0, mean_fp_gru_adamax_32_9_3 =  0.0, sum_fp_gru_adamax_32_9_3 =  0.0;
	float min_bp_gru_adamax_32_9_3 = 10.0, max_bp_gru_adamax_32_9_3 =  0.0, mean_bp_gru_adamax_32_9_3 =  0.0, sum_bp_gru_adamax_32_9_3 =  0.0;
	
	//static lstm my_lstm_sgd_64_9_3;
	//static lstm my_lstm_adam_64_9_3;
	//static lstm my_lstm_adamax_64_9_3;
	//float min_fp_lstm_sgd_64_9_3 = 10.0, max_fp_lstm_sgd_64_9_3 =  0.0, mean_fp_lstm_sgd_64_9_3 =  0.0, sum_fp_lstm_sgd_64_9_3 =  0.0;
	//float min_bp_lstm_sgd_64_9_3 = 10.0, max_bp_lstm_sgd_64_9_3 =  0.0, mean_bp_lstm_sgd_64_9_3 =  0.0, sum_bp_lstm_sgd_64_9_3 =  0.0;
	//float min_fp_lstm_adam_64_9_3 = 10.0, max_fp_lstm_adam_64_9_3 =  0.0, mean_fp_lstm_adam_64_9_3 =  0.0, sum_fp_lstm_adam_64_9_3 =  0.0;
	//float min_bp_lstm_adam_64_9_3 = 10.0, max_bp_lstm_adam_64_9_3 =  0.0, mean_bp_lstm_adam_64_9_3 =  0.0, sum_bp_lstm_adam_64_9_3 =  0.0;
	//float min_fp_lstm_adamax_64_9_3 = 10.0, max_fp_lstm_adamax_64_9_3 =  0.0, mean_fp_lstm_adamax_64_9_3 =  0.0, sum_fp_lstm_adamax_64_9_3 =  0.0;
	//float min_bp_lstm_adamax_64_9_3 = 10.0, max_bp_lstm_adamax_64_9_3 =  0.0, mean_bp_lstm_adamax_64_9_3 =  0.0, sum_bp_lstm_adamax_64_9_3 =  0.0;
	
	//static gru my_gru_sgd_64_9_3;
	//static gru my_gru_adam_64_9_3;
	//static gru my_gru_adamax_64_9_3;
	//float min_fp_gru_sgd_64_9_3 = 10.0, max_fp_gru_sgd_64_9_3 =  0.0, mean_fp_gru_sgd_64_9_3 =  0.0, sum_fp_gru_sgd_64_9_3 =  0.0;
	//float min_bp_gru_sgd_64_9_3 = 10.0, max_bp_gru_sgd_64_9_3 =  0.0, mean_bp_gru_sgd_64_9_3 =  0.0, sum_bp_gru_sgd_64_9_3 =  0.0;
	//float min_fp_gru_adam_64_9_3 = 10.0, max_fp_gru_adam_64_9_3 =  0.0, mean_fp_gru_adam_64_9_3 =  0.0, sum_fp_gru_adam_64_9_3 =  0.0;
	//float min_bp_gru_adam_64_9_3 = 10.0, max_bp_gru_adam_64_9_3 =  0.0, mean_bp_gru_adam_64_9_3 =  0.0, sum_bp_gru_adam_64_9_3 =  0.0;
	//float min_fp_gru_adamax_64_9_3 = 10.0, max_fp_gru_adamax_64_9_3 =  0.0, mean_fp_gru_adamax_64_9_3 =  0.0, sum_fp_gru_adamax_64_9_3 =  0.0;
	//float min_bp_gru_adamax_64_9_3 = 10.0, max_bp_gru_adamax_64_9_3 =  0.0, mean_bp_gru_adamax_64_9_3 =  0.0, sum_bp_gru_adamax_64_9_3 =  0.0;
	
	//static lstm my_lstm_sgd_16_9_1;
	//static lstm my_lstm_adam_16_9_1;
	//static lstm my_lstm_adamax_16_9_1;
	//float min_fp_lstm_sgd_16_9_1 = 10.0, max_fp_lstm_sgd_16_9_1 =  0.0, mean_fp_lstm_sgd_16_9_1 =  0.0, sum_fp_lstm_sgd_16_9_1 =  0.0;
	//float min_bp_lstm_sgd_16_9_1 = 10.0, max_bp_lstm_sgd_16_9_1 =  0.0, mean_bp_lstm_sgd_16_9_1 =  0.0, sum_bp_lstm_sgd_16_9_1 =  0.0;
	//float min_fp_lstm_adam_16_9_1 = 10.0, max_fp_lstm_adam_16_9_1 =  0.0, mean_fp_lstm_adam_16_9_1 =  0.0, sum_fp_lstm_adam_16_9_1 =  0.0;
	//float min_bp_lstm_adam_16_9_1 = 10.0, max_bp_lstm_adam_16_9_1 =  0.0, mean_bp_lstm_adam_16_9_1 =  0.0, sum_bp_lstm_adam_16_9_1 =  0.0;
	//float min_fp_lstm_adamax_16_9_1 = 10.0, max_fp_lstm_adamax_16_9_1 =  0.0, mean_fp_lstm_adamax_16_9_1 =  0.0, sum_fp_lstm_adamax_16_9_1 =  0.0;
	//float min_bp_lstm_adamax_16_9_1 = 10.0, max_bp_lstm_adamax_16_9_1 =  0.0, mean_bp_lstm_adamax_16_9_1 =  0.0, sum_bp_lstm_adamax_16_9_1 =  0.0;
	
	//static gru my_gru_sgd_16_9_1;
	//static gru my_gru_adam_16_9_1;
	//static gru my_gru_adamax_16_9_1;
	//float min_fp_gru_sgd_16_9_1 = 10.0, max_fp_gru_sgd_16_9_1 =  0.0, mean_fp_gru_sgd_16_9_1 =  0.0, sum_fp_gru_sgd_16_9_1 =  0.0;
	//float min_bp_gru_sgd_16_9_1 = 10.0, max_bp_gru_sgd_16_9_1 =  0.0, mean_bp_gru_sgd_16_9_1 =  0.0, sum_bp_gru_sgd_16_9_1 =  0.0;
	//float min_fp_gru_adam_16_9_1 = 10.0, max_fp_gru_adam_16_9_1 =  0.0, mean_fp_gru_adam_16_9_1 =  0.0, sum_fp_gru_adam_16_9_1 =  0.0;
	//float min_bp_gru_adam_16_9_1 = 10.0, max_bp_gru_adam_16_9_1 =  0.0, mean_bp_gru_adam_16_9_1 =  0.0, sum_bp_gru_adam_16_9_1 =  0.0;
	//float min_fp_gru_adamax_16_9_1 = 10.0, max_fp_gru_adamax_16_9_1 =  0.0, mean_fp_gru_adamax_16_9_1 =  0.0, sum_fp_gru_adamax_16_9_1 =  0.0;
	//float min_bp_gru_adamax_16_9_1 = 10.0, max_bp_gru_adamax_16_9_1 =  0.0, mean_bp_gru_adamax_16_9_1 =  0.0, sum_bp_gru_adamax_16_9_1 =  0.0;
	
	static lstm my_lstm_sgd_32_9_1;
	static lstm my_lstm_adam_32_9_1;
	static lstm my_lstm_adamax_32_9_1;
	float min_fp_lstm_sgd_32_9_1 = 10.0, max_fp_lstm_sgd_32_9_1 =  0.0, mean_fp_lstm_sgd_32_9_1 =  0.0, sum_fp_lstm_sgd_32_9_1 =  0.0;
	float min_bp_lstm_sgd_32_9_1 = 10.0, max_bp_lstm_sgd_32_9_1 =  0.0, mean_bp_lstm_sgd_32_9_1 =  0.0, sum_bp_lstm_sgd_32_9_1 =  0.0;
	float min_fp_lstm_adam_32_9_1 = 10.0, max_fp_lstm_adam_32_9_1 =  0.0, mean_fp_lstm_adam_32_9_1 =  0.0, sum_fp_lstm_adam_32_9_1 =  0.0;
	float min_bp_lstm_adam_32_9_1 = 10.0, max_bp_lstm_adam_32_9_1 =  0.0, mean_bp_lstm_adam_32_9_1 =  0.0, sum_bp_lstm_adam_32_9_1 =  0.0;
	float min_fp_lstm_adamax_32_9_1 = 10.0, max_fp_lstm_adamax_32_9_1 =  0.0, mean_fp_lstm_adamax_32_9_1 =  0.0, sum_fp_lstm_adamax_32_9_1 =  0.0;
	float min_bp_lstm_adamax_32_9_1 = 10.0, max_bp_lstm_adamax_32_9_1 =  0.0, mean_bp_lstm_adamax_32_9_1 =  0.0, sum_bp_lstm_adamax_32_9_1 =  0.0;
	
	static gru my_gru_sgd_32_9_1;
	static gru my_gru_adam_32_9_1;
	static gru my_gru_adamax_32_9_1;
	float min_fp_gru_sgd_32_9_1 = 10.0, max_fp_gru_sgd_32_9_1 =  0.0, mean_fp_gru_sgd_32_9_1 =  0.0, sum_fp_gru_sgd_32_9_1 =  0.0;
	float min_bp_gru_sgd_32_9_1 = 10.0, max_bp_gru_sgd_32_9_1 =  0.0, mean_bp_gru_sgd_32_9_1 =  0.0, sum_bp_gru_sgd_32_9_1 =  0.0;
	float min_fp_gru_adam_32_9_1 = 10.0, max_fp_gru_adam_32_9_1 =  0.0, mean_fp_gru_adam_32_9_1 =  0.0, sum_fp_gru_adam_32_9_1 =  0.0;
	float min_bp_gru_adam_32_9_1 = 10.0, max_bp_gru_adam_32_9_1 =  0.0, mean_bp_gru_adam_32_9_1 =  0.0, sum_bp_gru_adam_32_9_1 =  0.0;
	float min_fp_gru_adamax_32_9_1 = 10.0, max_fp_gru_adamax_32_9_1 =  0.0, mean_fp_gru_adamax_32_9_1 =  0.0, sum_fp_gru_adamax_32_9_1 =  0.0;
	float min_bp_gru_adamax_32_9_1 = 10.0, max_bp_gru_adamax_32_9_1 =  0.0, mean_bp_gru_adamax_32_9_1 =  0.0, sum_bp_gru_adamax_32_9_1 =  0.0;
	
	//static lstm my_lstm_sgd_64_9_1;
	//static lstm my_lstm_adam_64_9_1;
	//static lstm my_lstm_adamax_64_9_1;
	//float min_fp_lstm_sgd_64_9_1 = 10.0, max_fp_lstm_sgd_64_9_1 =  0.0, mean_fp_lstm_sgd_64_9_1 =  0.0, sum_fp_lstm_sgd_64_9_1 =  0.0;
	//float min_bp_lstm_sgd_64_9_1 = 10.0, max_bp_lstm_sgd_64_9_1 =  0.0, mean_bp_lstm_sgd_64_9_1 =  0.0, sum_bp_lstm_sgd_64_9_1 =  0.0;
	//float min_fp_lstm_adam_64_9_1 = 10.0, max_fp_lstm_adam_64_9_1 =  0.0, mean_fp_lstm_adam_64_9_1 =  0.0, sum_fp_lstm_adam_64_9_1 =  0.0;
	//float min_bp_lstm_adam_64_9_1 = 10.0, max_bp_lstm_adam_64_9_1 =  0.0, mean_bp_lstm_adam_64_9_1 =  0.0, sum_bp_lstm_adam_64_9_1 =  0.0;
	//float min_fp_lstm_adamax_64_9_1 = 10.0, max_fp_lstm_adamax_64_9_1 =  0.0, mean_fp_lstm_adamax_64_9_1 =  0.0, sum_fp_lstm_adamax_64_9_1 =  0.0;
	//float min_bp_lstm_adamax_64_9_1 = 10.0, max_bp_lstm_adamax_64_9_1 =  0.0, mean_bp_lstm_adamax_64_9_1 =  0.0, sum_bp_lstm_adamax_64_9_1 =  0.0;
	
	//static gru my_gru_sgd_64_9_1;
	//static gru my_gru_adam_64_9_1;
	//static gru my_gru_adamax_64_9_1;
	//float min_fp_gru_sgd_64_9_1 = 10.0, max_fp_gru_sgd_64_9_1 =  0.0, mean_fp_gru_sgd_64_9_1 =  0.0, sum_fp_gru_sgd_64_9_1 =  0.0;
	//float min_bp_gru_sgd_64_9_1 = 10.0, max_bp_gru_sgd_64_9_1 =  0.0, mean_bp_gru_sgd_64_9_1 =  0.0, sum_bp_gru_sgd_64_9_1 =  0.0;
	//float min_fp_gru_adam_64_9_1 = 10.0, max_fp_gru_adam_64_9_1 =  0.0, mean_fp_gru_adam_64_9_1 =  0.0, sum_fp_gru_adam_64_9_1 =  0.0;
	//float min_bp_gru_adam_64_9_1 = 10.0, max_bp_gru_adam_64_9_1 =  0.0, mean_bp_gru_adam_64_9_1 =  0.0, sum_bp_gru_adam_64_9_1 =  0.0;
	//float min_fp_gru_adamax_64_9_1 = 10.0, max_fp_gru_adamax_64_9_1 =  0.0, mean_fp_gru_adamax_64_9_1 =  0.0, sum_fp_gru_adamax_64_9_1 =  0.0;
	//float min_bp_gru_adamax_64_9_1 = 10.0, max_bp_gru_adamax_64_9_1 =  0.0, mean_bp_gru_adamax_64_9_1 =  0.0, sum_bp_gru_adamax_64_9_1 =  0.0;
	
	int i1,i2,i3,i4,i5,i6,i7,i8;
	int first_epoch;
	int logID;
	int nb_test_batches = 0, nb_train_batches = 0;
	float min_fp = 10.0;
	float max_fp =  0.0;
	float mean_fp =  0.0;
	float stddev_fp =  0.0;
	float sum_fp =  0.0;
	float min_bp = 10.0;
	float max_bp =  0.0;
	float mean_bp =  0.0;
	float stddev_bp =  0.0;
	float sum_bp =  0.0;
	
	std::string path_to_smarties_nn = "../";
	std::string log1_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_01.csv";
	std::string log2_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_02.csv";
	std::string log3_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_03.csv";
	std::string log4_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_04.csv";
	std::string log5_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_05.csv";
	std::string log6_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_06.csv";
	std::string log7_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_07.csv";
	std::string log8_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_08.csv";
	std::string log9_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_09.csv";
	std::string log10_path = path_to_smarties_nn + "datasets/px4/px4_quad_log_10.csv";
	std::string log11_path = path_to_smarties_nn + "datasets/px4/px4_quad_log_11.csv";
	std::string reportfilename = "report.csv";
	std::string optimizer;
	
	//training variables
	std::vector<std::vector<std::vector<float>>> train_input;
	std::vector<std::vector<std::vector<float>>> train_target;
	std::vector<std::vector<std::vector<float>>> train_output;
	std::vector<uint32_t> logs;

	//testing variables
	std::vector<std::vector<std::vector<float>>> test_input;
	std::vector<std::vector<std::vector<float>>> test_target;
	std::vector<std::vector<std::vector<float>>> test_output;
	float test_loss;
	
	//hyperparameters
	
	my_log_handler_16.load_log(log1_path.c_str());
	my_log_handler_16.update_adds();
	my_log_handler_16.normalize();
	my_log_handler_32.load_log(log1_path.c_str());
	my_log_handler_32.update_adds();
	my_log_handler_32.normalize();
	my_log_handler_64.load_log(log1_path.c_str());
	my_log_handler_64.update_adds();
	my_log_handler_64.normalize();
	

	my_log_handlers[0].load_log(log1_path.c_str());
	my_log_handlers[0].update_adds();
	my_log_handlers[0].normalize();
//	my_log_handlers[0].create_report(reportfilename.c_str());

	my_log_handlers[1].load_log(log2_path.c_str());
	my_log_handlers[1].update_adds();
	my_log_handlers[1].normalize();
//	my_log_handlers[1].create_report(reportfilename.c_str());

	my_log_handlers[2].load_log(log3_path.c_str());
	my_log_handlers[2].update_adds();
	my_log_handlers[2].normalize();
//	my_log_handlers[2].create_report(reportfilename.c_str());

	my_log_handlers[3].load_log(log4_path.c_str());
	my_log_handlers[3].update_adds();
	my_log_handlers[3].normalize();
//	my_log_handlers[3].create_report(reportfilename.c_str());

	my_log_handlers[4].load_log(log5_path.c_str());
	my_log_handlers[4].update_adds();
	my_log_handlers[4].normalize();
//	my_log_handlers[4].create_report(reportfilename.c_str());

	my_log_handlers[5].load_log(log6_path.c_str());
	my_log_handlers[5].update_adds();
	my_log_handlers[5].normalize();
//	my_log_handlers[5].create_report(reportfilename.c_str());

	my_log_handlers[6].load_log(log7_path.c_str());
	my_log_handlers[6].update_adds();
	my_log_handlers[6].normalize();
//	my_log_handlers[6].create_report(reportfilename.c_str());

	my_log_handlers[7].load_log(log8_path.c_str());
	my_log_handlers[7].update_adds();
	my_log_handlers[7].normalize();
//	my_log_handlers[7].create_report(reportfilename.c_str());

	my_log_handlers[8].load_log(log9_path.c_str());
	my_log_handlers[8].update_adds();
	my_log_handlers[8].normalize();
//	my_log_handlers[8].create_report(reportfilename.c_str());

	my_log_handlers[9].load_log(log10_path.c_str());
	my_log_handlers[9].update_adds();
	my_log_handlers[9].normalize();
//	my_log_handlers[9].create_report(reportfilename.c_str());

	my_log_handlers[10].load_log11(log11_path.c_str());
	my_log_handlers[10].update_adds();
	my_log_handlers[10].normalize();
//	my_log_handlers[10].create_report(reportfilename.c_str());

	logs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	
	std::string line;
	int nb_logs = logs.size();
	char net_filename[50], logResult[50];
	std::string filenameExtension = ""; //use to mark version for example
	uint32_t input_size  = 9;
	uint32_t output_size;// = 1;
	
	//my_lstm_sgd_16_9_3.init(16,9,3);
	//my_lstm_sgd_16_9_3.set_optimizer(1); // sgd
	//my_lstm_adam_16_9_3.init(16,9,3);
	//my_lstm_adam_16_9_3.set_optimizer(2); // adam
	//my_lstm_adamax_16_9_3.init(16,9,3);
	//my_lstm_adamax_16_9_3.set_optimizer(3); // adam
	my_lstm_sgd_32_9_3.init(32,9,3);
	my_lstm_sgd_32_9_3.set_optimizer(1); // sgd
	my_lstm_adam_32_9_3.init(32,9,3);
	my_lstm_adam_32_9_3.set_optimizer(2); // adam
	my_lstm_adamax_32_9_3.init(32,9,3);
	my_lstm_adamax_32_9_3.set_optimizer(3); // adam
	//my_lstm_sgd_64_9_3.init(64,9,3);
	//my_lstm_sgd_64_9_3.set_optimizer(1); // sgd
	//my_lstm_adam_64_9_3.init(64,9,3);
	//my_lstm_adam_64_9_3.set_optimizer(2); // adam
	//my_lstm_adamax_64_9_3.init(64,9,3);
	//my_lstm_adamax_64_9_3.set_optimizer(3); // adam
	//my_gru_sgd_16_9_3.init(16,9,3);
	//my_gru_sgd_16_9_3.set_optimizer(1); // sgd
	//my_gru_adam_16_9_3.init(16,9,3);
	//my_gru_adam_16_9_3.set_optimizer(2); // adam
	//my_gru_adamax_16_9_3.init(16,9,3);
	//my_gru_adamax_16_9_3.set_optimizer(3); // adam
	my_gru_sgd_32_9_3.init(32,9,3);
	my_gru_sgd_32_9_3.set_optimizer(1); // sgd
	my_gru_adam_32_9_3.init(32,9,3);
	my_gru_adam_32_9_3.set_optimizer(2); // adam
	my_gru_adamax_32_9_3.init(32,9,3);
	my_gru_adamax_32_9_3.set_optimizer(3); // adam
	//my_gru_sgd_64_9_3.init(64,9,3);
	//my_gru_sgd_64_9_3.set_optimizer(1); // sgd
	//my_gru_adam_64_9_3.init(64,9,3);
	//my_gru_adam_64_9_3.set_optimizer(2); // adam
	//my_gru_adamax_64_9_3.init(64,9,3);
	//my_gru_adamax_64_9_3.set_optimizer(3); // adam
	
	//my_lstm_sgd_16_9_1.init(16,9,1);
	//my_lstm_sgd_16_9_1.set_optimizer(1); // sgd
	//my_lstm_adam_16_9_1.init(16,9,1);
	//my_lstm_adam_16_9_1.set_optimizer(2); // adam
	//my_lstm_adamax_16_9_1.init(16,9,1);
	//my_lstm_adamax_16_9_1.set_optimizer(3); // adam
	my_lstm_sgd_32_9_1.init(32,9,1);
	my_lstm_sgd_32_9_1.set_optimizer(1); // sgd
	my_lstm_adam_32_9_1.init(32,9,1);
	my_lstm_adam_32_9_1.set_optimizer(2); // adam
	my_lstm_adamax_32_9_1.init(32,9,1);
	my_lstm_adamax_32_9_1.set_optimizer(3); // adam
	//my_lstm_sgd_64_9_1.init(64,9,1);
	//my_lstm_sgd_64_9_1.set_optimizer(1); // sgd
	//my_lstm_adam_64_9_1.init(64,9,1);
	//my_lstm_adam_64_9_1.set_optimizer(2); // adam
	//my_lstm_adamax_64_9_1.init(64,9,1);
	//my_lstm_adamax_64_9_1.set_optimizer(3); // adam
	//my_gru_sgd_16_9_1.init(16,9,1);
	//my_gru_sgd_16_9_1.set_optimizer(1); // sgd
	//my_gru_adam_16_9_1.init(16,9,1);
	//my_gru_adam_16_9_1.set_optimizer(2); // adam
	//my_gru_adamax_16_9_1.init(16,9,1);
	//my_gru_adamax_16_9_1.set_optimizer(3); // adam
	my_gru_sgd_32_9_1.init(32,9,1);
	my_gru_sgd_32_9_1.set_optimizer(1); // sgd
	my_gru_adam_32_9_1.init(32,9,1);
	my_gru_adam_32_9_1.set_optimizer(2); // adam
	my_gru_adamax_32_9_1.init(32,9,1);
	my_gru_adamax_32_9_1.set_optimizer(3); // adam
	//my_gru_sgd_64_9_1.init(64,9,1);
	//my_gru_sgd_64_9_1.set_optimizer(1); // sgd
	//my_gru_adam_64_9_1.init(64,9,1);
	//my_gru_adam_64_9_1.set_optimizer(2); // adam
	//my_gru_adamax_64_9_1.init(64,9,1);
	//my_gru_adamax_64_9_1.set_optimizer(3); // adam
	
	float train_loss = 0.0;
	
	my_log_handler_16.create_random_dataset(sample_nb, 16);
	my_log_handler_32.create_random_dataset(sample_nb, 32);
	my_log_handler_64.create_random_dataset(sample_nb, 64);

	
	// -----------------------------------------------------------------------------------------------------------------------------
	// TRAINING
	// -----------------------------------------------------------------------------------------------------------------------------
	
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "---------------------- TRAINING PERFORMANCES --------------------------" << std::endl;
	std::cout << "------------ Clear Momumtum/Feedforward/Backpropagation ---------------" << std::endl;
	std::cout << "------------------------100 iterations --------------------------------" << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;

	logID = 1; //logs[0];
	
	
	//max_samples = my_log_handler_16.get_max_train_samples();
	max_samples = 100;
	//std::cout << "------------ Samples_nb_16_timestep = " << max_samples << " ---------------" << std::endl;
	//std::cout << " A " << std::endl;
	//train_input  = my_log_handler_16.get_input_rdm_train(max_samples, 16, 9);
	//train_target = my_log_handler_16.get_target_rdm_train(max_samples, 16, 3);
	//std::cout << " A " << std::endl;
	//my_lstm_sgd_16_9_3.clear_grads();
	//my_lstm_adam_16_9_3.clear_grads();
	//my_lstm_adamax_16_9_3.clear_grads();
	//my_gru_sgd_16_9_3.clear_grads();
	//my_gru_adam_16_9_3.clear_grads();
	//my_gru_adamax_16_9_3.clear_grads();
	
	//std::cout << " A " << std::endl;

	//nb_train_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_16_9_3
		//auto start_bp_lstm_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_16_9_3.clear_momentums();
		//my_lstm_sgd_16_9_3.forward(train_input[i2]);
		//my_lstm_sgd_16_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_16_9_3 = finish_bp_lstm_sgd_16_9_3-start_bp_lstm_sgd_16_9_3;
		//float elapsed_ms_bp_lstm_sgd_16_9_3 = elapsed_bp_lstm_sgd_16_9_3.count();
		//if (elapsed_ms_bp_lstm_sgd_16_9_3<min_bp_lstm_sgd_16_9_3) min_bp_lstm_sgd_16_9_3 = elapsed_ms_bp_lstm_sgd_16_9_3;
		//if (elapsed_ms_bp_lstm_sgd_16_9_3>max_bp_lstm_sgd_16_9_3) max_bp_lstm_sgd_16_9_3 = elapsed_ms_bp_lstm_sgd_16_9_3;
		//sum_bp_lstm_sgd_16_9_3 = sum_bp_lstm_sgd_16_9_3 + elapsed_ms_bp_lstm_sgd_16_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adam_16_9_3
		//auto start_bp_lstm_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_16_9_3.clear_momentums();
		//my_lstm_adam_16_9_3.forward(train_input[i2]);
		//my_lstm_adam_16_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_16_9_3 = finish_bp_lstm_adam_16_9_3-start_bp_lstm_adam_16_9_3;
		//float elapsed_ms_bp_lstm_adam_16_9_3 = elapsed_bp_lstm_adam_16_9_3.count();
		//if (elapsed_ms_bp_lstm_adam_16_9_3<min_bp_lstm_adam_16_9_3) min_bp_lstm_adam_16_9_3 = elapsed_ms_bp_lstm_adam_16_9_3;
		//if (elapsed_ms_bp_lstm_adam_16_9_3>max_bp_lstm_adam_16_9_3) max_bp_lstm_adam_16_9_3 = elapsed_ms_bp_lstm_adam_16_9_3;
		//sum_bp_lstm_adam_16_9_3 = sum_bp_lstm_adam_16_9_3 + elapsed_ms_bp_lstm_adam_16_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_16_9_3
		//auto start_bp_lstm_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_16_9_3.clear_momentums();
		//my_lstm_adamax_16_9_3.forward(train_input[i2]);
		//my_lstm_adamax_16_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_16_9_3 = finish_bp_lstm_adamax_16_9_3-start_bp_lstm_adamax_16_9_3;
		//float elapsed_ms_bp_lstm_adamax_16_9_3 = elapsed_bp_lstm_adamax_16_9_3.count();
		//if (elapsed_ms_bp_lstm_adamax_16_9_3<min_bp_lstm_adamax_16_9_3) min_bp_lstm_adamax_16_9_3 = elapsed_ms_bp_lstm_adamax_16_9_3;
		//if (elapsed_ms_bp_lstm_adamax_16_9_3>max_bp_lstm_adamax_16_9_3) max_bp_lstm_adamax_16_9_3 = elapsed_ms_bp_lstm_adamax_16_9_3;
		//sum_bp_lstm_adamax_16_9_3 = sum_bp_lstm_adamax_16_9_3 + elapsed_ms_bp_lstm_adamax_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_sgd_16_9_3
		//auto start_bp_gru_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_16_9_3.clear_momentums();
		//my_gru_sgd_16_9_3.forward(train_input[i2]);
		//my_gru_sgd_16_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_16_9_3 = finish_bp_gru_sgd_16_9_3-start_bp_gru_sgd_16_9_3;
		//float elapsed_ms_bp_gru_sgd_16_9_3 = elapsed_bp_gru_sgd_16_9_3.count();
		//if (elapsed_ms_bp_gru_sgd_16_9_3<min_bp_gru_sgd_16_9_3) min_bp_gru_sgd_16_9_3 = elapsed_ms_bp_gru_sgd_16_9_3;
		//if (elapsed_ms_bp_gru_sgd_16_9_3>max_bp_gru_sgd_16_9_3) max_bp_gru_sgd_16_9_3 = elapsed_ms_bp_gru_sgd_16_9_3;
		//sum_bp_gru_sgd_16_9_3 = sum_bp_gru_sgd_16_9_3 + elapsed_ms_bp_gru_sgd_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adam_16_9_3
		//auto start_bp_gru_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_16_9_3.clear_momentums();
		//my_gru_adam_16_9_3.forward(train_input[i2]);
		//my_gru_adam_16_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_16_9_3 = finish_bp_gru_adam_16_9_3-start_bp_gru_adam_16_9_3;
		//float elapsed_ms_bp_gru_adam_16_9_3 = elapsed_bp_gru_adam_16_9_3.count();
		//if (elapsed_ms_bp_gru_adam_16_9_3<min_bp_gru_adam_16_9_3) min_bp_gru_adam_16_9_3 = elapsed_ms_bp_gru_adam_16_9_3;
		//if (elapsed_ms_bp_gru_adam_16_9_3>max_bp_gru_adam_16_9_3) max_bp_gru_adam_16_9_3 = elapsed_ms_bp_gru_adam_16_9_3;
		//sum_bp_gru_adam_16_9_3 = sum_bp_gru_adam_16_9_3 + elapsed_ms_bp_gru_adam_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adamax_16_9_3
		//auto start_bp_gru_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_16_9_3.clear_momentums();
		//my_gru_adamax_16_9_3.forward(train_input[i2]);
		//my_gru_adamax_16_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_16_9_3 = finish_bp_gru_adamax_16_9_3-start_bp_gru_adamax_16_9_3;
		//float elapsed_ms_bp_gru_adamax_16_9_3 = elapsed_bp_gru_adamax_16_9_3.count();
		//if (elapsed_ms_bp_gru_adamax_16_9_3<min_bp_gru_adamax_16_9_3) min_bp_gru_adamax_16_9_3 = elapsed_ms_bp_gru_adamax_16_9_3;
		//if (elapsed_ms_bp_gru_adamax_16_9_3>max_bp_gru_adamax_16_9_3) max_bp_gru_adamax_16_9_3 = elapsed_ms_bp_gru_adamax_16_9_3;
		//sum_bp_gru_adamax_16_9_3 = sum_bp_gru_adamax_16_9_3 + elapsed_ms_bp_gru_adamax_16_9_3;

		//nb_train_batches++;
	//}
	//train_loss /= nb_train_batches;
	//mean_bp_lstm_sgd_16_9_3=sum_bp_lstm_sgd_16_9_3/nb_train_batches;
	//mean_bp_lstm_adam_16_9_3=sum_bp_lstm_adam_16_9_3/nb_train_batches;
	//mean_bp_lstm_adamax_16_9_3=sum_bp_lstm_adamax_16_9_3/nb_train_batches;
	//mean_bp_gru_sgd_16_9_3=sum_bp_gru_sgd_16_9_3/nb_train_batches;
	//mean_bp_gru_adam_16_9_3=sum_bp_gru_adam_16_9_3/nb_train_batches;
	//mean_bp_gru_adamax_16_9_3=sum_bp_gru_adamax_16_9_3/nb_train_batches;
	
	//my_log_handlers[logID].randomize();
	//max_samples = my_log_handlers[logID].get_max_train_samples();
	//std::cout << "------------ Samples_nb_32_timestep = " << max_samples << " ---------------" << std::endl;
	train_input  = my_log_handler_32.get_input_rdm_train(max_samples, 32, 9);
	train_target = my_log_handler_32.get_target_rdm_train(max_samples, 32, 3);
	my_lstm_sgd_32_9_3.clear_grads();
	my_lstm_adam_32_9_3.clear_grads();
	my_lstm_adamax_32_9_3.clear_grads();
	my_gru_sgd_32_9_3.clear_grads();
	my_gru_adam_32_9_3.clear_grads();
	my_gru_adamax_32_9_3.clear_grads();

	nb_train_batches = 0;
	for (i2=0;i2<max_samples;i2++)
	{
		// -----------------------------------------------------------
		// my_lstm_sgd_32_9_3
		auto start_bp_lstm_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_sgd_32_9_3.clear_momentums();
		my_lstm_sgd_32_9_3.forward(train_input[i2]);
		my_lstm_sgd_32_9_3.backward(train_target[i2]);
		auto finish_bp_lstm_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_32_9_3 = finish_bp_lstm_sgd_32_9_3-start_bp_lstm_sgd_32_9_3;
		float elapsed_ms_bp_lstm_sgd_32_9_3 = elapsed_bp_lstm_sgd_32_9_3.count();
		if (elapsed_ms_bp_lstm_sgd_32_9_3<min_bp_lstm_sgd_32_9_3) min_bp_lstm_sgd_32_9_3 = elapsed_ms_bp_lstm_sgd_32_9_3;
		if (elapsed_ms_bp_lstm_sgd_32_9_3>max_bp_lstm_sgd_32_9_3) max_bp_lstm_sgd_32_9_3 = elapsed_ms_bp_lstm_sgd_32_9_3;
		sum_bp_lstm_sgd_32_9_3 = sum_bp_lstm_sgd_32_9_3 + elapsed_ms_bp_lstm_sgd_32_9_3;
		// -----------------------------------------------------------
		// my_lstm_adam_32_9_3
		auto start_bp_lstm_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_adam_32_9_3.clear_momentums();
		my_lstm_adam_32_9_3.forward(train_input[i2]);
		my_lstm_adam_32_9_3.backward(train_target[i2]);
		auto finish_bp_lstm_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_32_9_3 = finish_bp_lstm_adam_32_9_3-start_bp_lstm_adam_32_9_3;
		float elapsed_ms_bp_lstm_adam_32_9_3 = elapsed_bp_lstm_adam_32_9_3.count();
		if (elapsed_ms_bp_lstm_adam_32_9_3<min_bp_lstm_adam_32_9_3) min_bp_lstm_adam_32_9_3 = elapsed_ms_bp_lstm_adam_32_9_3;
		if (elapsed_ms_bp_lstm_adam_32_9_3>max_bp_lstm_adam_32_9_3) max_bp_lstm_adam_32_9_3 = elapsed_ms_bp_lstm_adam_32_9_3;
		sum_bp_lstm_adam_32_9_3 = sum_bp_lstm_adam_32_9_3 + elapsed_ms_bp_lstm_adam_32_9_3;
		// -----------------------------------------------------------
		// my_lstm_adamax_32_9_3
		auto start_bp_lstm_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_adamax_32_9_3.clear_momentums();
		my_lstm_adamax_32_9_3.forward(train_input[i2]);
		my_lstm_adamax_32_9_3.backward(train_target[i2]);
		auto finish_bp_lstm_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_32_9_3 = finish_bp_lstm_adamax_32_9_3-start_bp_lstm_adamax_32_9_3;
		float elapsed_ms_bp_lstm_adamax_32_9_3 = elapsed_bp_lstm_adamax_32_9_3.count();
		if (elapsed_ms_bp_lstm_adamax_32_9_3<min_bp_lstm_adamax_32_9_3) min_bp_lstm_adamax_32_9_3 = elapsed_ms_bp_lstm_adamax_32_9_3;
		if (elapsed_ms_bp_lstm_adamax_32_9_3>max_bp_lstm_adamax_32_9_3) max_bp_lstm_adamax_32_9_3 = elapsed_ms_bp_lstm_adamax_32_9_3;
		sum_bp_lstm_adamax_32_9_3 = sum_bp_lstm_adamax_32_9_3 + elapsed_ms_bp_lstm_adamax_32_9_3;
		// -----------------------------------------------------------
		// my_gru_sgd_32_9_3
		auto start_bp_gru_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_sgd_32_9_3.clear_momentums();
		my_gru_sgd_32_9_3.forward(train_input[i2]);
		my_gru_sgd_32_9_3.backward(train_target[i2]);
		auto finish_bp_gru_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_32_9_3 = finish_bp_gru_sgd_32_9_3-start_bp_gru_sgd_32_9_3;
		float elapsed_ms_bp_gru_sgd_32_9_3 = elapsed_bp_gru_sgd_32_9_3.count();
		if (elapsed_ms_bp_gru_sgd_32_9_3<min_bp_gru_sgd_32_9_3) min_bp_gru_sgd_32_9_3 = elapsed_ms_bp_gru_sgd_32_9_3;
		if (elapsed_ms_bp_gru_sgd_32_9_3>max_bp_gru_sgd_32_9_3) max_bp_gru_sgd_32_9_3 = elapsed_ms_bp_gru_sgd_32_9_3;
		sum_bp_gru_sgd_32_9_3 = sum_bp_gru_sgd_32_9_3 + elapsed_ms_bp_gru_sgd_32_9_3;
		// -----------------------------------------------------------
		// my_gru_adam_32_9_3
		auto start_bp_gru_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_adam_32_9_3.clear_momentums();
		my_gru_adam_32_9_3.forward(train_input[i2]);
		my_gru_adam_32_9_3.backward(train_target[i2]);
		auto finish_bp_gru_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_32_9_3 = finish_bp_gru_adam_32_9_3-start_bp_gru_adam_32_9_3;
		float elapsed_ms_bp_gru_adam_32_9_3 = elapsed_bp_gru_adam_32_9_3.count();
		if (elapsed_ms_bp_gru_adam_32_9_3<min_bp_gru_adam_32_9_3) min_bp_gru_adam_32_9_3 = elapsed_ms_bp_gru_adam_32_9_3;
		if (elapsed_ms_bp_gru_adam_32_9_3>max_bp_gru_adam_32_9_3) max_bp_gru_adam_32_9_3 = elapsed_ms_bp_gru_adam_32_9_3;
		sum_bp_gru_adam_32_9_3 = sum_bp_gru_adam_32_9_3 + elapsed_ms_bp_gru_adam_32_9_3;
		// -----------------------------------------------------------
		// my_gru_adamax_32_9_3
		auto start_bp_gru_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_adamax_32_9_3.clear_momentums();
		my_gru_adamax_32_9_3.forward(train_input[i2]);
		my_gru_adamax_32_9_3.backward(train_target[i2]);
		auto finish_bp_gru_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_32_9_3 = finish_bp_gru_adamax_32_9_3-start_bp_gru_adamax_32_9_3;
		float elapsed_ms_bp_gru_adamax_32_9_3 = elapsed_bp_gru_adamax_32_9_3.count();
		if (elapsed_ms_bp_gru_adamax_32_9_3<min_bp_gru_adamax_32_9_3) min_bp_gru_adamax_32_9_3 = elapsed_ms_bp_gru_adamax_32_9_3;
		if (elapsed_ms_bp_gru_adamax_32_9_3>max_bp_gru_adamax_32_9_3) max_bp_gru_adamax_32_9_3 = elapsed_ms_bp_gru_adamax_32_9_3;
		sum_bp_gru_adamax_32_9_3 = sum_bp_gru_adamax_32_9_3 + elapsed_ms_bp_gru_adamax_32_9_3;

		nb_train_batches++;
	}
	train_loss /= nb_train_batches;
	mean_bp_lstm_sgd_32_9_3=sum_bp_lstm_sgd_32_9_3/nb_train_batches;
	mean_bp_lstm_adam_32_9_3=sum_bp_lstm_adam_32_9_3/nb_train_batches;
	mean_bp_lstm_adamax_32_9_3=sum_bp_lstm_adamax_32_9_3/nb_train_batches;
	mean_bp_gru_sgd_32_9_3=sum_bp_gru_sgd_32_9_3/nb_train_batches;
	mean_bp_gru_adam_32_9_3=sum_bp_gru_adam_32_9_3/nb_train_batches;
	mean_bp_gru_adamax_32_9_3=sum_bp_gru_adamax_32_9_3/nb_train_batches;
	

	//std::cout << "------------ Samples_nb_64_timestep = " << max_samples << " ---------------" << std::endl;
	//train_input  = my_log_handler_64.get_input_rdm_train(max_samples, 64, 9);
	//train_target = my_log_handler_64.get_target_rdm_train(max_samples, 64, 3);
	//my_lstm_sgd_64_9_3.clear_grads();
	//my_lstm_adam_64_9_3.clear_grads();
	//my_lstm_adamax_64_9_3.clear_grads();
	//my_gru_sgd_64_9_3.clear_grads();
	//my_gru_adam_64_9_3.clear_grads();
	//my_gru_adamax_64_9_3.clear_grads();
	//nb_train_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_64_9_3
		//auto start_bp_lstm_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_64_9_3.clear_momentums();
		//my_lstm_sgd_64_9_3.forward(train_input[i2]);
		//my_lstm_sgd_64_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_64_9_3 = finish_bp_lstm_sgd_64_9_3-start_bp_lstm_sgd_64_9_3;
		//float elapsed_ms_bp_lstm_sgd_64_9_3 = elapsed_bp_lstm_sgd_64_9_3.count();
		//if (elapsed_ms_bp_lstm_sgd_64_9_3<min_bp_lstm_sgd_64_9_3) min_bp_lstm_sgd_64_9_3 = elapsed_ms_bp_lstm_sgd_64_9_3;
		//if (elapsed_ms_bp_lstm_sgd_64_9_3>max_bp_lstm_sgd_64_9_3) max_bp_lstm_sgd_64_9_3 = elapsed_ms_bp_lstm_sgd_64_9_3;
		//sum_bp_lstm_sgd_64_9_3 = sum_bp_lstm_sgd_64_9_3 + elapsed_ms_bp_lstm_sgd_64_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adam_64_9_3
		//auto start_bp_lstm_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_64_9_3.clear_momentums();
		//my_lstm_adam_64_9_3.forward(train_input[i2]);
		//my_lstm_adam_64_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_64_9_3 = finish_bp_lstm_adam_64_9_3-start_bp_lstm_adam_64_9_3;
		//float elapsed_ms_bp_lstm_adam_64_9_3 = elapsed_bp_lstm_adam_64_9_3.count();
		//if (elapsed_ms_bp_lstm_adam_64_9_3<min_bp_lstm_adam_64_9_3) min_bp_lstm_adam_64_9_3 = elapsed_ms_bp_lstm_adam_64_9_3;
		//if (elapsed_ms_bp_lstm_adam_64_9_3>max_bp_lstm_adam_64_9_3) max_bp_lstm_adam_64_9_3 = elapsed_ms_bp_lstm_adam_64_9_3;
		//sum_bp_lstm_adam_64_9_3 = sum_bp_lstm_adam_64_9_3 + elapsed_ms_bp_lstm_adam_64_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_64_9_3
		//auto start_bp_lstm_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_64_9_3.clear_momentums();
		//my_lstm_adamax_64_9_3.forward(train_input[i2]);
		//my_lstm_adamax_64_9_3.backward(train_target[i2]);
		//auto finish_bp_lstm_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_64_9_3 = finish_bp_lstm_adamax_64_9_3-start_bp_lstm_adamax_64_9_3;
		//float elapsed_ms_bp_lstm_adamax_64_9_3 = elapsed_bp_lstm_adamax_64_9_3.count();
		//if (elapsed_ms_bp_lstm_adamax_64_9_3<min_bp_lstm_adamax_64_9_3) min_bp_lstm_adamax_64_9_3 = elapsed_ms_bp_lstm_adamax_64_9_3;
		//if (elapsed_ms_bp_lstm_adamax_64_9_3>max_bp_lstm_adamax_64_9_3) max_bp_lstm_adamax_64_9_3 = elapsed_ms_bp_lstm_adamax_64_9_3;
		//sum_bp_lstm_adamax_64_9_3 = sum_bp_lstm_adamax_64_9_3 + elapsed_ms_bp_lstm_adamax_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_sgd_64_9_3
		//auto start_bp_gru_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_64_9_3.clear_momentums();
		//my_gru_sgd_64_9_3.forward(train_input[i2]);
		//my_gru_sgd_64_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_64_9_3 = finish_bp_gru_sgd_64_9_3-start_bp_gru_sgd_64_9_3;
		//float elapsed_ms_bp_gru_sgd_64_9_3 = elapsed_bp_gru_sgd_64_9_3.count();
		//if (elapsed_ms_bp_gru_sgd_64_9_3<min_bp_gru_sgd_64_9_3) min_bp_gru_sgd_64_9_3 = elapsed_ms_bp_gru_sgd_64_9_3;
		//if (elapsed_ms_bp_gru_sgd_64_9_3>max_bp_gru_sgd_64_9_3) max_bp_gru_sgd_64_9_3 = elapsed_ms_bp_gru_sgd_64_9_3;
		//sum_bp_gru_sgd_64_9_3 = sum_bp_gru_sgd_64_9_3 + elapsed_ms_bp_gru_sgd_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adam_64_9_3
		//auto start_bp_gru_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_64_9_3.clear_momentums();
		//my_gru_adam_64_9_3.forward(train_input[i2]);
		//my_gru_adam_64_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_64_9_3 = finish_bp_gru_adam_64_9_3-start_bp_gru_adam_64_9_3;
		//float elapsed_ms_bp_gru_adam_64_9_3 = elapsed_bp_gru_adam_64_9_3.count();
		//if (elapsed_ms_bp_gru_adam_64_9_3<min_bp_gru_adam_64_9_3) min_bp_gru_adam_64_9_3 = elapsed_ms_bp_gru_adam_64_9_3;
		//if (elapsed_ms_bp_gru_adam_64_9_3>max_bp_gru_adam_64_9_3) max_bp_gru_adam_64_9_3 = elapsed_ms_bp_gru_adam_64_9_3;
		//sum_bp_gru_adam_64_9_3 = sum_bp_gru_adam_64_9_3 + elapsed_ms_bp_gru_adam_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adamax_64_9_3
		//auto start_bp_gru_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_64_9_3.clear_momentums();
		//my_gru_adamax_64_9_3.forward(train_input[i2]);
		//my_gru_adamax_64_9_3.backward(train_target[i2]);
		//auto finish_bp_gru_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_64_9_3 = finish_bp_gru_adamax_64_9_3-start_bp_gru_adamax_64_9_3;
		//float elapsed_ms_bp_gru_adamax_64_9_3 = elapsed_bp_gru_adamax_64_9_3.count();
		//if (elapsed_ms_bp_gru_adamax_64_9_3<min_bp_gru_adamax_64_9_3) min_bp_gru_adamax_64_9_3 = elapsed_ms_bp_gru_adamax_64_9_3;
		//if (elapsed_ms_bp_gru_adamax_64_9_3>max_bp_gru_adamax_64_9_3) max_bp_gru_adamax_64_9_3 = elapsed_ms_bp_gru_adamax_64_9_3;
		//sum_bp_gru_adamax_64_9_3 = sum_bp_gru_adamax_64_9_3 + elapsed_ms_bp_gru_adamax_64_9_3;

		//nb_train_batches++;
	//}
	//train_loss /= nb_train_batches;
	//mean_bp_lstm_sgd_64_9_3=sum_bp_lstm_sgd_64_9_3/nb_train_batches;
	//mean_bp_lstm_adam_64_9_3=sum_bp_lstm_adam_64_9_3/nb_train_batches;
	//mean_bp_lstm_adamax_64_9_3=sum_bp_lstm_adamax_64_9_3/nb_train_batches;
	//mean_bp_gru_sgd_64_9_3=sum_bp_gru_sgd_64_9_3/nb_train_batches;
	//mean_bp_gru_adam_64_9_3=sum_bp_gru_adam_64_9_3/nb_train_batches;
	//mean_bp_gru_adamax_64_9_3=sum_bp_gru_adamax_64_9_3/nb_train_batches;
	
	
	//max_samples = my_log_handlers[logID].get_max_train_samples();
	//train_input  = my_log_handler_16.get_input_rdm_train(max_samples, 16, 9);
	//train_target = my_log_handler_16.get_target_rdm_train(max_samples, 16, 1);
	//my_lstm_sgd_16_9_1.clear_grads();
	//my_lstm_adam_16_9_1.clear_grads();
	//my_lstm_adamax_16_9_1.clear_grads();
	//my_gru_sgd_16_9_1.clear_grads();
	//my_gru_adam_16_9_1.clear_grads();
	//my_gru_adamax_16_9_1.clear_grads();

	//nb_train_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_16_9_1
		//auto start_bp_lstm_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_16_9_1.clear_momentums();
		//my_lstm_sgd_16_9_1.forward(train_input[i2]);
		//my_lstm_sgd_16_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_16_9_1 = finish_bp_lstm_sgd_16_9_1-start_bp_lstm_sgd_16_9_1;
		//float elapsed_ms_bp_lstm_sgd_16_9_1 = elapsed_bp_lstm_sgd_16_9_1.count();
		//if (elapsed_ms_bp_lstm_sgd_16_9_1<min_bp_lstm_sgd_16_9_1) min_bp_lstm_sgd_16_9_1 = elapsed_ms_bp_lstm_sgd_16_9_1;
		//if (elapsed_ms_bp_lstm_sgd_16_9_1>max_bp_lstm_sgd_16_9_1) max_bp_lstm_sgd_16_9_1 = elapsed_ms_bp_lstm_sgd_16_9_1;
		//sum_bp_lstm_sgd_16_9_1 = sum_bp_lstm_sgd_16_9_1 + elapsed_ms_bp_lstm_sgd_16_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adam_16_9_1
		//auto start_bp_lstm_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_16_9_1.clear_momentums();
		//my_lstm_adam_16_9_1.forward(train_input[i2]);
		//my_lstm_adam_16_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_16_9_1 = finish_bp_lstm_adam_16_9_1-start_bp_lstm_adam_16_9_1;
		//float elapsed_ms_bp_lstm_adam_16_9_1 = elapsed_bp_lstm_adam_16_9_1.count();
		//if (elapsed_ms_bp_lstm_adam_16_9_1<min_bp_lstm_adam_16_9_1) min_bp_lstm_adam_16_9_1 = elapsed_ms_bp_lstm_adam_16_9_1;
		//if (elapsed_ms_bp_lstm_adam_16_9_1>max_bp_lstm_adam_16_9_1) max_bp_lstm_adam_16_9_1 = elapsed_ms_bp_lstm_adam_16_9_1;
		//sum_bp_lstm_adam_16_9_1 = sum_bp_lstm_adam_16_9_1 + elapsed_ms_bp_lstm_adam_16_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_16_9_1
		//auto start_bp_lstm_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_16_9_1.clear_momentums();
		//my_lstm_adamax_16_9_1.forward(train_input[i2]);
		//my_lstm_adamax_16_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_16_9_1 = finish_bp_lstm_adamax_16_9_1-start_bp_lstm_adamax_16_9_1;
		//float elapsed_ms_bp_lstm_adamax_16_9_1 = elapsed_bp_lstm_adamax_16_9_1.count();
		//if (elapsed_ms_bp_lstm_adamax_16_9_1<min_bp_lstm_adamax_16_9_1) min_bp_lstm_adamax_16_9_1 = elapsed_ms_bp_lstm_adamax_16_9_1;
		//if (elapsed_ms_bp_lstm_adamax_16_9_1>max_bp_lstm_adamax_16_9_1) max_bp_lstm_adamax_16_9_1 = elapsed_ms_bp_lstm_adamax_16_9_1;
		//sum_bp_lstm_adamax_16_9_1 = sum_bp_lstm_adamax_16_9_1 + elapsed_ms_bp_lstm_adamax_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_sgd_16_9_1
		//auto start_bp_gru_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_16_9_1.clear_momentums();
		//my_gru_sgd_16_9_1.forward(train_input[i2]);
		//my_gru_sgd_16_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_16_9_1 = finish_bp_gru_sgd_16_9_1-start_bp_gru_sgd_16_9_1;
		//float elapsed_ms_bp_gru_sgd_16_9_1 = elapsed_bp_gru_sgd_16_9_1.count();
		//if (elapsed_ms_bp_gru_sgd_16_9_1<min_bp_gru_sgd_16_9_1) min_bp_gru_sgd_16_9_1 = elapsed_ms_bp_gru_sgd_16_9_1;
		//if (elapsed_ms_bp_gru_sgd_16_9_1>max_bp_gru_sgd_16_9_1) max_bp_gru_sgd_16_9_1 = elapsed_ms_bp_gru_sgd_16_9_1;
		//sum_bp_gru_sgd_16_9_1 = sum_bp_gru_sgd_16_9_1 + elapsed_ms_bp_gru_sgd_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adam_16_9_1
		//auto start_bp_gru_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_16_9_1.clear_momentums();
		//my_gru_adam_16_9_1.forward(train_input[i2]);
		//my_gru_adam_16_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_16_9_1 = finish_bp_gru_adam_16_9_1-start_bp_gru_adam_16_9_1;
		//float elapsed_ms_bp_gru_adam_16_9_1 = elapsed_bp_gru_adam_16_9_1.count();
		//if (elapsed_ms_bp_gru_adam_16_9_1<min_bp_gru_adam_16_9_1) min_bp_gru_adam_16_9_1 = elapsed_ms_bp_gru_adam_16_9_1;
		//if (elapsed_ms_bp_gru_adam_16_9_1>max_bp_gru_adam_16_9_1) max_bp_gru_adam_16_9_1 = elapsed_ms_bp_gru_adam_16_9_1;
		//sum_bp_gru_adam_16_9_1 = sum_bp_gru_adam_16_9_1 + elapsed_ms_bp_gru_adam_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adamax_16_9_1
		//auto start_bp_gru_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_16_9_1.clear_momentums();
		//my_gru_adamax_16_9_1.forward(train_input[i2]);
		//my_gru_adamax_16_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_16_9_1 = finish_bp_gru_adamax_16_9_1-start_bp_gru_adamax_16_9_1;
		//float elapsed_ms_bp_gru_adamax_16_9_1 = elapsed_bp_gru_adamax_16_9_1.count();
		//if (elapsed_ms_bp_gru_adamax_16_9_1<min_bp_gru_adamax_16_9_1) min_bp_gru_adamax_16_9_1 = elapsed_ms_bp_gru_adamax_16_9_1;
		//if (elapsed_ms_bp_gru_adamax_16_9_1>max_bp_gru_adamax_16_9_1) max_bp_gru_adamax_16_9_1 = elapsed_ms_bp_gru_adamax_16_9_1;
		//sum_bp_gru_adamax_16_9_1 = sum_bp_gru_adamax_16_9_1 + elapsed_ms_bp_gru_adamax_16_9_1;

		//nb_train_batches++;
	//}
	//train_loss /= nb_train_batches;
	//mean_bp_lstm_sgd_16_9_1=sum_bp_lstm_sgd_16_9_1/nb_train_batches;
	//mean_bp_lstm_adam_16_9_1=sum_bp_lstm_adam_16_9_1/nb_train_batches;
	//mean_bp_lstm_adamax_16_9_1=sum_bp_lstm_adamax_16_9_1/nb_train_batches;
	//mean_bp_gru_sgd_16_9_1=sum_bp_gru_sgd_16_9_1/nb_train_batches;
	//mean_bp_gru_adam_16_9_1=sum_bp_gru_adam_16_9_1/nb_train_batches;
	//mean_bp_gru_adamax_16_9_1=sum_bp_gru_adamax_16_9_1/nb_train_batches;
	
	//my_log_handlers[logID].randomize();
	//max_samples = my_log_handlers[logID].get_max_train_samples();
	train_input  = my_log_handler_32.get_input_rdm_train(max_samples, 32, 9);
	train_target = my_log_handler_32.get_target_rdm_train(max_samples, 32, 1);
	my_lstm_sgd_32_9_1.clear_grads();
	my_lstm_adam_32_9_1.clear_grads();
	my_lstm_adamax_32_9_1.clear_grads();
	my_gru_sgd_32_9_1.clear_grads();
	my_gru_adam_32_9_1.clear_grads();
	my_gru_adamax_32_9_1.clear_grads();

	nb_train_batches = 0;
	for (i2=0;i2<max_samples;i2++)
	{
		// -----------------------------------------------------------
		// my_lstm_sgd_32_9_1
		auto start_bp_lstm_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_sgd_32_9_1.clear_momentums();
		my_lstm_sgd_32_9_1.forward(train_input[i2]);
		my_lstm_sgd_32_9_1.backward(train_target[i2]);
		auto finish_bp_lstm_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_32_9_1 = finish_bp_lstm_sgd_32_9_1-start_bp_lstm_sgd_32_9_1;
		float elapsed_ms_bp_lstm_sgd_32_9_1 = elapsed_bp_lstm_sgd_32_9_1.count();
		if (elapsed_ms_bp_lstm_sgd_32_9_1<min_bp_lstm_sgd_32_9_1) min_bp_lstm_sgd_32_9_1 = elapsed_ms_bp_lstm_sgd_32_9_1;
		if (elapsed_ms_bp_lstm_sgd_32_9_1>max_bp_lstm_sgd_32_9_1) max_bp_lstm_sgd_32_9_1 = elapsed_ms_bp_lstm_sgd_32_9_1;
		sum_bp_lstm_sgd_32_9_1 = sum_bp_lstm_sgd_32_9_1 + elapsed_ms_bp_lstm_sgd_32_9_1;
		// -----------------------------------------------------------
		// my_lstm_adam_32_9_1
		auto start_bp_lstm_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_adam_32_9_1.clear_momentums();
		my_lstm_adam_32_9_1.forward(train_input[i2]);
		my_lstm_adam_32_9_1.backward(train_target[i2]);
		auto finish_bp_lstm_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_32_9_1 = finish_bp_lstm_adam_32_9_1-start_bp_lstm_adam_32_9_1;
		float elapsed_ms_bp_lstm_adam_32_9_1 = elapsed_bp_lstm_adam_32_9_1.count();
		if (elapsed_ms_bp_lstm_adam_32_9_1<min_bp_lstm_adam_32_9_1) min_bp_lstm_adam_32_9_1 = elapsed_ms_bp_lstm_adam_32_9_1;
		if (elapsed_ms_bp_lstm_adam_32_9_1>max_bp_lstm_adam_32_9_1) max_bp_lstm_adam_32_9_1 = elapsed_ms_bp_lstm_adam_32_9_1;
		sum_bp_lstm_adam_32_9_1 = sum_bp_lstm_adam_32_9_1 + elapsed_ms_bp_lstm_adam_32_9_1;
		// -----------------------------------------------------------
		// my_lstm_adamax_32_9_1
		auto start_bp_lstm_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_adamax_32_9_1.clear_momentums();
		my_lstm_adamax_32_9_1.forward(train_input[i2]);
		my_lstm_adamax_32_9_1.backward(train_target[i2]);
		auto finish_bp_lstm_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_32_9_1 = finish_bp_lstm_adamax_32_9_1-start_bp_lstm_adamax_32_9_1;
		float elapsed_ms_bp_lstm_adamax_32_9_1 = elapsed_bp_lstm_adamax_32_9_1.count();
		if (elapsed_ms_bp_lstm_adamax_32_9_1<min_bp_lstm_adamax_32_9_1) min_bp_lstm_adamax_32_9_1 = elapsed_ms_bp_lstm_adamax_32_9_1;
		if (elapsed_ms_bp_lstm_adamax_32_9_1>max_bp_lstm_adamax_32_9_1) max_bp_lstm_adamax_32_9_1 = elapsed_ms_bp_lstm_adamax_32_9_1;
		sum_bp_lstm_adamax_32_9_1 = sum_bp_lstm_adamax_32_9_1 + elapsed_ms_bp_lstm_adamax_32_9_1;
		// -----------------------------------------------------------
		// my_gru_sgd_32_9_1
		auto start_bp_gru_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_sgd_32_9_1.clear_momentums();
		my_gru_sgd_32_9_1.forward(train_input[i2]);
		my_gru_sgd_32_9_1.backward(train_target[i2]);
		auto finish_bp_gru_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_32_9_1 = finish_bp_gru_sgd_32_9_1-start_bp_gru_sgd_32_9_1;
		float elapsed_ms_bp_gru_sgd_32_9_1 = elapsed_bp_gru_sgd_32_9_1.count();
		if (elapsed_ms_bp_gru_sgd_32_9_1<min_bp_gru_sgd_32_9_1) min_bp_gru_sgd_32_9_1 = elapsed_ms_bp_gru_sgd_32_9_1;
		if (elapsed_ms_bp_gru_sgd_32_9_1>max_bp_gru_sgd_32_9_1) max_bp_gru_sgd_32_9_1 = elapsed_ms_bp_gru_sgd_32_9_1;
		sum_bp_gru_sgd_32_9_1 = sum_bp_gru_sgd_32_9_1 + elapsed_ms_bp_gru_sgd_32_9_1;
		// -----------------------------------------------------------
		// my_gru_adam_32_9_1
		auto start_bp_gru_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_adam_32_9_1.clear_momentums();
		my_gru_adam_32_9_1.forward(train_input[i2]);
		my_gru_adam_32_9_1.backward(train_target[i2]);
		auto finish_bp_gru_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_32_9_1 = finish_bp_gru_adam_32_9_1-start_bp_gru_adam_32_9_1;
		float elapsed_ms_bp_gru_adam_32_9_1 = elapsed_bp_gru_adam_32_9_1.count();
		if (elapsed_ms_bp_gru_adam_32_9_1<min_bp_gru_adam_32_9_1) min_bp_gru_adam_32_9_1 = elapsed_ms_bp_gru_adam_32_9_1;
		if (elapsed_ms_bp_gru_adam_32_9_1>max_bp_gru_adam_32_9_1) max_bp_gru_adam_32_9_1 = elapsed_ms_bp_gru_adam_32_9_1;
		sum_bp_gru_adam_32_9_1 = sum_bp_gru_adam_32_9_1 + elapsed_ms_bp_gru_adam_32_9_1;
		// -----------------------------------------------------------
		// my_gru_adamax_32_9_1
		auto start_bp_gru_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_adamax_32_9_1.clear_momentums();
		my_gru_adamax_32_9_1.forward(train_input[i2]);
		my_gru_adamax_32_9_1.backward(train_target[i2]);
		auto finish_bp_gru_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_32_9_1 = finish_bp_gru_adamax_32_9_1-start_bp_gru_adamax_32_9_1;
		float elapsed_ms_bp_gru_adamax_32_9_1 = elapsed_bp_gru_adamax_32_9_1.count();
		if (elapsed_ms_bp_gru_adamax_32_9_1<min_bp_gru_adamax_32_9_1) min_bp_gru_adamax_32_9_1 = elapsed_ms_bp_gru_adamax_32_9_1;
		if (elapsed_ms_bp_gru_adamax_32_9_1>max_bp_gru_adamax_32_9_1) max_bp_gru_adamax_32_9_1 = elapsed_ms_bp_gru_adamax_32_9_1;
		sum_bp_gru_adamax_32_9_1 = sum_bp_gru_adamax_32_9_1 + elapsed_ms_bp_gru_adamax_32_9_1;

		nb_train_batches++;
	}
	train_loss /= nb_train_batches;
	mean_bp_lstm_sgd_32_9_1=sum_bp_lstm_sgd_32_9_1/nb_train_batches;
	mean_bp_lstm_adam_32_9_1=sum_bp_lstm_adam_32_9_1/nb_train_batches;
	mean_bp_lstm_adamax_32_9_1=sum_bp_lstm_adamax_32_9_1/nb_train_batches;
	mean_bp_gru_sgd_32_9_1=sum_bp_gru_sgd_32_9_1/nb_train_batches;
	mean_bp_gru_adam_32_9_1=sum_bp_gru_adam_32_9_1/nb_train_batches;
	mean_bp_gru_adamax_32_9_1=sum_bp_gru_adamax_32_9_1/nb_train_batches;
	


	//train_input  = my_log_handler_64.get_input_rdm_train(max_samples, 64, 9);
	//train_target = my_log_handler_64.get_target_rdm_train(max_samples, 64, 3);
	//my_lstm_sgd_64_9_1.clear_grads();
	//my_lstm_adam_64_9_1.clear_grads();
	//my_lstm_adamax_64_9_1.clear_grads();
	//my_gru_sgd_64_9_1.clear_grads();
	//my_gru_adam_64_9_1.clear_grads();
	//my_gru_adamax_64_9_1.clear_grads();
	//nb_train_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_64_9_1
		//auto start_bp_lstm_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_64_9_1.clear_momentums();
		//my_lstm_sgd_64_9_1.forward(train_input[i2]);
		//my_lstm_sgd_64_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd_64_9_1 = finish_bp_lstm_sgd_64_9_1-start_bp_lstm_sgd_64_9_1;
		//float elapsed_ms_bp_lstm_sgd_64_9_1 = elapsed_bp_lstm_sgd_64_9_1.count();
		//if (elapsed_ms_bp_lstm_sgd_64_9_1<min_bp_lstm_sgd_64_9_1) min_bp_lstm_sgd_64_9_1 = elapsed_ms_bp_lstm_sgd_64_9_1;
		//if (elapsed_ms_bp_lstm_sgd_64_9_1>max_bp_lstm_sgd_64_9_1) max_bp_lstm_sgd_64_9_1 = elapsed_ms_bp_lstm_sgd_64_9_1;
		//sum_bp_lstm_sgd_64_9_1 = sum_bp_lstm_sgd_64_9_1 + elapsed_ms_bp_lstm_sgd_64_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adam_64_9_1
		//auto start_bp_lstm_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_64_9_1.clear_momentums();
		//my_lstm_adam_64_9_1.forward(train_input[i2]);
		//my_lstm_adam_64_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam_64_9_1 = finish_bp_lstm_adam_64_9_1-start_bp_lstm_adam_64_9_1;
		//float elapsed_ms_bp_lstm_adam_64_9_1 = elapsed_bp_lstm_adam_64_9_1.count();
		//if (elapsed_ms_bp_lstm_adam_64_9_1<min_bp_lstm_adam_64_9_1) min_bp_lstm_adam_64_9_1 = elapsed_ms_bp_lstm_adam_64_9_1;
		//if (elapsed_ms_bp_lstm_adam_64_9_1>max_bp_lstm_adam_64_9_1) max_bp_lstm_adam_64_9_1 = elapsed_ms_bp_lstm_adam_64_9_1;
		//sum_bp_lstm_adam_64_9_1 = sum_bp_lstm_adam_64_9_1 + elapsed_ms_bp_lstm_adam_64_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_64_9_1
		//auto start_bp_lstm_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_64_9_1.clear_momentums();
		//my_lstm_adamax_64_9_1.forward(train_input[i2]);
		//my_lstm_adamax_64_9_1.backward(train_target[i2]);
		//auto finish_bp_lstm_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_lstm_adamax_64_9_1 = finish_bp_lstm_adamax_64_9_1-start_bp_lstm_adamax_64_9_1;
		//float elapsed_ms_bp_lstm_adamax_64_9_1 = elapsed_bp_lstm_adamax_64_9_1.count();
		//if (elapsed_ms_bp_lstm_adamax_64_9_1<min_bp_lstm_adamax_64_9_1) min_bp_lstm_adamax_64_9_1 = elapsed_ms_bp_lstm_adamax_64_9_1;
		//if (elapsed_ms_bp_lstm_adamax_64_9_1>max_bp_lstm_adamax_64_9_1) max_bp_lstm_adamax_64_9_1 = elapsed_ms_bp_lstm_adamax_64_9_1;
		//sum_bp_lstm_adamax_64_9_1 = sum_bp_lstm_adamax_64_9_1 + elapsed_ms_bp_lstm_adamax_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_sgd_64_9_1
		//auto start_bp_gru_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_64_9_1.clear_momentums();
		//my_gru_sgd_64_9_1.forward(train_input[i2]);
		//my_gru_sgd_64_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd_64_9_1 = finish_bp_gru_sgd_64_9_1-start_bp_gru_sgd_64_9_1;
		//float elapsed_ms_bp_gru_sgd_64_9_1 = elapsed_bp_gru_sgd_64_9_1.count();
		//if (elapsed_ms_bp_gru_sgd_64_9_1<min_bp_gru_sgd_64_9_1) min_bp_gru_sgd_64_9_1 = elapsed_ms_bp_gru_sgd_64_9_1;
		//if (elapsed_ms_bp_gru_sgd_64_9_1>max_bp_gru_sgd_64_9_1) max_bp_gru_sgd_64_9_1 = elapsed_ms_bp_gru_sgd_64_9_1;
		//sum_bp_gru_sgd_64_9_1 = sum_bp_gru_sgd_64_9_1 + elapsed_ms_bp_gru_sgd_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adam_64_9_1
		//auto start_bp_gru_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_64_9_1.clear_momentums();
		//my_gru_adam_64_9_1.forward(train_input[i2]);
		//my_gru_adam_64_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adam_64_9_1 = finish_bp_gru_adam_64_9_1-start_bp_gru_adam_64_9_1;
		//float elapsed_ms_bp_gru_adam_64_9_1 = elapsed_bp_gru_adam_64_9_1.count();
		//if (elapsed_ms_bp_gru_adam_64_9_1<min_bp_gru_adam_64_9_1) min_bp_gru_adam_64_9_1 = elapsed_ms_bp_gru_adam_64_9_1;
		//if (elapsed_ms_bp_gru_adam_64_9_1>max_bp_gru_adam_64_9_1) max_bp_gru_adam_64_9_1 = elapsed_ms_bp_gru_adam_64_9_1;
		//sum_bp_gru_adam_64_9_1 = sum_bp_gru_adam_64_9_1 + elapsed_ms_bp_gru_adam_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adamax_64_9_1
		//auto start_bp_gru_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_64_9_1.clear_momentums();
		//my_gru_adamax_64_9_1.forward(train_input[i2]);
		//my_gru_adamax_64_9_1.backward(train_target[i2]);
		//auto finish_bp_gru_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_bp_gru_adamax_64_9_1 = finish_bp_gru_adamax_64_9_1-start_bp_gru_adamax_64_9_1;
		//float elapsed_ms_bp_gru_adamax_64_9_1 = elapsed_bp_gru_adamax_64_9_1.count();
		//if (elapsed_ms_bp_gru_adamax_64_9_1<min_bp_gru_adamax_64_9_1) min_bp_gru_adamax_64_9_1 = elapsed_ms_bp_gru_adamax_64_9_1;
		//if (elapsed_ms_bp_gru_adamax_64_9_1>max_bp_gru_adamax_64_9_1) max_bp_gru_adamax_64_9_1 = elapsed_ms_bp_gru_adamax_64_9_1;
		//sum_bp_gru_adamax_64_9_1 = sum_bp_gru_adamax_64_9_1 + elapsed_ms_bp_gru_adamax_64_9_1;

		//nb_train_batches++;
	//}
	//train_loss /= nb_train_batches;
	//mean_bp_lstm_sgd_64_9_1=sum_bp_lstm_sgd_64_9_1/nb_train_batches;
	//mean_bp_lstm_adam_64_9_1=sum_bp_lstm_adam_64_9_1/nb_train_batches;
	//mean_bp_lstm_adamax_64_9_1=sum_bp_lstm_adamax_64_9_1/nb_train_batches;
	//mean_bp_gru_sgd_64_9_1=sum_bp_gru_sgd_64_9_1/nb_train_batches;
	//mean_bp_gru_adam_64_9_1=sum_bp_gru_adam_64_9_1/nb_train_batches;
	//mean_bp_gru_adamax_64_9_1=sum_bp_gru_adamax_64_9_1/nb_train_batches;
	
	std::cout << "type | min_bp | max_bp | mean_bp" << std::endl;
	std::cout << "my_lstm_sgd_32_9_1 | " << min_bp_lstm_sgd_32_9_1 << " | " << max_bp_lstm_sgd_32_9_1 << " | " << mean_bp_lstm_sgd_32_9_1 << std::endl;
	std::cout << "my_lstm_adam_32_9_1 | " << min_bp_lstm_adam_32_9_1 << " | " << max_bp_lstm_adam_32_9_1 << " | " << mean_bp_lstm_adam_32_9_1 << std::endl;
	std::cout << "my_lstm_adamax_32_9_1 | " << min_bp_lstm_adamax_32_9_1 << " | " << max_bp_lstm_adamax_32_9_1 << " | " << mean_bp_lstm_adamax_32_9_1 << std::endl;
	std::cout << "my_gru_sgd_32_9_1 | " << min_bp_gru_sgd_32_9_1 << " | " << max_bp_gru_sgd_32_9_1 << " | " << mean_bp_gru_sgd_32_9_1 << std::endl;
	std::cout << "my_gru_adam_32_9_1 | " << min_bp_gru_adam_32_9_1 << " | " << max_bp_gru_adam_32_9_1 << " | " << mean_bp_gru_adam_32_9_1 << std::endl;
	std::cout << "my_gru_adamax_32_9_1 | " << min_bp_gru_adamax_32_9_1 << " | " << max_bp_gru_adamax_32_9_1 << " | " << mean_bp_gru_adamax_32_9_1 << std::endl;
	std::cout << "my_lstm_sgd_32_9_3 | " << min_bp_lstm_sgd_32_9_3 << " | " << max_bp_lstm_sgd_32_9_3 << " | " << mean_bp_lstm_sgd_32_9_3 << std::endl;
	std::cout << "my_lstm_adam_32_9_3 | " << min_bp_lstm_adam_32_9_3 << " | " << max_bp_lstm_adam_32_9_3 << " | " << mean_bp_lstm_adam_32_9_3 << std::endl;
	std::cout << "my_lstm_adamax_32_9_3 | " << min_bp_lstm_adamax_32_9_3 << " | " << max_bp_lstm_adamax_32_9_3 << " | " << mean_bp_lstm_adamax_32_9_3 << std::endl;
	std::cout << "my_gru_sgd_32_9_3 | " << min_bp_gru_sgd_32_9_3 << " | " << max_bp_gru_sgd_32_9_3 << " | " << mean_bp_gru_sgd_32_9_3 << std::endl;
	std::cout << "my_gru_adam_32_9_3 | " << min_bp_gru_adam_32_9_3 << " | " << max_bp_gru_adam_32_9_3 << " | " << mean_bp_gru_adam_32_9_3 << std::endl;
	std::cout << "my_gru_adamax_32_9_3 | " << min_bp_gru_adamax_32_9_3 << " | " << max_bp_gru_adamax_32_9_3 << " | " << mean_bp_gru_adamax_32_9_3 << std::endl;
	
	//std::cout << "-----------------------------------------------------------------------" << std::endl;
	//std::cout << "my_lstm_sgd_16_9_1 ( min = " << min_bp_lstm_sgd_16_9_1 << " | max = " << max_bp_lstm_sgd_16_9_1 << " | mean = " << mean_bp_lstm_sgd_16_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_16_9_1 ( min = " << min_bp_lstm_adam_16_9_1 << " | max = " << max_bp_lstm_adam_16_9_1 << " | mean = " << mean_bp_lstm_adam_16_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_16_9_1 ( min = " << min_bp_lstm_adamax_16_9_1 << " | max = " << max_bp_lstm_adamax_16_9_1 << " | mean = " << mean_bp_lstm_adamax_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_16_9_1 ( min = " << min_bp_gru_sgd_16_9_1 << " | max = " << max_bp_gru_sgd_16_9_1 << " | mean = " << mean_bp_gru_sgd_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_16_9_1 ( min = " << min_bp_gru_adam_16_9_1 << " | max = " << max_bp_gru_adam_16_9_1 << " | mean = " << mean_bp_gru_adam_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_16_9_1 ( min = " << min_bp_gru_adamax_16_9_1 << " | max = " << max_bp_gru_adamax_16_9_1 << " | mean = " << mean_bp_gru_adamax_16_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_32_9_1 ( min = " << min_bp_lstm_sgd_32_9_1 << " | max = " << max_bp_lstm_sgd_32_9_1 << " | mean = " << mean_bp_lstm_sgd_32_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_32_9_1 ( min = " << min_bp_lstm_adam_32_9_1 << " | max = " << max_bp_lstm_adam_32_9_1 << " | mean = " << mean_bp_lstm_adam_32_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_32_9_1 ( min = " << min_bp_lstm_adamax_32_9_1 << " | max = " << max_bp_lstm_adamax_32_9_1 << " | mean = " << mean_bp_lstm_adamax_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_32_9_1 ( min = " << min_bp_gru_sgd_32_9_1 << " | max = " << max_bp_gru_sgd_32_9_1 << " | mean = " << mean_bp_gru_sgd_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_32_9_1 ( min = " << min_bp_gru_adam_32_9_1 << " | max = " << max_bp_gru_adam_32_9_1 << " | mean = " << mean_bp_gru_adam_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_32_9_1 ( min = " << min_bp_gru_adamax_32_9_1 << " | max = " << max_bp_gru_adamax_32_9_1 << " | mean = " << mean_bp_gru_adamax_32_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_64_9_1 ( min = " << min_bp_lstm_sgd_64_9_1 << " | max = " << max_bp_lstm_sgd_64_9_1 << " | mean = " << mean_bp_lstm_sgd_64_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_64_9_1 ( min = " << min_bp_lstm_adam_64_9_1 << " | max = " << max_bp_lstm_adam_64_9_1 << " | mean = " << mean_bp_lstm_adam_64_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_64_9_1 ( min = " << min_bp_lstm_adamax_64_9_1 << " | max = " << max_bp_lstm_adamax_64_9_1 << " | mean = " << mean_bp_lstm_adamax_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_64_9_1 ( min = " << min_bp_gru_sgd_64_9_1 << " | max = " << max_bp_gru_sgd_64_9_1 << " | mean = " << mean_bp_gru_sgd_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_64_9_1 ( min = " << min_bp_gru_adam_64_9_1 << " | max = " << max_bp_gru_adam_64_9_1 << " | mean = " << mean_bp_gru_adam_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_64_9_1 ( min = " << min_bp_gru_adamax_64_9_1 << " | max = " << max_bp_gru_adamax_64_9_1 << " | mean = " << mean_bp_gru_adamax_64_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_16_9_3 ( min = " << min_bp_lstm_sgd_16_9_3 << " | max = " << max_bp_lstm_sgd_16_9_3 << " | mean = " << mean_bp_lstm_sgd_16_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_16_9_3 ( min = " << min_bp_lstm_adam_16_9_3 << " | max = " << max_bp_lstm_adam_16_9_3 << " | mean = " << mean_bp_lstm_adam_16_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_16_9_3 ( min = " << min_bp_lstm_adamax_16_9_3 << " | max = " << max_bp_lstm_adamax_16_9_3 << " | mean = " << mean_bp_lstm_adamax_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_16_9_3 ( min = " << min_bp_gru_sgd_16_9_3 << " | max = " << max_bp_gru_sgd_16_9_3 << " | mean = " << mean_bp_gru_sgd_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_16_9_3 ( min = " << min_bp_gru_adam_16_9_3 << " | max = " << max_bp_gru_adam_16_9_3 << " | mean = " << mean_bp_gru_adam_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_16_9_3 ( min = " << min_bp_gru_adamax_16_9_3 << " | max = " << max_bp_gru_adamax_16_9_3 << " | mean = " << mean_bp_gru_adamax_16_9_3 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_32_9_3 ( min = " << min_bp_lstm_sgd_32_9_3 << " | max = " << max_bp_lstm_sgd_32_9_3 << " | mean = " << mean_bp_lstm_sgd_32_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_32_9_3 ( min = " << min_bp_lstm_adam_32_9_3 << " | max = " << max_bp_lstm_adam_32_9_3 << " | mean = " << mean_bp_lstm_adam_32_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_32_9_3 ( min = " << min_bp_lstm_adamax_32_9_3 << " | max = " << max_bp_lstm_adamax_32_9_3 << " | mean = " << mean_bp_lstm_adamax_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_32_9_3 ( min = " << min_bp_gru_sgd_32_9_3 << " | max = " << max_bp_gru_sgd_32_9_3 << " | mean = " << mean_bp_gru_sgd_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_32_9_3 ( min = " << min_bp_gru_adam_32_9_3 << " | max = " << max_bp_gru_adam_32_9_3 << " | mean = " << mean_bp_gru_adam_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_32_9_3 ( min = " << min_bp_gru_adamax_32_9_3 << " | max = " << max_bp_gru_adamax_32_9_3 << " | mean = " << mean_bp_gru_adamax_32_9_3 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_64_9_3 ( min = " << min_bp_lstm_sgd_64_9_3 << " | max = " << max_bp_lstm_sgd_64_9_3 << " | mean = " << mean_bp_lstm_sgd_64_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_64_9_3 ( min = " << min_bp_lstm_adam_64_9_3 << " | max = " << max_bp_lstm_adam_64_9_3 << " | mean = " << mean_bp_lstm_adam_64_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_64_9_3 ( min = " << min_bp_lstm_adamax_64_9_3 << " | max = " << max_bp_lstm_adamax_64_9_3 << " | mean = " << mean_bp_lstm_adamax_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_64_9_3 ( min = " << min_bp_gru_sgd_64_9_3 << " | max = " << max_bp_gru_sgd_64_9_3 << " | mean = " << mean_bp_gru_sgd_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_64_9_3 ( min = " << min_bp_gru_adam_64_9_3 << " | max = " << max_bp_gru_adam_64_9_3 << " | mean = " << mean_bp_gru_adam_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_64_9_3 ( min = " << min_bp_gru_adamax_64_9_3 << " | max = " << max_bp_gru_adamax_64_9_3 << " | mean = " << mean_bp_gru_adamax_64_9_3 << " ) " << std::endl;

	
	// -----------------------------------------------------------------------------------------------------------------------------
	// TESTING
	// -----------------------------------------------------------------------------------------------------------------------------
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "---------------------- TESTING PERFORMANCES --------------------------" << std::endl;
	std::cout << "--------------------------- Feedforward -------------------------------" << std::endl;
	std::cout << "------------------------- 100 iterations ------------------------------" << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	//logID = logs[0];
	
	//my_log_handlers[logID].randomize();
	//max_samples = my_log_handlers[logID].get_max_test_samples();
	//test_input  = my_log_handler_16.get_input_rdm_test(max_samples, 16, 9);
	//test_target = my_log_handler_16.get_target_rdm_test(max_samples, 16, 3);
	//my_lstm_sgd_16_9_3.clear_grads();
	//my_lstm_adam_16_9_3.clear_grads();
	//my_lstm_adamax_16_9_3.clear_grads();
	//my_gru_sgd_16_9_3.clear_grads();
	//my_gru_adam_16_9_3.clear_grads();
	//my_gru_adamax_16_9_3.clear_grads();

	//nb_test_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_16_9_3
		//auto start_fp_lstm_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_16_9_3.clear_momentums();
		//my_lstm_sgd_16_9_3.forward(test_input[i2]);
		//my_lstm_sgd_16_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_16_9_3 = finish_fp_lstm_sgd_16_9_3-start_fp_lstm_sgd_16_9_3;
		//float elapsed_ms_fp_lstm_sgd_16_9_3 = elapsed_fp_lstm_sgd_16_9_3.count();
		//if (elapsed_ms_fp_lstm_sgd_16_9_3<min_fp_lstm_sgd_16_9_3) min_fp_lstm_sgd_16_9_3 = elapsed_ms_fp_lstm_sgd_16_9_3;
		//if (elapsed_ms_fp_lstm_sgd_16_9_3>max_fp_lstm_sgd_16_9_3) max_fp_lstm_sgd_16_9_3 = elapsed_ms_fp_lstm_sgd_16_9_3;
		//sum_fp_lstm_sgd_16_9_3 = sum_fp_lstm_sgd_16_9_3 + elapsed_ms_fp_lstm_sgd_16_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adam_16_9_3
		//auto start_fp_lstm_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_16_9_3.clear_momentums();
		//my_lstm_adam_16_9_3.forward(test_input[i2]);
		//my_lstm_adam_16_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_16_9_3 = finish_fp_lstm_adam_16_9_3-start_fp_lstm_adam_16_9_3;
		//float elapsed_ms_fp_lstm_adam_16_9_3 = elapsed_fp_lstm_adam_16_9_3.count();
		//if (elapsed_ms_fp_lstm_adam_16_9_3<min_fp_lstm_adam_16_9_3) min_fp_lstm_adam_16_9_3 = elapsed_ms_fp_lstm_adam_16_9_3;
		//if (elapsed_ms_fp_lstm_adam_16_9_3>max_fp_lstm_adam_16_9_3) max_fp_lstm_adam_16_9_3 = elapsed_ms_fp_lstm_adam_16_9_3;
		//sum_fp_lstm_adam_16_9_3 = sum_fp_lstm_adam_16_9_3 + elapsed_ms_fp_lstm_adam_16_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_16_9_3
		//auto start_fp_lstm_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_16_9_3.clear_momentums();
		//my_lstm_adamax_16_9_3.forward(test_input[i2]);
		//my_lstm_adamax_16_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_16_9_3 = finish_fp_lstm_adamax_16_9_3-start_fp_lstm_adamax_16_9_3;
		//float elapsed_ms_fp_lstm_adamax_16_9_3 = elapsed_fp_lstm_adamax_16_9_3.count();
		//if (elapsed_ms_fp_lstm_adamax_16_9_3<min_fp_lstm_adamax_16_9_3) min_fp_lstm_adamax_16_9_3 = elapsed_ms_fp_lstm_adamax_16_9_3;
		//if (elapsed_ms_fp_lstm_adamax_16_9_3>max_fp_lstm_adamax_16_9_3) max_fp_lstm_adamax_16_9_3 = elapsed_ms_fp_lstm_adamax_16_9_3;
		//sum_fp_lstm_adamax_16_9_3 = sum_fp_lstm_adamax_16_9_3 + elapsed_ms_fp_lstm_adamax_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_sgd_16_9_3
		//auto start_fp_gru_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_16_9_3.clear_momentums();
		//my_gru_sgd_16_9_3.forward(test_input[i2]);
		//my_gru_sgd_16_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_sgd_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_16_9_3 = finish_fp_gru_sgd_16_9_3-start_fp_gru_sgd_16_9_3;
		//float elapsed_ms_fp_gru_sgd_16_9_3 = elapsed_fp_gru_sgd_16_9_3.count();
		//if (elapsed_ms_fp_gru_sgd_16_9_3<min_fp_gru_sgd_16_9_3) min_fp_gru_sgd_16_9_3 = elapsed_ms_fp_gru_sgd_16_9_3;
		//if (elapsed_ms_fp_gru_sgd_16_9_3>max_fp_gru_sgd_16_9_3) max_fp_gru_sgd_16_9_3 = elapsed_ms_fp_gru_sgd_16_9_3;
		//sum_fp_gru_sgd_16_9_3 = sum_fp_gru_sgd_16_9_3 + elapsed_ms_fp_gru_sgd_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adam_16_9_3
		//auto start_fp_gru_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_16_9_3.clear_momentums();
		//my_gru_adam_16_9_3.forward(test_input[i2]);
		//my_gru_adam_16_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_adam_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_16_9_3 = finish_fp_gru_adam_16_9_3-start_fp_gru_adam_16_9_3;
		//float elapsed_ms_fp_gru_adam_16_9_3 = elapsed_fp_gru_adam_16_9_3.count();
		//if (elapsed_ms_fp_gru_adam_16_9_3<min_fp_gru_adam_16_9_3) min_fp_gru_adam_16_9_3 = elapsed_ms_fp_gru_adam_16_9_3;
		//if (elapsed_ms_fp_gru_adam_16_9_3>max_fp_gru_adam_16_9_3) max_fp_gru_adam_16_9_3 = elapsed_ms_fp_gru_adam_16_9_3;
		//sum_fp_gru_adam_16_9_3 = sum_fp_gru_adam_16_9_3 + elapsed_ms_fp_gru_adam_16_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adamax_16_9_3
		//auto start_fp_gru_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_16_9_3.clear_momentums();
		//my_gru_adamax_16_9_3.forward(test_input[i2]);
		//my_gru_adamax_16_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_adamax_16_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_16_9_3 = finish_fp_gru_adamax_16_9_3-start_fp_gru_adamax_16_9_3;
		//float elapsed_ms_fp_gru_adamax_16_9_3 = elapsed_fp_gru_adamax_16_9_3.count();
		//if (elapsed_ms_fp_gru_adamax_16_9_3<min_fp_gru_adamax_16_9_3) min_fp_gru_adamax_16_9_3 = elapsed_ms_fp_gru_adamax_16_9_3;
		//if (elapsed_ms_fp_gru_adamax_16_9_3>max_fp_gru_adamax_16_9_3) max_fp_gru_adamax_16_9_3 = elapsed_ms_fp_gru_adamax_16_9_3;
		//sum_fp_gru_adamax_16_9_3 = sum_fp_gru_adamax_16_9_3 + elapsed_ms_fp_gru_adamax_16_9_3;

		//nb_test_batches++;
	//}
	//test_loss /= nb_test_batches;
	//mean_fp_lstm_sgd_16_9_3=sum_fp_lstm_sgd_16_9_3/nb_test_batches;
	//mean_fp_lstm_adam_16_9_3=sum_fp_lstm_adam_16_9_3/nb_test_batches;
	//mean_fp_lstm_adamax_16_9_3=sum_fp_lstm_adamax_16_9_3/nb_test_batches;
	//mean_fp_gru_sgd_16_9_3=sum_fp_gru_sgd_16_9_3/nb_test_batches;
	//mean_fp_gru_adam_16_9_3=sum_fp_gru_adam_16_9_3/nb_test_batches;
	//mean_fp_gru_adamax_16_9_3=sum_fp_gru_adamax_16_9_3/nb_test_batches;
	
	//my_log_handlers[logID].randomize();
	//max_samples = my_log_handlers[logID].get_max_test_samples();
	test_input  = my_log_handler_32.get_input_rdm_test(max_samples, 32, 9);
	test_target = my_log_handler_32.get_target_rdm_test(max_samples, 32, 3);
	my_lstm_sgd_32_9_3.clear_grads();
	my_lstm_adam_32_9_3.clear_grads();
	my_lstm_adamax_32_9_3.clear_grads();
	my_gru_sgd_32_9_3.clear_grads();
	my_gru_adam_32_9_3.clear_grads();
	my_gru_adamax_32_9_3.clear_grads();

	nb_test_batches = 0;
	for (i2=0;i2<max_samples;i2++)
	{
		// -----------------------------------------------------------
		// my_lstm_sgd_32_9_3
		auto start_fp_lstm_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_sgd_32_9_3.forward(test_input[i2]);
		auto finish_fp_lstm_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_32_9_3 = finish_fp_lstm_sgd_32_9_3-start_fp_lstm_sgd_32_9_3;
		float elapsed_ms_fp_lstm_sgd_32_9_3 = elapsed_fp_lstm_sgd_32_9_3.count();
		if (elapsed_ms_fp_lstm_sgd_32_9_3<min_fp_lstm_sgd_32_9_3) min_fp_lstm_sgd_32_9_3 = elapsed_ms_fp_lstm_sgd_32_9_3;
		if (elapsed_ms_fp_lstm_sgd_32_9_3>max_fp_lstm_sgd_32_9_3) max_fp_lstm_sgd_32_9_3 = elapsed_ms_fp_lstm_sgd_32_9_3;
		sum_fp_lstm_sgd_32_9_3 = sum_fp_lstm_sgd_32_9_3 + elapsed_ms_fp_lstm_sgd_32_9_3;
		// -----------------------------------------------------------
		// my_lstm_adam_32_9_3
		auto start_fp_lstm_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_adam_32_9_3.forward(test_input[i2]);
		auto finish_fp_lstm_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_32_9_3 = finish_fp_lstm_adam_32_9_3-start_fp_lstm_adam_32_9_3;
		float elapsed_ms_fp_lstm_adam_32_9_3 = elapsed_fp_lstm_adam_32_9_3.count();
		if (elapsed_ms_fp_lstm_adam_32_9_3<min_fp_lstm_adam_32_9_3) min_fp_lstm_adam_32_9_3 = elapsed_ms_fp_lstm_adam_32_9_3;
		if (elapsed_ms_fp_lstm_adam_32_9_3>max_fp_lstm_adam_32_9_3) max_fp_lstm_adam_32_9_3 = elapsed_ms_fp_lstm_adam_32_9_3;
		sum_fp_lstm_adam_32_9_3 = sum_fp_lstm_adam_32_9_3 + elapsed_ms_fp_lstm_adam_32_9_3;
		// -----------------------------------------------------------
		// my_lstm_adamax_32_9_3
		auto start_fp_lstm_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		my_lstm_adamax_32_9_3.forward(test_input[i2]);
		auto finish_fp_lstm_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_32_9_3 = finish_fp_lstm_adamax_32_9_3-start_fp_lstm_adamax_32_9_3;
		float elapsed_ms_fp_lstm_adamax_32_9_3 = elapsed_fp_lstm_adamax_32_9_3.count();
		if (elapsed_ms_fp_lstm_adamax_32_9_3<min_fp_lstm_adamax_32_9_3) min_fp_lstm_adamax_32_9_3 = elapsed_ms_fp_lstm_adamax_32_9_3;
		if (elapsed_ms_fp_lstm_adamax_32_9_3>max_fp_lstm_adamax_32_9_3) max_fp_lstm_adamax_32_9_3 = elapsed_ms_fp_lstm_adamax_32_9_3;
		sum_fp_lstm_adamax_32_9_3 = sum_fp_lstm_adamax_32_9_3 + elapsed_ms_fp_lstm_adamax_32_9_3;
		// -----------------------------------------------------------
		// my_gru_sgd_32_9_3
		auto start_fp_gru_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_sgd_32_9_3.forward(test_input[i2]);
		auto finish_fp_gru_sgd_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_32_9_3 = finish_fp_gru_sgd_32_9_3-start_fp_gru_sgd_32_9_3;
		float elapsed_ms_fp_gru_sgd_32_9_3 = elapsed_fp_gru_sgd_32_9_3.count();
		if (elapsed_ms_fp_gru_sgd_32_9_3<min_fp_gru_sgd_32_9_3) min_fp_gru_sgd_32_9_3 = elapsed_ms_fp_gru_sgd_32_9_3;
		if (elapsed_ms_fp_gru_sgd_32_9_3>max_fp_gru_sgd_32_9_3) max_fp_gru_sgd_32_9_3 = elapsed_ms_fp_gru_sgd_32_9_3;
		sum_fp_gru_sgd_32_9_3 = sum_fp_gru_sgd_32_9_3 + elapsed_ms_fp_gru_sgd_32_9_3;
		// -----------------------------------------------------------
		// my_gru_adam_32_9_3
		auto start_fp_gru_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_adam_32_9_3.forward(test_input[i2]);
		auto finish_fp_gru_adam_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_32_9_3 = finish_fp_gru_adam_32_9_3-start_fp_gru_adam_32_9_3;
		float elapsed_ms_fp_gru_adam_32_9_3 = elapsed_fp_gru_adam_32_9_3.count();
		if (elapsed_ms_fp_gru_adam_32_9_3<min_fp_gru_adam_32_9_3) min_fp_gru_adam_32_9_3 = elapsed_ms_fp_gru_adam_32_9_3;
		if (elapsed_ms_fp_gru_adam_32_9_3>max_fp_gru_adam_32_9_3) max_fp_gru_adam_32_9_3 = elapsed_ms_fp_gru_adam_32_9_3;
		sum_fp_gru_adam_32_9_3 = sum_fp_gru_adam_32_9_3 + elapsed_ms_fp_gru_adam_32_9_3;
		// -----------------------------------------------------------
		// my_gru_adamax_32_9_3
		auto start_fp_gru_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		my_gru_adamax_32_9_3.forward(test_input[i2]);
		auto finish_fp_gru_adamax_32_9_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_32_9_3 = finish_fp_gru_adamax_32_9_3-start_fp_gru_adamax_32_9_3;
		float elapsed_ms_fp_gru_adamax_32_9_3 = elapsed_fp_gru_adamax_32_9_3.count();
		if (elapsed_ms_fp_gru_adamax_32_9_3<min_fp_gru_adamax_32_9_3) min_fp_gru_adamax_32_9_3 = elapsed_ms_fp_gru_adamax_32_9_3;
		if (elapsed_ms_fp_gru_adamax_32_9_3>max_fp_gru_adamax_32_9_3) max_fp_gru_adamax_32_9_3 = elapsed_ms_fp_gru_adamax_32_9_3;
		sum_fp_gru_adamax_32_9_3 = sum_fp_gru_adamax_32_9_3 + elapsed_ms_fp_gru_adamax_32_9_3;

		nb_test_batches++;
	}
	test_loss /= nb_test_batches;
	mean_fp_lstm_sgd_32_9_3=sum_fp_lstm_sgd_32_9_3/nb_test_batches;
	mean_fp_lstm_adam_32_9_3=sum_fp_lstm_adam_32_9_3/nb_test_batches;
	mean_fp_lstm_adamax_32_9_3=sum_fp_lstm_adamax_32_9_3/nb_test_batches;
	mean_fp_gru_sgd_32_9_3=sum_fp_gru_sgd_32_9_3/nb_test_batches;
	mean_fp_gru_adam_32_9_3=sum_fp_gru_adam_32_9_3/nb_test_batches;
	mean_fp_gru_adamax_32_9_3=sum_fp_gru_adamax_32_9_3/nb_test_batches;
	

	//my_log_handlers[logID].randomize();
	////max_samples = my_log_handlers[logID].get_max_test_samples();
	//test_input  = my_log_handler_64.get_input_rdm_test(max_samples, 64, 9);
	//test_target = my_log_handler_64.get_target_rdm_test(max_samples, 64, 3);
	//my_lstm_sgd_64_9_3.clear_grads();
	//my_lstm_adam_64_9_3.clear_grads();
	//my_lstm_adamax_64_9_3.clear_grads();
	//my_gru_sgd_64_9_3.clear_grads();
	//my_gru_adam_64_9_3.clear_grads();
	//my_gru_adamax_64_9_3.clear_grads();
	//nb_test_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_64_9_3
		//auto start_fp_lstm_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_64_9_3.clear_momentums();
		//my_lstm_sgd_64_9_3.forward(test_input[i2]);
		//my_lstm_sgd_64_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_64_9_3 = finish_fp_lstm_sgd_64_9_3-start_fp_lstm_sgd_64_9_3;
		//float elapsed_ms_fp_lstm_sgd_64_9_3 = elapsed_fp_lstm_sgd_64_9_3.count();
		//if (elapsed_ms_fp_lstm_sgd_64_9_3<min_fp_lstm_sgd_64_9_3) min_fp_lstm_sgd_64_9_3 = elapsed_ms_fp_lstm_sgd_64_9_3;
		//if (elapsed_ms_fp_lstm_sgd_64_9_3>max_fp_lstm_sgd_64_9_3) max_fp_lstm_sgd_64_9_3 = elapsed_ms_fp_lstm_sgd_64_9_3;
		//sum_fp_lstm_sgd_64_9_3 = sum_fp_lstm_sgd_64_9_3 + elapsed_ms_fp_lstm_sgd_64_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adam_64_9_3
		//auto start_fp_lstm_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_64_9_3.clear_momentums();
		//my_lstm_adam_64_9_3.forward(test_input[i2]);
		//my_lstm_adam_64_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_64_9_3 = finish_fp_lstm_adam_64_9_3-start_fp_lstm_adam_64_9_3;
		//float elapsed_ms_fp_lstm_adam_64_9_3 = elapsed_fp_lstm_adam_64_9_3.count();
		//if (elapsed_ms_fp_lstm_adam_64_9_3<min_fp_lstm_adam_64_9_3) min_fp_lstm_adam_64_9_3 = elapsed_ms_fp_lstm_adam_64_9_3;
		//if (elapsed_ms_fp_lstm_adam_64_9_3>max_fp_lstm_adam_64_9_3) max_fp_lstm_adam_64_9_3 = elapsed_ms_fp_lstm_adam_64_9_3;
		//sum_fp_lstm_adam_64_9_3 = sum_fp_lstm_adam_64_9_3 + elapsed_ms_fp_lstm_adam_64_9_3;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_64_9_3
		//auto start_fp_lstm_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_64_9_3.clear_momentums();
		//my_lstm_adamax_64_9_3.forward(test_input[i2]);
		//my_lstm_adamax_64_9_3.backward(test_target[i2]);
		//auto finish_fp_lstm_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_64_9_3 = finish_fp_lstm_adamax_64_9_3-start_fp_lstm_adamax_64_9_3;
		//float elapsed_ms_fp_lstm_adamax_64_9_3 = elapsed_fp_lstm_adamax_64_9_3.count();
		//if (elapsed_ms_fp_lstm_adamax_64_9_3<min_fp_lstm_adamax_64_9_3) min_fp_lstm_adamax_64_9_3 = elapsed_ms_fp_lstm_adamax_64_9_3;
		//if (elapsed_ms_fp_lstm_adamax_64_9_3>max_fp_lstm_adamax_64_9_3) max_fp_lstm_adamax_64_9_3 = elapsed_ms_fp_lstm_adamax_64_9_3;
		//sum_fp_lstm_adamax_64_9_3 = sum_fp_lstm_adamax_64_9_3 + elapsed_ms_fp_lstm_adamax_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_sgd_64_9_3
		//auto start_fp_gru_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_64_9_3.clear_momentums();
		//my_gru_sgd_64_9_3.forward(test_input[i2]);
		//my_gru_sgd_64_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_sgd_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_64_9_3 = finish_fp_gru_sgd_64_9_3-start_fp_gru_sgd_64_9_3;
		//float elapsed_ms_fp_gru_sgd_64_9_3 = elapsed_fp_gru_sgd_64_9_3.count();
		//if (elapsed_ms_fp_gru_sgd_64_9_3<min_fp_gru_sgd_64_9_3) min_fp_gru_sgd_64_9_3 = elapsed_ms_fp_gru_sgd_64_9_3;
		//if (elapsed_ms_fp_gru_sgd_64_9_3>max_fp_gru_sgd_64_9_3) max_fp_gru_sgd_64_9_3 = elapsed_ms_fp_gru_sgd_64_9_3;
		//sum_fp_gru_sgd_64_9_3 = sum_fp_gru_sgd_64_9_3 + elapsed_ms_fp_gru_sgd_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adam_64_9_3
		//auto start_fp_gru_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_64_9_3.clear_momentums();
		//my_gru_adam_64_9_3.forward(test_input[i2]);
		//my_gru_adam_64_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_adam_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_64_9_3 = finish_fp_gru_adam_64_9_3-start_fp_gru_adam_64_9_3;
		//float elapsed_ms_fp_gru_adam_64_9_3 = elapsed_fp_gru_adam_64_9_3.count();
		//if (elapsed_ms_fp_gru_adam_64_9_3<min_fp_gru_adam_64_9_3) min_fp_gru_adam_64_9_3 = elapsed_ms_fp_gru_adam_64_9_3;
		//if (elapsed_ms_fp_gru_adam_64_9_3>max_fp_gru_adam_64_9_3) max_fp_gru_adam_64_9_3 = elapsed_ms_fp_gru_adam_64_9_3;
		//sum_fp_gru_adam_64_9_3 = sum_fp_gru_adam_64_9_3 + elapsed_ms_fp_gru_adam_64_9_3;
		//// -----------------------------------------------------------
		//// my_gru_adamax_64_9_3
		//auto start_fp_gru_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_64_9_3.clear_momentums();
		//my_gru_adamax_64_9_3.forward(test_input[i2]);
		//my_gru_adamax_64_9_3.backward(test_target[i2]);
		//auto finish_fp_gru_adamax_64_9_3 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_64_9_3 = finish_fp_gru_adamax_64_9_3-start_fp_gru_adamax_64_9_3;
		//float elapsed_ms_fp_gru_adamax_64_9_3 = elapsed_fp_gru_adamax_64_9_3.count();
		//if (elapsed_ms_fp_gru_adamax_64_9_3<min_fp_gru_adamax_64_9_3) min_fp_gru_adamax_64_9_3 = elapsed_ms_fp_gru_adamax_64_9_3;
		//if (elapsed_ms_fp_gru_adamax_64_9_3>max_fp_gru_adamax_64_9_3) max_fp_gru_adamax_64_9_3 = elapsed_ms_fp_gru_adamax_64_9_3;
		//sum_fp_gru_adamax_64_9_3 = sum_fp_gru_adamax_64_9_3 + elapsed_ms_fp_gru_adamax_64_9_3;

		//nb_test_batches++;
	//}
	//test_loss /= nb_test_batches;
	//mean_fp_lstm_sgd_64_9_3=sum_fp_lstm_sgd_64_9_3/nb_test_batches;
	//mean_fp_lstm_adam_64_9_3=sum_fp_lstm_adam_64_9_3/nb_test_batches;
	//mean_fp_lstm_adamax_64_9_3=sum_fp_lstm_adamax_64_9_3/nb_test_batches;
	//mean_fp_gru_sgd_64_9_3=sum_fp_gru_sgd_64_9_3/nb_test_batches;
	//mean_fp_gru_adam_64_9_3=sum_fp_gru_adam_64_9_3/nb_test_batches;
	//mean_fp_gru_adamax_64_9_3=sum_fp_gru_adamax_64_9_3/nb_test_batches;
	
	
	//my_log_handlers[logID].randomize();
	////max_samples = my_log_handlers[logID].get_max_test_samples();
	//test_input  = my_log_handlers[logID].get_input_rdm_test(max_samples, 16, 9);
	//test_target = my_log_handlers[logID].get_target_rdm_test(max_samples, 16, 1);
	//my_lstm_sgd_16_9_1.clear_grads();
	//my_lstm_adam_16_9_1.clear_grads();
	//my_lstm_adamax_16_9_1.clear_grads();
	//my_gru_sgd_16_9_1.clear_grads();
	//my_gru_adam_16_9_1.clear_grads();
	//my_gru_adamax_16_9_1.clear_grads();

	//nb_test_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_16_9_1
		//auto start_fp_lstm_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_16_9_1.clear_momentums();
		//my_lstm_sgd_16_9_1.forward(test_input[i2]);
		//my_lstm_sgd_16_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_16_9_1 = finish_fp_lstm_sgd_16_9_1-start_fp_lstm_sgd_16_9_1;
		//float elapsed_ms_fp_lstm_sgd_16_9_1 = elapsed_fp_lstm_sgd_16_9_1.count();
		//if (elapsed_ms_fp_lstm_sgd_16_9_1<min_fp_lstm_sgd_16_9_1) min_fp_lstm_sgd_16_9_1 = elapsed_ms_fp_lstm_sgd_16_9_1;
		//if (elapsed_ms_fp_lstm_sgd_16_9_1>max_fp_lstm_sgd_16_9_1) max_fp_lstm_sgd_16_9_1 = elapsed_ms_fp_lstm_sgd_16_9_1;
		//sum_fp_lstm_sgd_16_9_1 = sum_fp_lstm_sgd_16_9_1 + elapsed_ms_fp_lstm_sgd_16_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adam_16_9_1
		//auto start_fp_lstm_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_16_9_1.clear_momentums();
		//my_lstm_adam_16_9_1.forward(test_input[i2]);
		//my_lstm_adam_16_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_16_9_1 = finish_fp_lstm_adam_16_9_1-start_fp_lstm_adam_16_9_1;
		//float elapsed_ms_fp_lstm_adam_16_9_1 = elapsed_fp_lstm_adam_16_9_1.count();
		//if (elapsed_ms_fp_lstm_adam_16_9_1<min_fp_lstm_adam_16_9_1) min_fp_lstm_adam_16_9_1 = elapsed_ms_fp_lstm_adam_16_9_1;
		//if (elapsed_ms_fp_lstm_adam_16_9_1>max_fp_lstm_adam_16_9_1) max_fp_lstm_adam_16_9_1 = elapsed_ms_fp_lstm_adam_16_9_1;
		//sum_fp_lstm_adam_16_9_1 = sum_fp_lstm_adam_16_9_1 + elapsed_ms_fp_lstm_adam_16_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_16_9_1
		//auto start_fp_lstm_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_16_9_1.clear_momentums();
		//my_lstm_adamax_16_9_1.forward(test_input[i2]);
		//my_lstm_adamax_16_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_16_9_1 = finish_fp_lstm_adamax_16_9_1-start_fp_lstm_adamax_16_9_1;
		//float elapsed_ms_fp_lstm_adamax_16_9_1 = elapsed_fp_lstm_adamax_16_9_1.count();
		//if (elapsed_ms_fp_lstm_adamax_16_9_1<min_fp_lstm_adamax_16_9_1) min_fp_lstm_adamax_16_9_1 = elapsed_ms_fp_lstm_adamax_16_9_1;
		//if (elapsed_ms_fp_lstm_adamax_16_9_1>max_fp_lstm_adamax_16_9_1) max_fp_lstm_adamax_16_9_1 = elapsed_ms_fp_lstm_adamax_16_9_1;
		//sum_fp_lstm_adamax_16_9_1 = sum_fp_lstm_adamax_16_9_1 + elapsed_ms_fp_lstm_adamax_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_sgd_16_9_1
		//auto start_fp_gru_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_16_9_1.clear_momentums();
		//my_gru_sgd_16_9_1.forward(test_input[i2]);
		//my_gru_sgd_16_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_sgd_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_16_9_1 = finish_fp_gru_sgd_16_9_1-start_fp_gru_sgd_16_9_1;
		//float elapsed_ms_fp_gru_sgd_16_9_1 = elapsed_fp_gru_sgd_16_9_1.count();
		//if (elapsed_ms_fp_gru_sgd_16_9_1<min_fp_gru_sgd_16_9_1) min_fp_gru_sgd_16_9_1 = elapsed_ms_fp_gru_sgd_16_9_1;
		//if (elapsed_ms_fp_gru_sgd_16_9_1>max_fp_gru_sgd_16_9_1) max_fp_gru_sgd_16_9_1 = elapsed_ms_fp_gru_sgd_16_9_1;
		//sum_fp_gru_sgd_16_9_1 = sum_fp_gru_sgd_16_9_1 + elapsed_ms_fp_gru_sgd_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adam_16_9_1
		//auto start_fp_gru_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_16_9_1.clear_momentums();
		//my_gru_adam_16_9_1.forward(test_input[i2]);
		//my_gru_adam_16_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_adam_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_16_9_1 = finish_fp_gru_adam_16_9_1-start_fp_gru_adam_16_9_1;
		//float elapsed_ms_fp_gru_adam_16_9_1 = elapsed_fp_gru_adam_16_9_1.count();
		//if (elapsed_ms_fp_gru_adam_16_9_1<min_fp_gru_adam_16_9_1) min_fp_gru_adam_16_9_1 = elapsed_ms_fp_gru_adam_16_9_1;
		//if (elapsed_ms_fp_gru_adam_16_9_1>max_fp_gru_adam_16_9_1) max_fp_gru_adam_16_9_1 = elapsed_ms_fp_gru_adam_16_9_1;
		//sum_fp_gru_adam_16_9_1 = sum_fp_gru_adam_16_9_1 + elapsed_ms_fp_gru_adam_16_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adamax_16_9_1
		//auto start_fp_gru_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_16_9_1.clear_momentums();
		//my_gru_adamax_16_9_1.forward(test_input[i2]);
		//my_gru_adamax_16_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_adamax_16_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_16_9_1 = finish_fp_gru_adamax_16_9_1-start_fp_gru_adamax_16_9_1;
		//float elapsed_ms_fp_gru_adamax_16_9_1 = elapsed_fp_gru_adamax_16_9_1.count();
		//if (elapsed_ms_fp_gru_adamax_16_9_1<min_fp_gru_adamax_16_9_1) min_fp_gru_adamax_16_9_1 = elapsed_ms_fp_gru_adamax_16_9_1;
		//if (elapsed_ms_fp_gru_adamax_16_9_1>max_fp_gru_adamax_16_9_1) max_fp_gru_adamax_16_9_1 = elapsed_ms_fp_gru_adamax_16_9_1;
		//sum_fp_gru_adamax_16_9_1 = sum_fp_gru_adamax_16_9_1 + elapsed_ms_fp_gru_adamax_16_9_1;

		//nb_test_batches++;
	//}
	//test_loss /= nb_test_batches;
	//mean_fp_lstm_sgd_16_9_1=sum_fp_lstm_sgd_16_9_1/nb_test_batches;
	//mean_fp_lstm_adam_16_9_1=sum_fp_lstm_adam_16_9_1/nb_test_batches;
	//mean_fp_lstm_adamax_16_9_1=sum_fp_lstm_adamax_16_9_1/nb_test_batches;
	//mean_fp_gru_sgd_16_9_1=sum_fp_gru_sgd_16_9_1/nb_test_batches;
	//mean_fp_gru_adam_16_9_1=sum_fp_gru_adam_16_9_1/nb_test_batches;
	//mean_fp_gru_adamax_16_9_1=sum_fp_gru_adamax_16_9_1/nb_test_batches;
	
	//max_samples = my_log_handlers[logID].get_max_test_samples();
	test_input  = my_log_handler_32.get_input_rdm_test(max_samples, 32, 9);
	test_target = my_log_handler_32.get_target_rdm_test(max_samples, 32, 1);
	my_lstm_sgd_32_9_1.clear_grads();
	my_lstm_adam_32_9_1.clear_grads();
	my_lstm_adamax_32_9_1.clear_grads();
	my_gru_sgd_32_9_1.clear_grads();
	my_gru_adam_32_9_1.clear_grads();
	my_gru_adamax_32_9_1.clear_grads();

	nb_test_batches = 0;
	for (i2=0;i2<max_samples;i2++)
	{
		// -----------------------------------------------------------
		// my_lstm_sgd_32_9_1
		auto start_fp_lstm_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_sgd_32_9_1.forward(test_input[i2]);
		auto finish_fp_lstm_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_32_9_1 = finish_fp_lstm_sgd_32_9_1-start_fp_lstm_sgd_32_9_1;
		float elapsed_ms_fp_lstm_sgd_32_9_1 = elapsed_fp_lstm_sgd_32_9_1.count();
		if (elapsed_ms_fp_lstm_sgd_32_9_1<min_fp_lstm_sgd_32_9_1) min_fp_lstm_sgd_32_9_1 = elapsed_ms_fp_lstm_sgd_32_9_1;
		if (elapsed_ms_fp_lstm_sgd_32_9_1>max_fp_lstm_sgd_32_9_1) max_fp_lstm_sgd_32_9_1 = elapsed_ms_fp_lstm_sgd_32_9_1;
		sum_fp_lstm_sgd_32_9_1 = sum_fp_lstm_sgd_32_9_1 + elapsed_ms_fp_lstm_sgd_32_9_1;
		// -----------------------------------------------------------
		// my_lstm_adam_32_9_1
		auto start_fp_lstm_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_adam_32_9_1.forward(test_input[i2]);
		auto finish_fp_lstm_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_32_9_1 = finish_fp_lstm_adam_32_9_1-start_fp_lstm_adam_32_9_1;
		float elapsed_ms_fp_lstm_adam_32_9_1 = elapsed_fp_lstm_adam_32_9_1.count();
		if (elapsed_ms_fp_lstm_adam_32_9_1<min_fp_lstm_adam_32_9_1) min_fp_lstm_adam_32_9_1 = elapsed_ms_fp_lstm_adam_32_9_1;
		if (elapsed_ms_fp_lstm_adam_32_9_1>max_fp_lstm_adam_32_9_1) max_fp_lstm_adam_32_9_1 = elapsed_ms_fp_lstm_adam_32_9_1;
		sum_fp_lstm_adam_32_9_1 = sum_fp_lstm_adam_32_9_1 + elapsed_ms_fp_lstm_adam_32_9_1;
		// -----------------------------------------------------------
		// my_lstm_adamax_32_9_1
		auto start_fp_lstm_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		my_lstm_adamax_32_9_1.forward(test_input[i2]);
		auto finish_fp_lstm_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_32_9_1 = finish_fp_lstm_adamax_32_9_1-start_fp_lstm_adamax_32_9_1;
		float elapsed_ms_fp_lstm_adamax_32_9_1 = elapsed_fp_lstm_adamax_32_9_1.count();
		if (elapsed_ms_fp_lstm_adamax_32_9_1<min_fp_lstm_adamax_32_9_1) min_fp_lstm_adamax_32_9_1 = elapsed_ms_fp_lstm_adamax_32_9_1;
		if (elapsed_ms_fp_lstm_adamax_32_9_1>max_fp_lstm_adamax_32_9_1) max_fp_lstm_adamax_32_9_1 = elapsed_ms_fp_lstm_adamax_32_9_1;
		sum_fp_lstm_adamax_32_9_1 = sum_fp_lstm_adamax_32_9_1 + elapsed_ms_fp_lstm_adamax_32_9_1;
		// -----------------------------------------------------------
		// my_gru_sgd_32_9_1
		auto start_fp_gru_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_sgd_32_9_1.forward(test_input[i2]);
		auto finish_fp_gru_sgd_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_32_9_1 = finish_fp_gru_sgd_32_9_1-start_fp_gru_sgd_32_9_1;
		float elapsed_ms_fp_gru_sgd_32_9_1 = elapsed_fp_gru_sgd_32_9_1.count();
		if (elapsed_ms_fp_gru_sgd_32_9_1<min_fp_gru_sgd_32_9_1) min_fp_gru_sgd_32_9_1 = elapsed_ms_fp_gru_sgd_32_9_1;
		if (elapsed_ms_fp_gru_sgd_32_9_1>max_fp_gru_sgd_32_9_1) max_fp_gru_sgd_32_9_1 = elapsed_ms_fp_gru_sgd_32_9_1;
		sum_fp_gru_sgd_32_9_1 = sum_fp_gru_sgd_32_9_1 + elapsed_ms_fp_gru_sgd_32_9_1;
		// -----------------------------------------------------------
		// my_gru_adam_32_9_1
		auto start_fp_gru_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_adam_32_9_1.forward(test_input[i2]);
		auto finish_fp_gru_adam_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_32_9_1 = finish_fp_gru_adam_32_9_1-start_fp_gru_adam_32_9_1;
		float elapsed_ms_fp_gru_adam_32_9_1 = elapsed_fp_gru_adam_32_9_1.count();
		if (elapsed_ms_fp_gru_adam_32_9_1<min_fp_gru_adam_32_9_1) min_fp_gru_adam_32_9_1 = elapsed_ms_fp_gru_adam_32_9_1;
		if (elapsed_ms_fp_gru_adam_32_9_1>max_fp_gru_adam_32_9_1) max_fp_gru_adam_32_9_1 = elapsed_ms_fp_gru_adam_32_9_1;
		sum_fp_gru_adam_32_9_1 = sum_fp_gru_adam_32_9_1 + elapsed_ms_fp_gru_adam_32_9_1;
		// -----------------------------------------------------------
		// my_gru_adamax_32_9_1
		auto start_fp_gru_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		my_gru_adamax_32_9_1.forward(test_input[i2]);
		auto finish_fp_gru_adamax_32_9_1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_32_9_1 = finish_fp_gru_adamax_32_9_1-start_fp_gru_adamax_32_9_1;
		float elapsed_ms_fp_gru_adamax_32_9_1 = elapsed_fp_gru_adamax_32_9_1.count();
		if (elapsed_ms_fp_gru_adamax_32_9_1<min_fp_gru_adamax_32_9_1) min_fp_gru_adamax_32_9_1 = elapsed_ms_fp_gru_adamax_32_9_1;
		if (elapsed_ms_fp_gru_adamax_32_9_1>max_fp_gru_adamax_32_9_1) max_fp_gru_adamax_32_9_1 = elapsed_ms_fp_gru_adamax_32_9_1;
		sum_fp_gru_adamax_32_9_1 = sum_fp_gru_adamax_32_9_1 + elapsed_ms_fp_gru_adamax_32_9_1;

		nb_test_batches++;
	}
	test_loss /= nb_test_batches;
	mean_fp_lstm_sgd_32_9_1=sum_fp_lstm_sgd_32_9_1/nb_test_batches;
	mean_fp_lstm_adam_32_9_1=sum_fp_lstm_adam_32_9_1/nb_test_batches;
	mean_fp_lstm_adamax_32_9_1=sum_fp_lstm_adamax_32_9_1/nb_test_batches;
	mean_fp_gru_sgd_32_9_1=sum_fp_gru_sgd_32_9_1/nb_test_batches;
	mean_fp_gru_adam_32_9_1=sum_fp_gru_adam_32_9_1/nb_test_batches;
	mean_fp_gru_adamax_32_9_1=sum_fp_gru_adamax_32_9_1/nb_test_batches;
	

	//my_log_handlers[logID].randomize();
	////max_samples = my_log_handlers[logID].get_max_test_samples();
	//test_input  = my_log_handlers[logID].get_input_rdm_test(max_samples, 64, 9);
	//test_target = my_log_handlers[logID].get_target_rdm_test(max_samples, 64, 3);
	//my_lstm_sgd_64_9_1.clear_grads();
	//my_lstm_adam_64_9_1.clear_grads();
	//my_lstm_adamax_64_9_1.clear_grads();
	//my_gru_sgd_64_9_1.clear_grads();
	//my_gru_adam_64_9_1.clear_grads();
	//my_gru_adamax_64_9_1.clear_grads();
	//nb_test_batches = 0;
	//for (i2=0;i2<max_samples;i2++)
	//{
		//// -----------------------------------------------------------
		//// my_lstm_sgd_64_9_1
		//auto start_fp_lstm_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_sgd_64_9_1.clear_momentums();
		//my_lstm_sgd_64_9_1.forward(test_input[i2]);
		//my_lstm_sgd_64_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_sgd_64_9_1 = finish_fp_lstm_sgd_64_9_1-start_fp_lstm_sgd_64_9_1;
		//float elapsed_ms_fp_lstm_sgd_64_9_1 = elapsed_fp_lstm_sgd_64_9_1.count();
		//if (elapsed_ms_fp_lstm_sgd_64_9_1<min_fp_lstm_sgd_64_9_1) min_fp_lstm_sgd_64_9_1 = elapsed_ms_fp_lstm_sgd_64_9_1;
		//if (elapsed_ms_fp_lstm_sgd_64_9_1>max_fp_lstm_sgd_64_9_1) max_fp_lstm_sgd_64_9_1 = elapsed_ms_fp_lstm_sgd_64_9_1;
		//sum_fp_lstm_sgd_64_9_1 = sum_fp_lstm_sgd_64_9_1 + elapsed_ms_fp_lstm_sgd_64_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adam_64_9_1
		//auto start_fp_lstm_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adam_64_9_1.clear_momentums();
		//my_lstm_adam_64_9_1.forward(test_input[i2]);
		//my_lstm_adam_64_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adam_64_9_1 = finish_fp_lstm_adam_64_9_1-start_fp_lstm_adam_64_9_1;
		//float elapsed_ms_fp_lstm_adam_64_9_1 = elapsed_fp_lstm_adam_64_9_1.count();
		//if (elapsed_ms_fp_lstm_adam_64_9_1<min_fp_lstm_adam_64_9_1) min_fp_lstm_adam_64_9_1 = elapsed_ms_fp_lstm_adam_64_9_1;
		//if (elapsed_ms_fp_lstm_adam_64_9_1>max_fp_lstm_adam_64_9_1) max_fp_lstm_adam_64_9_1 = elapsed_ms_fp_lstm_adam_64_9_1;
		//sum_fp_lstm_adam_64_9_1 = sum_fp_lstm_adam_64_9_1 + elapsed_ms_fp_lstm_adam_64_9_1;
		//// -----------------------------------------------------------
		//// my_lstm_adamax_64_9_1
		//auto start_fp_lstm_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_lstm_adamax_64_9_1.clear_momentums();
		//my_lstm_adamax_64_9_1.forward(test_input[i2]);
		//my_lstm_adamax_64_9_1.backward(test_target[i2]);
		//auto finish_fp_lstm_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_lstm_adamax_64_9_1 = finish_fp_lstm_adamax_64_9_1-start_fp_lstm_adamax_64_9_1;
		//float elapsed_ms_fp_lstm_adamax_64_9_1 = elapsed_fp_lstm_adamax_64_9_1.count();
		//if (elapsed_ms_fp_lstm_adamax_64_9_1<min_fp_lstm_adamax_64_9_1) min_fp_lstm_adamax_64_9_1 = elapsed_ms_fp_lstm_adamax_64_9_1;
		//if (elapsed_ms_fp_lstm_adamax_64_9_1>max_fp_lstm_adamax_64_9_1) max_fp_lstm_adamax_64_9_1 = elapsed_ms_fp_lstm_adamax_64_9_1;
		//sum_fp_lstm_adamax_64_9_1 = sum_fp_lstm_adamax_64_9_1 + elapsed_ms_fp_lstm_adamax_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_sgd_64_9_1
		//auto start_fp_gru_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_sgd_64_9_1.clear_momentums();
		//my_gru_sgd_64_9_1.forward(test_input[i2]);
		//my_gru_sgd_64_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_sgd_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_sgd_64_9_1 = finish_fp_gru_sgd_64_9_1-start_fp_gru_sgd_64_9_1;
		//float elapsed_ms_fp_gru_sgd_64_9_1 = elapsed_fp_gru_sgd_64_9_1.count();
		//if (elapsed_ms_fp_gru_sgd_64_9_1<min_fp_gru_sgd_64_9_1) min_fp_gru_sgd_64_9_1 = elapsed_ms_fp_gru_sgd_64_9_1;
		//if (elapsed_ms_fp_gru_sgd_64_9_1>max_fp_gru_sgd_64_9_1) max_fp_gru_sgd_64_9_1 = elapsed_ms_fp_gru_sgd_64_9_1;
		//sum_fp_gru_sgd_64_9_1 = sum_fp_gru_sgd_64_9_1 + elapsed_ms_fp_gru_sgd_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adam_64_9_1
		//auto start_fp_gru_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adam_64_9_1.clear_momentums();
		//my_gru_adam_64_9_1.forward(test_input[i2]);
		//my_gru_adam_64_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_adam_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adam_64_9_1 = finish_fp_gru_adam_64_9_1-start_fp_gru_adam_64_9_1;
		//float elapsed_ms_fp_gru_adam_64_9_1 = elapsed_fp_gru_adam_64_9_1.count();
		//if (elapsed_ms_fp_gru_adam_64_9_1<min_fp_gru_adam_64_9_1) min_fp_gru_adam_64_9_1 = elapsed_ms_fp_gru_adam_64_9_1;
		//if (elapsed_ms_fp_gru_adam_64_9_1>max_fp_gru_adam_64_9_1) max_fp_gru_adam_64_9_1 = elapsed_ms_fp_gru_adam_64_9_1;
		//sum_fp_gru_adam_64_9_1 = sum_fp_gru_adam_64_9_1 + elapsed_ms_fp_gru_adam_64_9_1;
		//// -----------------------------------------------------------
		//// my_gru_adamax_64_9_1
		//auto start_fp_gru_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//my_gru_adamax_64_9_1.clear_momentums();
		//my_gru_adamax_64_9_1.forward(test_input[i2]);
		//my_gru_adamax_64_9_1.backward(test_target[i2]);
		//auto finish_fp_gru_adamax_64_9_1 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed_fp_gru_adamax_64_9_1 = finish_fp_gru_adamax_64_9_1-start_fp_gru_adamax_64_9_1;
		//float elapsed_ms_fp_gru_adamax_64_9_1 = elapsed_fp_gru_adamax_64_9_1.count();
		//if (elapsed_ms_fp_gru_adamax_64_9_1<min_fp_gru_adamax_64_9_1) min_fp_gru_adamax_64_9_1 = elapsed_ms_fp_gru_adamax_64_9_1;
		//if (elapsed_ms_fp_gru_adamax_64_9_1>max_fp_gru_adamax_64_9_1) max_fp_gru_adamax_64_9_1 = elapsed_ms_fp_gru_adamax_64_9_1;
		//sum_fp_gru_adamax_64_9_1 = sum_fp_gru_adamax_64_9_1 + elapsed_ms_fp_gru_adamax_64_9_1;

		//nb_test_batches++;
	//}
	//test_loss /= nb_test_batches;
	//mean_fp_lstm_sgd_64_9_1=sum_fp_lstm_sgd_64_9_1/nb_test_batches;
	//mean_fp_lstm_adam_64_9_1=sum_fp_lstm_adam_64_9_1/nb_test_batches;
	//mean_fp_lstm_adamax_64_9_1=sum_fp_lstm_adamax_64_9_1/nb_test_batches;
	//mean_fp_gru_sgd_64_9_1=sum_fp_gru_sgd_64_9_1/nb_test_batches;
	//mean_fp_gru_adam_64_9_1=sum_fp_gru_adam_64_9_1/nb_test_batches;
	//mean_fp_gru_adamax_64_9_1=sum_fp_gru_adamax_64_9_1/nb_test_batches;

	std::cout << "type | min_fp | max_fp | mean_fp" << std::endl;
	std::cout << "my_lstm_sgd_32_9_1 | " << min_fp_lstm_sgd_32_9_1 << " | " << max_fp_lstm_sgd_32_9_1 << " | " << mean_fp_lstm_sgd_32_9_1 << std::endl;
	std::cout << "my_lstm_adam_32_9_1 | " << min_fp_lstm_adam_32_9_1 << " | " << max_fp_lstm_adam_32_9_1 << " | " << mean_fp_lstm_adam_32_9_1 << std::endl;
	std::cout << "my_lstm_adamax_32_9_1 | " << min_fp_lstm_adamax_32_9_1 << " | " << max_fp_lstm_adamax_32_9_1 << " | " << mean_fp_lstm_adamax_32_9_1 << std::endl;
	std::cout << "my_gru_sgd_32_9_1 | " << min_fp_gru_sgd_32_9_1 << " | " << max_fp_gru_sgd_32_9_1 << " | " << mean_fp_gru_sgd_32_9_1 << std::endl;
	std::cout << "my_gru_adam_32_9_1 | " << min_fp_gru_adam_32_9_1 << " | " << max_fp_gru_adam_32_9_1 << " | " << mean_fp_gru_adam_32_9_1 << std::endl;
	std::cout << "my_gru_adamax_32_9_1 | " << min_fp_gru_adamax_32_9_1 << " | " << max_fp_gru_adamax_32_9_1 << " | " << mean_fp_gru_adamax_32_9_1 << std::endl;
	std::cout << "my_lstm_sgd_32_9_3 | " << min_fp_lstm_sgd_32_9_3 << " | " << max_fp_lstm_sgd_32_9_3 << " | " << mean_fp_lstm_sgd_32_9_3 << std::endl;
	std::cout << "my_lstm_adam_32_9_3 | " << min_fp_lstm_adam_32_9_3 << " | " << max_fp_lstm_adam_32_9_3 << " | " << mean_fp_lstm_adam_32_9_3 << std::endl;
	std::cout << "my_lstm_adamax_32_9_3 | " << min_fp_lstm_adamax_32_9_3 << " | " << max_fp_lstm_adamax_32_9_3 << " | " << mean_fp_lstm_adamax_32_9_3 << std::endl;
	std::cout << "my_gru_sgd_32_9_3 | " << min_fp_gru_sgd_32_9_3 << " | " << max_fp_gru_sgd_32_9_3 << " | " << mean_fp_gru_sgd_32_9_3 << std::endl;
	std::cout << "my_gru_adam_32_9_3 | " << min_fp_gru_adam_32_9_3 << " | " << max_fp_gru_adam_32_9_3 << " | " << mean_fp_gru_adam_32_9_3 << std::endl;
	std::cout << "my_gru_adamax_32_9_3 | " << min_fp_gru_adamax_32_9_3 << " | " << max_fp_gru_adamax_32_9_3 << " | " << mean_fp_gru_adamax_32_9_3 << std::endl;
	
	//std::cout << "my_lstm_sgd_16_9_1 | " << min_fp_lstm_sgd_16_9_1 << " | max = " << max_fp_lstm_sgd_16_9_1 << " | mean = " << mean_fp_lstm_sgd_16_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_16_9_1 | " << min_fp_lstm_adam_16_9_1 << " | max = " << max_fp_lstm_adam_16_9_1 << " | mean = " << mean_fp_lstm_adam_16_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_16_9_1 | " << min_fp_lstm_adamax_16_9_1 << " | max = " << max_fp_lstm_adamax_16_9_1 << " | mean = " << mean_fp_lstm_adamax_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_16_9_1 | " << min_fp_gru_sgd_16_9_1 << " | max = " << max_fp_gru_sgd_16_9_1 << " | mean = " << mean_fp_gru_sgd_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_16_9_1, ( min = " << min_fp_gru_adam_16_9_1 << " | max = " << max_fp_gru_adam_16_9_1 << " | mean = " << mean_fp_gru_adam_16_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_16_9_1, ( min = " << min_fp_gru_adamax_16_9_1 << " | max = " << max_fp_gru_adamax_16_9_1 << " | mean = " << mean_fp_gru_adamax_16_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_32_9_1, ( min = " << min_fp_lstm_sgd_32_9_1 << " | max = " << max_fp_lstm_sgd_32_9_1 << " | mean = " << mean_fp_lstm_sgd_32_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_32_9_1, ( min = " << min_fp_lstm_adam_32_9_1 << " | max = " << max_fp_lstm_adam_32_9_1 << " | mean = " << mean_fp_lstm_adam_32_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_32_9_1, ( min = " << min_fp_lstm_adamax_32_9_1 << " | max = " << max_fp_lstm_adamax_32_9_1 << " | mean = " << mean_fp_lstm_adamax_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_32_9_1, ( min = " << min_fp_gru_sgd_32_9_1 << " | max = " << max_fp_gru_sgd_32_9_1 << " | mean = " << mean_fp_gru_sgd_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_32_9_1, ( min = " << min_fp_gru_adam_32_9_1 << " | max = " << max_fp_gru_adam_32_9_1 << " | mean = " << mean_fp_gru_adam_32_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_32_9_1, ( min = " << min_fp_gru_adamax_32_9_1 << " | max = " << max_fp_gru_adamax_32_9_1 << " | mean = " << mean_fp_gru_adamax_32_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_64_9_1, ( min = " << min_fp_lstm_sgd_64_9_1 << " | max = " << max_fp_lstm_sgd_64_9_1 << " | mean = " << mean_fp_lstm_sgd_64_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_64_9_1, ( min = " << min_fp_lstm_adam_64_9_1 << " | max = " << max_fp_lstm_adam_64_9_1 << " | mean = " << mean_fp_lstm_adam_64_9_1 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_64_9_1, ( min = " << min_fp_lstm_adamax_64_9_1 << " | max = " << max_fp_lstm_adamax_64_9_1 << " | mean = " << mean_fp_lstm_adamax_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_64_9_1, ( min = " << min_fp_gru_sgd_64_9_1 << " | max = " << max_fp_gru_sgd_64_9_1 << " | mean = " << mean_fp_gru_sgd_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adam_64_9_1, ( min = " << min_fp_gru_adam_64_9_1 << " | max = " << max_fp_gru_adam_64_9_1 << " | mean = " << mean_fp_gru_adam_64_9_1 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_64_9_1, ( min = " << min_fp_gru_adamax_64_9_1 << " | max = " << max_fp_gru_adamax_64_9_1 << " | mean = " << mean_fp_gru_adamax_64_9_1 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_16_9_3, ( min = " << min_fp_lstm_sgd_16_9_3 << " | max = " << max_fp_lstm_sgd_16_9_3 << " | mean = " << mean_fp_lstm_sgd_16_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_16_9_3, ( min = " << min_fp_lstm_adam_16_9_3 << " | max = " << max_fp_lstm_adam_16_9_3 << " | mean = " << mean_fp_lstm_adam_16_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_16_9_3, ( min = " << min_fp_lstm_adamax_16_9_3 << " | max = " << max_fp_lstm_adamax_16_9_3 << " | mean = " << mean_fp_lstm_adamax_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_16_9_3, ( min = " << min_fp_gru_sgd_16_9_3 << " | max = " << max_fp_gru_sgd_16_9_3 << " | mean = " << mean_fp_gru_sgd_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_16_9_3, ( min = " << min_fp_gru_adam_16_9_3 << " | max = " << max_fp_gru_adam_16_9_3 << " | mean = " << mean_fp_gru_adam_16_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_16_9_3, ( min = " << min_fp_gru_adamax_16_9_3 << " | max = " << max_fp_gru_adamax_16_9_3 << " | mean = " << mean_fp_gru_adamax_16_9_3 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_32_9_3, ( min = " << min_fp_lstm_sgd_32_9_3 << " | max = " << max_fp_lstm_sgd_32_9_3 << " | mean = " << mean_fp_lstm_sgd_32_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_32_9_3, ( min = " << min_fp_lstm_adam_32_9_3 << " | max = " << max_fp_lstm_adam_32_9_3 << " | mean = " << mean_fp_lstm_adam_32_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_32_9_3, ( min = " << min_fp_lstm_adamax_32_9_3 << " | max = " << max_fp_lstm_adamax_32_9_3 << " | mean = " << mean_fp_lstm_adamax_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_32_9_3, ( min = " << min_fp_gru_sgd_32_9_3 << " | max = " << max_fp_gru_sgd_32_9_3 << " | mean = " << mean_fp_gru_sgd_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_32_9_3, ( min = " << min_fp_gru_adam_32_9_3 << " | max = " << max_fp_gru_adam_32_9_3 << " | mean = " << mean_fp_gru_adam_32_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_32_9_3, ( min = " << min_fp_gru_adamax_32_9_3 << " | max = " << max_fp_gru_adamax_32_9_3 << " | mean = " << mean_fp_gru_adamax_32_9_3 << " ) " << std::endl;
	
	//std::cout << "my_lstm_sgd_64_9_3, ( min = " << min_fp_lstm_sgd_64_9_3 << " | max = " << max_fp_lstm_sgd_64_9_3 << " | mean = " << mean_fp_lstm_sgd_64_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adam_64_9_3, ( min = " << min_fp_lstm_adam_64_9_3 << " | max = " << max_fp_lstm_adam_64_9_3 << " | mean = " << mean_fp_lstm_adam_64_9_3 << " ) " << std::endl;
	//std::cout << "my_lstm_adamax_64_9_3, ( min = " << min_fp_lstm_adamax_64_9_3 << " | max = " << max_fp_lstm_adamax_64_9_3 << " | mean = " << mean_fp_lstm_adamax_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_sgd_64_9_3, ( min = " << min_fp_gru_sgd_64_9_3 << " | max = " << max_fp_gru_sgd_64_9_3 << " | mean = " << mean_fp_gru_sgd_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adam_64_9_3, ( min = " << min_fp_gru_adam_64_9_3 << " | max = " << max_fp_gru_adam_64_9_3 << " | mean = " << mean_fp_gru_adam_64_9_3 << " ) " << std::endl;
	//std::cout << "my_gru_adamax_64_9_3, ( min = " << min_fp_gru_adamax_64_9_3 << " | max = " << max_fp_gru_adamax_64_9_3 << " | mean = " << mean_fp_gru_adamax_64_9_3 << " ) " << std::endl;
	
	
    return 0 ;   
}
