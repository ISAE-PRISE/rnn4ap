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
#include <random>

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_int_distribution<int> dist (0,1);

/**
 * @brief LSTM network optimizer. Loads and normalizes IMU data as inputs from csv files
 * and trains a pretrained network (if one with the desired configuration exists) or
 * creates a new one. The network is trained to perform sensor fusion using the results of
 * a Kalman filter as targets.
 * The executable must be called followed by the optimizer to be used (sgd, adam or adamax),
 * the number of outputs to be considered (up to 3 in the order roll, pitch, yaw) and the
 * total number of training epochs.
 * 
 * @param optimizer: sgd, adam or adamax
 * @param output_size: 1 to 3, on the order roll, pitch, yaw
 * @param epoch_max: maximum number of training epochs
 * @return int 
 */
int main(int argc, char *argv[])
{	
	
	srand((int)time(NULL));
	static std::vector<px4_log_handler> my_log_handlers(11);
	
	static lstm my_lstm;
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
	//std::string log1_path  = path_to_smarties_nn + "datasets/px4/px4_quad_log_01.csv";
	std::string log1_path  = "px4_quad_log_01.csv";
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
	uint32_t epoch_max;// = 50;
	uint32_t nb_timestep = 32;
	uint32_t max_samples = 0;
	uint32_t sample_nb = 500;

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
	
	//optimizer = "sgd";
	optimizer=argv[1];
	std::string arg2(argv[2]);
	output_size = std::stoi(arg2);
	std::string arg3(argv[3]);
	epoch_max = std::stoi(arg3);

	sprintf( net_filename, "%snetFiles/lstm_N%dO%dI%d%s-%s.net.csv", path_to_smarties_nn.c_str(), nb_timestep, output_size, input_size, filenameExtension.c_str(), optimizer.c_str());
	sprintf( logResult   , "%snetFiles/lstm_N%dO%dI%d%s-%s.log.txt", path_to_smarties_nn.c_str(), nb_timestep, output_size, input_size, filenameExtension.c_str(), optimizer.c_str());
	std::ofstream logfile(logResult, std::ios_base::app);
	std::ofstream logbuffer;

	first_epoch = 0;
	if(file_exists(net_filename) && file_exists(logResult))//if the net file exists load it
	{
		my_lstm.load(net_filename, input_size, output_size);
		std::cout << "loaded network" << std::endl;

		std::ifstream f(logResult);
		if (f.is_open()){std::cout << f.rdbuf();} //print to terminal the train log lines already saved

		first_epoch = my_lstm.nbLines(logResult) - 2; //-2 because there are 2 lines in the header of the file
	}
	else
	{
		my_lstm.init(nb_timestep,input_size,output_size);
		std::cout << "--------LSTM optimized by " << optimizer << "--------" << std::endl;//print in the terminal
		logfile   << "--------LSTM optimized by " << optimizer << "--------" << std::endl;//print in the 
		std::cout << "epoch\t\t" << "training loss\t\t" << "testing loss\t\t" << "time (s)" << std::endl;
		logfile   << "epoch\t\t" << "training loss\t\t" << "testing loss\t\t" << "time (s)" << std::endl;
	}
	

	for(i8=0;i8<nb_logs;i8++){
		my_log_handlers[i8].create_random_dataset(sample_nb, nb_timestep);
	}

	void (lstm::*backward)(std::vector<std::vector<float>>); //specific operations depending on the chosen optimizer
	if      (optimizer == "sgd")	{my_lstm.set_lr_mu(0.0001, 0.9);				            backward = &lstm::backward_sgd;}
	else if (optimizer == "adam")	{my_lstm.set_lr_b1_b2_epsilon(0.001, 0.9, 0.999, 10.0);	backward = &lstm::backward_adam;}
	else if (optimizer == "adamax")	{my_lstm.set_lr_b1_b2_epsilon(0.001, 0.9, 0.999, 10.0);	backward = &lstm::backward_adamax;}
	
	float train_loss = 0.0;
	
	for(i1=first_epoch+1;i1<=epoch_max;i1++)//EPOCH
	{
		auto start = std::chrono::high_resolution_clock::now();
		train_loss = 0.0;
		nb_train_batches = 0;

		std::random_shuffle(logs.begin(), logs.end());
		min_bp = 10.0;
		max_bp =  0.0;
		mean_bp =  0.0;
		sum_bp =  0.0;
		for(i7 = 0; i7 < nb_logs; i7++){//TRAIN
			logID = logs[i7];
			my_log_handlers[logID].randomize();
			max_samples = my_log_handlers[logID].get_max_train_samples();
			train_input  = my_log_handlers[logID].get_input_rdm_train(max_samples, nb_timestep, input_size);
			train_target = my_log_handlers[logID].get_target_rdm_train(max_samples, nb_timestep, output_size);
			my_lstm.clear_grads();
			for (i2=0;i2<max_samples;i2++)
			{
				my_lstm.clear_momentums();
				auto start_bp = std::chrono::high_resolution_clock::now();
				my_lstm.forward(train_input[i2]);
				(my_lstm.*backward)(train_target[i2]);
				auto finish_bp = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float, std::milli> elapsed_bp = finish_bp-start_bp;
				float elapsed_float_bp = elapsed_bp.count();
				if (elapsed_float_bp<min_bp) min_bp = elapsed_float_bp;
				if (elapsed_float_bp>max_bp) max_bp = elapsed_float_bp;
				sum_bp = sum_bp + elapsed_float_bp;
				train_loss += my_lstm.get_mse();
				nb_train_batches++;
			}
		}
		train_loss /= nb_train_batches;
		mean_bp=sum_bp/nb_train_batches;


		test_loss = 0.0;
		nb_test_batches = 0;
		min_fp = 10.0;
		max_fp =  0.0;
		mean_fp =  0.0;
		stddev_fp =  0.0;
		sum_fp =  0.0;
		for(i6=0; i6<nb_logs; i6++) //TEST
		{
			max_samples = my_log_handlers[i6].get_max_test_samples();
			test_input  = my_log_handlers[i6].get_input_rdm_test(max_samples, nb_timestep, input_size);
			test_target = my_log_handlers[i6].get_target_rdm_test(max_samples, nb_timestep, output_size);

			for (i2=0;i2<max_samples;i2++)
			{
				auto start_fp = std::chrono::high_resolution_clock::now();
				my_lstm.forward(test_input[i2]);
				auto finish_fp = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float, std::milli> elapsed_fp = finish_fp-start_fp;
				float elapsed_float_fp = elapsed_fp.count();
				if (elapsed_float_fp<min_fp) min_fp = elapsed_float_fp;
				if (elapsed_float_fp>max_fp) max_fp = elapsed_float_fp;
				sum_fp = sum_fp + elapsed_float_fp;
				my_lstm.get_error(test_target[i2]);
				test_loss += my_lstm.get_mse();
				nb_test_batches++;
			}
		}
		test_loss/=nb_test_batches;
		mean_fp=sum_fp/nb_test_batches;	

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = finish - start;
		std::cout << i1 << "\t\t" << train_loss << "\t\t" << test_loss << "\t\t" << elapsed.count() << "\t\t";
		logfile   << i1 << "\t\t" << train_loss << "\t\t" << test_loss << "\t\t" << elapsed.count() << "\t\t";

		my_lstm.save(net_filename);
		std::cout << "saved" << std::endl;
		logfile   << "saved" << std::endl;
		
		std::cout << "forward ( min = " << min_fp << " | max = " << max_fp << " | mean = " << mean_fp << " ) " << std::endl;
		std::cout << "backward ( min = " << min_bp << " | max = " << max_bp << " | mean = " << mean_bp << " ) " << std::endl;
	}
    return 0 ;   
}
