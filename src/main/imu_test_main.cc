// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2022 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
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
#include <weights_handler.hh>
#include <imu_log_handler.hh>
#include <lstm_stk_flt.hh>
#include <gru_stk_flt.hh>
#include <random>

inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_int_distribution<int> dist (0,1);


 
// test 1
// ./IMU_TEST D1 1 5 20 10 5 0.0005 0.9 0.001 0.9 0.999 0

// test 2 (act interp) Warning interp not included in opensource version
// ./IMU_TEST D1 1 5 20 10 5 0.0005 0.9 0.001 0.9 0.999 1



int main(int argc, char *argv[])
{	
	srand((int)time(NULL));
	if (argc < 11)
    {
		// EXAMPLE --> >> ./IMU_TEST_JB D2 3 5 20 10 5 0.0005 0.9 0.001 0.9 0.999 0
        std::cerr << "Usage: " << argv[0] << "<dataset_id> <epochs> <seqlength> <size_l1> <size_l2> <size_l3> <lr_sgd> <mu_sgd> <lr_adam> <b1_adam> <b2_adam> <optimact>" << std::endl;
		return 1;
    }
    
    std::string dataset_id_string(argv[1]);
    std::string epochs_string(argv[2]);
    std::string seqlength_string(argv[3]);   
    std::string size_l1_string(argv[4]);  
    std::string size_l2_string(argv[5]); 
    std::string size_l3_string(argv[6]);
    std::string lr_sgd_string(argv[7]);
    std::string mu_sgd_string(argv[8]);
    std::string lr_adam_string(argv[9]);
    std::string b1_adam_string(argv[10]);
    std::string b2_adam_string(argv[11]);
    std::string optimact_string(argv[12]);
    
    // log handler and weight handler
	static imu_log_handler my_imu_log_handler;
	weights_handler my_weights_handler;

	//LSTM
	static lstm_stk_flt my_lstm_sgd_stk_flt;
	static lstm_stk_flt my_lstm_sgd_stk_flt_phi;
	static lstm_stk_flt my_lstm_adam_stk_flt;
	//static lstm_tdim_stk_flt my_lstm_sgd_stk_flt_tdim;
	
	// GRU
	static gru_stk_flt my_gru_sgd_stk_flt;
	static gru_stk_flt my_gru_adam_stk_flt;
	
	// Weights
	std::vector<std::vector<float>> W00, W10, W20, W30, W30_phi;
	std::vector<std::vector<float>> W01, W11, W21, W31, W31_phi;
	std::vector<std::vector<float>> W02, W12, W22, W32, W32_phi;
	std::vector<std::vector<float>> W03, W13, W23, W33, W33_phi;
	std::vector<std::vector<float>> U00, U10, U20, U30, U30_phi;
	std::vector<std::vector<float>> U01, U11, U21, U31, U31_phi;
	std::vector<std::vector<float>> U02, U12, U22, U32, U32_phi;
	std::vector<std::vector<float>> U03, U13, U23, U33, U33_phi;
	
	// For time measure
	float min_ff_lstm_sgd = 100.0, max_ff_lstm_sgd =  0.0, mean_ff_lstm_sgd =  0.0, sum_ff_lstm_sgd =  0.0;
	float min_bp_lstm_sgd = 100.0, max_bp_lstm_sgd =  0.0, mean_bp_lstm_sgd =  0.0, sum_bp_lstm_sgd =  0.0;
	float min_ff_lstm_adam = 100.0, max_ff_lstm_adam =  0.0, mean_ff_lstm_adam =  0.0, sum_ff_lstm_adam =  0.0;
	float min_bp_lstm_adam = 100.0, max_bp_lstm_adam =  0.0, mean_bp_lstm_adam =  0.0, sum_bp_lstm_adam =  0.0;
	float min_ff_lstm_adamax = 100.0, max_ff_lstm_adamax =  0.0, mean_ff_lstm_adamax =  0.0, sum_ff_lstm_adamax =  0.0;
	float min_bp_lstm_adamax = 100.0, max_bp_lstm_adamax =  0.0, mean_bp_lstm_adamax =  0.0, sum_bp_lstm_adamax =  0.0;
	
	float min_ff_gru_sgd = 100.0, max_ff_gru_sgd =  0.0, mean_ff_gru_sgd =  0.0, sum_ff_gru_sgd =  0.0;
	float min_bp_gru_sgd = 100.0, max_bp_gru_sgd =  0.0, mean_bp_gru_sgd =  0.0, sum_bp_gru_sgd =  0.0;
	float min_ff_gru_adam = 100.0, max_ff_gru_adam =  0.0, mean_ff_gru_adam =  0.0, sum_ff_gru_adam =  0.0;
	float min_bp_gru_adam = 100.0, max_bp_gru_adam =  0.0, mean_bp_gru_adam =  0.0, sum_bp_gru_adam =  0.0;
	float min_ff_gru_adamax = 100.0, max_ff_gru_adamax =  0.0, mean_ff_gru_adamax =  0.0, sum_ff_gru_adamax =  0.0;
	float min_bp_gru_adamax = 100.0, max_bp_gru_adamax =  0.0, mean_bp_gru_adamax =  0.0, sum_bp_gru_adamax =  0.0;
	
	std::string csv_ext = ".csv";
	std::string arguments1 = dataset_id_string + "_" + epochs_string + "_" + seqlength_string + "_" + size_l1_string + "_" + size_l2_string + "_" + size_l3_string;
	std::string arguments2 = lr_sgd_string + "_" + mu_sgd_string + "_" + lr_adam_string + "_" + b1_adam_string + "_" + b2_adam_string + "_" + optimact_string;

	std::string path_to_rnnap_files = "../";
	if (dataset_id_string.compare("D1"))
	{
		std::string log_path  = path_to_rnnap_files + "datasets/imu/D1_scaled_pytorch.csv";
		my_imu_log_handler.load_imu_log(log_path.c_str());
	}
	else if (dataset_id_string.compare("D2"))
	{
		std::string log_path  = path_to_rnnap_files + "datasets/imu/D2_scaled_pytorch.csv";
		my_imu_log_handler.load_imu_log(log_path.c_str());
	}
	else if (dataset_id_string.compare("D3"))
	{
		std::string log_path  = path_to_rnnap_files + "datasets/imu/D3_scaled_pytorch.csv";
		my_imu_log_handler.load_imu_log(log_path.c_str());
	}
	else if (dataset_id_string.compare("D4"))
	{
		std::string log_path  = path_to_rnnap_files + "datasets/imu/D4_scaled_pytorch.csv";
		my_imu_log_handler.load_imu_log(log_path.c_str());
	}
	else
	{
		std::cerr << "No proper log file has been found !!!!!" << std::endl;
		return 1;
	}
	

	std::string reportfilenameOutputs = "outputs_" + arguments1 + arguments2 + csv_ext;
	std::string reportfilenameLosses = "losses_" + arguments1 + arguments2 + csv_ext;
	std::string reportfilenameTimings = "timings_" + arguments1 + arguments2 + csv_ext;

	//RNN SIZE
	uint32_t inputs_size = 9;  // ax,ay,az,gx,gy,gz,mx,my,mz
	uint32_t outputs_size = 3; // phi,theta,psi
	uint32_t epoch_max = (uint32_t) atoi(argv[2]);
    uint32_t seqlength = (uint32_t) atoi(argv[3]); // number of timesteps
	
	uint32_t size_layer0 = inputs_size;
	uint32_t size_layer1 = (uint32_t) atoi(argv[4]);
    uint32_t size_layer2 = (uint32_t) atoi(argv[5]);
    uint32_t size_layer3 = (uint32_t) atoi(argv[6]);
	uint32_t size_layer4 = outputs_size;
	
	float lr_sgd  = (float) atof(argv[7]);
    float mu_sgd  = (float) atof(argv[8]);
    float lr_adam = (float) atof(argv[9]);
    float b1_adam = (float) atof(argv[10]);
    float b2_adam = (float) atof(argv[11]);
    bool act_interp_fag = (bool) atoi(argv[12]);
	
	W00 = my_weights_handler.get_2d_weights_flt_xavier(size_layer0,size_layer1,size_layer0,size_layer1);
	W01 = my_weights_handler.get_2d_weights_flt_xavier(size_layer0,size_layer1,size_layer0,size_layer1);
	W02 = my_weights_handler.get_2d_weights_flt_xavier(size_layer0,size_layer1,size_layer0,size_layer1);
	W03 = my_weights_handler.get_2d_weights_flt_xavier(size_layer0,size_layer1,size_layer0,size_layer1);
	U00 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer1,size_layer1,size_layer1);
	U01 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer1,size_layer1,size_layer1);
	U02 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer1,size_layer1,size_layer1);
	U03 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer1,size_layer1,size_layer1);
	
	W10 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer2,size_layer1,size_layer2);
	W11 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer2,size_layer1,size_layer2);
	W12 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer2,size_layer1,size_layer2);
	W13 = my_weights_handler.get_2d_weights_flt_xavier(size_layer1,size_layer2,size_layer1,size_layer2);
	U10 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer2,size_layer2,size_layer2);
	U11 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer2,size_layer2,size_layer2);
	U12 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer2,size_layer2,size_layer2);
	U13 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer2,size_layer2,size_layer2);
	
	W20 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer3,size_layer2,size_layer3);
	W21 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer3,size_layer2,size_layer3);
	W22 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer3,size_layer2,size_layer3);
	W23 = my_weights_handler.get_2d_weights_flt_xavier(size_layer2,size_layer3,size_layer2,size_layer3);
	U20 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer3,size_layer3,size_layer3);
	U21 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer3,size_layer3,size_layer3);
	U22 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer3,size_layer3,size_layer3);
	U23 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer3,size_layer3,size_layer3);

	W30 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W31 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W32 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W33 = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	U30 = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U31 = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U32 = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U33 = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	
	std::vector<uint32_t> stk_layers_size = {size_layer0, size_layer1, size_layer2, size_layer3, size_layer4};
	
	size_layer4 = 1;
	
	W30_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W31_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W32_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	W33_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer3,size_layer4,size_layer3,size_layer4);
	U30_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U31_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U32_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	U33_phi = my_weights_handler.get_2d_weights_flt_xavier(size_layer4,size_layer4,size_layer4,size_layer4);
	
	std::vector<uint32_t> stk_layers_size_phi = {size_layer0, size_layer1, size_layer2, size_layer3, size_layer4};
	
	my_lstm_sgd_stk_flt.init(seqlength,stk_layers_size);
	my_lstm_sgd_stk_flt.set_lr_mu(lr_sgd, mu_sgd);
	my_lstm_sgd_stk_flt.set_Wa(W00,0);
	my_lstm_sgd_stk_flt.set_Wi(W01,0);
	my_lstm_sgd_stk_flt.set_Wf(W02,0);
	my_lstm_sgd_stk_flt.set_Wo(W03,0);
	my_lstm_sgd_stk_flt.set_Wa(W10,1);
	my_lstm_sgd_stk_flt.set_Wi(W11,1);
	my_lstm_sgd_stk_flt.set_Wf(W12,1);
	my_lstm_sgd_stk_flt.set_Wo(W13,1);
	my_lstm_sgd_stk_flt.set_Wa(W20,2);
	my_lstm_sgd_stk_flt.set_Wi(W21,2);
	my_lstm_sgd_stk_flt.set_Wf(W22,2);
	my_lstm_sgd_stk_flt.set_Wo(W23,2);
	my_lstm_sgd_stk_flt.set_Wa(W30,3);
	my_lstm_sgd_stk_flt.set_Wi(W31,3);
	my_lstm_sgd_stk_flt.set_Wf(W32,3);
	my_lstm_sgd_stk_flt.set_Wo(W33,3);
	my_lstm_sgd_stk_flt.set_Ua(U00,0);
	my_lstm_sgd_stk_flt.set_Ui(U01,0);
	my_lstm_sgd_stk_flt.set_Uf(U02,0);
	my_lstm_sgd_stk_flt.set_Uo(U03,0);
	my_lstm_sgd_stk_flt.set_Ua(U10,1);
	my_lstm_sgd_stk_flt.set_Ui(U11,1);
	my_lstm_sgd_stk_flt.set_Uf(U12,1);
	my_lstm_sgd_stk_flt.set_Uo(U13,1);
	my_lstm_sgd_stk_flt.set_Ua(U20,2);
	my_lstm_sgd_stk_flt.set_Ui(U21,2);
	my_lstm_sgd_stk_flt.set_Uf(U22,2);
	my_lstm_sgd_stk_flt.set_Uo(U23,2);
	my_lstm_sgd_stk_flt.set_Ua(U30,3);
	my_lstm_sgd_stk_flt.set_Ui(U31,3);
	my_lstm_sgd_stk_flt.set_Uf(U32,3);
	my_lstm_sgd_stk_flt.set_Uo(U33,3);
	my_lstm_sgd_stk_flt.set_interp_act(act_interp_fag);
	
	my_lstm_sgd_stk_flt_phi.init(seqlength,stk_layers_size_phi);
	my_lstm_sgd_stk_flt_phi.set_lr_mu(lr_sgd, mu_sgd);
	my_lstm_sgd_stk_flt_phi.set_Wa(W00,0);
	my_lstm_sgd_stk_flt_phi.set_Wi(W01,0);
	my_lstm_sgd_stk_flt_phi.set_Wf(W02,0);
	my_lstm_sgd_stk_flt_phi.set_Wo(W03,0);
	my_lstm_sgd_stk_flt_phi.set_Wa(W10,1);
	my_lstm_sgd_stk_flt_phi.set_Wi(W11,1);
	my_lstm_sgd_stk_flt_phi.set_Wf(W12,1);
	my_lstm_sgd_stk_flt_phi.set_Wo(W13,1);
	my_lstm_sgd_stk_flt_phi.set_Wa(W20,2);
	my_lstm_sgd_stk_flt_phi.set_Wi(W21,2);
	my_lstm_sgd_stk_flt_phi.set_Wf(W22,2);
	my_lstm_sgd_stk_flt_phi.set_Wo(W23,2);
	my_lstm_sgd_stk_flt_phi.set_Wa(W30_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Wi(W31_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Wf(W32_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Wo(W33_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Ua(U00,0);
	my_lstm_sgd_stk_flt_phi.set_Ui(U01,0);
	my_lstm_sgd_stk_flt_phi.set_Uf(U02,0);
	my_lstm_sgd_stk_flt_phi.set_Uo(U03,0);
	my_lstm_sgd_stk_flt_phi.set_Ua(U10,1);
	my_lstm_sgd_stk_flt_phi.set_Ui(U11,1);
	my_lstm_sgd_stk_flt_phi.set_Uf(U12,1);
	my_lstm_sgd_stk_flt_phi.set_Uo(U13,1);
	my_lstm_sgd_stk_flt_phi.set_Ua(U20,2);
	my_lstm_sgd_stk_flt_phi.set_Ui(U21,2);
	my_lstm_sgd_stk_flt_phi.set_Uf(U22,2);
	my_lstm_sgd_stk_flt_phi.set_Uo(U23,2);
	my_lstm_sgd_stk_flt_phi.set_Ua(U30_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Ui(U31_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Uf(U32_phi,3);
	my_lstm_sgd_stk_flt_phi.set_Uo(U33_phi,3);
	my_lstm_sgd_stk_flt_phi.set_interp_act(act_interp_fag);
	
	my_lstm_adam_stk_flt.init(seqlength,stk_layers_size);
	my_lstm_adam_stk_flt.set_lr_b1_b2_epsilon(lr_adam, b1_adam, b2_adam, 1e-08);
	my_lstm_adam_stk_flt.set_Wa(W00,0);
	my_lstm_adam_stk_flt.set_Wi(W01,0);
	my_lstm_adam_stk_flt.set_Wf(W02,0);
	my_lstm_adam_stk_flt.set_Wo(W03,0);
	my_lstm_adam_stk_flt.set_Wa(W10,1);
	my_lstm_adam_stk_flt.set_Wi(W11,1);
	my_lstm_adam_stk_flt.set_Wf(W12,1);
	my_lstm_adam_stk_flt.set_Wo(W13,1);
	my_lstm_adam_stk_flt.set_Wa(W20,2);
	my_lstm_adam_stk_flt.set_Wi(W21,2);
	my_lstm_adam_stk_flt.set_Wf(W22,2);
	my_lstm_adam_stk_flt.set_Wo(W23,2);
	my_lstm_adam_stk_flt.set_Wa(W30,3);
	my_lstm_adam_stk_flt.set_Wi(W31,3);
	my_lstm_adam_stk_flt.set_Wf(W32,3);
	my_lstm_adam_stk_flt.set_Wo(W33,3);
	my_lstm_adam_stk_flt.set_Ua(U00,0);
	my_lstm_adam_stk_flt.set_Ui(U01,0);
	my_lstm_adam_stk_flt.set_Uf(U02,0);
	my_lstm_adam_stk_flt.set_Uo(U03,0);
	my_lstm_adam_stk_flt.set_Ua(U10,1);
	my_lstm_adam_stk_flt.set_Ui(U11,1);
	my_lstm_adam_stk_flt.set_Uf(U12,1);
	my_lstm_adam_stk_flt.set_Uo(U13,1);
	my_lstm_adam_stk_flt.set_Ua(U20,2);
	my_lstm_adam_stk_flt.set_Ui(U21,2);
	my_lstm_adam_stk_flt.set_Uf(U22,2);
	my_lstm_adam_stk_flt.set_Uo(U23,2);
	my_lstm_adam_stk_flt.set_Ua(U30,3);
	my_lstm_adam_stk_flt.set_Ui(U31,3);
	my_lstm_adam_stk_flt.set_Uf(U32,3);
	my_lstm_adam_stk_flt.set_Uo(U33,3);
	my_lstm_adam_stk_flt.set_interp_act(act_interp_fag);

	my_gru_sgd_stk_flt.init(seqlength,stk_layers_size);
	my_gru_sgd_stk_flt.set_lr_mu(lr_sgd, mu_sgd);
	my_gru_sgd_stk_flt.set_Wz(W01,0);
	my_gru_sgd_stk_flt.set_Wr(W02,0);
	my_gru_sgd_stk_flt.set_Wh(W03,0);
	my_gru_sgd_stk_flt.set_Wz(W11,1);
	my_gru_sgd_stk_flt.set_Wr(W12,1);
	my_gru_sgd_stk_flt.set_Wh(W13,1);
	my_gru_sgd_stk_flt.set_Wz(W21,2);
	my_gru_sgd_stk_flt.set_Wr(W22,2);
	my_gru_sgd_stk_flt.set_Wh(W23,2);
	my_gru_sgd_stk_flt.set_Wz(W31,3);
	my_gru_sgd_stk_flt.set_Wr(W32,3);
	my_gru_sgd_stk_flt.set_Wh(W33,3);
	my_gru_sgd_stk_flt.set_Uz(U01,0);
	my_gru_sgd_stk_flt.set_Ur(U02,0);
	my_gru_sgd_stk_flt.set_Uh(U03,0);
	my_gru_sgd_stk_flt.set_Uz(U11,1);
	my_gru_sgd_stk_flt.set_Ur(U12,1);
	my_gru_sgd_stk_flt.set_Uh(U13,1);
	my_gru_sgd_stk_flt.set_Uz(U21,2);
	my_gru_sgd_stk_flt.set_Ur(U22,2);
	my_gru_sgd_stk_flt.set_Uh(U23,2);
	my_gru_sgd_stk_flt.set_Uz(U31,3);
	my_gru_sgd_stk_flt.set_Ur(U32,3);
	my_gru_sgd_stk_flt.set_Uh(U33,3);
	my_gru_sgd_stk_flt.set_interp_act(act_interp_fag);
	
	my_gru_adam_stk_flt.init(seqlength,stk_layers_size);
	my_gru_adam_stk_flt.set_lr_b1_b2_epsilon(lr_adam, b1_adam, b2_adam, 1e-08);
	my_gru_adam_stk_flt.set_Wz(W01,0);
	my_gru_adam_stk_flt.set_Wr(W02,0);
	my_gru_adam_stk_flt.set_Wh(W03,0);
	my_gru_adam_stk_flt.set_Wz(W11,1);
	my_gru_adam_stk_flt.set_Wr(W12,1);
	my_gru_adam_stk_flt.set_Wh(W13,1);
	my_gru_adam_stk_flt.set_Wz(W21,2);
	my_gru_adam_stk_flt.set_Wr(W22,2);
	my_gru_adam_stk_flt.set_Wh(W23,2);
	my_gru_adam_stk_flt.set_Wz(W31,3);
	my_gru_adam_stk_flt.set_Wr(W32,3);
	my_gru_adam_stk_flt.set_Wh(W33,3);
	my_gru_adam_stk_flt.set_Uz(U01,0);
	my_gru_adam_stk_flt.set_Ur(U02,0);
	my_gru_adam_stk_flt.set_Uh(U03,0);
	my_gru_adam_stk_flt.set_Uz(U11,1);
	my_gru_adam_stk_flt.set_Ur(U12,1);
	my_gru_adam_stk_flt.set_Uh(U13,1);
	my_gru_adam_stk_flt.set_Uz(U21,2);
	my_gru_adam_stk_flt.set_Ur(U22,2);
	my_gru_adam_stk_flt.set_Uh(U23,2);
	my_gru_adam_stk_flt.set_Uz(U31,3);
	my_gru_adam_stk_flt.set_Ur(U32,3);
	my_gru_adam_stk_flt.set_Uh(U33,3);
	my_gru_adam_stk_flt.set_interp_act(act_interp_fag);
	
	
	// TRAINING TUNNING
	uint32_t nb_train_samples = 0;
	uint32_t nb_test_samples = 0;

	uint32_t it1, it2, it3, it4;
	
	float lstm_train_sgd_loss_stk = 0.0;
	float lstm_test_sgd_loss_stk = 0.0;
	float lstm_train_adam_loss_stk = 0.0;
	float lstm_test_adam_loss_stk = 0.0;
	float lstm_train_sgd_loss_stk_phi = 0.0;
	float lstm_test_sgd_loss_stk_phi = 0.0;
	//float lstm_train_sgd_loss_stk_tdim = 0.0;
	//float lstm_test_sgd_loss_stk_tdim = 0.0;
	
	float gru_train_sgd_loss_stk = 0.0;
	float gru_test_sgd_loss_stk = 0.0;
	float gru_train_adam_loss_stk = 0.0;
	float gru_test_adam_loss_stk = 0.0;

	std::vector<std::vector<std::vector<float>>> rnn_input_train_rdm;
	std::vector<std::vector<std::vector<float>>> rnn_target_train_rdm;
	std::vector<std::vector<std::vector<float>>> rnn_target_train_rdm_phi;
	std::vector<std::vector<std::vector<float>>> rnn_input_test;
	std::vector<std::vector<std::vector<float>>> rnn_target_test;
	std::vector<std::vector<std::vector<float>>> rnn_target_test_phi;
	
	std::vector<float> lstm_sgd_stk_output;
	std::vector<float> lstm_sgd_stk_output_phi;
	//std::vector<float> lstm_sgd_stk_output_tdim;
	std::vector<float> lstm_adam_stk_output;
	
	std::vector<float> gru_sgd_stk_output;
	std::vector<float> gru_adam_stk_output;
	

	my_imu_log_handler.create_split_and_create_random_dataset(0.3);
	nb_train_samples = my_imu_log_handler.get_max_train_samples();
	nb_test_samples = my_imu_log_handler.get_max_test_samples();
	
	for(it1=0; it1<epoch_max; it1++)//EPOCH
	{
		lstm_train_sgd_loss_stk = 0.0;
		lstm_test_sgd_loss_stk = 0.0;
		lstm_train_adam_loss_stk = 0.0;
		lstm_test_adam_loss_stk = 0.0;
		lstm_train_sgd_loss_stk_phi = 0.0;
		lstm_test_sgd_loss_stk_phi = 0.0;
		
		//lstm_train_sgd_loss_stk_tdim = 0.0;
		//lstm_test_sgd_loss_stk_tdim = 0.0;
		
		gru_train_sgd_loss_stk = 0.0;
		gru_test_sgd_loss_stk = 0.0;
		gru_train_adam_loss_stk = 0.0;
		gru_test_adam_loss_stk = 0.0;
		
		
		sum_ff_lstm_sgd =  sum_bp_lstm_sgd = sum_ff_lstm_adam = sum_bp_lstm_adam = sum_ff_lstm_adamax = sum_bp_lstm_adamax = 0.0;
		sum_ff_gru_sgd =  sum_bp_gru_sgd = sum_ff_gru_adam = sum_bp_gru_adam = sum_ff_gru_adamax = sum_bp_gru_adamax = 0.0;

		auto start = std::chrono::high_resolution_clock::now();
		my_imu_log_handler.randomize_training_dataset();

		// Training 
		rnn_input_train_rdm  = my_imu_log_handler.get_input_rdm_train(seqlength);
		rnn_target_train_rdm = my_imu_log_handler.get_target_rdm_train(seqlength);
		rnn_target_train_rdm_phi = my_imu_log_handler.get_target_rdm_train(seqlength);
		my_lstm_adam_stk_flt.clear_momentums();
		for (it2=0;it2<nb_train_samples;it2++)
		{
			// LSTM SGD ALL
			auto start_bp_lstm_sgd = std::chrono::high_resolution_clock::now();
			my_lstm_sgd_stk_flt.forward_train(rnn_input_train_rdm[it2]);
			my_lstm_sgd_stk_flt.backward_sgd_train(rnn_target_train_rdm[it2]);
			auto finish_bp_lstm_sgd = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_bp_lstm_sgd = finish_bp_lstm_sgd-start_bp_lstm_sgd;
			float elapsed_ms_bp_lstm_sgd = elapsed_bp_lstm_sgd.count();
			if (elapsed_ms_bp_lstm_sgd<min_bp_lstm_sgd) min_bp_lstm_sgd = elapsed_ms_bp_lstm_sgd;
			if (elapsed_ms_bp_lstm_sgd>max_bp_lstm_sgd) max_bp_lstm_sgd = elapsed_ms_bp_lstm_sgd;
			sum_bp_lstm_sgd = sum_bp_lstm_sgd + elapsed_ms_bp_lstm_sgd;
			
			// LSTM ADAM ALL
			auto start_bp_lstm_adam = std::chrono::high_resolution_clock::now();
			my_lstm_adam_stk_flt.forward_train(rnn_input_train_rdm[it2]);
			my_lstm_adam_stk_flt.backward_adam_train(rnn_target_train_rdm[it2]); // call here adam bp
			auto finish_bp_lstm_adam = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_bp_lstm_adam = finish_bp_lstm_adam-start_bp_lstm_adam;
			float elapsed_ms_bp_lstm_adam = elapsed_bp_lstm_adam.count();
			if (elapsed_ms_bp_lstm_adam<min_bp_lstm_adam) min_bp_lstm_adam = elapsed_ms_bp_lstm_adam;
			if (elapsed_ms_bp_lstm_adam>max_bp_lstm_adam) max_bp_lstm_adam = elapsed_ms_bp_lstm_adam;
			sum_bp_lstm_adam = sum_bp_lstm_adam + elapsed_ms_bp_lstm_adam;
			
			// LSTM SGD PHI
			my_lstm_sgd_stk_flt_phi.forward_train(rnn_input_train_rdm[it2]);
			my_lstm_sgd_stk_flt_phi.backward_sgd_train(rnn_target_train_rdm_phi[it2]);
			
			// GRU SGD ALL
			auto start_bp_gru_sgd = std::chrono::high_resolution_clock::now();
			my_gru_sgd_stk_flt.forward_train(rnn_input_train_rdm[it2]);
			my_gru_sgd_stk_flt.backward_sgd_train(rnn_target_train_rdm[it2]);
			auto finish_bp_gru_sgd = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_bp_gru_sgd = finish_bp_gru_sgd-start_bp_gru_sgd;
			float elapsed_ms_bp_gru_sgd = elapsed_bp_gru_sgd.count();
			if (elapsed_ms_bp_gru_sgd<min_bp_gru_sgd) min_bp_gru_sgd = elapsed_ms_bp_gru_sgd;
			if (elapsed_ms_bp_gru_sgd>max_bp_gru_sgd) max_bp_gru_sgd = elapsed_ms_bp_gru_sgd;
			sum_bp_gru_sgd = sum_bp_gru_sgd + elapsed_ms_bp_gru_sgd;
			// GRU ADAM ALL
			auto start_bp_gru_adam = std::chrono::high_resolution_clock::now();
			my_gru_adam_stk_flt.forward_train(rnn_input_train_rdm[it2]);
			my_gru_adam_stk_flt.backward_adam_train(rnn_target_train_rdm[it2]); // call here adam bp
			auto finish_bp_gru_adam = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_bp_gru_adam = finish_bp_gru_adam-start_bp_gru_adam;
			float elapsed_ms_bp_gru_adam = elapsed_bp_gru_adam.count();
			if (elapsed_ms_bp_gru_adam<min_bp_gru_adam) min_bp_gru_adam = elapsed_ms_bp_gru_adam;
			if (elapsed_ms_bp_gru_adam>max_bp_gru_adam) max_bp_gru_adam = elapsed_ms_bp_gru_adam;
			sum_bp_gru_adam = sum_bp_gru_adam + elapsed_ms_bp_gru_adam;
			
			// ERRORS ACCUMULATIONS
			lstm_train_sgd_loss_stk += my_lstm_sgd_stk_flt.get_mse();
			lstm_train_adam_loss_stk += my_lstm_adam_stk_flt.get_mse();
			lstm_train_sgd_loss_stk_phi += my_lstm_sgd_stk_flt_phi.get_mse();
			//lstm_train_sgd_loss_stk_tdim += my_lstm_sgd_stk_flt_tdim.get_mse();
			gru_train_sgd_loss_stk += my_gru_sgd_stk_flt.get_mse();
			gru_train_adam_loss_stk += my_gru_adam_stk_flt.get_mse();
		}
		lstm_train_sgd_loss_stk /= nb_train_samples;
		lstm_train_adam_loss_stk /= nb_train_samples;
		lstm_train_sgd_loss_stk_phi /= nb_train_samples; 
		gru_train_sgd_loss_stk /= nb_train_samples;
		gru_train_adam_loss_stk /= nb_train_samples; 
		
		// TIMINGS
		mean_bp_lstm_sgd=sum_bp_lstm_sgd/nb_train_samples;
		mean_bp_lstm_adam=sum_bp_lstm_adam/nb_train_samples;
		mean_bp_gru_sgd=sum_bp_gru_sgd/nb_train_samples;
		mean_bp_gru_adam=sum_bp_gru_adam/nb_train_samples;
		
		std::vector<float> timings; timings.push_back(0.0); timings.push_back(0.0); timings.push_back(0.0); 
		timings[0] = min_bp_lstm_sgd; timings[1] = max_bp_lstm_sgd; timings[2] = mean_bp_lstm_sgd; 
		my_imu_log_handler.add_epoch_train_time_for_lstm_sgd_all(timings);
		timings[0] = min_bp_gru_sgd; timings[1] = max_bp_gru_sgd; timings[2] = mean_bp_gru_sgd; 
		my_imu_log_handler.add_epoch_train_time_for_gru_sgd_all(timings);
		timings[0] = min_bp_lstm_adam; timings[1] = max_bp_lstm_adam; timings[2] = mean_bp_lstm_adam; 
		my_imu_log_handler.add_epoch_train_time_for_lstm_adam_all(timings);
		timings[0] = min_bp_gru_adam; timings[1] = max_bp_gru_adam; timings[2] = mean_bp_gru_adam; 
		my_imu_log_handler.add_epoch_train_time_for_gru_adam_all(timings);
		
		// WRITE TRAIN LOSSES IN THE LOG
		my_imu_log_handler.add_epoch_train_error_for_lstm_sgd_all(lstm_train_sgd_loss_stk);
		my_imu_log_handler.add_epoch_train_error_for_gru_sgd_all(gru_train_sgd_loss_stk);
		my_imu_log_handler.add_epoch_train_error_for_lstm_adam_all(lstm_train_adam_loss_stk);
		my_imu_log_handler.add_epoch_train_error_for_gru_adam_all(gru_train_adam_loss_stk);
		
		// Testing 
		rnn_input_test  = my_imu_log_handler.get_input_test(seqlength);
		rnn_target_test = my_imu_log_handler.get_target_test(seqlength);
		rnn_target_test_phi = my_imu_log_handler.get_target_phi_test(seqlength);
		//std::cout << "rnn_target_test.size()=" << rnn_target_test.size() << std::endl;
		std::vector<std::vector<float>> lstm_sgd_stk_output_epoch;
		std::vector<std::vector<float>> lstm_sgd_stk_output_epoch_phi;
		std::vector<std::vector<float>> lstm_adam_stk_output_epoch;
		std::vector<std::vector<float>> gru_sgd_stk_output_epoch;
		std::vector<std::vector<float>> gru_adam_stk_output_epoch;
		
		for (it2=0;it2<nb_test_samples;it2++)
		{
			auto start_ff_lstm_sgd = std::chrono::high_resolution_clock::now();
			my_lstm_sgd_stk_flt.forward_train(rnn_input_test[it2]);
			lstm_sgd_stk_output = my_lstm_sgd_stk_flt.get_output();
			lstm_sgd_stk_output_epoch.push_back(lstm_sgd_stk_output);
			auto finish_ff_lstm_sgd = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_ff_lstm_sgd = finish_ff_lstm_sgd-start_ff_lstm_sgd;
			float elapsed_ms_ff_lstm_sgd = elapsed_ff_lstm_sgd.count();
			if (elapsed_ms_ff_lstm_sgd<min_ff_lstm_sgd) min_ff_lstm_sgd = elapsed_ms_ff_lstm_sgd;
			if (elapsed_ms_ff_lstm_sgd>max_ff_lstm_sgd) max_ff_lstm_sgd = elapsed_ms_ff_lstm_sgd;
			sum_ff_lstm_sgd = sum_ff_lstm_sgd + elapsed_ms_ff_lstm_sgd;
			
			my_lstm_sgd_stk_flt_phi.forward_train(rnn_input_test[it2]);
			lstm_sgd_stk_output_phi = my_lstm_sgd_stk_flt_phi.get_output();
			lstm_sgd_stk_output_epoch_phi.push_back(lstm_sgd_stk_output_phi);
			
			auto start_ff_lstm_adam = std::chrono::high_resolution_clock::now();
			my_lstm_adam_stk_flt.forward_train(rnn_input_test[it2]);
			lstm_adam_stk_output = my_lstm_adam_stk_flt.get_output();
			lstm_adam_stk_output_epoch.push_back(lstm_adam_stk_output);
			auto finish_ff_lstm_adam = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_ff_lstm_adam = finish_ff_lstm_adam-start_ff_lstm_adam;
			float elapsed_ms_ff_lstm_adam = elapsed_ff_lstm_adam.count();
			if (elapsed_ms_ff_lstm_adam<min_ff_lstm_adam) min_ff_lstm_adam = elapsed_ms_ff_lstm_adam;
			if (elapsed_ms_ff_lstm_adam>max_ff_lstm_adam) max_ff_lstm_adam = elapsed_ms_ff_lstm_adam;
			sum_ff_lstm_adam = sum_ff_lstm_adam + elapsed_ms_ff_lstm_adam;
			
			auto start_ff_gru_sgd = std::chrono::high_resolution_clock::now();
			my_gru_sgd_stk_flt.forward_train(rnn_input_test[it2]);
			gru_sgd_stk_output = my_gru_sgd_stk_flt.get_output();
			gru_sgd_stk_output_epoch.push_back(gru_sgd_stk_output);
			auto finish_ff_gru_sgd = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_ff_gru_sgd = finish_ff_gru_sgd-start_ff_gru_sgd;
			float elapsed_ms_ff_gru_sgd = elapsed_ff_gru_sgd.count();
			if (elapsed_ms_ff_gru_sgd<min_ff_gru_sgd) min_ff_gru_sgd = elapsed_ms_ff_gru_sgd;
			if (elapsed_ms_ff_gru_sgd>max_ff_gru_sgd) max_ff_gru_sgd = elapsed_ms_ff_gru_sgd;
			sum_ff_gru_sgd = sum_ff_gru_sgd + elapsed_ms_ff_gru_sgd;
			
			auto start_ff_gru_adam = std::chrono::high_resolution_clock::now();
			my_gru_adam_stk_flt.forward_train(rnn_input_test[it2]);
			gru_adam_stk_output = my_gru_adam_stk_flt.get_output();
			gru_adam_stk_output_epoch.push_back(gru_adam_stk_output);
			auto finish_ff_gru_adam = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> elapsed_ff_gru_adam = finish_ff_gru_adam-start_ff_gru_adam;
			float elapsed_ms_ff_gru_adam = elapsed_ff_gru_adam.count();
			if (elapsed_ms_ff_gru_adam<min_ff_gru_adam) min_ff_gru_adam = elapsed_ms_ff_gru_adam;
			if (elapsed_ms_ff_gru_adam>max_ff_gru_adam) max_ff_gru_adam = elapsed_ms_ff_gru_adam;
			sum_ff_gru_adam = sum_ff_gru_adam + elapsed_ms_ff_gru_adam;
			

			lstm_test_sgd_loss_stk += my_lstm_sgd_stk_flt.get_mse_testing(rnn_target_test[it2]);
			lstm_test_sgd_loss_stk_phi += my_lstm_sgd_stk_flt_phi.get_mse_testing(rnn_target_test_phi[it2]);
			lstm_test_adam_loss_stk += my_lstm_adam_stk_flt.get_mse_testing(rnn_target_test[it2]);
			gru_test_sgd_loss_stk += my_gru_sgd_stk_flt.get_mse_testing(rnn_target_test[it2]);
			gru_test_adam_loss_stk += my_gru_adam_stk_flt.get_mse_testing(rnn_target_test[it2]);
		}
		lstm_test_sgd_loss_stk /= nb_test_samples;
		lstm_test_sgd_loss_stk_phi /= nb_test_samples;
		lstm_test_adam_loss_stk /= nb_test_samples;
		gru_test_sgd_loss_stk /= nb_test_samples;
		gru_test_adam_loss_stk /= nb_test_samples;
		
		// TIMINGS
		mean_ff_lstm_sgd=sum_ff_lstm_sgd/nb_test_samples;
		mean_ff_lstm_adam=sum_ff_lstm_adam/nb_test_samples;
		mean_ff_gru_sgd=sum_ff_gru_sgd/nb_test_samples;
		mean_ff_gru_adam=sum_ff_gru_adam/nb_test_samples;
		
		timings[0] = min_ff_lstm_sgd; timings[1] = max_ff_lstm_sgd; timings[2] = mean_ff_lstm_sgd; 
		my_imu_log_handler.add_epoch_test_time_for_lstm_sgd_all(timings);
		timings[0] = min_ff_gru_sgd; timings[1] = max_ff_gru_sgd; timings[2] = mean_ff_gru_sgd; 
		my_imu_log_handler.add_epoch_test_time_for_gru_sgd_all(timings);
		timings[0] = min_ff_lstm_adam; timings[1] = max_ff_lstm_adam; timings[2] = mean_ff_lstm_adam; 
		my_imu_log_handler.add_epoch_test_time_for_lstm_adam_all(timings);
		timings[0] = min_ff_gru_adam; timings[1] = max_ff_gru_adam; timings[2] = mean_ff_gru_adam; 
		my_imu_log_handler.add_epoch_test_time_for_gru_adam_all(timings);
		
		// WRITE NN OUTPUTS IN THE LOG
		my_imu_log_handler.add_one_epoch_for_lstm_sgd_all(lstm_sgd_stk_output_epoch);
		my_imu_log_handler.add_one_epoch_for_lstm_sgd_phi(lstm_sgd_stk_output_epoch_phi);
		my_imu_log_handler.add_one_epoch_for_lstm_adam_all(lstm_adam_stk_output_epoch);
		my_imu_log_handler.add_one_epoch_for_gru_sgd_all(gru_sgd_stk_output_epoch);
		my_imu_log_handler.add_one_epoch_for_gru_adam_all(gru_adam_stk_output_epoch);
		
		// WRITE TEST LOSSES IN THE LOG
		my_imu_log_handler.add_epoch_test_error_for_lstm_sgd_all(lstm_test_sgd_loss_stk);
		my_imu_log_handler.add_epoch_test_error_for_gru_sgd_all(gru_test_sgd_loss_stk);
		my_imu_log_handler.add_epoch_test_error_for_lstm_adam_all(lstm_test_adam_loss_stk);
		my_imu_log_handler.add_epoch_test_error_for_gru_adam_all(gru_test_adam_loss_stk);
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = finish - start;
		std::cout << "-------------------------------------------------------------------------------------" << std::endl;
		std::cout << "LSTM-ALL SGD  epoch " << it1 << " - train_loss = " << std::setprecision(9) << lstm_train_sgd_loss_stk << " - test_loss = " << std::setprecision(9) << lstm_test_sgd_loss_stk <<" - time = " << std::setprecision(5) << elapsed.count() << std::endl;
		std::cout << "LSTM-ALL ADAM epoch " << it1 << " - train_loss = " << std::setprecision(9) << lstm_train_adam_loss_stk << " - test_loss = " << std::setprecision(9) << lstm_test_adam_loss_stk <<" - time = " << std::setprecision(5) << elapsed.count() << std::endl;
		std::cout << "LSTM-PHI SGD  epoch " << it1 << " - train_loss = " << std::setprecision(9) << lstm_train_sgd_loss_stk_phi << " - test_loss = " << std::setprecision(9) << lstm_test_sgd_loss_stk_phi <<" - time = " << std::setprecision(5) << elapsed.count() << std::endl;
		std::cout << "GRU-ALL  SGD  epoch " << it1 << " - train_loss = " << std::setprecision(9) << gru_train_sgd_loss_stk  << " - test_loss = " << std::setprecision(9) << gru_test_sgd_loss_stk  <<" - time = " << std::setprecision(5) << elapsed.count() << std::endl;
		std::cout << "GRU-ALL  ADAM epoch " << it1 << " - train_loss = " << std::setprecision(9) << gru_train_adam_loss_stk  << " - test_loss = " << std::setprecision(9) << gru_test_adam_loss_stk <<" - time = " << std::setprecision(5) << elapsed.count() << std::endl;
		std::cout << " ... " << std::endl;
	}
	
	my_imu_log_handler.create_report(reportfilenameOutputs.c_str());
	my_imu_log_handler.create_loss_report(reportfilenameLosses.c_str());
	my_imu_log_handler.create_time_report(reportfilenameTimings.c_str());
		
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "---------------------- TRAINING PERFORMANCES --------------------------" << std::endl;
	std::cout << "------------              Backpropagation (BP)          ---------------" << std::endl;
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	std::cout << "type | min_bp | max_bp | mean_bp" << std::endl;
	std::cout << "bp_lstm_sgd | " << min_bp_lstm_sgd << " | " << max_bp_lstm_sgd << " | " << mean_bp_lstm_sgd << std::endl;
	std::cout << "bp_lstm_adam | " << min_bp_lstm_adam << " | " << max_bp_lstm_adam << " | " << mean_bp_lstm_adam << std::endl;
	std::cout << "bp_gru_sgd | " << min_bp_gru_sgd << " | " << max_bp_gru_sgd << " | " << mean_bp_gru_sgd << std::endl;
	std::cout << "bp_gru_adam | " << min_bp_gru_adam << " | " << max_bp_gru_adam << " | " << mean_bp_gru_adam << std::endl;
	
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << "---------------------- TESTING PERFORMANCES --------------------------" << std::endl;
	std::cout << "------------             Feedforward (FF)              ---------------" << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << "type | min_bp | max_bp | mean_bp" << std::endl;
	std::cout << "ff_lstm_sgd | " << min_ff_lstm_sgd << " | " << max_ff_lstm_sgd << " | " << mean_ff_lstm_sgd << std::endl;
	std::cout << "ff_lstm_adam | " << min_ff_lstm_adam << " | " << max_ff_lstm_adam << " | " << mean_ff_lstm_adam << std::endl;
	std::cout << "ff_gru_sgd | " << min_ff_gru_sgd << " | " << max_ff_gru_sgd << " | " << mean_ff_gru_sgd << std::endl;
	std::cout << "ff_gru_adam | " << min_ff_gru_adam << " | " << max_ff_gru_adam << " | " << mean_ff_gru_adam << std::endl;
	
    return 0 ;   
}
