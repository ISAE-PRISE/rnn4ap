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

#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream> 
#include <sstream>
#include <iomanip>      // std::setprecision

#ifndef __GRU_STK_FLT_HH__
#define __GRU_STK_FLT_HH__

class gru_stk_flt
{
	public:
		gru_stk_flt();
		~gru_stk_flt();
		
		void init(uint32_t nb_timestep, std::vector<uint32_t> layers_size);
		void set_lr_mu(float lr, float mu);
		void set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon);
		void forward(std::vector<float> input);
		void forward_train(std::vector<std::vector<float>> input);
		void backward_sgd_train(std::vector<std::vector<float>> target);
		void backward_sgd_train_mnist(std::vector<float> target);
		void backward_adam_train(std::vector<std::vector<float>> target);
		void backward_adamax_train(std::vector<std::vector<float>> target);
		float get_mse();
		float get_mse_testing(std::vector<std::vector<float>> target);
		float get_mse_mlp();
		float randomize(float min, float max);
		void clear_grads();
		void clear_momentums();
		std::vector<float> get_output();
		std::vector<float> get_output_mlp();
		std::vector<float> get_error();
		//float get_mse();
		
		// set/get weights and bias
		std::vector<std::vector<float>> get_Wz(uint32_t i){return _Wz[i];};
		std::vector<std::vector<float>> get_Wr(uint32_t i){return _Wr[i];};
		std::vector<std::vector<float>> get_Wh(uint32_t i){return _Wh[i];};
		
		std::vector<std::vector<float>> get_Uz(uint32_t i){return _Uz[i];};
		std::vector<std::vector<float>> get_Ur(uint32_t i){return _Ur[i];};
		std::vector<std::vector<float>> get_Uh(uint32_t i){return _Uh[i];};

		std::vector<float> get_bz(uint32_t i){return _bz[i];};
		std::vector<float> get_br(uint32_t i){return _br[i];};
		std::vector<float> get_bh(uint32_t i){return _bh[i];};
		
		void set_Wz(std::vector<std::vector<float>> Wz, uint32_t i){_Wz[i] = Wz;};
		void set_Wr(std::vector<std::vector<float>> Wr, uint32_t i){_Wr[i] = Wr;};
		void set_Wh(std::vector<std::vector<float>> Wh, uint32_t i){_Wh[i] = Wh;};
		
		void set_Uz(std::vector<std::vector<float>> Uz, uint32_t i){_Uz[i] = Uz;};
		void set_Ur(std::vector<std::vector<float>> Ur, uint32_t i){_Ur[i] = Ur;};
		void set_Uh(std::vector<std::vector<float>> Uh, uint32_t i){_Uh[i] = Uh;};
		void set_Uo(std::vector<std::vector<float>> Uo, uint32_t i){_Uo[i] = Uo;};
		
		void set_bz(std::vector<float> bz, uint32_t i){_bz[i] = bz;};
		void set_br(std::vector<float> br, uint32_t i){_br[i] = br;};
		void set_bh(std::vector<float> bh, uint32_t i){_bh[i] = bh;};
		
		//activation functions
		float sigmoid(float x);
		float dsigmoid(float x);
		float dtanh(float x);
		float leakyrelu(float x);
		float dleakyrelu(float x);
		float softplus(float x);
		float dsoftplus(float x);
		float ddtanh(float x);
		//use optimized activations
		float tanh_interp(float x);
		float sigmoid_interp(float x);
		void set_interp_act(bool val);
		
		std::vector<float> softmax(std::vector<float> input);
		std::vector<float> dsoftmax(std::vector<float> input);
		
		// naming stuff
		void set_name(std::string name){_id_name = name;};
		
		void forward_mlp_layer(std::vector<float> input);
		void backward_mlp_layer(std::vector<float> target);
		
	private: 
	
		std::string _id_name;
		std::ofstream _header_file;
		std::string _header_file_name;
		std::ofstream _main_file;
		std::string _main_file_name;
	
		// flags
		bool _init_ok;
		bool _loaded;
		bool _use_interp_act;
	
		// sizes 
		std::vector<uint32_t> _layers_size;
		uint32_t _total_size;
		uint32_t _nb_timestep, _input_size, _output_size;
		uint32_t _load_cnt;
		
		// optimizer choice 1=SGB / 2=ADAM / 3=ADAMAX
		uint8_t _optim_id;
	
		// hyper-parameters
		float _lr; //learning rate
		float _mu; //momentum
		float _b1; 
		float _b2; 
		float _epsilon;
		
		// softmax
		std::vector<float> _softmax;  // values
		std::vector<float> _dsoftmax;  // values
		
		// layers
		std::vector<std::vector<std::vector<float>>> _layers;  // values
		std::vector<std::vector<std::vector<float>>> _dlayers; // total gradients 
		std::vector<std::vector<std::vector<float>>> _dxlayers; // gradients for input
		std::vector<std::vector<std::vector<float>>> _dhlayers; // gradients for output (re-injected h)
		
		// gates
		std::vector<std::vector<std::vector<float>>> _z_gate;
		std::vector<std::vector<std::vector<float>>> _r_gate;
		std::vector<std::vector<std::vector<float>>> _h_gate;
		
		std::vector<std::vector<std::vector<float>>> _dz_gate;
		std::vector<std::vector<std::vector<float>>> _dr_gate;
		std::vector<std::vector<std::vector<float>>> _dh_gate;
		
		std::vector<std::vector<std::vector<float>>> _o_gate;
		std::vector<std::vector<std::vector<float>>> _do_gate;
		
		// bias
		std::vector<std::vector<float>> _bz;
		std::vector<std::vector<float>> _br;
		std::vector<std::vector<float>> _bh;
		std::vector<std::vector<float>> _dbz;
		std::vector<std::vector<float>> _dbr;
		std::vector<std::vector<float>> _dbh;
		std::vector<std::vector<float>> _mbz;
		std::vector<std::vector<float>> _mbr;
		std::vector<std::vector<float>> _mbh;
		std::vector<std::vector<float>> _vbz;
		std::vector<std::vector<float>> _vbr;
		std::vector<std::vector<float>> _vbh;
		
		// W weights
		std::vector<std::vector<std::vector<float>>> _Wz;
		std::vector<std::vector<std::vector<float>>> _Wr;
		std::vector<std::vector<std::vector<float>>> _Wh;
		std::vector<std::vector<std::vector<float>>> _dWz;
		std::vector<std::vector<std::vector<float>>> _dWr;
		std::vector<std::vector<std::vector<float>>> _dWh;
		std::vector<std::vector<std::vector<float>>> _mWz;
		std::vector<std::vector<std::vector<float>>> _mWr;
		std::vector<std::vector<std::vector<float>>> _mWh;
		std::vector<std::vector<std::vector<float>>> _vWz;
		std::vector<std::vector<std::vector<float>>> _vWr;
		std::vector<std::vector<std::vector<float>>> _vWh;
		
		// U weights
		std::vector<std::vector<std::vector<float>>> _Uz;
		std::vector<std::vector<std::vector<float>>> _Ur;
		std::vector<std::vector<std::vector<float>>> _Uh;
		std::vector<std::vector<std::vector<float>>> _dUz;
		std::vector<std::vector<std::vector<float>>> _dUr;
		std::vector<std::vector<std::vector<float>>> _dUh;
		std::vector<std::vector<std::vector<float>>> _mUz;
		std::vector<std::vector<std::vector<float>>> _mUr;
		std::vector<std::vector<std::vector<float>>> _mUh;
		std::vector<std::vector<std::vector<float>>> _vUz;
		std::vector<std::vector<std::vector<float>>> _vUr;
		std::vector<std::vector<std::vector<float>>> _vUh;
		
		std::vector<std::vector<std::vector<float>>> _Uo;
		std::vector<std::vector<std::vector<float>>> _dUo;
		
		std::vector<std::vector<float>> _bo;
		std::vector<std::vector<float>> _dbo;
		
		std::vector<std::vector<float>> _errors;
			
};

#endif // __LSTM_STD_FLT_HH__
