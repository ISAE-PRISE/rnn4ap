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

#ifndef __LSTM_STK_FLT_HH__
#define __LSTM_STK_FLT_HH__

// cuda impl: https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt

class lstm_stk_flt
{
	public:
		lstm_stk_flt();
		~lstm_stk_flt();
		
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
		std::vector<std::vector<float>> get_Wa(uint32_t i){return _Wa[i];};
		std::vector<std::vector<float>> get_Wi(uint32_t i){return _Wi[i];};
		std::vector<std::vector<float>> get_Wf(uint32_t i){return _Wf[i];};
		std::vector<std::vector<float>> get_Wo(uint32_t i){return _Wo[i];};
		std::vector<std::vector<float>> get_Ua(uint32_t i){return _Ua[i];};
		std::vector<std::vector<float>> get_Ui(uint32_t i){return _Ui[i];};
		std::vector<std::vector<float>> get_Uf(uint32_t i){return _Uf[i];};
		std::vector<std::vector<float>> get_Uo(uint32_t i){return _Uo[i];};
		std::vector<float> get_ba(uint32_t i){return _ba[i];};
		std::vector<float> get_bi(uint32_t i){return _bi[i];};
		std::vector<float> get_bf(uint32_t i){return _bf[i];};
		std::vector<float> get_bo(uint32_t i){return _bo[i];};
		
		void set_Wa(std::vector<std::vector<float>> Wa, uint32_t i){_Wa[i] = Wa;};
		void set_Wi(std::vector<std::vector<float>> Wi, uint32_t i){_Wi[i] = Wi;};
		void set_Wf(std::vector<std::vector<float>> Wf, uint32_t i){_Wf[i] = Wf;};
		void set_Wo(std::vector<std::vector<float>> Wo, uint32_t i){_Wo[i] = Wo;};
		void set_Ua(std::vector<std::vector<float>> Ua, uint32_t i){_Ua[i] = Ua;};
		void set_Ui(std::vector<std::vector<float>> Ui, uint32_t i){_Ui[i] = Ui;};
		void set_Uf(std::vector<std::vector<float>> Uf, uint32_t i){_Uf[i] = Uf;};
		void set_Uo(std::vector<std::vector<float>> Uo, uint32_t i){_Uo[i] = Uo;};
		void set_ba(std::vector<float> ba, uint32_t i){_ba[i] = ba;};
		void set_bi(std::vector<float> bi, uint32_t i){_bi[i] = bi;};
		void set_bf(std::vector<float> bf, uint32_t i){_bf[i] = bf;};
		void set_bo(std::vector<float> bo, uint32_t i){_bo[i] = bo;};
		
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
		
		bool _use_interp_act;
		
		// layers
		std::vector<std::vector<std::vector<float>>> _layers;  // values
		std::vector<std::vector<std::vector<float>>> _dlayers; // total gradients 
		std::vector<std::vector<std::vector<float>>> _dxlayers; // gradients for input
		std::vector<std::vector<std::vector<float>>> _dhlayers; // gradients for output (re-injected h)
		
		// gates
		std::vector<std::vector<std::vector<float>>> _a_gate;
		std::vector<std::vector<std::vector<float>>> _i_gate;
		std::vector<std::vector<std::vector<float>>> _f_gate;
		std::vector<std::vector<std::vector<float>>> _o_gate;
		std::vector<std::vector<std::vector<float>>> _da_gate;
		std::vector<std::vector<std::vector<float>>> _di_gate;
		std::vector<std::vector<std::vector<float>>> _df_gate;
		std::vector<std::vector<std::vector<float>>> _do_gate;
		std::vector<std::vector<std::vector<float>>> _state;
		std::vector<std::vector<std::vector<float>>> _dstate;
		
		// bias
		std::vector<std::vector<float>> _ba;
		std::vector<std::vector<float>> _bi;
		std::vector<std::vector<float>> _bf;
		std::vector<std::vector<float>> _bo;
		std::vector<std::vector<float>> _dba;
		std::vector<std::vector<float>> _dbi;
		std::vector<std::vector<float>> _dbf;
		std::vector<std::vector<float>> _dbo;
		std::vector<std::vector<float>> _mba;
		std::vector<std::vector<float>> _mbi;
		std::vector<std::vector<float>> _mbf;
		std::vector<std::vector<float>> _mbo;
		std::vector<std::vector<float>> _vba;
		std::vector<std::vector<float>> _vbi;
		std::vector<std::vector<float>> _vbf;
		std::vector<std::vector<float>> _vbo;
		
		// W weights
		std::vector<std::vector<std::vector<float>>> _Wa;
		std::vector<std::vector<std::vector<float>>> _Wi;
		std::vector<std::vector<std::vector<float>>> _Wf;
		std::vector<std::vector<std::vector<float>>> _Wo;
		std::vector<std::vector<std::vector<float>>> _dWa;
		std::vector<std::vector<std::vector<float>>> _dWi;
		std::vector<std::vector<std::vector<float>>> _dWf;
		std::vector<std::vector<std::vector<float>>> _dWo;
		std::vector<std::vector<std::vector<float>>> _mWa;
		std::vector<std::vector<std::vector<float>>> _mWi;
		std::vector<std::vector<std::vector<float>>> _mWf;
		std::vector<std::vector<std::vector<float>>> _mWo;
		std::vector<std::vector<std::vector<float>>> _vWa;
		std::vector<std::vector<std::vector<float>>> _vWi;
		std::vector<std::vector<std::vector<float>>> _vWf;
		std::vector<std::vector<std::vector<float>>> _vWo;
		
		// U weights
		std::vector<std::vector<std::vector<float>>> _Ua;
		std::vector<std::vector<std::vector<float>>> _Ui;
		std::vector<std::vector<std::vector<float>>> _Uf;
		std::vector<std::vector<std::vector<float>>> _Uo;
		std::vector<std::vector<std::vector<float>>> _dUa;
		std::vector<std::vector<std::vector<float>>> _dUi;
		std::vector<std::vector<std::vector<float>>> _dUf;
		std::vector<std::vector<std::vector<float>>> _dUo;
		std::vector<std::vector<std::vector<float>>> _mUa;
		std::vector<std::vector<std::vector<float>>> _mUi;
		std::vector<std::vector<std::vector<float>>> _mUf;
		std::vector<std::vector<std::vector<float>>> _mUo;
		std::vector<std::vector<std::vector<float>>> _vUa;
		std::vector<std::vector<std::vector<float>>> _vUi;
		std::vector<std::vector<std::vector<float>>> _vUf;
		std::vector<std::vector<std::vector<float>>> _vUo;
		
		std::vector<std::vector<float>> _errors;
			
};

#endif // __LSTM_STD_FLT_HH__
