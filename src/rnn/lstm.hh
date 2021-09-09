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

#ifndef __LSTM_HH__
#define __LSTM_HH__

class lstm
{
	public:
		lstm();
		~lstm();
		
		void init(uint32_t nb_timestep, uint32_t input_size, uint32_t output_size);
		void set_lr_mu(float lr, float mu);
		void set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon);
		void set_optimizer(uint8_t optim_id);
		void forward(std::vector<std::vector<float>> input);
		void backward(std::vector<std::vector<float>> target);
		void backward_sgd(std::vector<std::vector<float>> target);
		void backward_adam(std::vector<std::vector<float>> target);
		void backward_adamax(std::vector<std::vector<float>> target);
		float get_mse();
		float get_mse(std::vector<std::vector<float>> target, std::vector<std::vector<float>> output);
		float randomize(float min, float max);
		void clear_grads();
		void clear_momentums();
		int nbLines(std::string filename);
		void save(const char* filename);
		void load(const char* lstmlogfilename, uint32_t input_size, uint32_t output_size);
		std::vector<std::vector<float>> get_output(){return _out;};

		
		//activation functions
		float sigmoid(float x);
		float dsigmoid(float x);
		float dtanh(float x);
		float leakyrelu(float x);
		float dleakyrelu(float x);
		float softplus(float x);
		float dsoftplus(float x);
		
		// gradient descent
		std::vector<std::vector<float>> sgd(std::vector<std::vector<float>> W, std::vector<std::vector<float>> dW, float lr);
		void get_error(std::vector<std::vector<float>> target);
		std::vector<std::vector<float>> get_inference_error(){return _errors;};
		
		

	private: 
	
		// flags
		bool _init_ok;
	
		// sizes 
		uint32_t _nb_timestep, _input_size, _output_size;
		
		// optimizer choice 1=SGB / 2=ADAM / 3=ADAMAX
		uint8_t _optim_id;
	
		// hyper-parameters
		float _lr; //learning rate
		float _mu; //momentum
		float _b1; 
		float _b2; 
		float _epsilon;

		//optimization momentums
		std::vector<std::vector<float>> _m_ba, _m_bi, _m_bf, _m_bo, _v_ba, _v_bi, _v_bf, _v_bo;
		std::vector<std::vector<std::vector<float>>> _m_Wa, _m_Wi, _m_Wf, _m_Wo, _v_Wa, _v_Wi, _v_Wf, _v_Wo;
		std::vector<std::vector<std::vector<float>>> _m_Ua, _m_Ui, _m_Uf, _m_Uo, _v_Ua, _v_Ui, _v_Uf, _v_Uo;
		
		// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
		// inputs
		std::vector<std::vector<float>> _x;
		std::vector<std::vector<float>> _dx;
		
		std::vector<std::vector<float>> _errors;
		std::vector<float> _errors_report;
		
		// gates
		std::vector<std::vector<float>> _a_gate;
		std::vector<std::vector<float>> _i_gate;
		std::vector<std::vector<float>> _f_gate;
		std::vector<std::vector<float>> _o_gate;
		std::vector<std::vector<float>> _da_gate;
		std::vector<std::vector<float>> _di_gate;
		std::vector<std::vector<float>> _df_gate;
		std::vector<std::vector<float>> _do_gate;
		
		std::vector<std::vector<float>> _state;
		std::vector<std::vector<float>> _dstate;
		
		std::vector<std::vector<float>> _out;
		std::vector<std::vector<float>> _dout;
		std::vector<std::vector<float>> _Dout;

		std::vector<std::vector<float>> _ba;
		std::vector<std::vector<float>> _bi;
		std::vector<std::vector<float>> _bf;
		std::vector<std::vector<float>> _bo;
		std::vector<std::vector<float>> _dba;
		std::vector<std::vector<float>> _dbi;
		std::vector<std::vector<float>> _dbf;
		std::vector<std::vector<float>> _dbo;
		std::vector<std::vector<std::vector<float>>> _Wa;
		std::vector<std::vector<std::vector<float>>> _Wi;
		std::vector<std::vector<std::vector<float>>> _Wf;
		std::vector<std::vector<std::vector<float>>> _Wo;
		std::vector<std::vector<std::vector<float>>> _Ua;
		std::vector<std::vector<std::vector<float>>> _Ui;
		std::vector<std::vector<std::vector<float>>> _Uf;
		std::vector<std::vector<std::vector<float>>> _Uo;
		std::vector<std::vector<std::vector<float>>> _dWa;
		std::vector<std::vector<std::vector<float>>> _dWi;
		std::vector<std::vector<std::vector<float>>> _dWf;
		std::vector<std::vector<std::vector<float>>> _dWo;
		std::vector<std::vector<std::vector<float>>> _dUa;
		std::vector<std::vector<std::vector<float>>> _dUi;
		std::vector<std::vector<std::vector<float>>> _dUf;
		std::vector<std::vector<std::vector<float>>> _dUo;
		
		// https://arxiv.org/pdf/1702.00071.pdf (weights initializations)
		// https://arxiv.org/pdf/1701.05923.pdf (gru for mnist)
		// https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8404457 (gru F16 example)
			
};

#endif 
