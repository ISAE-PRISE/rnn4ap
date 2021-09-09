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

#ifndef __GRU_HH__
#define __GRU_HH__

class gru
{
	public:
		gru();
		~gru();
		
		void init(uint32_t nb_timestep, uint32_t input_size, uint32_t output_size);
		void set_lr_mu(float lr, float mu);
		void set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon);
		void set_optimizer(uint8_t optim_id);
		void forward(std::vector<std::vector<float>> input);
		void backward(std::vector<std::vector<float>> target);
		void backward_sgd(std::vector<std::vector<float>> target);
		void backward_adam(std::vector<std::vector<float>> target);
		void backward_adamax(std::vector<std::vector<float>> target);
		void get_error(std::vector<std::vector<float>> target);
		float get_mse();
		float get_mse(std::vector<std::vector<float>> target, std::vector<std::vector<float>> output);
		float sigmoid(float x);
		float dsigmoid(float x);
		float dtanh(float x);
		float randomize(float min, float max);
		void clear_grads();
		void clear_momentums();
		void save(const char* filename);
		void load(const char* grulogfilename, uint32_t input_size, uint32_t output_size);
		int nbLines(std::string filename);
		std::vector<std::vector<float>> get_output(){return _out;};
		std::vector<std::vector<float>> get_inference_error(){return _errors;};
		
		// gradient descent
		std::vector<std::vector<float>> sgd(std::vector<std::vector<float>> W, std::vector<std::vector<float>> dW, float lr);
		

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
		std::vector<std::vector<float>> _m_bz, _m_br, _m_bh, _v_bz, _v_br, _v_bh;
		std::vector<std::vector<std::vector<float>>> _m_Wz, _m_Wr, _m_Wh, _v_Wz, _v_Wr, _v_Wh;
		std::vector<std::vector<std::vector<float>>> _m_Uz, _m_Ur, _m_Uh, _v_Uz, _v_Ur, _v_Uh;
		
		// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
		// inputs
		std::vector<std::vector<float>> _x;
		std::vector<std::vector<float>> _dx;
		
		std::vector<std::vector<float>> _errors;
		std::vector<float> _errors_report;
		
		// gates
		std::vector<std::vector<float>> _z_gate;
		std::vector<std::vector<float>> _dz_gate;
		std::vector<std::vector<float>> _r_gate;
		std::vector<std::vector<float>> _dr_gate;
		std::vector<std::vector<float>> _h_gate;
		std::vector<std::vector<float>> _dh_gate;
		
		std::vector<std::vector<float>> _out;
		std::vector<std::vector<float>> _dout;
		std::vector<std::vector<float>> _Dout;

		std::vector<std::vector<float>> _bz;
		std::vector<std::vector<float>> _br;
		std::vector<std::vector<float>> _bh;
		std::vector<std::vector<float>> _dbz;
		std::vector<std::vector<float>> _dbr;
		std::vector<std::vector<float>> _dbh;
		
		std::vector<std::vector<std::vector<float>>> _Wz;
		std::vector<std::vector<std::vector<float>>> _Wr;
		std::vector<std::vector<std::vector<float>>> _Wh;
		std::vector<std::vector<std::vector<float>>> _Uz;
		std::vector<std::vector<std::vector<float>>> _Ur;
		std::vector<std::vector<std::vector<float>>> _Uh;
		std::vector<std::vector<std::vector<float>>> _dWz;
		std::vector<std::vector<std::vector<float>>> _dWr;
		std::vector<std::vector<std::vector<float>>> _dWh;
		std::vector<std::vector<std::vector<float>>> _dUz;
		std::vector<std::vector<std::vector<float>>> _dUr;
		std::vector<std::vector<std::vector<float>>> _dUh;
			
};

#endif 
