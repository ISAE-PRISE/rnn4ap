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


#ifndef __WEIGHTS_HANDLER_HH__
#define __WEIGHTS_HANDLER_HH__

class weights_handler
{
	public:
		weights_handler();
		~weights_handler();
		
		// init randomly between min and max
		std::vector<std::vector<float>>   get_2d_weights_flt_minmax(uint32_t row_size, uint32_t col_size, float min, float max);
		
		// init using Xavier
		std::vector<std::vector<float>>   get_2d_weights_flt_xavier(uint32_t row_size, uint32_t col_size, uint32_t input_size, uint32_t ouput_size);
		
		// init using Xavier uniform
		std::vector<std::vector<float>>   get_2d_weights_flt_xavier_uni(uint32_t row_size, uint32_t col_size, uint32_t input_size, uint32_t ouput_size);
		
		float   random_flt(float min, float max);
		float   xavier_flt(uint32_t input_size, uint32_t ouput_size);
		float   xavier_uni_flt(uint32_t input_size, uint32_t ouput_size);

	private: 
	
			
};

#endif 
