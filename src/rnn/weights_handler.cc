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
#include <weights_handler.hh>

//---------------------------------------------------------------------
weights_handler::weights_handler()
{
	
}

//---------------------------------------------------------------------
weights_handler::~weights_handler()
{
	
}

//---------------------------------------------------------------------
float weights_handler::random_flt(float min, float max)
{
    float value = ((float) rand()/ RAND_MAX);
    return min + value * (max - min);
}

//---------------------------------------------------------------------
float weights_handler::xavier_flt(uint32_t input_size, uint32_t ouput_size)
{

    float sqrt_val = (float) (2.0/(input_size + ouput_size));
    float min =  -sqrt(sqrt_val);
    float max =   sqrt(sqrt_val);
    float value = ((float) rand()/ RAND_MAX);
    float value_flt = min + value * (max - min);
    return value_flt;
}

//---------------------------------------------------------------------
float weights_handler::xavier_uni_flt(uint32_t input_size, uint32_t ouput_size)
{

    float sqrt_val = (float) (6.0/(input_size + ouput_size));
    float min =  -sqrt(sqrt_val);
    float max =   sqrt(sqrt_val);
    float value = ((float) rand()/ RAND_MAX);
    float value_flt = min + value * (max - min);
    return value_flt;
}

//---------------------------------------------------------------------
std::vector<std::vector<float>> weights_handler::get_2d_weights_flt_minmax(uint32_t row_size, uint32_t col_size, float min, float max)
{
	uint32_t i1, i2; //iterators
	float weight_value = 0.0;
	std::vector<std::vector<float>> weights;
	for (i1 = 0; i1 < col_size; i1++)
	{
		std::vector<float> weights_i2;	
		for (i2 = 0; i2 < row_size; i2++)
		{
			weight_value = random_flt(min, max);
			weights_i2.push_back(weight_value);
		}
		weights.push_back(weights_i2);
	}
	return weights;
}


//---------------------------------------------------------------------
std::vector<std::vector<float>> weights_handler::get_2d_weights_flt_xavier(uint32_t row_size, uint32_t col_size, uint32_t input_size, uint32_t ouput_size)
{
	uint32_t i1, i2; //iterators
	float weight_value = 0.0;
	std::vector<std::vector<float>> weights;
	for (i1 = 0; i1 < col_size; i1++)
	{
		std::vector<float> weights_i2;	
		for (i2 = 0; i2 < row_size; i2++)
		{
			weight_value = xavier_flt(input_size,ouput_size);
			weights_i2.push_back(weight_value);
		}
		weights.push_back(weights_i2);
	}
	return weights;
}

//---------------------------------------------------------------------
std::vector<std::vector<float>> weights_handler::get_2d_weights_flt_xavier_uni(uint32_t row_size, uint32_t col_size, uint32_t input_size, uint32_t ouput_size)
{
	uint32_t i1, i2; //iterators
	float weight_value = 0.0;
	std::vector<std::vector<float>> weights;
	for (i1 = 0; i1 < col_size; i1++)
	{
		std::vector<float> weights_i2;	
		for (i2 = 0; i2 < row_size; i2++)
		{
			weight_value = xavier_uni_flt(input_size,ouput_size);
			weights_i2.push_back(weight_value);
		}
		weights.push_back(weights_i2);
	}
	return weights;
}


