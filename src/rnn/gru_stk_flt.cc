// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2021-2022 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// ---------------------------------------------------------------------

#include <gru_stk_flt.hh>

//---------------------------------------------------------------------
gru_stk_flt::gru_stk_flt()
{
	_nb_timestep = 0;
	_input_size = 0;
	_output_size = 0;
	_init_ok = false;
	_loaded = false;
	_load_cnt = 0;
	_use_interp_act = false;
	
	// defaut lr
	_lr = 0.1;
	_mu = 0.0;
	
	_id_name = "lstm";
}

//---------------------------------------------------------------------
gru_stk_flt::~gru_stk_flt()
{
	
}

//---------------------------------------------------------------------
void gru_stk_flt::set_lr_mu(float lr, float mu)
{
	_lr = lr;
	_mu = mu;
}

//---------------------------------------------------------------------
void gru_stk_flt::set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon)
{
	_lr = lr;
	_b1 = b1;
	_b2 = b2;
	_epsilon = epsilon;
}


//---------------------------------------------------------------------
void gru_stk_flt::init(uint32_t nb_timestep, std::vector<uint32_t> layers_size)
{
	uint32_t l, t, i1, i2; //iterators
	
	float val_tmp  = 0.0;
	float init_min = -0.4;
	float init_max = 0.4;
	
	_layers_size = layers_size;
	_total_size = _layers_size.size();
	_nb_timestep = nb_timestep;
	_input_size = _layers_size[0];
	_output_size = _layers_size[_total_size-1];
	
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<float> errors_tmp;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			errors_tmp.push_back(0.0);
		}
		_errors.push_back(errors_tmp);
	}
	
	for (i1 = 0; i1 < _output_size; i1++)
	{
		_softmax.push_back(0.0);
		_dsoftmax.push_back(0.0);
	}
	
	// layers/layers gradients init all is init at 0
	for (l = 0; l < _total_size; l++)
	{
		std::vector<std::vector<float>> layers_tmp_l;
		for (t = 0; t < _nb_timestep; t++)
		{
			std::vector<float> layers_tmp_t;
			for (i1 = 0; i1 < _layers_size[l]; i1++)
			{
				layers_tmp_t.push_back(0.0);
			}
			layers_tmp_l.push_back(layers_tmp_t);
		}
		_layers.push_back(layers_tmp_l);
		_dlayers.push_back(layers_tmp_l);
		_dxlayers.push_back(layers_tmp_l);
		_dhlayers.push_back(layers_tmp_l);
		_o_gate.push_back(layers_tmp_l);
		_z_gate.push_back(layers_tmp_l);
		_r_gate.push_back(layers_tmp_l);
		_h_gate.push_back(layers_tmp_l);
		_do_gate.push_back(layers_tmp_l);
		_dz_gate.push_back(layers_tmp_l);
		_dr_gate.push_back(layers_tmp_l);
		_dh_gate.push_back(layers_tmp_l);
	}
	
	//  W weights init (Weights are random and delta weight are 0)
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<std::vector<float>> Wz_tmp_l;
		std::vector<std::vector<float>> Wr_tmp_l;
		std::vector<std::vector<float>> Wh_tmp_l;
		std::vector<std::vector<float>> dWx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			std::vector<float> Wz_tmp_i1;
			std::vector<float> Wr_tmp_i1;
			std::vector<float> Wh_tmp_i1;
			std::vector<float> dWx_tmp_i1;
			for (i2 = 0; i2 < _layers_size[l]; i2++)
			{
				val_tmp = randomize(init_min, init_max);
				Wz_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Wr_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Wh_tmp_i1.push_back(val_tmp);
				dWx_tmp_i1.push_back(0.0);
			}
			Wz_tmp_l.push_back(Wz_tmp_i1);
			Wr_tmp_l.push_back(Wr_tmp_i1);
			Wh_tmp_l.push_back(Wh_tmp_i1);
			dWx_tmp_l.push_back(dWx_tmp_i1);
		}
		_Wz.push_back(Wz_tmp_l);
		_Wr.push_back(Wr_tmp_l);
		_Wh.push_back(Wh_tmp_l);
		_dWz.push_back(dWx_tmp_l);
		_dWr.push_back(dWx_tmp_l);
		_dWh.push_back(dWx_tmp_l);
		_mWz.push_back(dWx_tmp_l);
		_mWr.push_back(dWx_tmp_l);
		_mWh.push_back(dWx_tmp_l);
		_vWz.push_back(dWx_tmp_l);
		_vWr.push_back(dWx_tmp_l);
		_vWh.push_back(dWx_tmp_l);
	}
	
	//  U weights init (Weights are random and delta weight are 0)
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<std::vector<float>> Uz_tmp_l;
		std::vector<std::vector<float>> Ur_tmp_l;
		std::vector<std::vector<float>> Uh_tmp_l;
		std::vector<std::vector<float>> dUx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			std::vector<float> Uz_tmp_i1;
			std::vector<float> Ur_tmp_i1;
			std::vector<float> Uh_tmp_i1;
			std::vector<float> dUx_tmp_i1;
			for (i2 = 0; i2 < _layers_size[l+1]; i2++)
			{
				val_tmp = randomize(init_min, init_max);
				Uz_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Ur_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Uh_tmp_i1.push_back(val_tmp);
				dUx_tmp_i1.push_back(0.0);
			}
			Uz_tmp_l.push_back(Uz_tmp_i1);
			Ur_tmp_l.push_back(Ur_tmp_i1);
			Uh_tmp_l.push_back(Uh_tmp_i1);
			dUx_tmp_l.push_back(dUx_tmp_i1);
		}
		_Uz.push_back(Uz_tmp_l);
		_Ur.push_back(Ur_tmp_l);
		_Uh.push_back(Uh_tmp_l);
		_dUz.push_back(dUx_tmp_l);
		_dUr.push_back(dUx_tmp_l);
		_dUh.push_back(dUx_tmp_l);
		_mUz.push_back(dUx_tmp_l);
		_mUr.push_back(dUx_tmp_l);
		_mUh.push_back(dUx_tmp_l);
		_vUz.push_back(dUx_tmp_l);
		_vUr.push_back(dUx_tmp_l);
		_vUh.push_back(dUx_tmp_l);
	}
	
	//  Bias init 
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<float> bz_tmp_l;
		std::vector<float> br_tmp_l;
		std::vector<float> bh_tmp_l;
		std::vector<float> dbx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			val_tmp = randomize(init_min, init_max);
			bz_tmp_l.push_back(0.0);
			val_tmp = randomize(init_min, init_max);
			br_tmp_l.push_back(0.0);
			val_tmp = randomize(init_min, init_max);
			bh_tmp_l.push_back(0.0);
			dbx_tmp_l.push_back(0.0);
		}
		_bz.push_back(bz_tmp_l);
		_br.push_back(br_tmp_l);
		_bh.push_back(bh_tmp_l);
		_dbz.push_back(dbx_tmp_l);
		_dbr.push_back(dbx_tmp_l);
		_dbh.push_back(dbx_tmp_l);
		_mbz.push_back(dbx_tmp_l);
		_mbr.push_back(dbx_tmp_l);
		_mbh.push_back(dbx_tmp_l);
		_vbz.push_back(dbx_tmp_l);
		_vbr.push_back(dbx_tmp_l);
		_vbh.push_back(dbx_tmp_l);
	}
		
}

//---------------------------------------------------------------------
std::vector<float> gru_stk_flt::get_output()
{
	uint32_t i1;
	std::vector<float> output;
	for (i1=0; i1<_output_size; i1++)
	{
		output.push_back(_layers[_total_size-1][_nb_timestep-1][i1]);
		// output.push_back(_softmax[i1]);
	}
	return output;
}

//---------------------------------------------------------------------
// 
float gru_stk_flt::get_mse()
{
	uint32_t i1;
	float mse_tmp = 0.0;
	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += _errors[_nb_timestep-1][i1]*_errors[_nb_timestep-1][i1];
		//std::cout << " mse_tmp " << mse_tmp <<  std::endl;
	}
	//std::cout << " mse_tmp " << mse_tmp/_output_size <<  std::endl;
	return mse_tmp/_output_size;
}

//---------------------------------------------------------------------
// 
float gru_stk_flt::get_mse_testing(std::vector<std::vector<float>> target)
{
	uint32_t i1,t;
	float mse_tmp = 0.0;
	for (t=0;t<_nb_timestep;t++)
	{
		if (t==_nb_timestep-1)
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  target[t][i1] - _layers[_total_size-1][t][i1];
			}
		}
		else
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  0.0;
			}
		}
	}
	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += _errors[_nb_timestep-1][i1]*_errors[_nb_timestep-1][i1];
	}
	return mse_tmp/_output_size;
}


//---------------------------------------------------------------------
// 
void gru_stk_flt::forward_train(std::vector<std::vector<float>> input)
{
	int l, t, i1, i2; //iterators
	float sum_z;
	float sum_r;
	float sum_h;
	float sum_o;
	
	_layers[0] = input;
	
	for (l = 1; l < _total_size; l++)
	{
		// --- T = 0 ---
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_z = 0.0f;
			sum_r = 0.0f;
			sum_h = 0.0f;
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_z += _Wz[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_r += _Wr[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_h += _Wh[l-1][i1][i2]*_layers[l-1][0][i2];
			}
			
			
			if (_use_interp_act)
			{
				_z_gate[l][0][i1] = sigmoid_interp(sum_z + _bz[l-1][i1]);
				_r_gate[l][0][i1] = sigmoid_interp(sum_r + _br[l-1][i1]);
				_h_gate[l][0][i1] = tanh_interp(sum_h + _bh[l-1][i1]);
				_layers[l][0][i1] = _z_gate[l][0][i1]*_h_gate[l][0][i1];
			}
			else
			{
				_z_gate[l][0][i1] = sigmoid(sum_z + _bz[l-1][i1]);
				_r_gate[l][0][i1] = sigmoid(sum_r + _br[l-1][i1]);
				_h_gate[l][0][i1] = tanh(sum_h + _bh[l-1][i1]);
				_layers[l][0][i1] = _z_gate[l][0][i1]*_h_gate[l][0][i1];
			}
		}
		// --- T = OTHERS TIMESTEPS ---
		for (t=1; t<_nb_timestep; t++)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				//The input layer is broadcast to the hidden layer
				sum_z = 0.0f;
				sum_r = 0.0f;
				sum_h = 0.0f;
				for (i2=0; i2<_layers_size[l-1]; i2++)
				{
					sum_z += _Wz[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_r += _Wr[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_h += _Wh[l-1][i1][i2]*_layers[l-1][t][i2];
				}
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_z += _Uz[l-1][i1][i2]*_layers[l][t-1][i2]; // t-1 because of BP from 0 to N-1!
					sum_r += _Ur[l-1][i1][i2]*_layers[l][t-1][i2];
				}
				
				if (_use_interp_act)
				{
					_z_gate[l][t][i1] = sigmoid_interp(sum_z + _bz[l-1][i1]);
					_r_gate[l][t][i1] = sigmoid_interp(sum_r + _br[l-1][i1]);
				}
				else
				{
					_z_gate[l][t][i1] = sigmoid(sum_z + _bz[l-1][i1]);
					_r_gate[l][t][i1] = sigmoid(sum_r + _br[l-1][i1]);
				}

				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_h += _Uh[l-1][i1][i2]*(_layers[l][t-1][i2]*_r_gate[l][t][i2]);
				}
				if (_use_interp_act)
				{
					_h_gate[l][t][i1] = tanh_interp(sum_h + _bh[l-1][i1]);
				}
				else
				{
					_h_gate[l][t][i1] = tanh(sum_h + _bh[l-1][i1]);
				}
				
				// _layers[l][t][i1] = tanh((1.0f -_z_gate[l][t][i1])*_layers[l][t-1][i1]+_z_gate[l][t][i1]*_h_gate[l][t][i1]);
				_layers[l][t][i1] = (1.0f -_z_gate[l][t][i1])*_layers[l][t-1][i1]+_z_gate[l][t][i1]*_h_gate[l][t][i1];
			}
		}
	}
}

//---------------------------------------------------------------------
// // https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void gru_stk_flt::forward(std::vector<float> input)
{
	uint32_t l, t, i1, i2; //iterators
	float sum_z;
	float sum_r;
	float sum_h;
	float sum_o;
	
	_layers[0].erase(_layers[0].begin());
	_layers[0].push_back(input);
	
	for (l = 1; l < _total_size; l++)
	{
		// --- T = 0 ---
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_z = 0.0f;
			sum_r = 0.0f;
			sum_h = 0.0f;
			sum_o = 0.0f;
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_z += _Wz[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_r += _Wr[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_h += _Wh[l-1][i1][i2]*_layers[l-1][0][i2];
			}
			_z_gate[l][0][i1] = sigmoid(sum_z + _bz[l-1][i1]);
			_r_gate[l][0][i1] = sigmoid(sum_r + _br[l-1][i1]);
			_h_gate[l][0][i1] = tanh(sum_h + _bh[l-1][i1]);
			_layers[l][0][i1] = _z_gate[l][0][i1]*_h_gate[l][0][i1];
		}
		// --- T = OTHERS TIMESTEPS ---
		for (t=1; t<_nb_timestep; t++)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				//The input layer is broadcast to the hidden layer
				sum_z = 0.0f;
				sum_r = 0.0f;
				sum_h = 0.0f;
				for (i2=0; i2<_layers_size[l-1]; i2++)
				{
					sum_z += _Wz[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_r += _Wr[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_h += _Wh[l-1][i1][i2]*_layers[l-1][t][i2];
				}
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_z += _Uz[l-1][i1][i2]*_layers[l][t-1][i2]; // t-1 because of BP from 0 to N-1!
					sum_r += _Ur[l-1][i1][i2]*_layers[l][t-1][i2]; 
				}
				_z_gate[l][t][i1] = sigmoid(sum_z + _bz[l-1][i1]);
				_r_gate[l][t][i1] = sigmoid(sum_r + _br[l-1][i1]);

				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_h += _Uh[l-1][i1][i2]*(_layers[l][t-1][i2]*_r_gate[l][t][i2]);
				}
				_h_gate[l][t][i1] = tanh(sum_h + _bh[l-1][i1]);
				
				//_o_gate[l][t][i1] = (1.0f-_z_gate[l][t][i1])*_layers[l][t-1][i1]+_z_gate[l][t][i1]*_h_gate[l][t][i1];
				
				//sum_o = 0.0f;
				//for (i2=0; i2<_layers_size[l]; i2++)
				//{
					//sum_o += _Uo[l-1][i1][i2]*_o_gate[l][t-1][i2]; // t-1 because of BP from 0 to N-1!
				//}
				//_layers[l][t][i1] = tanh(sum_o + _bo[l-1][i1]);
				
				// ORI
				// _layers[l][t][i1] = tanh((1.0f -_z_gate[l][t][i1])*_layers[l][t-1][i1]+_z_gate[l][t][i1]*_h_gate[l][t][i1]);
				_layers[l][t][i1] = (1.0f -_z_gate[l][t][i1])*_layers[l][t-1][i1]+_z_gate[l][t][i1]*_h_gate[l][t][i1];
			
			}
		}
	}	
}


//---------------------------------------------------------------------
// 
void gru_stk_flt::backward_sgd_train(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t l, i1, i2, i3; //iterators
	float sum_dz, sum_bz, sum_Wz, sum_Uz;
	float sum_dr, sum_br, sum_Wr, sum_Ur;
	float sum_dh, sum_bh, sum_Wh, sum_Uh;
	float m;
	//std::cout << " STEP backward_sgd_train_mnist init " << std::endl;
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		if (t==_nb_timestep-1)
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  target[t][i1] - _layers[_total_size-1][t][i1];
			}
		}
		else
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  0.0;
			}
		}
	}
	
	//for (t=0;t<_nb_timestep;t++)
	//{
		//for (i1=0; i1<_output_size; i1++)
		//{
			//_errors[t][i1] =  target[t][i1] - _layers[_total_size-1][t][i1];
		//}
	//}
	
	// std::cout << " ERROR = " << _errors[_nb_timestep-1][_output_size-1] << std::endl;
	_dlayers[_total_size-1] = _errors;
	//std::cout << " STEP 2 " << std::endl;
	for (l = _total_size-1; l > 0; l--)
	{
		// --------------------------------
		// T = TIMESTEP - 1 (because it starts at 0)
		t = _nb_timestep-1;
		for (i2=0; i2<_layers_size[l]; i2++)
		{
			// _dhlayers[l][t][i2] = _dlayers[l][t][i2] * dtanh(_layers[l][t][i2]);
			_dhlayers[l][t][i2] = _dlayers[l][t][i2];
			//_dhlayers[l][t][i2] = _dlayers[l][t][i2] ;
			_dz_gate[l][t][i2] = dsigmoid(_z_gate[l][t][i2]) * (_dhlayers[l][t][i2] * (_h_gate[l][t][i2] - _layers[l][t-1][i2]));
			_dh_gate[l][t][i2] = dtanh(_h_gate[l][t][i2]) * _dhlayers[l][t][i2] * _z_gate[l][t][i2];
			m=0.0;
			for(i3=0; i3<_output_size; i3++)
			{
				m += _Uh[l-1][i3][i2] * _dh_gate[l][t][i3];
			}
			_dr_gate[l][t][i2] = dsigmoid(_r_gate[l][t][i2]) * _r_gate[l][t][i2] * m ;
		}
		//std::cout << " STEP 3 " << std::endl;
		// T=TIMESTEP-2 TO 0
		for (t=_nb_timestep-2; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				sum_dz = 0.0f;
				sum_dr = 0.0f;
				sum_dh = 0.0f;
				//std::cout << " STEP 31 => " << _nb_timestep << std::endl;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_dz += _Uz[l-1][i2][i1]*_dz_gate[l][t+1][i2];
					sum_dr += _Ur[l-1][i2][i1]*_dr_gate[l][t+1][i2];
					sum_dh += _Uh[l-1][i2][i1]*_dh_gate[l][t+1][i2];
				}
				//std::cout << " STEP 311 " << std::endl;
				if (l == _total_size-1)
				{
					_dhlayers[l][t][i1] =  _dlayers[l][t+1][i1]; // sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
				else
				{
					_dhlayers[l][t][i1] = _dlayers[l][t+1][i1]* (1 -_z_gate[l][t][i1]) + sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
			
				// OK
				// _dhlayers[l][t][i1] = _dhlayers[l][t+1][i1]* (1 -_z_gate[l][t][i1]) + sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				_dz_gate[l][t][i1] = 0.0f;
				if(t>0)
				{
					_dz_gate[l][t][i1] = dsigmoid(_z_gate[l][t][i1]) * (_dhlayers[l][t][i1] * (_h_gate[l][t][i1] - _layers[l][t-1][i1]));
				}
				_dh_gate[l][t][i1] = dtanh(_h_gate[l][t][i1]) * _dhlayers[l][t][i1] * _z_gate[l][t][i1];
				
				_dr_gate[l][t][i1] = 0.0f;
				if(t>0)
				{
					m=0.0f;
					for(i3=0; i3<_output_size; i3++)
					{
						m += _Uh[l-1][i3][i1] * _dh_gate[l][t][i3];
					}
					_dr_gate[l][t][i1] = dsigmoid(_r_gate[l][t][i1]) * _r_gate[l][t][i1] * m ;
				}
			}
		}
		
		// Update Input layers gradient
		for (t=_nb_timestep-1; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l-1]; i1++)
			{
				sum_dz = 0.0f;
				sum_dr = 0.0f;
				sum_dh = 0.0f;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_dz += _Wz[l-1][i2][i1]*_dz_gate[l][t][i2];
					sum_dr += _Wr[l-1][i2][i1]*_dr_gate[l][t][i2];
					sum_dh += _Wh[l-1][i2][i1]*_dh_gate[l][t][i2];
				}
				_dlayers[l-1][t][i1] = sum_dz + sum_dr + sum_dh;
			}
		}

		for (i1=0; i1<_layers_size[l]; i1++)
		{	
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_Wz = 0.0f;
				sum_Wr = 0.0f;
				sum_Wh = 0.0f;
				for (t=0; t<_nb_timestep; t++)
				{
					sum_Wz += _dz_gate[l][t][i1]*_layers[l-1][t][i2]; //input
					sum_Wr += _dr_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wh += _dh_gate[l][t][i1]*_layers[l-1][t][i2];		
				} 
				
				_dWz[l-1][i1][i2] = _lr * sum_Wz + _mu * _dWz[l-1][i1][i2];
				_dWr[l-1][i1][i2] = _lr * sum_Wr + _mu * _dWr[l-1][i1][i2];
				_dWh[l-1][i1][i2] = _lr * sum_Wh + _mu * _dWh[l-1][i1][i2];

				_Wh[l-1][i1][i2] = _Wh[l-1][i1][i2] + _dWh[l-1][i1][i2];
				_Wr[l-1][i1][i2] = _Wr[l-1][i1][i2] + _dWr[l-1][i1][i2];
				_Wz[l-1][i1][i2] = _Wz[l-1][i1][i2] + _dWz[l-1][i1][i2];

			}
			// output size loop	
			for (i2=0; i2<_layers_size[l]; i2++)
			{
				sum_Uz = 0.0f;
				sum_Ur = 0.0f;
				sum_Uh = 0.0f;
				for (t=0; t<_nb_timestep-1; t++)
				{
					sum_Uz += _dz_gate[l][t+1][i1]*_layers[l][t][i2]; //output
					sum_Ur += _dr_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uh += _dh_gate[l][t+1][i1]*_layers[l][t][i2]*_r_gate[l][t+1][i2];
				}
				_dUz[l-1][i1][i2] = _lr * sum_Uz + _mu * _dUz[l-1][i1][i2];
				_dUr[l-1][i1][i2] = _lr * sum_Ur + _mu * _dUr[l-1][i1][i2];
				_dUh[l-1][i1][i2] = _lr * sum_Uh + _mu * _dUh[l-1][i1][i2];

				_Uz[l-1][i1][i2] = _Uz[l-1][i1][i2] + _dUz[l-1][i1][i2];
				_Ur[l-1][i1][i2] = _Ur[l-1][i1][i2] + _dUr[l-1][i1][i2];
				_Uh[l-1][i1][i2] = _Uh[l-1][i1][i2] + _dUh[l-1][i1][i2];
			}
			sum_bz = 0.0f;
			sum_br = 0.0f;
			sum_bh = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_bz += _dz_gate[l][t][i1];
				sum_br += _dr_gate[l][t][i1];
				sum_bh += _dh_gate[l][t][i1];
			}
			_dbz[l-1][i1] = _lr * sum_bz + _mu * _dbz[l-1][i1];
			_dbr[l-1][i1] = _lr * sum_br + _mu * _dbr[l-1][i1];
			_dbh[l-1][i1] = _lr * sum_bh + _mu * _dbh[l-1][i1];

			_bz[l-1][i1] = _bz[l-1][i1] + _dbz[l-1][i1];
			_br[l-1][i1] = _br[l-1][i1] + _dbr[l-1][i1];
			_bh[l-1][i1] = _bh[l-1][i1] + _dbh[l-1][i1];
		}
	}
}

//---------------------------------------------------------------------
// 
void gru_stk_flt::backward_adam_train(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t l, i1, i2, i3; //iterators
	float sum_dz, sum_bz, sum_Wz, sum_Uz;
	float sum_dr, sum_br, sum_Wr, sum_Ur;
	float sum_dh, sum_bh, sum_Wh, sum_Uh;
	float m;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		if (t==_nb_timestep-1)
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  target[t][i1] - _layers[_total_size-1][t][i1];
			}
		}
		else
		{
			for (i1=0; i1<_output_size; i1++)
			{
				_errors[t][i1] =  0.0;
			}
		}
	}
	_dlayers[_total_size-1] = _errors;
	
	for (l = _total_size-1; l > 0; l--)
	{
		// --------------------------------
		// T = TIMESTEP - 1 (because it starts at 0)
		t = _nb_timestep-1;
		for (i2=0; i2<_layers_size[l]; i2++)
		{
			// _dhlayers[l][t][i2] = _dlayers[l][t][i2] * dtanh(_layers[l][t][i2]);
			_dhlayers[l][t][i2] = _dlayers[l][t][i2];
			//_dhlayers[l][t][i2] = _dlayers[l][t][i2] ;
			_dz_gate[l][t][i2] = dsigmoid(_z_gate[l][t][i2]) * (_dhlayers[l][t][i2] * (_h_gate[l][t][i2] - _layers[l][t-1][i2]));
			_dh_gate[l][t][i2] = dtanh(_h_gate[l][t][i2]) * _dhlayers[l][t][i2] * _z_gate[l][t][i2];
			m=0.0;
			for(i3=0; i3<_output_size; i3++)
			{
				m += _Uh[l-1][i3][i2] * _dh_gate[l][t][i3];
			}
			_dr_gate[l][t][i2] = dsigmoid(_r_gate[l][t][i2]) * _r_gate[l][t][i2] * m ;
		}
		//std::cout << " STEP 3 " << std::endl;
		// T=TIMESTEP-2 TO 0
		for (t=_nb_timestep-2; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				sum_dz = 0.0f;
				sum_dr = 0.0f;
				sum_dh = 0.0f;
				//std::cout << " STEP 31 => " << _nb_timestep << std::endl;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_dz += _Uz[l-1][i2][i1]*_dz_gate[l][t+1][i2];
					sum_dr += _Ur[l-1][i2][i1]*_dr_gate[l][t+1][i2];
					sum_dh += _Uh[l-1][i2][i1]*_dh_gate[l][t+1][i2];
				}
				//std::cout << " STEP 311 " << std::endl;
				if (l == _total_size-1)
				{
					_dhlayers[l][t][i1] =  _dlayers[l][t+1][i1]; // sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
				else
				{
					_dhlayers[l][t][i1] = _dlayers[l][t+1][i1]* (1 -_z_gate[l][t][i1]) + sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
			
				// OK
				// _dhlayers[l][t][i1] = _dhlayers[l][t+1][i1]* (1 -_z_gate[l][t][i1]) + sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				_dz_gate[l][t][i1] = 0.0f;
				if(t>0)
				{
					_dz_gate[l][t][i1] = dsigmoid(_z_gate[l][t][i1]) * (_dhlayers[l][t][i1] * (_h_gate[l][t][i1] - _layers[l][t-1][i1]));
				}
				_dh_gate[l][t][i1] = dtanh(_h_gate[l][t][i1]) * _dhlayers[l][t][i1] * _z_gate[l][t][i1];
				
				_dr_gate[l][t][i1] = 0.0f;
				if(t>0)
				{
					m=0.0f;
					for(i3=0; i3<_output_size; i3++)
					{
						m += _Uh[l-1][i3][i1] * _dh_gate[l][t][i3];
					}
					_dr_gate[l][t][i1] = dsigmoid(_r_gate[l][t][i1]) * _r_gate[l][t][i1] * m ;
				}
			}
		}
		
		// Update Input layers gradient
		for (t=_nb_timestep-1; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l-1]; i1++)
			{
				sum_dz = 0.0f;
				sum_dr = 0.0f;
				sum_dh = 0.0f;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_dz += _Wz[l-1][i2][i1]*_dz_gate[l][t][i2];
					sum_dr += _Wr[l-1][i2][i1]*_dr_gate[l][t][i2];
					sum_dh += _Wh[l-1][i2][i1]*_dh_gate[l][t][i2];
				}
				_dlayers[l-1][t][i1] = sum_dz + sum_dr + sum_dh;
			}
		}

		for (i1=0; i1<_layers_size[l]; i1++)
		{	
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_Wz = 0.0f;
				sum_Wr = 0.0f;
				sum_Wh = 0.0f;
				for (t=0; t<_nb_timestep; t++)
				{
					sum_Wz += _dz_gate[l][t][i1]*_layers[l-1][t][i2]; //input
					sum_Wr += _dr_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wh += _dh_gate[l][t][i1]*_layers[l-1][t][i2];		
				} 
				
				_mWz[l-1][i1][i2] = _b1 * _mWz[l-1][i1][i2] + (1 - _b1) * sum_Wz;
				_mWr[l-1][i1][i2] = _b1 * _mWr[l-1][i1][i2] + (1 - _b1) * sum_Wr;
				_mWh[l-1][i1][i2] = _b1 * _mWh[l-1][i1][i2] + (1 - _b1) * sum_Wh;

				_vWz[l-1][i1][i2] = _b2 * _vWz[l-1][i1][i2] + (1 -_b2) * sum_Wz * sum_Wz;
				_vWr[l-1][i1][i2] = _b2 * _vWr[l-1][i1][i2] + (1 -_b2) * sum_Wr * sum_Wr;
				_vWh[l-1][i1][i2] = _b2 * _vWh[l-1][i1][i2] + (1 -_b2) * sum_Wh * sum_Wh;

				_dWz[l-1][i1][i2] = _lr * (_mWz[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWz[l-1][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWr[l-1][i1][i2] = _lr * (_mWr[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWr[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dWh[l-1][i1][i2] = _lr * (_mWh[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWh[l-1][i1][i2] / (1 - _b2)) + _epsilon);

				_Wz[l-1][i1][i2] = _Wz[l-1][i1][i2] + _dWz[l-1][i1][i2];
				_Wr[l-1][i1][i2] = _Wr[l-1][i1][i2] + _dWr[l-1][i1][i2];
				_Wh[l-1][i1][i2] = _Wh[l-1][i1][i2] + _dWh[l-1][i1][i2];

			}
			// output size loop	
			for (i2=0; i2<_layers_size[l]; i2++)
			{
				sum_Uz = 0.0f;
				sum_Ur = 0.0f;
				sum_Uh = 0.0f;
				for (t=0; t<_nb_timestep-1; t++)
				{
					sum_Uz += _dz_gate[l][t+1][i1]*_layers[l][t][i2]; //output
					sum_Ur += _dr_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uh += _dh_gate[l][t+1][i1]*_layers[l][t][i2]*_r_gate[l][t+1][i2];
				}
				
				_mUz[l-1][i1][i2] = _b1 * _mUz[l-1][i1][i2] + (1 - _b1) * sum_Uz;
				_mUr[l-1][i1][i2] = _b1 * _mUr[l-1][i1][i2] + (1 - _b1) * sum_Ur;
				_mUh[l-1][i1][i2] = _b1 * _mUh[l-1][i1][i2] + (1 - _b1) * sum_Uh;

				_vUz[l-1][i1][i2] = _b2 * _vUz[l-1][i1][i2] + (1 -_b2) * sum_Uz * sum_Uz;
				_vUr[l-1][i1][i2] = _b2 * _vUr[l-1][i1][i2] + (1 -_b2) * sum_Ur * sum_Ur;
				_vUh[l-1][i1][i2] = _b2 * _vUh[l-1][i1][i2] + (1 -_b2) * sum_Uh * sum_Uh;

				_dUz[l-1][i1][i2] = _lr * (_mUz[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUz[l-1][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dUr[l-1][i1][i2] = _lr * (_mUr[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUr[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dUh[l-1][i1][i2] = _lr * (_mUh[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUh[l-1][i1][i2] / (1 - _b2)) + _epsilon);

				_Uz[l-1][i1][i2] = _Uz[l-1][i1][i2] + _dUz[l-1][i1][i2];
				_Ur[l-1][i1][i2] = _Ur[l-1][i1][i2] + _dUr[l-1][i1][i2];
				_Uh[l-1][i1][i2] = _Uh[l-1][i1][i2] + _dUh[l-1][i1][i2];
			}
			sum_bz = 0.0f;
			sum_br = 0.0f;
			sum_bh = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_bz += _dz_gate[l][t][i1];
				sum_br += _dr_gate[l][t][i1];
				sum_bh += _dh_gate[l][t][i1];
			}
			
			_mbz[l-1][i1] = _b1 * _mbz[l-1][i1] + (1 - _b1) * sum_bz;
			_mbr[l-1][i1] = _b1 * _mbr[l-1][i1] + (1 - _b1) * sum_br;
			_mbh[l-1][i1] = _b1 * _mbh[l-1][i1] + (1 - _b1) * sum_bh;

			_vbz[l-1][i1] = _b2 * _vbz[l-1][i1] + (1 -_b2) * sum_bz * sum_bz;
			_vbr[l-1][i1] = _b2 * _vbr[l-1][i1] + (1 -_b2) * sum_br * sum_br;
			_vbh[l-1][i1] = _b2 * _vbh[l-1][i1] + (1 -_b2) * sum_bh * sum_bh;

			_dbz[l-1][i1] = _lr * (_mbz[l-1][i1] / (1 - _b1)) / (sqrt(_vbz[l-1][i1] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbr[l-1][i1] = _lr * (_mbr[l-1][i1] / (1 - _b1)) / (sqrt(_vbr[l-1][i1] / (1 - _b2)) + _epsilon);
			_dbh[l-1][i1] = _lr * (_mbh[l-1][i1] / (1 - _b1)) / (sqrt(_vbh[l-1][i1] / (1 - _b2)) + _epsilon);
			
			_bz[l-1][i1] = _bz[l-1][i1] + _dbz[l-1][i1];
			_br[l-1][i1] = _br[l-1][i1] + _dbr[l-1][i1];
			_bh[l-1][i1] = _bh[l-1][i1] + _dbh[l-1][i1];
		}
	}
}


//---------------------------------------------------------------------
float gru_stk_flt::sigmoid(float x)
{
	return (1.0 / (1.0 + exp(-x)));
}

//---------------------------------------------------------------------
float gru_stk_flt::dsigmoid(float x)
{
	return (x*(1.0 - x));
}

//---------------------------------------------------------------------
float gru_stk_flt::dtanh(float x)
{
	float y = tanh(x);
	return (1.0 - y*y);
}

//---------------------------------------------------------------------
float gru_stk_flt::ddtanh(float x)
{
	//float y = tanh(x);
	return (1.0 - x*x);
}

//---------------------------------------------------------------------
float gru_stk_flt::leakyrelu(float x) 
{
    return (x < 0.0 ? (0.001 * x) : x);
}

//---------------------------------------------------------------------
float gru_stk_flt::dleakyrelu(float x) 
{
   return (x < 0.0 ? 0.001 : x);
}

//---------------------------------------------------------------------
float gru_stk_flt::softplus(float x)
{
    return log(1.0+(exp(x)));
}

//---------------------------------------------------------------------
float gru_stk_flt::dsoftplus(float x)
{
    return 1.0/(1.0+(exp(-x)));
}

//---------------------------------------------------------------------
float gru_stk_flt::randomize(float min, float max){
    float f = ((float) rand()) / RAND_MAX;
    return min + f * (max - min);
}

//---------------------------------------------------------------------
std::vector<float> gru_stk_flt::softmax(std::vector<float> input)
{
	std::vector<float> output;
	float sum = 0;
	int size_in = input.size();
	for (int n=0; n<size_in; n++)
	{
		sum += exp(input[n]);
	}
	for (int n=0; n<size_in; n++)
	{
		output.push_back(exp(input[n]) / sum);
	}
}

//---------------------------------------------------------------------
std::vector<float> gru_stk_flt::dsoftmax(std::vector<float> input)
{
	std::vector<float> output;
	float sum = 0.0,val=0.0;
	int size_in = input.size();
	for (int n=0; n<size_in; n++)
	{
		val = 0.0;
		for (int k=0; k<size_in; k++)
		{
			val += input[n] * ( (float)(k==n) - input[k] );
		}
		output.push_back(val);
	}
}

//---------------------------------------------------------------------------------------------------
float gru_stk_flt::sigmoid_interp(float x)
{
	std::cout << "!! Warning !! inperpolated version not included in opensource version - back to normal sigmoid" << std::endl;
	float y = sigmoid(x);
	return y;
}

//---------------------------------------------------------------------------------------------------
float gru_stk_flt::tanh_interp(float x)
{
	std::cout << "!! Warning !! inperpolated version not included in opensource version - back to normal tanh" << std::endl;
	float y = tanh(x);
	return y;
}

//---------------------------------------------------------------------
void gru_stk_flt::set_interp_act(bool val)
{
	_use_interp_act = val;
}


