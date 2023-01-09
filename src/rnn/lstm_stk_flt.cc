// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2021-2022 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// goncalo.fontes-neves@student.isae-supaero.fr
// ---------------------------------------------------------------------

#include <lstm_stk_flt.hh>

//---------------------------------------------------------------------
lstm_stk_flt::lstm_stk_flt()
{
	_nb_timestep = 0;
	_input_size = 0;
	_output_size = 0;
	_init_ok = false;
	_loaded = false;
	_use_interp_act = false;
	_load_cnt = 0;
	
	// defaut lr
	_lr = 0.1;
	_mu = 0.0;
	
	_id_name = "lstm";
}

//---------------------------------------------------------------------
lstm_stk_flt::~lstm_stk_flt()
{
	
}

//---------------------------------------------------------------------
void lstm_stk_flt::set_lr_mu(float lr, float mu)
{
	_lr = lr;
	_mu = mu;
}

//---------------------------------------------------------------------
void lstm_stk_flt::set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon)
{
	_lr = lr;
	_b1 = b1;
	_b2 = b2;
	_epsilon = epsilon;
}


//---------------------------------------------------------------------
void lstm_stk_flt::init(uint32_t nb_timestep, std::vector<uint32_t> layers_size)
{
	uint32_t l, t, i1, i2; //iterators
	
	float val_tmp  = 0.0;
	float init_min = -1.0;
	float init_max = 1.0;
	
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
		_a_gate.push_back(layers_tmp_l);
		_i_gate.push_back(layers_tmp_l);
		_f_gate.push_back(layers_tmp_l);
		_o_gate.push_back(layers_tmp_l);
		_da_gate.push_back(layers_tmp_l);
		_di_gate.push_back(layers_tmp_l);
		_df_gate.push_back(layers_tmp_l);
		_do_gate.push_back(layers_tmp_l);
		
		_state.push_back(layers_tmp_l);
		_dstate.push_back(layers_tmp_l);
	}
	
	//  W weights init (Weights are random and delta weight are 0)
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<std::vector<float>> Wa_tmp_l;
		std::vector<std::vector<float>> Wi_tmp_l;
		std::vector<std::vector<float>> Wf_tmp_l;
		std::vector<std::vector<float>> Wo_tmp_l;
		std::vector<std::vector<float>> dWx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			std::vector<float> Wa_tmp_i1;
			std::vector<float> Wi_tmp_i1;
			std::vector<float> Wf_tmp_i1;
			std::vector<float> Wo_tmp_i1;
			std::vector<float> dWx_tmp_i1;
			for (i2 = 0; i2 < _layers_size[l]; i2++)
			{
				val_tmp = randomize(init_min, init_max);
				Wa_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Wi_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Wf_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Wo_tmp_i1.push_back(val_tmp);
				dWx_tmp_i1.push_back(0.0);
			}
			Wa_tmp_l.push_back(Wa_tmp_i1);
			Wi_tmp_l.push_back(Wi_tmp_i1);
			Wf_tmp_l.push_back(Wf_tmp_i1);
			Wo_tmp_l.push_back(Wo_tmp_i1);
			dWx_tmp_l.push_back(dWx_tmp_i1);
		}
		_Wa.push_back(Wa_tmp_l);
		_Wi.push_back(Wi_tmp_l);
		_Wf.push_back(Wf_tmp_l);
		_Wo.push_back(Wo_tmp_l);
		_dWa.push_back(dWx_tmp_l);
		_dWi.push_back(dWx_tmp_l);
		_dWf.push_back(dWx_tmp_l);
		_dWo.push_back(dWx_tmp_l);
		_vWa.push_back(dWx_tmp_l);
		_vWi.push_back(dWx_tmp_l);
		_vWf.push_back(dWx_tmp_l);
		_vWo.push_back(dWx_tmp_l);
		_mWa.push_back(dWx_tmp_l);
		_mWi.push_back(dWx_tmp_l);
		_mWf.push_back(dWx_tmp_l);
		_mWo.push_back(dWx_tmp_l);
	}
	
	//  U weights init (Weights are random and delta weight are 0)
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<std::vector<float>> Ua_tmp_l;
		std::vector<std::vector<float>> Ui_tmp_l;
		std::vector<std::vector<float>> Uf_tmp_l;
		std::vector<std::vector<float>> Uo_tmp_l;
		std::vector<std::vector<float>> dUx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			std::vector<float> Ua_tmp_i1;
			std::vector<float> Ui_tmp_i1;
			std::vector<float> Uf_tmp_i1;
			std::vector<float> Uo_tmp_i1;
			std::vector<float> dUx_tmp_i1;
			for (i2 = 0; i2 < _layers_size[l+1]; i2++)
			{
				val_tmp = randomize(init_min, init_max);
				Ua_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Ui_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Uf_tmp_i1.push_back(val_tmp);
				val_tmp = randomize(init_min, init_max);
				Uo_tmp_i1.push_back(val_tmp);
				dUx_tmp_i1.push_back(0.0);
			}
			Ua_tmp_l.push_back(Ua_tmp_i1);
			Ui_tmp_l.push_back(Ui_tmp_i1);
			Uf_tmp_l.push_back(Uf_tmp_i1);
			Uo_tmp_l.push_back(Uo_tmp_i1);
			dUx_tmp_l.push_back(dUx_tmp_i1);
		}
		_Ua.push_back(Ua_tmp_l);
		_Ui.push_back(Ui_tmp_l);
		_Uf.push_back(Uf_tmp_l);
		_Uo.push_back(Uo_tmp_l);
		_dUa.push_back(dUx_tmp_l);
		_dUi.push_back(dUx_tmp_l);
		_dUf.push_back(dUx_tmp_l);
		_dUo.push_back(dUx_tmp_l);
		_vUa.push_back(dUx_tmp_l);
		_vUi.push_back(dUx_tmp_l);
		_vUf.push_back(dUx_tmp_l);
		_vUo.push_back(dUx_tmp_l);
		_mUa.push_back(dUx_tmp_l);
		_mUi.push_back(dUx_tmp_l);
		_mUf.push_back(dUx_tmp_l);
		_mUo.push_back(dUx_tmp_l);
	}
	
	//  Bias init 
	for (l = 0; l < _total_size-1; l++)
	{
		std::vector<float> ba_tmp_l;
		std::vector<float> bi_tmp_l;
		std::vector<float> bf_tmp_l;
		std::vector<float> bo_tmp_l;
		std::vector<float> dbx_tmp_l;
		for (i1 = 0; i1 < _layers_size[l+1]; i1++)
		{
			val_tmp = randomize(init_min, init_max);
			ba_tmp_l.push_back(0.0);
			val_tmp = randomize(init_min, init_max);
			bi_tmp_l.push_back(0.0);
			val_tmp = randomize(init_min, init_max);
			bf_tmp_l.push_back(0.0);
			val_tmp = randomize(init_min, init_max);
			bo_tmp_l.push_back(0.0);
			dbx_tmp_l.push_back(0.0);
		}
		_ba.push_back(ba_tmp_l);
		_bi.push_back(bi_tmp_l);
		_bf.push_back(bf_tmp_l);
		_bo.push_back(bo_tmp_l);
		_dba.push_back(dbx_tmp_l);
		_dbi.push_back(dbx_tmp_l);
		_dbf.push_back(dbx_tmp_l);
		_dbo.push_back(dbx_tmp_l);
		_vba.push_back(dbx_tmp_l);
		_vbi.push_back(dbx_tmp_l);
		_vbf.push_back(dbx_tmp_l);
		_vbo.push_back(dbx_tmp_l);
		_mba.push_back(dbx_tmp_l);
		_mbi.push_back(dbx_tmp_l);
		_mbf.push_back(dbx_tmp_l);
		_mbo.push_back(dbx_tmp_l);
	}
		
}

//---------------------------------------------------------------------
std::vector<float> lstm_stk_flt::get_output()
{
	uint32_t i1;
	std::vector<float> output;
	for (i1=0; i1<_output_size; i1++)
	{
		output.push_back(_layers[_total_size-1][_nb_timestep-1][i1]);
	}
	return output;
}

//---------------------------------------------------------------------
// 
float lstm_stk_flt::get_mse()
{
	uint32_t i1;
	float mse_tmp = 0.0;
	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += _errors[_nb_timestep-1][i1]*_errors[_nb_timestep-1][i1];
	}
	return mse_tmp/_output_size;
}

//---------------------------------------------------------------------
// 
float lstm_stk_flt::get_mse_testing(std::vector<std::vector<float>> target)
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
// // https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void lstm_stk_flt::forward_train(std::vector<std::vector<float>> input)
{
	int l, t, i1, i2; //iterators
	float sum_a;
	float sum_i;
	float sum_f;
	float sum_o;
	
	_layers[0] = input;
	
	for (l = 1; l < _total_size; l++)
	{
		// --- T = 0 ---
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_a = 0.0f;
			sum_i = 0.0f;
			sum_f = 0.0f;
			sum_o = 0.0f;
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_a += _Wa[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_i += _Wi[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_f += _Wf[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_o += _Wo[l-1][i1][i2]*_layers[l-1][0][i2];
			}
			if (_use_interp_act)
			{
				_a_gate[l][0][i1] = tanh_interp(sum_a + _ba[l-1][i1]);
				_i_gate[l][0][i1] = sigmoid_interp(sum_i + _bi[l-1][i1]);
				_f_gate[l][0][i1] = sigmoid_interp(sum_f + _bf[l-1][i1]);
				_o_gate[l][0][i1] = sigmoid_interp(sum_o + _bo[l-1][i1]);
				_state[l][0][i1] = _a_gate[l][0][i1]*_i_gate[l][0][i1];
				_layers[l][0][i1] = tanh_interp(_state[l][0][i1])*_o_gate[l][0][i1];
			}
			else
			{
				_a_gate[l][0][i1] = tanh(sum_a + _ba[l-1][i1]);
				_i_gate[l][0][i1] = sigmoid(sum_i + _bi[l-1][i1]);
				_f_gate[l][0][i1] = sigmoid(sum_f + _bf[l-1][i1]);
				_o_gate[l][0][i1] = sigmoid(sum_o + _bo[l-1][i1]);
				_state[l][0][i1] = _a_gate[l][0][i1]*_i_gate[l][0][i1];
				_layers[l][0][i1] = tanh(_state[l][0][i1])*_o_gate[l][0][i1];
			}
		}
		// --- T = OTHERS TIMESTEPS ---
		for (t=1; t<_nb_timestep; t++)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				//The input layer is broadcast to the hidden layer
				sum_a = 0.0f;
				sum_i = 0.0f;
				sum_f = 0.0f;
				sum_o = 0.0f;
				for (i2=0; i2<_layers_size[l-1]; i2++)
				{
					sum_a += _Wa[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_i += _Wi[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_f += _Wf[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_o += _Wo[l-1][i1][i2]*_layers[l-1][t][i2];
				}
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_a += _Ua[l-1][i1][i2]*_layers[l][t-1][i2]; // t-1 because of BP from 0 to N-1!
					sum_i += _Ui[l-1][i1][i2]*_layers[l][t-1][i2];
					sum_f += _Uf[l-1][i1][i2]*_layers[l][t-1][i2];
					sum_o += _Uo[l-1][i1][i2]*_layers[l][t-1][i2];
				}
				
				
				if (_use_interp_act)
				{
					_a_gate[l][t][i1] = tanh_interp(sum_a + _ba[l-1][i1]);
					_i_gate[l][t][i1] = sigmoid_interp(sum_i + _bi[l-1][i1]);
					_f_gate[l][t][i1] = sigmoid_interp(sum_f + _bf[l-1][i1]);
					_o_gate[l][t][i1] = sigmoid_interp(sum_o + _bo[l-1][i1]);
					_state[l][t][i1] = _a_gate[l][t][i1]*_i_gate[l][t][i1] + _f_gate[l][t][i1]*_state[l][t-1][i1];
					_layers[l][t][i1] = tanh_interp(_state[l][t][i1])*_o_gate[l][t][i1];
				}
				else
				{
					_a_gate[l][t][i1] = tanh(sum_a + _ba[l-1][i1]);
					_i_gate[l][t][i1] = sigmoid(sum_i + _bi[l-1][i1]);
					_f_gate[l][t][i1] = sigmoid(sum_f + _bf[l-1][i1]);
					_o_gate[l][t][i1] = sigmoid(sum_o + _bo[l-1][i1]);
					_state[l][t][i1] = _a_gate[l][t][i1]*_i_gate[l][t][i1] + _f_gate[l][t][i1]*_state[l][t-1][i1];
					_layers[l][t][i1] = tanh(_state[l][t][i1])*_o_gate[l][t][i1];
				}
			}
		}
	}
	
}

//---------------------------------------------------------------------
// // https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void lstm_stk_flt::forward(std::vector<float> input)
{
	uint32_t l, t, i1, i2; //iterators
	float sum_a;
	float sum_i;
	float sum_f;
	float sum_o;
	
	_layers[0].erase(_layers[0].begin());
	_layers[0].push_back(input);
	
	for (l = 1; l < _total_size; l++)
	{
		// --- T = 0 ---
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_a = 0.0f;
			sum_i = 0.0f;
			sum_f = 0.0f;
			sum_o = 0.0f;
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_a += _Wa[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_i += _Wi[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_f += _Wf[l-1][i1][i2]*_layers[l-1][0][i2];
				sum_o += _Wo[l-1][i1][i2]*_layers[l-1][0][i2];
			}
			_a_gate[l][0][i1] = tanh(sum_a + _ba[l-1][i1]);
			_i_gate[l][0][i1] = sigmoid(sum_i + _bi[l-1][i1]);
			_f_gate[l][0][i1] = sigmoid(sum_f + _bf[l-1][i1]);
			_o_gate[l][0][i1] = sigmoid(sum_o + _bo[l-1][i1]);
			_state[l][0][i1] = _a_gate[l][0][i1]*_i_gate[l][0][i1];
			_layers[l][0][i1] = tanh(_state[l][0][i1])*_o_gate[l][0][i1];
		}
		// --- T = OTHERS TIMESTEPS ---
		for (t=1; t<_nb_timestep; t++)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				//The input layer is broadcast to the hidden layer
				sum_a = 0.0f;
				sum_i = 0.0f;
				sum_f = 0.0f;
				sum_o = 0.0f;
				for (i2=0; i2<_layers_size[l-1]; i2++)
				{
					sum_a += _Wa[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_i += _Wi[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_f += _Wf[l-1][i1][i2]*_layers[l-1][t][i2];
					sum_o += _Wo[l-1][i1][i2]*_layers[l-1][t][i2];
				}
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_a += _Ua[l-1][i1][i2]*_layers[l][t-1][i2]; // t-1 because of BP from 0 to N-1!
					sum_i += _Ui[l-1][i1][i2]*_layers[l][t-1][i2];
					sum_f += _Uf[l-1][i1][i2]*_layers[l][t-1][i2];
					sum_o += _Uo[l-1][i1][i2]*_layers[l][t-1][i2];
				}
				_a_gate[l][t][i1] = tanh(sum_a + _ba[l-1][i1]);
				_i_gate[l][t][i1] = sigmoid(sum_i + _bi[l-1][i1]);
				_f_gate[l][t][i1] = sigmoid(sum_f + _bf[l-1][i1]);
				_o_gate[l][t][i1] = sigmoid(sum_o + _bo[l-1][i1]);
				_state[l][t][i1] = _a_gate[l][t][i1]*_i_gate[l][t][i1] + _f_gate[l][t][i1]*_state[l][t-1][i1];
				_layers[l][t][i1] = tanh(_state[l][t][i1])*_o_gate[l][t][i1];
			}
		}
	}
	
}


//---------------------------------------------------------------------
// 
void lstm_stk_flt::backward_sgd_train(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t l, i1, i2; //iterators
	float sum_da, sum_ba, sum_Wa, sum_Ua;
	float sum_di, sum_bi, sum_Wi, sum_Ui;
	float sum_df, sum_bf, sum_Wf, sum_Uf;
	float sum_do, sum_bo, sum_Wo, sum_Uo;
	//std::cout << " STEP 0 " << std::endl;
	// get error
	//for (t=0;t<_nb_timestep;t++)
	//{
		//for (i1=0; i1<_output_size; i1++)
		//{
			//_errors[t][i1] =  target[t][i1] - _layers[_total_size-1][t][i1];
		//}
	//}
	
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
			_dhlayers[l][t][i2] = _dlayers[l][t][i2];
			_dstate[l][t][i2]  = _dhlayers[l][t][i2]*_o_gate[l][t][i2]*dtanh(_state[l][t][i2]);	
			_da_gate[l][t][i2] = _dstate[l][t][i2]*_i_gate[l][t][i2]*(1.0-(_a_gate[l][t][i2]*_a_gate[l][t][i2]));
			_di_gate[l][t][i2] = _dstate[l][t][i2]*_a_gate[l][t][i2]*_i_gate[l][t][i2]*(1.0-_i_gate[l][t][i2]);
			_df_gate[l][t][i2] = _dstate[l][t][i2]*_state[l][t-1][i2]*_f_gate[l][t][i2]*(1.0-_f_gate[l][t][i2]);
			_do_gate[l][t][i2] = _dhlayers[l][t][i2]*tanh(_state[l][t][i2])*_o_gate[l][t][i2]*(1.0-_o_gate[l][t][i2]);
		}
		//std::cout << " STEP 3 " << std::endl;
		// T=TIMESTEP-2 TO 0
		for (t=_nb_timestep-2; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				sum_da = 0.0f;
				sum_di = 0.0f;
				sum_df = 0.0f;
				sum_do = 0.0f;
				//std::cout << " STEP 31 => " << _nb_timestep << std::endl;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_da += _Ua[l-1][i2][i1]*_da_gate[l][t+1][i2];
					sum_di += _Ui[l-1][i2][i1]*_di_gate[l][t+1][i2];
					sum_df += _Uf[l-1][i2][i1]*_df_gate[l][t+1][i2];
					sum_do += _Uo[l-1][i2][i1]*_do_gate[l][t+1][i2];
				}
				//std::cout << " STEP 311 " << std::endl;
				if (l == _total_size-1)
				{
					_dhlayers[l][t][i1] =  _dlayers[l][t+1][i1]; // sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
				else
				{
					_dhlayers[l][t][i1] = _dlayers[l][t+1][i1] + sum_da + sum_di + sum_df + sum_do;
				}
				//std::cout << " STEP 313 " << std::endl;
				// _dout[t][i1] = _Dout[t][i1];
				_dstate[l][t][i1] = _dhlayers[l][t][i1]*_o_gate[l][t][i1]*dtanh(_state[l][t][i1]) + _dstate[l][t+1][i1]*_f_gate[l][t+1][i1];	
				_da_gate[l][t][i1] = _dstate[l][t][i1]*_i_gate[l][t][i1]*(1.0-(_a_gate[l][t][i1]*_a_gate[l][t][i1]));
				_di_gate[l][t][i1] = _dstate[l][t][i1]*_a_gate[l][t][i1]*_i_gate[l][t][i1]*(1.0-_i_gate[l][t][i1]);
				_df_gate[l][t][i1] = 0.0;
				//std::cout << " STEP 32 " << std::endl;
				if (t>0)
				{
					_df_gate[l][t][i1] = _dstate[l][t][i1]*_state[l][t-1][i1]*_f_gate[l][t][i1]*(1.0-_f_gate[l][t][i1]);
				}
				_do_gate[l][t][i1] = _dhlayers[l][t][i1]*tanh(_state[l][t][i1])*_o_gate[l][t][i1]*(1.0-_o_gate[l][t][i1]);
				//std::cout << " STEP 33 " << std::endl;
			}
		}
		//std::cout << " STEP 4 " << std::endl;
		
		// Update Input layers gradient
		for (t=_nb_timestep-1; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l-1]; i1++)
			{
				sum_da = 0.0f;
				sum_di = 0.0f;
				sum_df = 0.0f;
				sum_do = 0.0f;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_da += _Wa[l-1][i2][i1]*_da_gate[l][t][i2];
					sum_di += _Wi[l-1][i2][i1]*_di_gate[l][t][i2];
					sum_df += _Wf[l-1][i2][i1]*_df_gate[l][t][i2];
					sum_do += _Wo[l-1][i2][i1]*_do_gate[l][t][i2];
				}
				_dlayers[l-1][t][i1] = sum_da + sum_di + sum_df + sum_do;
			}
		}

		for (i1=0; i1<_layers_size[l]; i1++)
		{	
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_Wa = 0.0f;
				sum_Wi = 0.0f;
				sum_Wf = 0.0f;
				sum_Wo = 0.0f;
				for (t=0; t<_nb_timestep; t++)
				{
					sum_Wa += _da_gate[l][t][i1]*_layers[l-1][t][i2]; //input
					sum_Wi += _di_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wf += _df_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wo += _do_gate[l][t][i1]*_layers[l-1][t][i2];			
				} 
				
				_dWa[l-1][i1][i2] = _lr * sum_Wa + _mu * _dWa[l-1][i1][i2];
				_dWi[l-1][i1][i2] = _lr * sum_Wi + _mu * _dWi[l-1][i1][i2];
				_dWf[l-1][i1][i2] = _lr * sum_Wf + _mu * _dWf[l-1][i1][i2];
				_dWo[l-1][i1][i2] = _lr * sum_Wo + _mu * _dWo[l-1][i1][i2];

				_Wf[l-1][i1][i2] = _Wf[l-1][i1][i2] + _dWf[l-1][i1][i2];
				_Wi[l-1][i1][i2] = _Wi[l-1][i1][i2] + _dWi[l-1][i1][i2];
				_Wa[l-1][i1][i2] = _Wa[l-1][i1][i2] + _dWa[l-1][i1][i2];
				_Wo[l-1][i1][i2] = _Wo[l-1][i1][i2] + _dWo[l-1][i1][i2];

			}
			// output size loop	
			for (i2=0; i2<_layers_size[l]; i2++)
			{
				sum_Ua = 0.0f;
				sum_Ui = 0.0f;
				sum_Uf = 0.0f;
				sum_Uo = 0.0f;
				for (t=0; t<_nb_timestep-1; t++)
				{
					sum_Ua += _da_gate[l][t+1][i1]*_layers[l][t][i2]; //output
					sum_Ui += _di_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uf += _df_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uo += _do_gate[l][t+1][i1]*_layers[l][t][i2];
				}
				_dUa[l-1][i1][i2] = _lr * sum_Ua + _mu * _dUa[l-1][i1][i2];
				_dUi[l-1][i1][i2] = _lr * sum_Ui + _mu * _dUi[l-1][i1][i2];
				_dUf[l-1][i1][i2] = _lr * sum_Uf + _mu * _dUf[l-1][i1][i2];
				_dUo[l-1][i1][i2] = _lr * sum_Uo + _mu * _dUo[l-1][i1][i2];

				_Ua[l-1][i1][i2] = _Ua[l-1][i1][i2] + _dUa[l-1][i1][i2];
				_Ui[l-1][i1][i2] = _Ui[l-1][i1][i2] + _dUi[l-1][i1][i2];
				_Uf[l-1][i1][i2] = _Uf[l-1][i1][i2] + _dUf[l-1][i1][i2];
				_Uo[l-1][i1][i2] = _Uo[l-1][i1][i2] + _dUo[l-1][i1][i2];
			}
			sum_ba = 0.0f;
			sum_bi = 0.0f;
			sum_bf = 0.0f;
			sum_bo = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_ba += _da_gate[l][t][i1];
				sum_bi += _di_gate[l][t][i1];
				sum_bf += _df_gate[l][t][i1];
				sum_bo += _do_gate[l][t][i1];
			}
			_dba[l-1][i1] = _lr * sum_ba + _mu * _dba[l-1][i1];
			_dbi[l-1][i1] = _lr * sum_bi + _mu * _dbi[l-1][i1];
			_dbf[l-1][i1] = _lr * sum_bf + _mu * _dbf[l-1][i1];
			_dbo[l-1][i1] = _lr * sum_bo + _mu * _dbo[l-1][i1];

			_ba[l-1][i1] = _ba[l-1][i1] + _dba[l-1][i1];
			_bi[l-1][i1] = _bi[l-1][i1] + _dbi[l-1][i1];
			_bf[l-1][i1] = _bf[l-1][i1] + _dbf[l-1][i1];
			_bo[l-1][i1] = _bo[l-1][i1] + _dbo[l-1][i1];
		}
	}
}


//---------------------------------------------------------------------
// 
void lstm_stk_flt::backward_adam_train(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t l, i1, i2; //iterators
	float sum_da, sum_ba, sum_Wa, sum_Ua;
	float sum_di, sum_bi, sum_Wi, sum_Ui;
	float sum_df, sum_bf, sum_Wf, sum_Uf;
	float sum_do, sum_bo, sum_Wo, sum_Uo;

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
			_dhlayers[l][t][i2] = _dlayers[l][t][i2];
			_dstate[l][t][i2]  = _dhlayers[l][t][i2]*_o_gate[l][t][i2]*dtanh(_state[l][t][i2]);	
			_da_gate[l][t][i2] = _dstate[l][t][i2]*_i_gate[l][t][i2]*(1.0-(_a_gate[l][t][i2]*_a_gate[l][t][i2]));
			_di_gate[l][t][i2] = _dstate[l][t][i2]*_a_gate[l][t][i2]*_i_gate[l][t][i2]*(1.0-_i_gate[l][t][i2]);
			_df_gate[l][t][i2] = _dstate[l][t][i2]*_state[l][t-1][i2]*_f_gate[l][t][i2]*(1.0-_f_gate[l][t][i2]);
			_do_gate[l][t][i2] = _dhlayers[l][t][i2]*tanh(_state[l][t][i2])*_o_gate[l][t][i2]*(1.0-_o_gate[l][t][i2]);
		}
		//std::cout << " STEP 3 " << std::endl;
		// T=TIMESTEP-2 TO 0
		for (t=_nb_timestep-2; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l]; i1++)
			{
				sum_da = 0.0f;
				sum_di = 0.0f;
				sum_df = 0.0f;
				sum_do = 0.0f;
				//std::cout << " STEP 31 => " << _nb_timestep << std::endl;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_da += _Ua[l-1][i2][i1]*_da_gate[l][t+1][i2];
					sum_di += _Ui[l-1][i2][i1]*_di_gate[l][t+1][i2];
					sum_df += _Uf[l-1][i2][i1]*_df_gate[l][t+1][i2];
					sum_do += _Uo[l-1][i2][i1]*_do_gate[l][t+1][i2];
				}
				//std::cout << " STEP 311 " << std::endl;
				//_dhlayers[l][t][i1] = sum_da + sum_di + sum_df + sum_do;// + _dlayers[l][t][i1];
				//std::cout << " STEP 312 " << std::endl;
				//_dhlayers[l][t][i1] = _dhlayers[l][t][i1] + _dlayers[l][t][i1] ;
				
				// OK
				//_dhlayers[l][t][i1] = _dlayers[l][t][i1] ;
			
				//std::cout << " STEP 311 " << std::endl;
				if (l == _total_size-1)
				{
					_dhlayers[l][t][i1] =  _dlayers[l][t+1][i1]; // sum_dz + sum_dr + sum_dh*_r_gate[l][t][i1];
				}
				else
				{
					_dhlayers[l][t][i1] = _dlayers[l][t+1][i1] + sum_da + sum_di + sum_df + sum_do;
				}
				
				//_dhlayers[l][t][i1] = _dlayers[l][t+1][i1] + sum_da + sum_di + sum_df + sum_do;
				//std::cout << " STEP 313 " << std::endl;
				// _dout[t][i1] = _Dout[t][i1];
				_dstate[l][t][i1] = _dhlayers[l][t][i1]*_o_gate[l][t][i1]*dtanh(_state[l][t][i1]) + _dstate[l][t+1][i1]*_f_gate[l][t+1][i1];	
				_da_gate[l][t][i1] = _dstate[l][t][i1]*_i_gate[l][t][i1]*(1.0-(_a_gate[l][t][i1]*_a_gate[l][t][i1]));
				_di_gate[l][t][i1] = _dstate[l][t][i1]*_a_gate[l][t][i1]*_i_gate[l][t][i1]*(1.0-_i_gate[l][t][i1]);
				_df_gate[l][t][i1] = 0.0;
				//std::cout << " STEP 32 " << std::endl;
				if (t>0)
				{
					_df_gate[l][t][i1] = _dstate[l][t][i1]*_state[l][t-1][i1]*_f_gate[l][t][i1]*(1.0-_f_gate[l][t][i1]);
				}
				_do_gate[l][t][i1] = _dhlayers[l][t][i1]*tanh(_state[l][t][i1])*_o_gate[l][t][i1]*(1.0-_o_gate[l][t][i1]);
				//std::cout << " STEP 33 " << std::endl;
			}
		}
		//std::cout << " STEP 4 " << std::endl;
		
		// Update Input layers gradient
		for (t=_nb_timestep-1; t>=0; t--)
		{
			for (i1=0; i1<_layers_size[l-1]; i1++)
			{
				sum_da = 0.0f;
				sum_di = 0.0f;
				sum_df = 0.0f;
				sum_do = 0.0f;
				for (i2=0; i2<_layers_size[l]; i2++)
				{
					sum_da += _Wa[l-1][i2][i1]*_da_gate[l][t][i2];
					sum_di += _Wi[l-1][i2][i1]*_di_gate[l][t][i2];
					sum_df += _Wf[l-1][i2][i1]*_df_gate[l][t][i2];
					sum_do += _Wo[l-1][i2][i1]*_do_gate[l][t][i2];
				}
				_dlayers[l-1][t][i1] = sum_da + sum_di + sum_df + sum_do;
			}
		}

		for (i1=0; i1<_layers_size[l]; i1++)
		{	
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				sum_Wa = 0.0f;
				sum_Wi = 0.0f;
				sum_Wf = 0.0f;
				sum_Wo = 0.0f;
				for (t=0; t<_nb_timestep; t++)
				{
					sum_Wa += _da_gate[l][t][i1]*_layers[l-1][t][i2]; //input
					sum_Wi += _di_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wf += _df_gate[l][t][i1]*_layers[l-1][t][i2];
					sum_Wo += _do_gate[l][t][i1]*_layers[l-1][t][i2];			
				} 
				
				_mWa[l-1][i1][i2] = _b1 * _mWa[l-1][i1][i2] + (1 - _b1) * sum_Wa;
				_mWi[l-1][i1][i2] = _b1 * _mWi[l-1][i1][i2] + (1 - _b1) * sum_Wi;
				_mWf[l-1][i1][i2] = _b1 * _mWf[l-1][i1][i2] + (1 - _b1) * sum_Wf;
				_mWo[l-1][i1][i2] = _b1 * _mWo[l-1][i1][i2] + (1 - _b1) * sum_Wo;

				_vWa[l-1][i1][i2] = _b2 * _vWa[l-1][i1][i2] + (1 -_b2) * sum_Wa * sum_Wa;
				_vWi[l-1][i1][i2] = _b2 * _vWi[l-1][i1][i2] + (1 -_b2) * sum_Wi * sum_Wi;
				_vWf[l-1][i1][i2] = _b2 * _vWf[l-1][i1][i2] + (1 -_b2) * sum_Wf * sum_Wf;
				_vWo[l-1][i1][i2] = _b2 * _vWo[l-1][i1][i2] + (1 -_b2) * sum_Wo * sum_Wo;

				_dWa[l-1][i1][i2] = _lr * (_mWa[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWa[l-1][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWi[l-1][i1][i2] = _lr * (_mWi[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWi[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dWf[l-1][i1][i2] = _lr * (_mWf[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWf[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dWo[l-1][i1][i2] = _lr * (_mWo[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vWo[l-1][i1][i2] / (1 - _b2)) + _epsilon);

				_Wa[l-1][i1][i2] = _Wa[l-1][i1][i2] + _dWa[l-1][i1][i2];
				_Wi[l-1][i1][i2] = _Wi[l-1][i1][i2] + _dWi[l-1][i1][i2];
				_Wf[l-1][i1][i2] = _Wf[l-1][i1][i2] + _dWf[l-1][i1][i2];
				_Wo[l-1][i1][i2] = _Wo[l-1][i1][i2] + _dWo[l-1][i1][i2];

			}
			// output size loop	
			for (i2=0; i2<_layers_size[l]; i2++)
			{
				sum_Ua = 0.0f;
				sum_Ui = 0.0f;
				sum_Uf = 0.0f;
				sum_Uo = 0.0f;
				for (t=0; t<_nb_timestep-1; t++)
				{
					sum_Ua += _da_gate[l][t+1][i1]*_layers[l][t][i2]; //output
					sum_Ui += _di_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uf += _df_gate[l][t+1][i1]*_layers[l][t][i2];
					sum_Uo += _do_gate[l][t+1][i1]*_layers[l][t][i2];
				}
				
				_mUa[l-1][i1][i2] = _b1 * _mUa[l-1][i1][i2] + (1 - _b1) * sum_Ua;
				_mUi[l-1][i1][i2] = _b1 * _mUi[l-1][i1][i2] + (1 - _b1) * sum_Ui;
				_mUf[l-1][i1][i2] = _b1 * _mUf[l-1][i1][i2] + (1 - _b1) * sum_Uf;
				_mUo[l-1][i1][i2] = _b1 * _mUo[l-1][i1][i2] + (1 - _b1) * sum_Uo;

				_vUa[l-1][i1][i2] = _b2 * _vUa[l-1][i1][i2] + (1 -_b2) * sum_Ua * sum_Ua;
				_vUi[l-1][i1][i2] = _b2 * _vUi[l-1][i1][i2] + (1 -_b2) * sum_Ui * sum_Ui;
				_vUf[l-1][i1][i2] = _b2 * _vUf[l-1][i1][i2] + (1 -_b2) * sum_Uf * sum_Uf;
				_vUo[l-1][i1][i2] = _b2 * _vUo[l-1][i1][i2] + (1 -_b2) * sum_Uo * sum_Uo;

				_dUa[l-1][i1][i2] = _lr * (_mUa[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUa[l-1][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dUi[l-1][i1][i2] = _lr * (_mUi[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUi[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dUf[l-1][i1][i2] = _lr * (_mUf[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUf[l-1][i1][i2] / (1 - _b2)) + _epsilon);
				_dUo[l-1][i1][i2] = _lr * (_mUo[l-1][i1][i2] / (1 - _b1)) / (sqrt(_vUo[l-1][i1][i2] / (1 - _b2)) + _epsilon);

				_Ua[l-1][i1][i2] = _Ua[l-1][i1][i2] + _dUa[l-1][i1][i2];
				_Ui[l-1][i1][i2] = _Ui[l-1][i1][i2] + _dUi[l-1][i1][i2];
				_Uf[l-1][i1][i2] = _Uf[l-1][i1][i2] + _dUf[l-1][i1][i2];
				_Uo[l-1][i1][i2] = _Uo[l-1][i1][i2] + _dUo[l-1][i1][i2];
			}
			sum_ba = 0.0f;
			sum_bi = 0.0f;
			sum_bf = 0.0f;
			sum_bo = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_ba += _da_gate[l][t][i1];
				sum_bi += _di_gate[l][t][i1];
				sum_bf += _df_gate[l][t][i1];
				sum_bo += _do_gate[l][t][i1];
			}
			
			_mba[l-1][i1] = _b1 * _mba[l-1][i1] + (1 - _b1) * sum_ba;
			_mbi[l-1][i1] = _b1 * _mbi[l-1][i1] + (1 - _b1) * sum_bi;
			_mbf[l-1][i1] = _b1 * _mbf[l-1][i1] + (1 - _b1) * sum_bf;
			_mbo[l-1][i1] = _b1 * _mbo[l-1][i1] + (1 - _b1) * sum_bo;

			_vba[l-1][i1] = _b2 * _vba[l-1][i1] + (1 -_b2) * sum_ba * sum_ba;
			_vbi[l-1][i1] = _b2 * _vbi[l-1][i1] + (1 -_b2) * sum_bi * sum_bi;
			_vbf[l-1][i1] = _b2 * _vbf[l-1][i1] + (1 -_b2) * sum_bf * sum_bf;
			_vbo[l-1][i1] = _b2 * _vbo[l-1][i1] + (1 -_b2) * sum_bo * sum_bo;

			_dba[l-1][i1] = _lr * (_mba[l-1][i1] / (1 - _b1)) / (sqrt(_vba[l-1][i1] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbi[l-1][i1] = _lr * (_mbi[l-1][i1] / (1 - _b1)) / (sqrt(_vbi[l-1][i1] / (1 - _b2)) + _epsilon);
			_dbf[l-1][i1] = _lr * (_mbf[l-1][i1] / (1 - _b1)) / (sqrt(_vbf[l-1][i1] / (1 - _b2)) + _epsilon);
			_dbo[l-1][i1] = _lr * (_mbo[l-1][i1] / (1 - _b1)) / (sqrt(_vbo[l-1][i1] / (1 - _b2)) + _epsilon);

			_ba[l-1][i1] = _ba[l-1][i1] + _dba[l-1][i1];
			_bi[l-1][i1] = _bi[l-1][i1] + _dbi[l-1][i1];
			_bf[l-1][i1] = _bf[l-1][i1] + _dbf[l-1][i1];
			_bo[l-1][i1] = _bo[l-1][i1] + _dbo[l-1][i1];
		}
	}
}

//----------------------------------------------------------------------

void lstm_stk_flt::clear_momentums()
{
	uint32_t l, i1, i2; //iterators
	for (l = 1; l < _total_size; l++)
	{
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			for (i2=0; i2<_layers_size[l-1]; i2++)
			{
				_mWa[l-1][i1][i2] = 0.0;
				_mWi[l-1][i1][i2] = 0.0;
				_mWf[l-1][i1][i2] = 0.0;
				_mWo[l-1][i1][i2] = 0.0;
				_vWa[l-1][i1][i2] = 0.0;
				_vWi[l-1][i1][i2] = 0.0;
				_vWf[l-1][i1][i2] = 0.0;
				_vWo[l-1][i1][i2] = 0.0;
			}
		}
		
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			for (i2=0; i2<_layers_size[l]; i2++)
			{
				_mUa[l-1][i1][i2] = 0.0;
				_mUi[l-1][i1][i2] = 0.0;
				_mUf[l-1][i1][i2] = 0.0;
				_mUo[l-1][i1][i2] = 0.0;
				_vUa[l-1][i1][i2] = 0.0;
				_vUi[l-1][i1][i2] = 0.0;
				_vUf[l-1][i1][i2] = 0.0;
				_vUo[l-1][i1][i2] = 0.0;
			}
		}
		for (i1=0; i1<_layers_size[l]; i1++)
		{
			_mba[l-1][i1] = 0.0;
			_mbi[l-1][i1] = 0.0;
			_mbf[l-1][i1] = 0.0;
			_mbo[l-1][i1] = 0.0;
			_vba[l-1][i1] = 0.0;
			_vbi[l-1][i1] = 0.0;
			_vbf[l-1][i1] = 0.0;
			_vbo[l-1][i1] = 0.0;
		}
	}
}


//---------------------------------------------------------------------
float lstm_stk_flt::sigmoid(float x)
{
	return (1.0 / (1.0 + exp(-x)));
}

//---------------------------------------------------------------------
float lstm_stk_flt::dsigmoid(float x)
{
	return (x*(1.0 - x));
}

//---------------------------------------------------------------------
float lstm_stk_flt::dtanh(float x)
{
	float y = tanh(x);
	return (1.0 - y*y);
}

//---------------------------------------------------------------------
float lstm_stk_flt::ddtanh(float x)
{
	//float y = tanh(x);
	return (1.0 - x*x);
}

//---------------------------------------------------------------------
float lstm_stk_flt::leakyrelu(float x) 
{
    return (x < 0.0 ? (0.001 * x) : x);
}

//---------------------------------------------------------------------
float lstm_stk_flt::dleakyrelu(float x) 
{
   return (x < 0.0 ? 0.001 : x);
}

//---------------------------------------------------------------------
float lstm_stk_flt::softplus(float x)
{
    return log(1.0+(exp(x)));
}

//---------------------------------------------------------------------
float lstm_stk_flt::dsoftplus(float x)
{
    return 1.0/(1.0+(exp(-x)));
}

//---------------------------------------------------------------------
float lstm_stk_flt::randomize(float min, float max){
    float f = ((float) rand()) / RAND_MAX;
    return min + f * (max - min);
}

//---------------------------------------------------------------------------------------------------
float lstm_stk_flt::sigmoid_interp(float x)
{
	std::cout << "!! Warning !! inperpolated version not included in opensource version - back to normal sigmoid" << std::endl;
	float y = sigmoid(x);
	return y;
}

//---------------------------------------------------------------------------------------------------
float lstm_stk_flt::tanh_interp(float x)
{
	std::cout << "!! Warning !! inperpolated version not included in opensource version - back to normal tanh" << std::endl;
	float y = tanh(x);
	return y;
}

//---------------------------------------------------------------------
void lstm_stk_flt::set_interp_act(bool val)
{
	_use_interp_act = val;
}

