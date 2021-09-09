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

#include <lstm.hh>

//---------------------------------------------------------------------
lstm::lstm()
{
	_nb_timestep = 0;
	_input_size = 0;
	_output_size = 0;
	_init_ok = false;
	
	// defaut lr
	_lr = 0.1;
	_mu = 0.0;
	
	// defaut optim id
	_optim_id = 0;
	
	// set error report
	_errors_report.push_back(0.0); // mse
	_errors_report.push_back(0.0); // rmse
	//_errors_report.push_back(0.0) // m
}

//---------------------------------------------------------------------
lstm::~lstm()
{
	
}

//---------------------------------------------------------------------
void lstm::set_lr_mu(float lr, float mu)
{
	_lr = lr;
	_mu = mu;
}
//---------------------------------------------------------------------
void lstm::set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon)
{
	_lr = lr;
	_b1 = b1;
	_b2 = b2;
	_epsilon = epsilon;
}

//---------------------------------------------------------------------
void lstm::set_optimizer(uint8_t optim_id)
{
	_optim_id = optim_id;
}


//---------------------------------------------------------------------
void lstm::init(uint32_t nb_timestep, uint32_t input_size, uint32_t output_size)
{
	float val_tmp;
	uint32_t t, i1, i2; //iterators
	_nb_timestep = nb_timestep;
	_input_size = input_size;
	_output_size = output_size;
	
	// cell inits
	val_tmp = 0.0;
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<float> x_tmp;
		std::vector<float> dx_tmp;
		std::vector<float> a_tmp;
		std::vector<float> i_tmp;
		std::vector<float> f_tmp;
		std::vector<float> o_tmp;
		std::vector<float> state_tmp;
		std::vector<float> out_tmp;
		std::vector<float> da_tmp;
		std::vector<float> di_tmp;
		std::vector<float> df_tmp;
		std::vector<float> do_tmp;
		std::vector<float> dstate_tmp;
		std::vector<float> dout_tmp;
		std::vector<float> Dout_tmp;
		std::vector<float> errors_tmp;
		for (i1 = 0; i1 < _input_size; i1++)
		{
			x_tmp.push_back(val_tmp);
			dx_tmp.push_back(val_tmp);
		}
		_x.push_back(x_tmp); 
		_dx.push_back(dx_tmp); 
		for (i1 = 0; i1 < _output_size; i1++)
		{
			
			a_tmp.push_back(val_tmp); 
			i_tmp.push_back(val_tmp); 
			f_tmp.push_back(val_tmp);
			o_tmp.push_back(val_tmp);  
			state_tmp.push_back(val_tmp); 
			out_tmp.push_back(val_tmp); 
			da_tmp.push_back(val_tmp);  
			di_tmp.push_back(val_tmp); 
			df_tmp.push_back(val_tmp); 
			do_tmp.push_back(val_tmp); 
			dstate_tmp.push_back(val_tmp); 
			dout_tmp.push_back(val_tmp); 
			Dout_tmp.push_back(val_tmp); 
			errors_tmp.push_back(val_tmp); 
		}
		_a_gate.push_back(a_tmp); 
		_i_gate.push_back(i_tmp); 
		_f_gate.push_back(f_tmp);
		_o_gate.push_back(o_tmp);  
		_state.push_back(state_tmp); 
		_out.push_back(out_tmp); 
		_da_gate.push_back(da_tmp);  
		_di_gate.push_back(di_tmp); 
		_df_gate.push_back(df_tmp); 
		_do_gate.push_back(do_tmp); 
		_dstate.push_back(dstate_tmp); 
		_dout.push_back(dout_tmp); 
		_Dout.push_back(Dout_tmp); 
		_errors.push_back(errors_tmp); 
	}
	// Bias
	val_tmp = 0.0;
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<float> ba_tmp;
		std::vector<float> bi_tmp;
		std::vector<float> bf_tmp;
		std::vector<float> bo_tmp;
		std::vector<float> dba_tmp;
		std::vector<float> dbi_tmp;
		std::vector<float> dbf_tmp;
		std::vector<float> dbo_tmp;
		std::vector<float> m_ba_tmp;
		std::vector<float> m_bi_tmp;
		std::vector<float> m_bf_tmp;
		std::vector<float> m_bo_tmp;
		std::vector<float> v_ba_tmp;
		std::vector<float> v_bi_tmp;
		std::vector<float> v_bf_tmp;
		std::vector<float> v_bo_tmp;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			val_tmp = randomize(-1.0, 1.0);
			ba_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0);
			bi_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0); 
			bf_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0);
			bo_tmp.push_back(val_tmp);  
			dba_tmp.push_back(0.0); 
			dbi_tmp.push_back(0.0); 
			dbf_tmp.push_back(0.0);
			dbo_tmp.push_back(0.0);
			m_ba_tmp.push_back(0.0);
			m_bi_tmp.push_back(0.0);
			m_bf_tmp.push_back(0.0);
			m_bo_tmp.push_back(0.0);
			v_ba_tmp.push_back(0.0);
			v_bi_tmp.push_back(0.0);
			v_bf_tmp.push_back(0.0);
			v_bo_tmp.push_back(0.0);  
		}
		_ba.push_back(ba_tmp); 
		_bi.push_back(bi_tmp); 
		_bf.push_back(bf_tmp);
		_bo.push_back(bo_tmp);
		_dba.push_back(dba_tmp); 
		_dbi.push_back(dbi_tmp); 
		_dbf.push_back(dbf_tmp);
		_dbo.push_back(dbo_tmp);
		
		_m_ba.push_back(m_ba_tmp);
		_m_bi.push_back(m_bi_tmp);
		_m_bf.push_back(m_bf_tmp);
		_m_bo.push_back(m_bo_tmp);
		_v_ba.push_back(v_ba_tmp);
		_v_bi.push_back(v_bi_tmp);
		_v_bf.push_back(v_bf_tmp);
		_v_bo.push_back(v_bo_tmp);
	}

	// W weights
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<std::vector<float>> Wa_tmp_i1;
		std::vector<std::vector<float>> Wi_tmp_i1;
		std::vector<std::vector<float>> Wf_tmp_i1;
		std::vector<std::vector<float>> Wo_tmp_i1;
		
		std::vector<std::vector<float>> dWa_tmp_i1;
		std::vector<std::vector<float>> dWi_tmp_i1;
		std::vector<std::vector<float>> dWf_tmp_i1;
		std::vector<std::vector<float>> dWo_tmp_i1;

		std::vector<std::vector<float>> m_Wa_tmp_i1;
		std::vector<std::vector<float>> m_Wi_tmp_i1;
		std::vector<std::vector<float>> m_Wf_tmp_i1;
		std::vector<std::vector<float>> m_Wo_tmp_i1;
		std::vector<std::vector<float>> v_Wa_tmp_i1;
		std::vector<std::vector<float>> v_Wi_tmp_i1;
		std::vector<std::vector<float>> v_Wf_tmp_i1;
		std::vector<std::vector<float>> v_Wo_tmp_i1;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			std::vector<float> Wa_tmp_i2;
			std::vector<float> Wi_tmp_i2;
			std::vector<float> Wf_tmp_i2;
			std::vector<float> Wo_tmp_i2;
			std::vector<float> dWa_tmp_i2;
			std::vector<float> dWi_tmp_i2;
			std::vector<float> dWf_tmp_i2;
			std::vector<float> dWo_tmp_i2;
			std::vector<float> m_Wa_tmp_i2;
			std::vector<float> m_Wi_tmp_i2;
			std::vector<float> m_Wf_tmp_i2;
			std::vector<float> m_Wo_tmp_i2;
			std::vector<float> v_Wa_tmp_i2;
			std::vector<float> v_Wi_tmp_i2;
			std::vector<float> v_Wf_tmp_i2;
			std::vector<float> v_Wo_tmp_i2;
			for (i2 = 0; i2 < _input_size; i2++)
			{
				val_tmp = randomize(-1.0, 1.0);
				Wa_tmp_i2.push_back(val_tmp);
				dWa_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wi_tmp_i2.push_back(val_tmp);
				dWi_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wf_tmp_i2.push_back(val_tmp);
				dWf_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wo_tmp_i2.push_back(val_tmp);
				dWo_tmp_i2.push_back(0.0);

				m_Wa_tmp_i2.push_back(0.0);
				m_Wi_tmp_i2.push_back(0.0);
				m_Wf_tmp_i2.push_back(0.0);
				m_Wo_tmp_i2.push_back(0.0);
				v_Wa_tmp_i2.push_back(0.0);
				v_Wi_tmp_i2.push_back(0.0);
				v_Wf_tmp_i2.push_back(0.0);
				v_Wo_tmp_i2.push_back(0.0);
			}
			Wa_tmp_i1.push_back(Wa_tmp_i2);
			Wi_tmp_i1.push_back(Wi_tmp_i2);
			Wf_tmp_i1.push_back(Wf_tmp_i2);
			Wo_tmp_i1.push_back(Wo_tmp_i2);
			dWa_tmp_i1.push_back(dWa_tmp_i2);
			dWi_tmp_i1.push_back(dWi_tmp_i2);
			dWf_tmp_i1.push_back(dWf_tmp_i2);
			dWo_tmp_i1.push_back(dWo_tmp_i2);
			m_Wa_tmp_i1.push_back(m_Wa_tmp_i2);
			m_Wi_tmp_i1.push_back(m_Wi_tmp_i2);
			m_Wf_tmp_i1.push_back(m_Wf_tmp_i2);
			m_Wo_tmp_i1.push_back(m_Wo_tmp_i2);
			v_Wa_tmp_i1.push_back(v_Wa_tmp_i2);
			v_Wi_tmp_i1.push_back(v_Wi_tmp_i2);
			v_Wf_tmp_i1.push_back(v_Wf_tmp_i2);
			v_Wo_tmp_i1.push_back(v_Wo_tmp_i2);
			
		}
		_Wa.push_back(Wa_tmp_i1);
		_Wi.push_back(Wi_tmp_i1);
		_Wf.push_back(Wf_tmp_i1);
		_Wo.push_back(Wo_tmp_i1);
		_dWa.push_back(dWa_tmp_i1);
		_dWi.push_back(dWi_tmp_i1);
		_dWf.push_back(dWf_tmp_i1);
		_dWo.push_back(dWo_tmp_i1);

		_m_Wa.push_back(m_Wa_tmp_i1);
		_m_Wi.push_back(m_Wi_tmp_i1);
		_m_Wf.push_back(m_Wf_tmp_i1);
		_m_Wo.push_back(m_Wo_tmp_i1);
		_v_Wa.push_back(v_Wa_tmp_i1);
		_v_Wi.push_back(v_Wi_tmp_i1);
		_v_Wf.push_back(v_Wf_tmp_i1);
		_v_Wo.push_back(v_Wo_tmp_i1);
	}


	
	// U weights
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<std::vector<float>> Ua_tmp_i1;
		std::vector<std::vector<float>> Ui_tmp_i1;
		std::vector<std::vector<float>> Uf_tmp_i1;
		std::vector<std::vector<float>> Uo_tmp_i1;

		std::vector<std::vector<float>> dUa_tmp_i1;
		std::vector<std::vector<float>> dUi_tmp_i1;
		std::vector<std::vector<float>> dUf_tmp_i1;
		std::vector<std::vector<float>> dUo_tmp_i1;

		std::vector<std::vector<float>> m_Ua_tmp_i1;
		std::vector<std::vector<float>> m_Ui_tmp_i1;
		std::vector<std::vector<float>> m_Uf_tmp_i1;
		std::vector<std::vector<float>> m_Uo_tmp_i1;
		std::vector<std::vector<float>> v_Ua_tmp_i1;
		std::vector<std::vector<float>> v_Ui_tmp_i1;
		std::vector<std::vector<float>> v_Uf_tmp_i1;
		std::vector<std::vector<float>> v_Uo_tmp_i1;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			std::vector<float> Ua_tmp_i2;
			std::vector<float> Ui_tmp_i2;
			std::vector<float> Uf_tmp_i2;
			std::vector<float> Uo_tmp_i2;
			std::vector<float> dUa_tmp_i2;
			std::vector<float> dUi_tmp_i2;
			std::vector<float> dUf_tmp_i2;
			std::vector<float> dUo_tmp_i2;
			std::vector<float> m_Ua_tmp_i2;
			std::vector<float> m_Ui_tmp_i2;
			std::vector<float> m_Uf_tmp_i2;
			std::vector<float> m_Uo_tmp_i2;
			std::vector<float> v_Ua_tmp_i2;
			std::vector<float> v_Ui_tmp_i2;
			std::vector<float> v_Uf_tmp_i2;
			std::vector<float> v_Uo_tmp_i2;
			for (i2 = 0; i2 < _output_size; i2++)
			{
				val_tmp = randomize(-1.0, 1.0);
				Ua_tmp_i2.push_back(val_tmp);
				dUa_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Ui_tmp_i2.push_back(val_tmp);
				dUi_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Uf_tmp_i2.push_back(val_tmp);
				dUf_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Uo_tmp_i2.push_back(val_tmp);
				dUo_tmp_i2.push_back(0.0);

				m_Ua_tmp_i2.push_back(0.0);
				m_Ui_tmp_i2.push_back(0.0);
				m_Uf_tmp_i2.push_back(0.0);
				m_Uo_tmp_i2.push_back(0.0);
				v_Ua_tmp_i2.push_back(0.0);
				v_Ui_tmp_i2.push_back(0.0);
				v_Uf_tmp_i2.push_back(0.0);
				v_Uo_tmp_i2.push_back(0.0);
			}
			Ua_tmp_i1.push_back(Ua_tmp_i2);
			Ui_tmp_i1.push_back(Ui_tmp_i2);
			Uf_tmp_i1.push_back(Uf_tmp_i2);
			Uo_tmp_i1.push_back(Uo_tmp_i2);
			dUa_tmp_i1.push_back(dUa_tmp_i2);
			dUi_tmp_i1.push_back(dUi_tmp_i2);
			dUf_tmp_i1.push_back(dUf_tmp_i2);
			dUo_tmp_i1.push_back(dUo_tmp_i2);

			m_Ua_tmp_i1.push_back(dUa_tmp_i2);
			m_Ui_tmp_i1.push_back(dUi_tmp_i2);
			m_Uf_tmp_i1.push_back(dUf_tmp_i2);
			m_Uo_tmp_i1.push_back(dUo_tmp_i2);
			v_Ua_tmp_i1.push_back(dUa_tmp_i2);
			v_Ui_tmp_i1.push_back(dUi_tmp_i2);
			v_Uf_tmp_i1.push_back(dUf_tmp_i2);
			v_Uo_tmp_i1.push_back(dUo_tmp_i2);
		}
		_Ua.push_back(Ua_tmp_i1);
		_Ui.push_back(Ui_tmp_i1);
		_Uf.push_back(Uf_tmp_i1);
		_Uo.push_back(Uo_tmp_i1);
		_dUa.push_back(dUa_tmp_i1);
		_dUi.push_back(dUi_tmp_i1);
		_dUf.push_back(dUf_tmp_i1);
		_dUo.push_back(dUo_tmp_i1);

		_m_Ua.push_back(m_Ua_tmp_i1);
		_m_Ui.push_back(m_Ui_tmp_i1);
		_m_Uf.push_back(m_Uf_tmp_i1);
		_m_Uo.push_back(m_Uo_tmp_i1);
		_v_Ua.push_back(v_Ua_tmp_i1);
		_v_Ui.push_back(v_Ui_tmp_i1);
		_v_Uf.push_back(v_Uf_tmp_i1);
		_v_Uo.push_back(v_Uo_tmp_i1);
	}
}

//---------------------------------------------------------------------
// // https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void lstm::forward(std::vector<std::vector<float>> input)
{
	uint32_t t, i1, i2; //iterators
	float sum_a;
	float sum_i;
	float sum_f;
	float sum_o;
	std::vector<std::vector<float>> output;
	
	_x = input;
	// --------------------------------
	// T = 0
	// W Weights matrix sums
	for (i1=0; i1<_output_size; i1++)
	{
		//The input layer is broadcast to the hidden layer
		sum_a = 0.0f;
		sum_i = 0.0f;
		sum_f = 0.0f;
		sum_o = 0.0f;
		for (i2=0; i2<_input_size; i2++)
		{
			sum_a += _Wa[0][i1][i2]*_x[0][i2];
			sum_i += _Wi[0][i1][i2]*_x[0][i2];
			sum_f += _Wf[0][i1][i2]*_x[0][i2];
			sum_o += _Wo[0][i1][i2]*_x[0][i2];
		}
		_a_gate[0][i1] = tanh(sum_a + _ba[0][i1]);
		_i_gate[0][i1] = sigmoid(sum_i + _bi[0][i1]);
		_f_gate[0][i1] = sigmoid(sum_f + _bf[0][i1]);
		_o_gate[0][i1] = sigmoid(sum_o + _bo[0][i1]);
		_state[0][i1] = _a_gate[0][i1]*_i_gate[0][i1];
		_out[0][i1] = tanh(_state[0][i1])*_o_gate[0][i1];
	}
	// --------------------------------
	// T = OTHERS TIMESTEPS 
	for (t=1; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_a = 0.0f;
			sum_i = 0.0f;
			sum_f = 0.0f;
			sum_o = 0.0f;
			for (i2=0; i2<_input_size; i2++)
			{
				sum_a += _Wa[t][i1][i2]*_x[t][i2];
				sum_i += _Wi[t][i1][i2]*_x[t][i2];
				sum_f += _Wf[t][i1][i2]*_x[t][i2];
				sum_o += _Wo[t][i1][i2]*_x[t][i2];
			}
			for (i2=0; i2<_output_size; i2++)
			{
				sum_a += _Ua[t-1][i1][i2]*_out[t-1][i2]; // t-1 because of BP from 0 to N-1!
				sum_i += _Ui[t-1][i1][i2]*_out[t-1][i2];
				sum_f += _Uf[t-1][i1][i2]*_out[t-1][i2];
				sum_o += _Uo[t-1][i1][i2]*_out[t-1][i2];
			}
			_a_gate[t][i1] = tanh(sum_a + _ba[t][i1]);
			_i_gate[t][i1] = sigmoid(sum_i + _bi[t][i1]);
			_f_gate[t][i1] = sigmoid(sum_f + _bf[t][i1]);
			_o_gate[t][i1] = sigmoid(sum_o + _bo[t][i1]);
			_state[t][i1] = _a_gate[t][i1]*_i_gate[t][i1] + _f_gate[t][i1]*_state[t-1][i1];
			_out[t][i1] = tanh(_state[t][i1])*_o_gate[t][i1];
		}
	}
}
//---------------------------------------------------------------------
void lstm::get_error(std::vector<std::vector<float>> target){
	uint32_t t, i1;
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
		}
	}
}

//---------------------------------------------------------------------
// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void lstm::backward_sgd(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2; //iterators
	float sum_da, sum_ba, sum_Wa, sum_Ua;
	float sum_di, sum_bi, sum_Wi, sum_Ui;
	float sum_df, sum_bf, sum_Wf, sum_Uf;
	float sum_do, sum_bo, sum_Wo, sum_Uo;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];
		_dstate[t][i2] = _dout[t][i2]*_o_gate[t][i2]*dtanh(_state[t][i2]);	
		_da_gate[t][i2] = _dstate[t][i2]*_i_gate[t][i2]*(1.0-(_a_gate[t][i2]*_a_gate[t][i2]));
		_di_gate[t][i2] = _dstate[t][i2]*_a_gate[t][i2]*_i_gate[t][i2]*(1.0-_i_gate[t][i2]);
		_df_gate[t][i2] = _dstate[t][i2]*_state[t-1][i2]*_f_gate[t][i2]*(1.0-_f_gate[t][i2]);
		_do_gate[t][i2] = _dout[t][i2]*tanh(_state[t][i2])*_o_gate[t][i2]*(1.0-_o_gate[t][i2]);
	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_da = 0.0f;
			sum_di = 0.0f;
			sum_df = 0.0f;
			sum_do = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_da += _Ua[t][i2][i1]*_da_gate[t+1][i2];
				sum_di += _Ui[t][i2][i1]*_di_gate[t+1][i2];
				sum_df += _Uf[t][i2][i1]*_df_gate[t+1][i2];
				sum_do += _Uo[t][i2][i1]*_do_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_da + sum_di + sum_df + sum_do;
			//_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];
			_dout[t][i1] = _Dout[t][i1];
			_dstate[t][i1] = _dout[t][i1]*_o_gate[t][i1]*dtanh(_state[t][i1]) + _dstate[t+1][i1]*_f_gate[t+1][i1];	
			_da_gate[t][i1] = _dstate[t][i1]*_i_gate[t][i1]*(1.0-(_a_gate[t][i1]*_a_gate[t][i1]));
			_di_gate[t][i1] = _dstate[t][i1]*_a_gate[t][i1]*_i_gate[t][i1]*(1.0-_i_gate[t][i1]);
			_df_gate[t][i1] = 0.0;
			if (t>0)
			{
				_df_gate[t][i1] = _dstate[t][i1]*_state[t-1][i1]*_f_gate[t][i1]*(1.0-_f_gate[t][i1]);
			}
			_do_gate[t][i1] = _dout[t][i1]*tanh(_state[t][i1])*_o_gate[t][i1]*(1.0-_o_gate[t][i1]);
		}
	}

	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wa = 0.0f;
			sum_Wi = 0.0f;
			sum_Wf = 0.0f;
			sum_Wo = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wa = _da_gate[t][i1]*_x[t][i2];
				sum_Wi = _di_gate[t][i1]*_x[t][i2];
				sum_Wf = _df_gate[t][i1]*_x[t][i2];
				sum_Wo = _do_gate[t][i1]*_x[t][i2];

				_dWa[t][i1][i2] = _lr * ((_mu-1)*sum_Wa + _mu * _dWa[t][i1][i2]);
				_dWi[t][i1][i2] = _lr * ((_mu-1)*sum_Wi + _mu * _dWi[t][i1][i2]);
				_dWf[t][i1][i2] = _lr * ((_mu-1)*sum_Wf + _mu * _dWf[t][i1][i2]);
				_dWo[t][i1][i2] = _lr * ((_mu-1)*sum_Wo + _mu * _dWo[t][i1][i2]);

				_Wf[t][i1][i2] = _Wf[t][i1][i2] - _dWf[t][i1][i2];
				_Wi[t][i1][i2] = _Wi[t][i1][i2] - _dWi[t][i1][i2];
				_Wa[t][i1][i2] = _Wa[t][i1][i2] - _dWa[t][i1][i2];
				_Wo[t][i1][i2] = _Wo[t][i1][i2] - _dWo[t][i1][i2];
				
			} 

		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Ua = 0.0f;
			sum_Ui = 0.0f;
			sum_Uf = 0.0f;
			sum_Uo = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Ua = _da_gate[t+1][i1]*_out[t][i2];
				sum_Ui = _di_gate[t+1][i1]*_out[t][i2];
				sum_Uf = _df_gate[t+1][i1]*_out[t][i2];
				sum_Uo = _do_gate[t+1][i1]*_out[t][i2];

				_dUa[t][i1][i2] = _lr * ((_mu-1)*sum_Ua + _mu * _dUa[t][i1][i2]);
				_dUi[t][i1][i2] = _lr * ((_mu-1)*sum_Ui + _mu * _dUi[t][i1][i2]);
				_dUf[t][i1][i2] = _lr * ((_mu-1)*sum_Uf + _mu * _dUf[t][i1][i2]);
				_dUo[t][i1][i2] = _lr * ((_mu-1)*sum_Uo + _mu * _dUo[t][i1][i2]);

				_Ua[t][i1][i2] = _Ua[t][i1][i2] - _dUa[t][i1][i2];
				_Ui[t][i1][i2] = _Ui[t][i1][i2] - _dUi[t][i1][i2];
				_Uf[t][i1][i2] = _Uf[t][i1][i2] - _dUf[t][i1][i2];
				_Uo[t][i1][i2] = _Uo[t][i1][i2] - _dUo[t][i1][i2];
			}
		}
		sum_ba = 0.0f;
		sum_bi = 0.0f;
		sum_bf = 0.0f;
		sum_bo = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_ba = _da_gate[t][i1];
			sum_bi = _di_gate[t][i1];
			sum_bf = _df_gate[t][i1];
			sum_bo = _do_gate[t][i1];

			_dba[t][i1] = _lr * ((_mu-1)*sum_ba + _mu * _dba[t][i1]);
			_dbi[t][i1] = _lr * ((_mu-1)*sum_bi + _mu * _dbi[t][i1]);
			_dbf[t][i1] = _lr * ((_mu-1)*sum_bf + _mu * _dbf[t][i1]);
			_dbo[t][i1] = _lr * ((_mu-1)*sum_bo + _mu * _dbo[t][i1]);

			_ba[t][i1] = _ba[t][i1] - _dba[t][i1];
			_bi[t][i1] = _bi[t][i1] - _dbi[t][i1];
			_bf[t][i1] = _bf[t][i1] - _dbf[t][i1];
			_bo[t][i1] = _bo[t][i1] - _dbo[t][i1];
		}
	}
}
//---------------------------------------------------------------------
void lstm::backward_adam(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2; //iterators
	float sum_da, sum_ba, sum_Wa, sum_Ua;
	float sum_di, sum_bi, sum_Wi, sum_Ui;
	float sum_df, sum_bf, sum_Wf, sum_Uf;
	float sum_do, sum_bo, sum_Wo, sum_Uo;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];
		_dstate[t][i2] = _dout[t][i2]*_o_gate[t][i2]*dtanh(_state[t][i2]);	
		_da_gate[t][i2] = _dstate[t][i2]*_i_gate[t][i2]*(1.0-(_a_gate[t][i2]*_a_gate[t][i2]));
		_di_gate[t][i2] = _dstate[t][i2]*_a_gate[t][i2]*_i_gate[t][i2]*(1.0-_i_gate[t][i2]);
		_df_gate[t][i2] = _dstate[t][i2]*_state[t-1][i2]*_f_gate[t][i2]*(1.0-_f_gate[t][i2]);
		_do_gate[t][i2] = _dout[t][i2]*tanh(_state[t][i2])*_o_gate[t][i2]*(1.0-_o_gate[t][i2]);
	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_da = 0.0f;
			sum_di = 0.0f;
			sum_df = 0.0f;
			sum_do = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_da += _Ua[t][i2][i1]*_da_gate[t+1][i2];
				sum_di += _Ui[t][i2][i1]*_di_gate[t+1][i2];
				sum_df += _Uf[t][i2][i1]*_df_gate[t+1][i2];
				sum_do += _Uo[t][i2][i1]*_do_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_da + sum_di + sum_df + sum_do;
			//_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];	
			_dout[t][i1] = _Dout[t][i1];	
			_dstate[t][i1] = _dout[t][i1]*_o_gate[t][i1]*dtanh(_state[t][i1]) + _dstate[t+1][i1]*_f_gate[t+1][i1];	
			_da_gate[t][i1] = _dstate[t][i1]*_i_gate[t][i1]*(1.0-(_a_gate[t][i1]*_a_gate[t][i1]));
			_di_gate[t][i1] = _dstate[t][i1]*_a_gate[t][i1]*_i_gate[t][i1]*(1.0-_i_gate[t][i1]);
			_df_gate[t][i1] = 0.0;
			if (t>0)
			{
				_df_gate[t][i1] = _dstate[t][i1]*_state[t-1][i1]*_f_gate[t][i1]*(1.0-_f_gate[t][i1]);
			}
			_do_gate[t][i1] = _dout[t][i1]*tanh(_state[t][i1])*_o_gate[t][i1]*(1.0-_o_gate[t][i1]);
		}
	}

	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wa = 0.0f;
			sum_Wi = 0.0f;
			sum_Wf = 0.0f;
			sum_Wo = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wa = _da_gate[t][i1]*_x[t][i2];
				sum_Wi = _di_gate[t][i1]*_x[t][i2];
				sum_Wf = _df_gate[t][i1]*_x[t][i2];
				sum_Wo = _do_gate[t][i1]*_x[t][i2];
				
				_m_Wa[t][i1][i2] = _b1 * _m_Wa[t][i1][i2] + (1 - _b1) * sum_Wa;
				_m_Wi[t][i1][i2] = _b1 * _m_Wi[t][i1][i2] + (1 - _b1) * sum_Wi;
				_m_Wf[t][i1][i2] = _b1 * _m_Wf[t][i1][i2] + (1 - _b1) * sum_Wf;
				_m_Wo[t][i1][i2] = _b1 * _m_Wo[t][i1][i2] + (1 - _b1) * sum_Wo;

				_v_Wa[t][i1][i2] = _b2 * _v_Wa[t][i1][i2] + (1 -_b2) * sum_Wa * sum_Wa;
				_v_Wi[t][i1][i2] = _b2 * _v_Wi[t][i1][i2] + (1 -_b2) * sum_Wi * sum_Wi;
				_v_Wf[t][i1][i2] = _b2 * _v_Wf[t][i1][i2] + (1 -_b2) * sum_Wf * sum_Wf;
				_v_Wo[t][i1][i2] = _b2 * _v_Wo[t][i1][i2] + (1 -_b2) * sum_Wo * sum_Wo;

				_dWa[t][i1][i2] = _lr * (_m_Wa[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wa[t][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWi[t][i1][i2] = _lr * (_m_Wi[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wi[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dWf[t][i1][i2] = _lr * (_m_Wf[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wf[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dWo[t][i1][i2] = _lr * (_m_Wo[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wo[t][i1][i2] / (1 - _b2)) + _epsilon);

				_Wa[t][i1][i2] = _Wa[t][i1][i2] + _dWa[t][i1][i2];
				_Wi[t][i1][i2] = _Wi[t][i1][i2] + _dWi[t][i1][i2];
				_Wf[t][i1][i2] = _Wf[t][i1][i2] + _dWf[t][i1][i2];
				_Wo[t][i1][i2] = _Wo[t][i1][i2] + _dWo[t][i1][i2];
				
			} 

		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Ua = 0.0f;
			sum_Ui = 0.0f;
			sum_Uf = 0.0f;
			sum_Uo = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Ua = _da_gate[t+1][i1]*_out[t][i2];
				sum_Ui = _di_gate[t+1][i1]*_out[t][i2];
				sum_Uf = _df_gate[t+1][i1]*_out[t][i2];
				sum_Uo = _do_gate[t+1][i1]*_out[t][i2];
				
				_m_Ua[t][i1][i2] = _b1 * _m_Ua[t][i1][i2] + (1 - _b1) * sum_Ua;
				_m_Ui[t][i1][i2] = _b1 * _m_Ui[t][i1][i2] + (1 - _b1) * sum_Ui;
				_m_Uf[t][i1][i2] = _b1 * _m_Uf[t][i1][i2] + (1 - _b1) * sum_Uf;
				_m_Uo[t][i1][i2] = _b1 * _m_Uo[t][i1][i2] + (1 - _b1) * sum_Uo;

				_v_Ua[t][i1][i2] = _b2 * _v_Ua[t][i1][i2] + (1 -_b2) * sum_Ua * sum_Ua;
				_v_Ui[t][i1][i2] = _b2 * _v_Ui[t][i1][i2] + (1 -_b2) * sum_Ui * sum_Ui;
				_v_Uf[t][i1][i2] = _b2 * _v_Uf[t][i1][i2] + (1 -_b2) * sum_Uf * sum_Uf;
				_v_Uo[t][i1][i2] = _b2 * _v_Uo[t][i1][i2] + (1 -_b2) * sum_Uo * sum_Uo;

				_dUa[t][i1][i2] = _lr * (_m_Ua[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Ua[t][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dUi[t][i1][i2] = _lr * (_m_Ui[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Ui[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dUf[t][i1][i2] = _lr * (_m_Uf[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Uf[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dUo[t][i1][i2] = _lr * (_m_Uo[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Uo[t][i1][i2] / (1 - _b2)) + _epsilon);

				_Ua[t][i1][i2] = _Ua[t][i1][i2] + _dUa[t][i1][i2];
				_Ui[t][i1][i2] = _Ui[t][i1][i2] + _dUi[t][i1][i2];
				_Uf[t][i1][i2] = _Uf[t][i1][i2] + _dUf[t][i1][i2];
				_Uo[t][i1][i2] = _Uo[t][i1][i2] + _dUo[t][i1][i2];
			}
		}
		sum_ba = 0.0f;
		sum_bi = 0.0f;
		sum_bf = 0.0f;
		sum_bo = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_ba = _da_gate[t][i1];
			sum_bi = _di_gate[t][i1];
			sum_bf = _df_gate[t][i1];
			sum_bo = _do_gate[t][i1];
			
			_m_ba[t][i1] = _b1 * _m_ba[t][i1] + (1 - _b1) * sum_ba;
			_m_bi[t][i1] = _b1 * _m_bi[t][i1] + (1 - _b1) * sum_bi;
			_m_bf[t][i1] = _b1 * _m_bf[t][i1] + (1 - _b1) * sum_bf;
			_m_bo[t][i1] = _b1 * _m_bo[t][i1] + (1 - _b1) * sum_bo;

			_v_ba[t][i1] = _b2 * _v_ba[t][i1] + (1 -_b2) * sum_ba * sum_ba;
			_v_bi[t][i1] = _b2 * _v_bi[t][i1] + (1 -_b2) * sum_bi * sum_bi;
			_v_bf[t][i1] = _b2 * _v_bf[t][i1] + (1 -_b2) * sum_bf * sum_bf;
			_v_bo[t][i1] = _b2 * _v_bo[t][i1] + (1 -_b2) * sum_bo * sum_bo;

			_dba[t][i1] = _lr * (_m_ba[t][i1] / (1 - _b1)) / (sqrt(_v_ba[t][i1] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbi[t][i1] = _lr * (_m_bi[t][i1] / (1 - _b1)) / (sqrt(_v_bi[t][i1] / (1 - _b2)) + _epsilon);
			_dbf[t][i1] = _lr * (_m_bf[t][i1] / (1 - _b1)) / (sqrt(_v_bf[t][i1] / (1 - _b2)) + _epsilon);
			_dbo[t][i1] = _lr * (_m_bo[t][i1] / (1 - _b1)) / (sqrt(_v_bo[t][i1] / (1 - _b2)) + _epsilon);

			_ba[t][i1] = _ba[t][i1] + _dba[t][i1];
			_bi[t][i1] = _bi[t][i1] + _dbi[t][i1];
			_bf[t][i1] = _bf[t][i1] + _dbf[t][i1];
			_bo[t][i1] = _bo[t][i1] + _dbo[t][i1];
		}
	}
}
//---------------------------------------------------------------------
void lstm::backward_adamax(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2; //iterators
	float sum_da, sum_ba, sum_Wa, sum_Ua;
	float sum_di, sum_bi, sum_Wi, sum_Ui;
	float sum_df, sum_bf, sum_Wf, sum_Uf;
	float sum_do, sum_bo, sum_Wo, sum_Uo;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];
		_dstate[t][i2] = _dout[t][i2]*_o_gate[t][i2]*dtanh(_state[t][i2]);	
		_da_gate[t][i2] = _dstate[t][i2]*_i_gate[t][i2]*(1.0-(_a_gate[t][i2]*_a_gate[t][i2]));
		_di_gate[t][i2] = _dstate[t][i2]*_a_gate[t][i2]*_i_gate[t][i2]*(1.0-_i_gate[t][i2]);
		_df_gate[t][i2] = _dstate[t][i2]*_state[t-1][i2]*_f_gate[t][i2]*(1.0-_f_gate[t][i2]);
		_do_gate[t][i2] = _dout[t][i2]*tanh(_state[t][i2])*_o_gate[t][i2]*(1.0-_o_gate[t][i2]);
	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_da = 0.0f;
			sum_di = 0.0f;
			sum_df = 0.0f;
			sum_do = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_da += _Ua[t][i2][i1]*_da_gate[t+1][i2];
				sum_di += _Ui[t][i2][i1]*_di_gate[t+1][i2];
				sum_df += _Uf[t][i2][i1]*_df_gate[t+1][i2];
				sum_do += _Uo[t][i2][i1]*_do_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_da + sum_di + sum_df + sum_do;
			//_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];	
			_dout[t][i1] = _Dout[t][i1];	
			_dstate[t][i1] = _dout[t][i1]*_o_gate[t][i1]*dtanh(_state[t][i1]) + _dstate[t+1][i1]*_f_gate[t+1][i1];	
			_da_gate[t][i1] = _dstate[t][i1]*_i_gate[t][i1]*(1.0-(_a_gate[t][i1]*_a_gate[t][i1]));
			_di_gate[t][i1] = _dstate[t][i1]*_a_gate[t][i1]*_i_gate[t][i1]*(1.0-_i_gate[t][i1]);
			_df_gate[t][i1] = 0.0;
			//_df_gate[t][i1] = _dstate[t][i1]*_f_gate[t][i1]*(1.0-_f_gate[t][i1]);
			if (t>0)
			{
				_df_gate[t][i1] *= _state[t-1][i1] * _dstate[t][i1]*_f_gate[t][i1]*(1.0-_f_gate[t][i1]);
			}
			_do_gate[t][i1] = _dout[t][i1]*tanh(_state[t][i1])*_o_gate[t][i1]*(1.0-_o_gate[t][i1]);
		}
	}

	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wa = 0.0f;
			sum_Wi = 0.0f;
			sum_Wf = 0.0f;
			sum_Wo = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wa = _da_gate[t][i1]*_x[t][i2];
				sum_Wi = _di_gate[t][i1]*_x[t][i2];
				sum_Wf = _df_gate[t][i1]*_x[t][i2];
				sum_Wo = _do_gate[t][i1]*_x[t][i2];
				
				_m_Wa[t][i1][i2] = _b1 * _m_Wa[t][i1][i2] + (1 - _b1) * sum_Wa;
				_m_Wi[t][i1][i2] = _b1 * _m_Wi[t][i1][i2] + (1 - _b1) * sum_Wi;
				_m_Wf[t][i1][i2] = _b1 * _m_Wf[t][i1][i2] + (1 - _b1) * sum_Wf;
				_m_Wo[t][i1][i2] = _b1 * _m_Wo[t][i1][i2] + (1 - _b1) * sum_Wo;

				_v_Wa[t][i1][i2] = std::max(_v_Wa[t][i1][i2]*_b2, std::abs(sum_Wa));
				_v_Wi[t][i1][i2] = std::max(_v_Wi[t][i1][i2]*_b2, std::abs(sum_Wi));
				_v_Wf[t][i1][i2] = std::max(_v_Wf[t][i1][i2]*_b2, std::abs(sum_Wf));
				_v_Wo[t][i1][i2] = std::max(_v_Wo[t][i1][i2]*_b2, std::abs(sum_Wo));

				_dWa[t][i1][i2] = _lr * (_m_Wa[t][i1][i2] / (1 - _b1)) / (_v_Wa[t][i1][i2] + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWi[t][i1][i2] = _lr * (_m_Wi[t][i1][i2] / (1 - _b1)) / (_v_Wi[t][i1][i2] + _epsilon);
				_dWf[t][i1][i2] = _lr * (_m_Wf[t][i1][i2] / (1 - _b1)) / (_v_Wf[t][i1][i2] + _epsilon);
				_dWo[t][i1][i2] = _lr * (_m_Wo[t][i1][i2] / (1 - _b1)) / (_v_Wo[t][i1][i2] + _epsilon);

				_Wa[t][i1][i2] = _Wa[t][i1][i2] + _dWa[t][i1][i2];
				_Wi[t][i1][i2] = _Wi[t][i1][i2] + _dWi[t][i1][i2];
				_Wf[t][i1][i2] = _Wf[t][i1][i2] + _dWf[t][i1][i2];
				_Wo[t][i1][i2] = _Wo[t][i1][i2] + _dWo[t][i1][i2];
				
			} 

		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Ua = 0.0f;
			sum_Ui = 0.0f;
			sum_Uf = 0.0f;
			sum_Uo = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Ua = _da_gate[t+1][i1]*_out[t][i2];
				sum_Ui = _di_gate[t+1][i1]*_out[t][i2];
				sum_Uf = _df_gate[t+1][i1]*_out[t][i2];
				sum_Uo = _do_gate[t+1][i1]*_out[t][i2];
				
				_m_Ua[t][i1][i2] = _b1 * _m_Ua[t][i1][i2] + (1 - _b1) * sum_Ua;
				_m_Ui[t][i1][i2] = _b1 * _m_Ui[t][i1][i2] + (1 - _b1) * sum_Ui;
				_m_Uf[t][i1][i2] = _b1 * _m_Uf[t][i1][i2] + (1 - _b1) * sum_Uf;
				_m_Uo[t][i1][i2] = _b1 * _m_Uo[t][i1][i2] + (1 - _b1) * sum_Uo;

				_v_Ua[t][i1][i2] = std::max(_v_Ua[t][i1][i2]*_b2, std::abs(sum_Ua));
				_v_Ui[t][i1][i2] = std::max(_v_Ui[t][i1][i2]*_b2, std::abs(sum_Ui));
				_v_Uf[t][i1][i2] = std::max(_v_Uf[t][i1][i2]*_b2, std::abs(sum_Uf));
				_v_Uo[t][i1][i2] = std::max(_v_Uo[t][i1][i2]*_b2, std::abs(sum_Uo));

				_dUa[t][i1][i2] = _lr * (_m_Ua[t][i1][i2] / (1 - _b1)) / (_v_Ua[t][i1][i2] + _epsilon);
				_dUi[t][i1][i2] = _lr * (_m_Ui[t][i1][i2] / (1 - _b1)) / (_v_Ui[t][i1][i2] + _epsilon);
				_dUf[t][i1][i2] = _lr * (_m_Uf[t][i1][i2] / (1 - _b1)) / (_v_Uf[t][i1][i2] + _epsilon);
				_dUo[t][i1][i2] = _lr * (_m_Uo[t][i1][i2] / (1 - _b1)) / (_v_Uo[t][i1][i2] + _epsilon);

				_Ua[t][i1][i2] = _Ua[t][i1][i2] + _dUa[t][i1][i2];
				_Ui[t][i1][i2] = _Ui[t][i1][i2] + _dUi[t][i1][i2];
				_Uf[t][i1][i2] = _Uf[t][i1][i2] + _dUf[t][i1][i2];
				_Uo[t][i1][i2] = _Uo[t][i1][i2] + _dUo[t][i1][i2];
			}
		}
		sum_ba = 0.0f;
		sum_bi = 0.0f;
		sum_bf = 0.0f;
		sum_bo = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_ba = _da_gate[t][i1];
			sum_bi = _di_gate[t][i1];
			sum_bf = _df_gate[t][i1];
			sum_bo = _do_gate[t][i1];
			
			_m_ba[t][i1] = _b1 * _m_ba[t][i1] + (1 - _b1) * sum_ba;
			_m_bi[t][i1] = _b1 * _m_bi[t][i1] + (1 - _b1) * sum_bi;
			_m_bf[t][i1] = _b1 * _m_bf[t][i1] + (1 - _b1) * sum_bf;
			_m_bo[t][i1] = _b1 * _m_bo[t][i1] + (1 - _b1) * sum_bo;

			_v_ba[t][i1] = std::max(_v_ba[t][i1]*_b2, std::abs(sum_ba));
			_v_bi[t][i1] = std::max(_v_bi[t][i1]*_b2, std::abs(sum_bi));
			_v_bf[t][i1] = std::max(_v_bf[t][i1]*_b2, std::abs(sum_bf));
			_v_bo[t][i1] = std::max(_v_bo[t][i1]*_b2, std::abs(sum_bo));

			_dba[t][i1] = _lr * (_m_ba[t][i1] / (1 - _b1)) / (_v_ba[t][i1] + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbi[t][i1] = _lr * (_m_bi[t][i1] / (1 - _b1)) / (_v_bi[t][i1] + _epsilon);
			_dbf[t][i1] = _lr * (_m_bf[t][i1] / (1 - _b1)) / (_v_bf[t][i1] + _epsilon);
			_dbo[t][i1] = _lr * (_m_bo[t][i1] / (1 - _b1)) / (_v_bo[t][i1] + _epsilon);

			_ba[t][i1] = _ba[t][i1] + _dba[t][i1];
			_bi[t][i1] = _bi[t][i1] + _dbi[t][i1];
			_bf[t][i1] = _bf[t][i1] + _dbf[t][i1];
			_bo[t][i1] = _bo[t][i1] + _dbo[t][i1];
		}
	}
}

//---------------------------------------------------------------------
// generic backward
void lstm::backward(std::vector<std::vector<float>> target)
{
	if(_optim_id = 1)
	{
		backward_sgd(target);
	}
	else if(_optim_id = 2)
	{
		backward_adam(target);
	}
	else if(_optim_id = 3)
	{
		backward_adamax(target);
	}
}

//---------------------------------------------------------------------
// 
float lstm::get_mse()
{
	uint32_t i1;
	float mse_tmp = 0.0;
	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += _errors[_nb_timestep-1][i1]*_errors[_nb_timestep-1][i1];
	}
	return mse_tmp/_output_size;
}

float lstm::get_mse(std::vector<std::vector<float>> target, std::vector<std::vector<float>> output)
{
	uint32_t i1;
	float mse_tmp = 0.0;
	uint32_t last = target.size()-1;
	std::vector<float> error=target[last];


	for (i1=0; i1<_output_size; i1++)
	{
		error[i1] -= output[last][i1];
	}

	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += error[i1]*error[i1];
	}
	return mse_tmp/_output_size;
}

//---------------------------------------------------------------------
void lstm::clear_grads()
{
	uint32_t t, i1, i2; //iterators
	for (t=0; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_input_size; i2++)
			{
				_dWa[t][i1][i2] = 0.0;
				_dWi[t][i1][i2] = 0.0;
				_dWf[t][i1][i2] = 0.0;
				_dWo[t][i1][i2] = 0.0;
			}
		}
		
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_output_size; i2++)
			{
				_dUa[t][i1][i2] = 0.0;
				_dUi[t][i1][i2] = 0.0;
				_dUf[t][i1][i2] = 0.0;
				_dUo[t][i1][i2] = 0.0;
			}
		}
		for (i1=0; i1<_output_size; i1++)
		{
			_dba[t][i1] = 0.0;
			_dbi[t][i1] = 0.0;
			_dbf[t][i1] = 0.0;
			_dbo[t][i1] = 0.0;
		}
	}
}
//----------------------------------------------------------------------

void lstm::clear_momentums()
{
	uint32_t t, i1, i2; //iterators
	for (t=0; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_input_size; i2++)
			{
				_m_Wa[t][i1][i2] = 0.0;
				_m_Wi[t][i1][i2] = 0.0;
				_m_Wf[t][i1][i2] = 0.0;
				_m_Wo[t][i1][i2] = 0.0;
				_v_Wa[t][i1][i2] = 0.0;
				_v_Wi[t][i1][i2] = 0.0;
				_v_Wf[t][i1][i2] = 0.0;
				_v_Wo[t][i1][i2] = 0.0;
			}
		}
		
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_output_size; i2++)
			{
				_m_Ua[t][i1][i2] = 0.0;
				_m_Ui[t][i1][i2] = 0.0;
				_m_Uf[t][i1][i2] = 0.0;
				_m_Uo[t][i1][i2] = 0.0;
				_v_Ua[t][i1][i2] = 0.0;
				_v_Ui[t][i1][i2] = 0.0;
				_v_Uf[t][i1][i2] = 0.0;
				_v_Uo[t][i1][i2] = 0.0;
			}
		}
		for (i1=0; i1<_output_size; i1++)
		{
			_m_ba[t][i1] = 0.0;
			_m_bi[t][i1] = 0.0;
			_m_bf[t][i1] = 0.0;
			_m_bo[t][i1] = 0.0;
			_v_ba[t][i1] = 0.0;
			_v_bi[t][i1] = 0.0;
			_v_bf[t][i1] = 0.0;
			_v_bo[t][i1] = 0.0;
		}
	}
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------

void lstm::save(const char* filename)
{
	std::ofstream reportfile;
	
	reportfile.open(filename);

	// inputs
	std::string x;
	std::string dx;
	
	std::string errors;
	
	// gates
	std::string a_gate;
	std::string i_gate;
	std::string f_gate;
	std::string o_gate;
	std::string da_gate;
	std::string di_gate;
	std::string df_gate;
	std::string do_gate;
	
	std::string state;
	std::string dstate;
	
	std::string out;
	std::string dout;
	std::string Dout;

	std::string ba;
	std::string bi;
	std::string bf;
	std::string bo;
	std::string dba;
	std::string dbi;
	std::string dbf;
	std::string dbo;
	std::string Wa;
	std::string Wi;
	std::string Wf;
	std::string Wo;
	std::string Ua;
	std::string Ui;
	std::string Uf;
	std::string Uo;
	std::string dWa;
	std::string dWi;
	std::string dWf;
	std::string dWo;
	std::string dUa;
	std::string dUi;
	std::string dUf;
	std::string dUo;


	uint32_t t, i1, i2; //iterators

	for (i1 = 0; i1 < _input_size; i1++){x+=("x[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _input_size; i1++){dx+=("dx[" + std::to_string(i1) + "], ");}
	
	for (i1 = 0; i1 < _output_size; i1++){a_gate +=("a_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){i_gate +=("i_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){f_gate +=("f_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){o_gate +=("o_gate[" + std::to_string(i1) + "], ");}

	for (i1 = 0; i1 < _output_size; i1++){ state  +=("state[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ out    +=("out[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ da_gate+=("da_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ di_gate+=("di_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ df_gate+=("df_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ do_gate+=("do_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dstate +=("dstate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dout   +=("dout[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ Dout   +=("Dout[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ errors +=("errors[" + std::to_string(i1) + "], ");}
	
	// Bias
	for (i1 = 0; i1 < _output_size; i1++){ ba+=("ba[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ bi+=("bi[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ bf+=("bf[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ bo+=("bo[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dba+=("dba[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbi+=("dbi[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbf+=("dbf[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbo+=("dbo[" + std::to_string(i1) + "], ");}

	// W weights
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wa+=("Wa[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wi+=("Wi[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wf+=("Wf[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wo+=("Wo[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWa+=("dWa[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWi+=("dWi[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWf+=("dWf[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWo+=("dWo[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	
	
	// U weights
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ua+=("Ua[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ui+=("Ui[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uf+=("Uf[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uo+=("Uo[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUa+=("dUa[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUi+=("dUi[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUf+=("dUf[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUo+=("dUo[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}

	reportfile << 
	"#Neuron, " <<
	x << dx <<
	a_gate << i_gate << f_gate << o_gate << 
	state << out << 
	da_gate << di_gate << df_gate << do_gate <<
	dstate << dout << Dout << errors <<
	ba << bi << bf << bo << 
	dba << dbi << dbf << dbo << 
	Wa << Wi << Wf << Wo << 
	dWa << dWi << dWf << dWo << 
	Ua << Ui <<	Uf << Uo << 
	dUa << dUi << dUf << dUo <<
	std::endl;

	for (t = 0; t < _nb_timestep; t++)
	{
		x.clear(); dx.clear(); errors.clear();
		a_gate.clear(); i_gate.clear(); f_gate.clear(); o_gate.clear(); 
		da_gate.clear(); di_gate.clear(); df_gate.clear(); do_gate.clear();
		state.clear(); dstate.clear(); out.clear(); dout.clear(); Dout.clear(); 
		ba.clear(); bi.clear(); bf.clear(); bo.clear(); 
		dba.clear(); dbi.clear(); dbf.clear(); dbo.clear(); 
		Wa.clear(); Wi.clear(); Wf.clear(); Wo.clear(); 
		Ua.clear(); Ui.clear();	Uf.clear(); Uo.clear(); 
		dWa.clear(); dWi.clear(); dWf.clear(); dWo.clear(); 
		dUa.clear(); dUi.clear(); dUf.clear(); dUo.clear();
 
		for (i1 = 0; i1 < _input_size; i1++){x+=(std::to_string(_x[t][i1]) + ", ");}
		for (i1 = 0; i1 < _input_size; i1++){dx+=(std::to_string( _dx[t][i1]) + ", ");}
		
		for (i1 = 0; i1 < _output_size; i1++){a_gate +=(std::to_string( _a_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){i_gate +=(std::to_string( _i_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){f_gate +=(std::to_string( _f_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){o_gate +=(std::to_string( _o_gate[t][i1]) + ", ");}

		for (i1 = 0; i1 < _output_size; i1++){ state  +=(std::to_string( _state[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ out    +=(std::to_string( _out[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ da_gate+=(std::to_string( _da_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ di_gate+=(std::to_string( _di_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ df_gate+=(std::to_string( _df_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ do_gate+=(std::to_string( _do_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dstate +=(std::to_string( _dstate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dout   +=(std::to_string( _dout[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ Dout   +=(std::to_string( _Dout[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ errors +=(std::to_string( _errors[t][i1]) + ", ");}
		
		// Bias
		for (i1 = 0; i1 < _output_size; i1++){ ba+=(std::to_string( _ba[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ bi+=(std::to_string( _bi[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ bf+=(std::to_string( _bf[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ bo+=(std::to_string( _bo[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dba+=(std::to_string( _dba[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbi+=(std::to_string( _dbi[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbf+=(std::to_string( _dbf[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbo+=(std::to_string( _dbo[t][i1]) + ", ");}

		// W weights
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wa+=(std::to_string( _Wa[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wi+=(std::to_string( _Wi[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wf+=(std::to_string( _Wf[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wo+=(std::to_string( _Wo[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWa+=(std::to_string( _dWa[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWi+=(std::to_string( _dWi[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWf+=(std::to_string( _dWf[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWo+=(std::to_string( _dWo[t][i1][i2]) + ", ");}}
		
		
		// U weights
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ua+=(std::to_string( _Ua[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ui+=(std::to_string( _Ui[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uf+=(std::to_string( _Uf[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uo+=(std::to_string( _Uo[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUa+=(std::to_string( _dUa[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUi+=(std::to_string( _dUi[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUf+=(std::to_string( _dUf[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUo+=(std::to_string( _dUo[t][i1][i2]) + ", ");}}

		reportfile << 
		std::to_string(t) << ", " << 
		x << dx <<
		a_gate << i_gate << f_gate << o_gate << 
		state << out << 
		da_gate << di_gate << df_gate << do_gate <<
		dstate << dout << Dout << errors <<
		ba << bi << bf << bo << 
		dba << dbi << dbf << dbo << 
		Wa << Wi << Wf << Wo << 
		dWa << dWi << dWf << dWo << 
		Ua << Ui <<	Uf << Uo << 
		dUa << dUi << dUf << dUo <<
		std::endl;
	}
	reportfile.close();
}
//----------------------------------------------------------------------------------
inline int lstm::nbLines(std::string filename)
{
    // Create an input filestream
    std::ifstream inFile(filename);

    int nol = 0; //number of lines
    std::string l; //line

    // Make sure the file is open
    //if(!inFile.is_open()) throw std::runtime_error("Could not open file");

    while (std::getline(inFile, l)){
        ++nol;
    }
    inFile.close();
    return nol;
}


void lstm::load(const char* lstmlogfilename, uint32_t input_size, uint32_t output_size)
{
	int nb_timesteps = nbLines(lstmlogfilename) - 1;
	int i1,i2;
	int t = 0;

	this->init(nb_timesteps, input_size, output_size);
	// stream for the log
    std::ifstream lstmlogfile; 
    // string to store each file line
    std::string line; 
    std::string delim;
	std::string number;
	lstmlogfile.open(lstmlogfilename);

	if (!lstmlogfile.is_open()) 
    {
        std::cerr << "There was a problem opening the LSTM LOG input file " << lstmlogfilename << std::endl;
        exit(1);
    }
    else
    {
		std::getline(lstmlogfile, line); // read the first line only for column title
	}
    while(std::getline(lstmlogfile, line))
	{
		std::stringstream lineStream(line);
		lineStream >> number;
		for (i1 = 0; i1 < _input_size; i1++){lineStream >>  _x [ t ] [ i1 ];lineStream >> delim;}
		for (i1 = 0; i1 < _input_size; i1++){lineStream >>  _dx [ t ] [ i1 ];  lineStream >> delim;}
		
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _a_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _i_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _f_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _o_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _state [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _out [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _da_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _di_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _df_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _do_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dstate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dout [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _Dout [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _errors [ t ] [ i1 ];  lineStream >> delim;}
		
		// Bias
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _ba [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _bi [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _bf [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _bo [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dba [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbi [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbf [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbo [ t ] [ i1 ];  lineStream >> delim;}

		// W weights
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wa [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wi [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wf [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wo [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWa [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWi [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWf [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWo [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		
		
		// U weights
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Ua [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Ui [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Uf [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Uo [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUa [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUi [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUf [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUo [ t ] [ i1 ]  [ i1 ];}}

		t++;
	}
	lstmlogfile.close();
}

//---------------------------------------------------------------------
float lstm::sigmoid(float x)
{
	return (1.0 / (1.0 + exp(-x)));
}

//---------------------------------------------------------------------
float lstm::dsigmoid(float x)
{
	return (x*(1.0 - x));
}

//---------------------------------------------------------------------
float lstm::dtanh(float x)
{
	float y = tanh(x);
	return (1.0 - y*y);
}

//---------------------------------------------------------------------
float lstm::leakyrelu(float x) 
{
    return (x < 0.0 ? (0.001 * x) : x);
}

//---------------------------------------------------------------------
float lstm::dleakyrelu(float x) 
{
   return (x < 0.0 ? 0.001 : x);
}

//---------------------------------------------------------------------
float lstm::softplus(float x)
{
    return log(1.0+(exp(x)));
}

//---------------------------------------------------------------------
float lstm::dsoftplus(float x)
{
    return 1.0/(1.0+(exp(-x)));
}

//---------------------------------------------------------------------
float lstm::randomize(float min, float max){
    float f = ((float) rand()) / RAND_MAX;
    return min + f * (max - min);
}
