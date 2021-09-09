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

#include <gru.hh>

//---------------------------------------------------------------------
gru::gru()
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
gru::~gru()
{
	
}

//---------------------------------------------------------------------
void gru::set_lr_mu(float lr, float mu)
{
	_lr = lr;
	_mu = mu;
}
//---------------------------------------------------------------------
void gru::set_lr_b1_b2_epsilon(float lr, float b1, float b2, float epsilon)
{
	_lr = lr;
	_b1 = b1;
	_b2 = b2;
	_epsilon = epsilon;
}

//---------------------------------------------------------------------
void gru::set_optimizer(uint8_t optim_id)
{
	_optim_id = optim_id;
}

//---------------------------------------------------------------------
void gru::init(uint32_t nb_timestep, uint32_t input_size, uint32_t output_size)
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
		
		std::vector<float> z_tmp;
		std::vector<float> r_tmp;
		std::vector<float> h_tmp;
		
		std::vector<float> out_tmp;
		
		std::vector<float> dz_tmp;
		std::vector<float> dr_tmp;
		std::vector<float> dh_tmp;
		
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
			z_tmp.push_back(val_tmp); 
			r_tmp.push_back(val_tmp); 
			h_tmp.push_back(val_tmp); 
			out_tmp.push_back(val_tmp);
			 
			dz_tmp.push_back(val_tmp);  
			dr_tmp.push_back(val_tmp); 
			dh_tmp.push_back(val_tmp); 
			dout_tmp.push_back(val_tmp); 
			Dout_tmp.push_back(val_tmp); 
			errors_tmp.push_back(val_tmp); 
		}
		_z_gate.push_back(z_tmp); 
		_r_gate.push_back(r_tmp);  
		_h_gate.push_back(h_tmp); 
		_out.push_back(out_tmp); 
		
		_dz_gate.push_back(dz_tmp);  
		_dr_gate.push_back(dr_tmp); 
		_dh_gate.push_back(dh_tmp); 
		_dout.push_back(dout_tmp); 
		_Dout.push_back(Dout_tmp); 
		_errors.push_back(errors_tmp); 
	}
	// Bias
	// Bias
	val_tmp = 0.0;
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<float> bz_tmp;
		std::vector<float> br_tmp;
		std::vector<float> bh_tmp;
		std::vector<float> dbz_tmp;
		std::vector<float> dbr_tmp;
		std::vector<float> dbh_tmp;
		std::vector<float> m_bz_tmp;
		std::vector<float> m_br_tmp;
		std::vector<float> m_bh_tmp;
		std::vector<float> v_bz_tmp;
		std::vector<float> v_br_tmp;
		std::vector<float> v_bh_tmp;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			val_tmp = randomize(-1.0, 1.0);
			bz_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0);
			br_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0); 
			bh_tmp.push_back(val_tmp);
			val_tmp = randomize(-1.0, 1.0);
			dbz_tmp.push_back(0.0); 
			dbr_tmp.push_back(0.0); 
			dbh_tmp.push_back(0.0);
			m_bz_tmp.push_back(0.0);
			m_br_tmp.push_back(0.0);
			m_bh_tmp.push_back(0.0);
			v_bz_tmp.push_back(0.0);
			v_br_tmp.push_back(0.0);
			v_bh_tmp.push_back(0.0);
		}
		_bz.push_back(bz_tmp); 
		_br.push_back(br_tmp); 
		_bh.push_back(bh_tmp);
		_dbz.push_back(dbz_tmp); 
		_dbr.push_back(dbr_tmp); 
		_dbh.push_back(dbh_tmp);
		
		_m_bz.push_back(m_bz_tmp);
		_m_br.push_back(m_br_tmp);
		_m_bh.push_back(m_bh_tmp);
		_v_bz.push_back(v_bz_tmp);
		_v_br.push_back(v_br_tmp);
		_v_bh.push_back(v_bh_tmp);
	}

	// W weights
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<std::vector<float>> Wz_tmp_i1;
		std::vector<std::vector<float>> Wr_tmp_i1;
		std::vector<std::vector<float>> Wh_tmp_i1;
		std::vector<std::vector<float>> Wo_tmp_i1;
		
		std::vector<std::vector<float>> dWz_tmp_i1;
		std::vector<std::vector<float>> dWr_tmp_i1;
		std::vector<std::vector<float>> dWh_tmp_i1;
		std::vector<std::vector<float>> dWo_tmp_i1;

		std::vector<std::vector<float>> m_Wz_tmp_i1;
		std::vector<std::vector<float>> m_Wr_tmp_i1;
		std::vector<std::vector<float>> m_Wh_tmp_i1;
		std::vector<std::vector<float>> m_Wo_tmp_i1;
		std::vector<std::vector<float>> v_Wz_tmp_i1;
		std::vector<std::vector<float>> v_Wr_tmp_i1;
		std::vector<std::vector<float>> v_Wh_tmp_i1;
		std::vector<std::vector<float>> v_Wo_tmp_i1;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			std::vector<float> Wz_tmp_i2;
			std::vector<float> Wr_tmp_i2;
			std::vector<float> Wh_tmp_i2;
			std::vector<float> Wo_tmp_i2;
			std::vector<float> dWz_tmp_i2;
			std::vector<float> dWr_tmp_i2;
			std::vector<float> dWh_tmp_i2;
			std::vector<float> dWo_tmp_i2;
			std::vector<float> m_Wz_tmp_i2;
			std::vector<float> m_Wr_tmp_i2;
			std::vector<float> m_Wh_tmp_i2;
			std::vector<float> m_Wo_tmp_i2;
			std::vector<float> v_Wz_tmp_i2;
			std::vector<float> v_Wr_tmp_i2;
			std::vector<float> v_Wh_tmp_i2;
			std::vector<float> v_Wo_tmp_i2;
			for (i2 = 0; i2 < _input_size; i2++)
			{
				val_tmp = randomize(-1.0, 1.0);
				Wz_tmp_i2.push_back(val_tmp);
				dWz_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wr_tmp_i2.push_back(val_tmp);
				dWr_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wh_tmp_i2.push_back(val_tmp);
				dWh_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Wo_tmp_i2.push_back(val_tmp);
				dWo_tmp_i2.push_back(0.0);

				m_Wz_tmp_i2.push_back(0.0);
				m_Wr_tmp_i2.push_back(0.0);
				m_Wh_tmp_i2.push_back(0.0);
				m_Wo_tmp_i2.push_back(0.0);
				v_Wz_tmp_i2.push_back(0.0);
				v_Wr_tmp_i2.push_back(0.0);
				v_Wh_tmp_i2.push_back(0.0);
				v_Wo_tmp_i2.push_back(0.0);
			}
			Wz_tmp_i1.push_back(Wz_tmp_i2);
			Wr_tmp_i1.push_back(Wr_tmp_i2);
			Wh_tmp_i1.push_back(Wh_tmp_i2);
			Wo_tmp_i1.push_back(Wo_tmp_i2);
			dWz_tmp_i1.push_back(dWz_tmp_i2);
			dWr_tmp_i1.push_back(dWr_tmp_i2);
			dWh_tmp_i1.push_back(dWh_tmp_i2);
			dWo_tmp_i1.push_back(dWo_tmp_i2);
			m_Wz_tmp_i1.push_back(m_Wz_tmp_i2);
			m_Wr_tmp_i1.push_back(m_Wr_tmp_i2);
			m_Wh_tmp_i1.push_back(m_Wh_tmp_i2);
			m_Wo_tmp_i1.push_back(m_Wo_tmp_i2);
			v_Wz_tmp_i1.push_back(v_Wz_tmp_i2);
			v_Wr_tmp_i1.push_back(v_Wr_tmp_i2);
			v_Wh_tmp_i1.push_back(v_Wh_tmp_i2);
			v_Wo_tmp_i1.push_back(v_Wo_tmp_i2);
			
		}
		_Wz.push_back(Wz_tmp_i1);
		_Wr.push_back(Wr_tmp_i1);
		_Wh.push_back(Wh_tmp_i1);
		_dWz.push_back(dWz_tmp_i1);
		_dWr.push_back(dWr_tmp_i1);
		_dWh.push_back(dWh_tmp_i1);

		_m_Wz.push_back(m_Wz_tmp_i1);
		_m_Wr.push_back(m_Wr_tmp_i1);
		_m_Wh.push_back(m_Wh_tmp_i1);
		_v_Wz.push_back(v_Wz_tmp_i1);
		_v_Wr.push_back(v_Wr_tmp_i1);
		_v_Wh.push_back(v_Wh_tmp_i1);
	}


	
	// U weights
	for (t = 0; t < _nb_timestep; t++)
	{
		std::vector<std::vector<float>> Uz_tmp_i1;
		std::vector<std::vector<float>> Ur_tmp_i1;
		std::vector<std::vector<float>> Uh_tmp_i1;
		std::vector<std::vector<float>> Uo_tmp_i1;

		std::vector<std::vector<float>> dUz_tmp_i1;
		std::vector<std::vector<float>> dUr_tmp_i1;
		std::vector<std::vector<float>> dUh_tmp_i1;
		std::vector<std::vector<float>> dUo_tmp_i1;

		std::vector<std::vector<float>> m_Uz_tmp_i1;
		std::vector<std::vector<float>> m_Ur_tmp_i1;
		std::vector<std::vector<float>> m_Uh_tmp_i1;
		std::vector<std::vector<float>> m_Uo_tmp_i1;
		std::vector<std::vector<float>> v_Uz_tmp_i1;
		std::vector<std::vector<float>> v_Ur_tmp_i1;
		std::vector<std::vector<float>> v_Uh_tmp_i1;
		std::vector<std::vector<float>> v_Uo_tmp_i1;
		for (i1 = 0; i1 < _output_size; i1++)
		{
			std::vector<float> Uz_tmp_i2;
			std::vector<float> Ur_tmp_i2;
			std::vector<float> Uh_tmp_i2;
			std::vector<float> Uo_tmp_i2;
			std::vector<float> dUz_tmp_i2;
			std::vector<float> dUr_tmp_i2;
			std::vector<float> dUh_tmp_i2;
			std::vector<float> dUo_tmp_i2;
			std::vector<float> m_Uz_tmp_i2;
			std::vector<float> m_Ur_tmp_i2;
			std::vector<float> m_Uh_tmp_i2;
			std::vector<float> m_Uo_tmp_i2;
			std::vector<float> v_Uz_tmp_i2;
			std::vector<float> v_Ur_tmp_i2;
			std::vector<float> v_Uh_tmp_i2;
			std::vector<float> v_Uo_tmp_i2;
			for (i2 = 0; i2 < _output_size; i2++)
			{
				val_tmp = randomize(-1.0, 1.0);
				Uz_tmp_i2.push_back(val_tmp);
				dUz_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Ur_tmp_i2.push_back(val_tmp);
				dUr_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Uh_tmp_i2.push_back(val_tmp);
				dUh_tmp_i2.push_back(0.0);
				val_tmp = randomize(-1.0, 1.0);
				Uo_tmp_i2.push_back(val_tmp);
				dUo_tmp_i2.push_back(0.0);

				m_Uz_tmp_i2.push_back(0.0);
				m_Ur_tmp_i2.push_back(0.0);
				m_Uh_tmp_i2.push_back(0.0);
				m_Uo_tmp_i2.push_back(0.0);
				v_Uz_tmp_i2.push_back(0.0);
				v_Ur_tmp_i2.push_back(0.0);
				v_Uh_tmp_i2.push_back(0.0);
				v_Uo_tmp_i2.push_back(0.0);
			}
			Uz_tmp_i1.push_back(Uz_tmp_i2);
			Ur_tmp_i1.push_back(Ur_tmp_i2);
			Uh_tmp_i1.push_back(Uh_tmp_i2);
			Uo_tmp_i1.push_back(Uo_tmp_i2);
			dUz_tmp_i1.push_back(dUz_tmp_i2);
			dUr_tmp_i1.push_back(dUr_tmp_i2);
			dUh_tmp_i1.push_back(dUh_tmp_i2);
			dUo_tmp_i1.push_back(dUo_tmp_i2);

			m_Uz_tmp_i1.push_back(dUz_tmp_i2);
			m_Ur_tmp_i1.push_back(dUr_tmp_i2);
			m_Uh_tmp_i1.push_back(dUh_tmp_i2);
			m_Uo_tmp_i1.push_back(dUo_tmp_i2);
			v_Uz_tmp_i1.push_back(dUz_tmp_i2);
			v_Ur_tmp_i1.push_back(dUr_tmp_i2);
			v_Uh_tmp_i1.push_back(dUh_tmp_i2);
			v_Uo_tmp_i1.push_back(dUo_tmp_i2);
		}
		_Uz.push_back(Uz_tmp_i1);
		_Ur.push_back(Ur_tmp_i1);
		_Uh.push_back(Uh_tmp_i1);
		_dUz.push_back(dUz_tmp_i1);
		_dUr.push_back(dUr_tmp_i1);
		_dUh.push_back(dUh_tmp_i1);

		_m_Uz.push_back(m_Uz_tmp_i1);
		_m_Ur.push_back(m_Ur_tmp_i1);
		_m_Uh.push_back(m_Uh_tmp_i1);
		_v_Uz.push_back(v_Uz_tmp_i1);
		_v_Ur.push_back(v_Ur_tmp_i1);
		_v_Uh.push_back(v_Uh_tmp_i1);
	}
}

//---------------------------------------------------------------------
// // https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void gru::forward(std::vector<std::vector<float>> input)
{
	uint32_t t, i1, i2; //iterators
	float sum_z;
	float sum_r;
	float sum_h;
	std::vector<std::vector<float>> output;
	
	_x = input;
	// --------------------------------
	// T = 0
	// W Weights matrix sums
	for (i1=0; i1<_output_size; i1++)
	{
		//The input layer is broadcast to the hidden layer
		sum_z = 0.0f;
		sum_r = 0.0f;
		sum_h = 0.0f;
		for (i2=0; i2<_input_size; i2++)
		{
			sum_z += _Wz[0][i1][i2]*_x[0][i2];
			sum_r += _Wr[0][i1][i2]*_x[0][i2];
			sum_h += _Wh[0][i1][i2]*_x[0][i2];
		}
		_z_gate[0][i1] = sigmoid(sum_z + _bz[0][i1]);
		_r_gate[0][i1] = sigmoid(sum_r + _br[0][i1]);
		_h_gate[0][i1] = tanh(sum_h + _bh[0][i1]);
		_out[0][i1] = _z_gate[0][i1]*_h_gate[0][i1];
		//_out[0][i1] = (1.0-_z_gate[0][i1])*_h_gate[0][i1];
	}
	// --------------------------------
	// T = OTHERS TIMESTEPS 
	for (t=1; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			//The input layer is broadcast to the hidden layer
			sum_z = 0.0f;
			sum_r = 0.0f;
			sum_h = 0.0f;
			for (i2=0; i2<_input_size; i2++)
			{
				sum_z += _Wz[t][i1][i2]*_x[t][i2];
				sum_r += _Wr[t][i1][i2]*_x[t][i2];
				sum_h += _Wh[t][i1][i2]*_x[t][i2];
			}
			for (i2=0; i2<_output_size; i2++)
			{
				sum_z += _Uz[t][i1][i2]*_out[t-1][i2];
				sum_r += _Ur[t][i1][i2]*_out[t-1][i2];
			}
			_z_gate[t][i1] = sigmoid(sum_z + _bz[t][i1]);
			_r_gate[t][i1] = sigmoid(sum_r + _br[t][i1]);

			for (i2=0; i2<_output_size; i2++)
			{
				sum_h += _Uh[t][i1][i2]*(_out[t-1][i2]*_r_gate[t][i2]);
			}
			_h_gate[t][i1] = tanh(sum_h + _bh[t][i1]);
			_out[t][i1] = (1.0-_z_gate[t][i1])*_out[t-1][i1]+_z_gate[t][i1]*_h_gate[t][i1];

			//_out[t][i1] = (1.0-_z_gate[t][i1])*_h_gate[t][i1]+_z_gate[t][i1]*_out[t-1][i1];
		}
	}
}
//---------------------------------------------------------------------
void gru::get_error(std::vector<std::vector<float>> target){
	uint32_t t, i1;
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
		}
	}
}

bool isNan(std::vector<std::vector<float>> in){
	for(int i = 0; i < in.size(); i++){
		for(int n = 0; n < in[i].size(); n++){
			if (isnan(in[i][n])){
				printf("i=%d; n=%d\n",i,n);
				return true;
			}
		}
	}
	return false;
}
//--------------------------------------------------------------------
// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
void gru::backward_sgd(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2, i3; //iterators
	float sum_dz, sum_Wz, sum_Uz, sum_bz;
	float sum_dr, sum_Wr, sum_Ur, sum_br;
	float sum_dh, sum_Wh, sum_Uh, sum_bh;
	float drout;
	float m;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
			//std::cout << " _errors[" << t << "][" << i1 << "]= " << _errors[t][i1] << std::endl;
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];

		_dz_gate[t][i2] = dsigmoid(_z_gate[t][i2]) * (_dout[t][i2] * (_h_gate[t][i2] - _out[t-1][i2]));
		_dh_gate[t][i2] = dtanh(_h_gate[t][i2]) * _dout[t][i2] * _z_gate[t][i2];
		m=0.0;
		for(i3=0; i3<_output_size; i3++)
		{
			m += _Uh[t][i3][i2] * _dh_gate[t][i3];
			//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
		}
		_dr_gate[t][i2] = dsigmoid(_r_gate[t][i2]) * m * _out[t-1][i2];
		//printf("t=%d\n",t);

	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_dh = 0.0f;
			sum_dz = 0.0f;
			sum_dr = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_dz += _Uz[t][i2][i1]*_dz_gate[t+1][i2];
				sum_dr += _Ur[t][i2][i1]*_dr_gate[t+1][i2];
				sum_dh += _Uh[t][i2][i1]*_dh_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_dz + sum_dr + sum_dh*_r_gate[t][i1];
			//_Dout[t][i1] = sum_dz + sum_dr;
            //_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];
			_dout[t][i1] = _dout[t+1][i1] * (1 -_z_gate[t][i1]) + _Dout[t][i1];
			//_dout[t][i1] = _dout[t+1][i1] * (1 -_z_gate[t][i1]) + _Dout[t][i1] + _errors[t][i1];

			_dz_gate[t][i1] = 0.0f;
			if(t>0)
			{
				//_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * _dout[t][i1] * (_h_gate[t][i1] * _out[t-1][i1]);
				_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * (_dout[t][i1] * (_h_gate[t][i1] - _out[t-1][i1]));
			}
			
			_dh_gate[t][i1] = dtanh(_h_gate[t][i1]) * _dout[t][i1] * _z_gate[t][i1];
			
			_dr_gate[t][i1] = 0.0f;
			if(t>0)
			{
				m=0.0;
				for(i3=0; i3<_output_size; i3++)
				{
					m += _Uh[t][i3][i1] * _dh_gate[t][i3];
					//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
				}
				_dr_gate[t][i1] = dsigmoid(_r_gate[t][i1]) * m * _out[t-1][i1];
			}
		}
	}

	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wz = 0.0f;
			sum_Wr = 0.0f;
			sum_Wh = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wz = _dz_gate[t][i1]*_x[t][i2];
				sum_Wr = _dr_gate[t][i1]*_x[t][i2];
				sum_Wh = _dh_gate[t][i1]*_x[t][i2];

				_dWz[t][i1][i2] = _lr * ((_mu-1)*sum_Wz + _mu * _dWz[t][i1][i2]);
				_dWr[t][i1][i2] = _lr * ((_mu-1)*sum_Wr + _mu * _dWr[t][i1][i2]);
				_dWh[t][i1][i2] = _lr * ((_mu-1)*sum_Wh + _mu * _dWh[t][i1][i2]);

				_Wz[t][i1][i2] = _Wz[t][i1][i2] - _dWz[t][i1][i2];
				_Wr[t][i1][i2] = _Wr[t][i1][i2] - _dWr[t][i1][i2];
				_Wh[t][i1][i2] = _Wh[t][i1][i2] - _dWh[t][i1][i2];
			}
		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Uz = 0.0f;
			sum_Ur = 0.0f;
			sum_Uh = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Uz = _dz_gate[t+1][i1]*_out[t][i2];
				sum_Ur = _dr_gate[t+1][i1]*_out[t][i2];
				//sum_Uh += _dh_gate[t+1][i1]*_out[t][i2];
				if(t>0){
					sum_Uh = _dh_gate[t+1][i1]*_out[t-1][i2]*_r_gate[t][i2];
				}

				_dUz[t][i1][i2] = _lr * ((_mu-1)*sum_Uz + _mu * _dUz[t][i1][i2]);
				_dUr[t][i1][i2] = _lr * ((_mu-1)*sum_Ur + _mu * _dUr[t][i1][i2]);
				_dUh[t][i1][i2] = _lr * ((_mu-1)*sum_Uh + _mu * _dUh[t][i1][i2]);

				_Uz[t][i1][i2] = _Uz[t][i1][i2] - _dUz[t][i1][i2];
				_Ur[t][i1][i2] = _Ur[t][i1][i2] - _dUr[t][i1][i2];
				_Uh[t][i1][i2] = _Uh[t][i1][i2] - _dUh[t][i1][i2];
			}
		}
		
		sum_bz = 0.0f;
		sum_br = 0.0f;
		sum_bh = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_bz = _dz_gate[t][i1];
			sum_br = _dr_gate[t][i1];
			sum_bh = _dh_gate[t][i1];

			_dbz[t][i1] = _lr * ((_mu-1) * sum_bz + _mu * _dbz[t][i1]);
			_dbr[t][i1] = _lr * ((_mu-1) * sum_br + _mu * _dbr[t][i1]);
			_dbh[t][i1] = _lr * ((_mu-1) * sum_bh + _mu * _dbh[t][i1]);
			
			_bz[t][i1] = _bz[t][i1] - _dbz[t][i1];
			_br[t][i1] = _br[t][i1] - _dbr[t][i1];
			_bh[t][i1] = _bh[t][i1] - _dbh[t][i1];
		}
	}
}
//---------------------------------------------------------------------
void gru::backward_adam(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2, i3; //iterators
	float sum_dz, sum_Wz, sum_Uz, sum_bz;
	float sum_dr, sum_Wr, sum_Ur, sum_br;
	float sum_dh, sum_Wh, sum_Uh, sum_bh;
	float drout;
	float m;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
			//std::cout << " _errors[" << t << "][" << i1 << "]= " << _errors[t][i1] << std::endl;
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];

		_dz_gate[t][i2] = dsigmoid(_z_gate[t][i2]) * (_dout[t][i2] * (_h_gate[t][i2] - _out[t-1][i2]));
		_dh_gate[t][i2] = dtanh(_h_gate[t][i2]) * _dout[t][i2] * _z_gate[t][i2];
		m=0.0;
		for(i3=0; i3<_output_size; i3++)
		{
			m += _Uh[t][i3][i2] * _dh_gate[t][i3];
			//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
		}
		_dr_gate[t][i2] = dsigmoid(_r_gate[t][i2]) * m * _out[t-1][i2];
		//printf("t=%d\n",t);

	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_dh = 0.0f;
			sum_dz = 0.0f;
			sum_dr = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_dz += _Uz[t][i2][i1]*_dz_gate[t+1][i2];
				sum_dr += _Ur[t][i2][i1]*_dr_gate[t+1][i2];
				sum_dh += _Uh[t][i2][i1]*_dh_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_dz + sum_dr + sum_dh*_r_gate[t][i1];
			//_Dout[t][i1] = sum_dz + sum_dr;
            //_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];
			_dout[t][i1] = _dout[t+1][i1] * (1 -_z_gate[t][i1]) + _Dout[t][i1];

			_dz_gate[t][i1] = 0.0f;
			if(t>0)
			{
				//_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * _dout[t][i1] * (_h_gate[t][i1] * _out[t-1][i1]);
				_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * (_dout[t][i1] * (_h_gate[t][i1] - _out[t-1][i1]));
			}
			
			_dh_gate[t][i1] = dtanh(_h_gate[t][i1]) * _dout[t][i1] * _z_gate[t][i1];
			
			_dr_gate[t][i1] = 0.0f;
			if(t>0)
			{
				m=0.0;
				for(i3=0; i3<_output_size; i3++)
				{
					m += _Uh[t][i3][i1] * _dh_gate[t][i3];
					//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
				}
				_dr_gate[t][i1] = dsigmoid(_r_gate[t][i1]) * m * _out[t-1][i1];
			}
		}
	}
	
	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wz = 0.0f;
			sum_Wr = 0.0f;
			sum_Wh = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wz = _dz_gate[t][i1]*_x[t][i2];
				sum_Wr = _dr_gate[t][i1]*_x[t][i2];
				sum_Wh = _dh_gate[t][i1]*_x[t][i2];

				_m_Wz[t][i1][i2] = _b1 * _m_Wz[t][i1][i2] + (1 - _b1) * sum_Wz;
				_m_Wr[t][i1][i2] = _b1 * _m_Wr[t][i1][i2] + (1 - _b1) * sum_Wr;
				_m_Wh[t][i1][i2] = _b1 * _m_Wh[t][i1][i2] + (1 - _b1) * sum_Wh;

				_v_Wz[t][i1][i2] = _b2 * _v_Wz[t][i1][i2] + (1 -_b2) * sum_Wz * sum_Wz;
				_v_Wr[t][i1][i2] = _b2 * _v_Wr[t][i1][i2] + (1 -_b2) * sum_Wr * sum_Wr;
				_v_Wh[t][i1][i2] = _b2 * _v_Wh[t][i1][i2] + (1 -_b2) * sum_Wh * sum_Wh;

				_dWz[t][i1][i2] = _lr * (_m_Wz[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wz[t][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWr[t][i1][i2] = _lr * (_m_Wr[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wr[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dWh[t][i1][i2] = _lr * (_m_Wh[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Wh[t][i1][i2] / (1 - _b2)) + _epsilon);

				_Wz[t][i1][i2] = _Wz[t][i1][i2] + _dWz[t][i1][i2];
				_Wr[t][i1][i2] = _Wr[t][i1][i2] + _dWr[t][i1][i2];
				_Wh[t][i1][i2] = _Wh[t][i1][i2] + _dWh[t][i1][i2];
			}
		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Uz = 0.0f;
			sum_Ur = 0.0f;
			sum_Uh = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Uz = _dz_gate[t+1][i1]*_out[t][i2];
				sum_Ur = _dr_gate[t+1][i1]*_out[t][i2];
				if(t>0){
					sum_Uh = _dh_gate[t+1][i1]*_out[t-1][i2]*_r_gate[t][i2];
				}
			
				_m_Uz[t][i1][i2] = _b1 * _m_Uz[t][i1][i2] + (1 - _b1) * sum_Uz;
				_m_Ur[t][i1][i2] = _b1 * _m_Ur[t][i1][i2] + (1 - _b1) * sum_Ur;
				_m_Uh[t][i1][i2] = _b1 * _m_Uh[t][i1][i2] + (1 - _b1) * sum_Uh;

				_v_Uz[t][i1][i2] = _b2 * _v_Uz[t][i1][i2] + (1 -_b2) * sum_Uz * sum_Uz;
				_v_Ur[t][i1][i2] = _b2 * _v_Ur[t][i1][i2] + (1 -_b2) * sum_Ur * sum_Ur;
				_v_Uh[t][i1][i2] = _b2 * _v_Uh[t][i1][i2] + (1 -_b2) * sum_Uh * sum_Uh;

				_dUz[t][i1][i2] = _lr * (_m_Uz[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Uz[t][i1][i2] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dUr[t][i1][i2] = _lr * (_m_Ur[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Ur[t][i1][i2] / (1 - _b2)) + _epsilon);
				_dUh[t][i1][i2] = _lr * (_m_Uh[t][i1][i2] / (1 - _b1)) / (sqrt(_v_Uh[t][i1][i2] / (1 - _b2)) + _epsilon);

				_Uz[t][i1][i2] = _Uz[t][i1][i2] + _dUz[t][i1][i2];
				_Ur[t][i1][i2] = _Ur[t][i1][i2] + _dUr[t][i1][i2];
				_Uh[t][i1][i2] = _Uh[t][i1][i2] + _dUh[t][i1][i2];
			}
		}
		
		sum_bz = 0.0f;
		sum_br = 0.0f;
		sum_bh = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_bz = _dz_gate[t][i1];
			sum_br = _dr_gate[t][i1];
			sum_bh = _dh_gate[t][i1];

			_m_bz[t][i1] = _b1 * _m_bz[t][i1] + (1 - _b1) * sum_bz;
			_m_br[t][i1] = _b1 * _m_br[t][i1] + (1 - _b1) * sum_br;
			_m_bh[t][i1] = _b1 * _m_bh[t][i1] + (1 - _b1) * sum_bh;

			_v_bz[t][i1] = _b2 * _v_bz[t][i1] + (1 -_b2) * sum_bz * sum_bz;
			_v_br[t][i1] = _b2 * _v_br[t][i1] + (1 -_b2) * sum_br * sum_br;
			_v_bh[t][i1] = _b2 * _v_bh[t][i1] + (1 -_b2) * sum_bh * sum_bh;

			_dbz[t][i1] = _lr * (_m_bz[t][i1] / (1 - _b1)) / (sqrt(_v_bz[t][i1] / (1 - _b2)) + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbr[t][i1] = _lr * (_m_br[t][i1] / (1 - _b1)) / (sqrt(_v_br[t][i1] / (1 - _b2)) + _epsilon);
			_dbh[t][i1] = _lr * (_m_bh[t][i1] / (1 - _b1)) / (sqrt(_v_bh[t][i1] / (1 - _b2)) + _epsilon);

			_bz[t][i1] = _bz[t][i1] + _dbz[t][i1];
			_br[t][i1] = _br[t][i1] + _dbr[t][i1];
			_bh[t][i1] = _bh[t][i1] + _dbh[t][i1];
		}
	}
}
//---------------------------------------------------------------------
void gru::backward_adamax(std::vector<std::vector<float>> target)
{
	int t;
	uint32_t i1, i2, i3; //iterators
	float sum_dz, sum_Wz, sum_Uz, sum_bz;
	float sum_dr, sum_Wr, sum_Ur, sum_br;
	float sum_dh, sum_Wh, sum_Uh, sum_bh;
	float drout;
	float m;
	
	// get error
	for (t=0;t<_nb_timestep;t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			_errors[t][i1] = target[t][i1] - _out[t][i1];
			//std::cout << " _errors[" << t << "][" << i1 << "]= " << _errors[t][i1] << std::endl;
		}
	}

	// --------------------------------
	// T = TIMESTEP - 1 (because it starts at 0)
	t = _nb_timestep-1;
	for (i2=0; i2<_output_size; i2++)
	{
		_Dout[t][i2] = 0.0f;
		_dout[t][i2] = _errors[t][i2];

		_dz_gate[t][i2] = dsigmoid(_z_gate[t][i2]) * (_dout[t][i2] * (_h_gate[t][i2] - _out[t-1][i2]));
		_dh_gate[t][i2] = dtanh(_h_gate[t][i2]) * _dout[t][i2] * _z_gate[t][i2];
		m=0.0;
		for(i3=0; i3<_output_size; i3++)
		{
			m += _Uh[t][i3][i2] * _dh_gate[t][i3];
			//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
		}
		_dr_gate[t][i2] = dsigmoid(_r_gate[t][i2]) * m * _out[t-1][i2];
		//printf("t=%d\n",t);

	}
	// T=TIMESTEP-2 TO 0
	for (t=_nb_timestep-2; t>=0; t--)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			sum_dh = 0.0f;
			sum_dz = 0.0f;
			sum_dr = 0.0f;
			for (i2=0; i2<_output_size; i2++)
			{
				sum_dz += _Uz[t][i2][i1]*_dz_gate[t+1][i2];
				sum_dr += _Ur[t][i2][i1]*_dr_gate[t+1][i2];
				sum_dh += _Uh[t][i2][i1]*_dh_gate[t+1][i2];
			}
			_Dout[t][i1] = sum_dz + sum_dr + sum_dh*_r_gate[t][i1];
			//_Dout[t][i1] = sum_dz + sum_dr;
            //_dout[t][i1] = _errors[t][i1] + _Dout[t][i1];
			_dout[t][i1] = _dout[t+1][i1] * (1 -_z_gate[t][i1]) + _Dout[t][i1];

			_dz_gate[t][i1] = 0.0f;
			if(t>0)
			{
				//_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * _dout[t][i1] * (_h_gate[t][i1] * _out[t-1][i1]);
				_dz_gate[t][i1] = dsigmoid(_z_gate[t][i1]) * (_dout[t][i1] * (_h_gate[t][i1] - _out[t-1][i1]));
			}
			
			_dh_gate[t][i1] = dtanh(_h_gate[t][i1]) * _dout[t][i1] * _z_gate[t][i1];
			
			_dr_gate[t][i1] = 0.0f;
			if(t>0)
			{
				m=0.0;
				for(i3=0; i3<_output_size; i3++)
				{
					m += _Uh[t][i3][i1] * _dh_gate[t][i3];
					//_dr_gate[t][i2] += dsigmoid(_r_gate[t][i2]) * (_Uh[t][i3][i2] * _dh_gate[t][i3]) * _out[t-1][i2];
				}
				_dr_gate[t][i1] = dsigmoid(_r_gate[t][i1]) * m * _out[t-1][i1];
			}
		}
	}
	
	for (i1=0; i1<_output_size; i1++)
	{	
		for (i2=0; i2<_input_size; i2++)
		{
			sum_Wz = 0.0f;
			sum_Wr = 0.0f;
			sum_Wh = 0.0f;
			for (t=0; t<_nb_timestep; t++)
			{
				sum_Wz = _dz_gate[t][i1]*_x[t][i2];
				sum_Wr = _dr_gate[t][i1]*_x[t][i2];
				sum_Wh = _dh_gate[t][i1]*_x[t][i2];

				_m_Wz[t][i1][i2] = _b1 * _m_Wz[t][i1][i2] + (1 - _b1) * sum_Wz;
				_m_Wr[t][i1][i2] = _b1 * _m_Wr[t][i1][i2] + (1 - _b1) * sum_Wr;
				_m_Wh[t][i1][i2] = _b1 * _m_Wh[t][i1][i2] + (1 - _b1) * sum_Wh;

				_v_Wz[t][i1][i2] = std::max(_v_Wz[t][i1][i2]*_b2, std::abs(sum_Wz));
				_v_Wr[t][i1][i2] = std::max(_v_Wr[t][i1][i2]*_b2, std::abs(sum_Wr));
				_v_Wh[t][i1][i2] = std::max(_v_Wh[t][i1][i2]*_b2, std::abs(sum_Wh));

				_dWz[t][i1][i2] = _lr * (_m_Wz[t][i1][i2] / (1 - _b1)) / (_v_Wz[t][i1][i2] + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
				_dWr[t][i1][i2] = _lr * (_m_Wr[t][i1][i2] / (1 - _b1)) / (_v_Wr[t][i1][i2] + _epsilon);
				_dWh[t][i1][i2] = _lr * (_m_Wh[t][i1][i2] / (1 - _b1)) / (_v_Wh[t][i1][i2] + _epsilon);

				_Wz[t][i1][i2] = _Wz[t][i1][i2] + _dWz[t][i1][i2];
				_Wr[t][i1][i2] = _Wr[t][i1][i2] + _dWr[t][i1][i2];
				_Wh[t][i1][i2] = _Wh[t][i1][i2] + _dWh[t][i1][i2];
			}
		}
	}
	// output size loop
	for (i1=0; i1<_output_size; i1++)
	{
		// output size loop	
		for (i2=0; i2<_output_size; i2++)
		{
			sum_Uz = 0.0f;
			sum_Ur = 0.0f;
			sum_Uh = 0.0f;
			for (t=0; t<_nb_timestep-1; t++)
			{
				sum_Uz = _dz_gate[t+1][i1]*_out[t][i2];
				sum_Ur = _dr_gate[t+1][i1]*_out[t][i2];
				if(t>0){
					sum_Uh = _dh_gate[t+1][i1]*_out[t-1][i2]*_r_gate[t][i2];
				}
			
				_m_Uz[t][i1][i2] = _b1 * _m_Uz[t][i1][i2] + (1 - _b1) * sum_Uz;
				_m_Ur[t][i1][i2] = _b1 * _m_Ur[t][i1][i2] + (1 - _b1) * sum_Ur;
				_m_Uh[t][i1][i2] = _b1 * _m_Uh[t][i1][i2] + (1 - _b1) * sum_Uh;

				_v_Uz[t][i1][i2] = std::max(_v_Uz[t][i1][i2]*_b2, std::abs(sum_Uz));
				_v_Ur[t][i1][i2] = std::max(_v_Ur[t][i1][i2]*_b2, std::abs(sum_Ur));
				_v_Uh[t][i1][i2] = std::max(_v_Uh[t][i1][i2]*_b2, std::abs(sum_Uh));

				_dUz[t][i1][i2] = _lr * (_m_Uz[t][i1][i2] / (1 - _b1)) / (_v_Uz[t][i1][i2] + _epsilon);
				_dUr[t][i1][i2] = _lr * (_m_Ur[t][i1][i2] / (1 - _b1)) / (_v_Ur[t][i1][i2] + _epsilon);
				_dUh[t][i1][i2] = _lr * (_m_Uh[t][i1][i2] / (1 - _b1)) / (_v_Uh[t][i1][i2] + _epsilon);

				_Uz[t][i1][i2] = _Uz[t][i1][i2] + _dUz[t][i1][i2];
				_Ur[t][i1][i2] = _Ur[t][i1][i2] + _dUr[t][i1][i2];
				_Uh[t][i1][i2] = _Uh[t][i1][i2] + _dUh[t][i1][i2];
			}
		}
		
		sum_bz = 0.0f;
		sum_br = 0.0f;
		sum_bh = 0.0f;
		for (t=0; t<_nb_timestep; t++)
		{
			sum_bz = _dz_gate[t][i1];
			sum_br = _dr_gate[t][i1];
			sum_bh = _dh_gate[t][i1];

			_m_bz[t][i1] = _b1 * _m_bz[t][i1] + (1 - _b1) * sum_bz;
			_m_br[t][i1] = _b1 * _m_br[t][i1] + (1 - _b1) * sum_br;
			_m_bh[t][i1] = _b1 * _m_bh[t][i1] + (1 - _b1) * sum_bh;

			_v_bz[t][i1] = std::max(_v_bz[t][i1]*_b2, std::abs(sum_bz));
			_v_br[t][i1] = std::max(_v_br[t][i1]*_b2, std::abs(sum_br));
			_v_bh[t][i1] = std::max(_v_bh[t][i1]*_b2, std::abs(sum_bh));

			_dbz[t][i1] = _lr * (_m_bz[t][i1] / (1 - _b1)) / (_v_bz[t][i1] + _epsilon); //lr*m~/(sqrt(v~)+epsilon)
			_dbr[t][i1] = _lr * (_m_br[t][i1] / (1 - _b1)) / (_v_br[t][i1] + _epsilon);
			_dbh[t][i1] = _lr * (_m_bh[t][i1] / (1 - _b1)) / (_v_bh[t][i1] + _epsilon);

			_bz[t][i1] = _bz[t][i1] + _dbz[t][i1];
			_br[t][i1] = _br[t][i1] + _dbr[t][i1];
			_bh[t][i1] = _bh[t][i1] + _dbh[t][i1];
		}
	}
}

//---------------------------------------------------------------------
// generic backward
void gru::backward(std::vector<std::vector<float>> target)
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
float gru::get_mse()
{
	uint32_t i1;
	float mse_tmp = 0.0;
	for (i1=0; i1<_output_size; i1++)
	{
		mse_tmp += _errors[_nb_timestep-1][i1]*_errors[_nb_timestep-1][i1];
	}
	return mse_tmp/_output_size;
}

float gru::get_mse(std::vector<std::vector<float>> target, std::vector<std::vector<float>> output)
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
	return mse_tmp;
}
//----------------------------------------------------
void gru::clear_grads()
{
	uint32_t t, i1, i2; //iterators
	for (t=0; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_input_size; i2++)
			{
				_dWz[t][i1][i2] = 0.0;
				_dWr[t][i1][i2] = 0.0;
				_dWh[t][i1][i2] = 0.0;
			}
		}
		
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_output_size; i2++)
			{
				_dUz[t][i1][i2] = 0.0;
				_dUr[t][i1][i2] = 0.0;
				_dUh[t][i1][i2] = 0.0;
			}
		}
		for (i1=0; i1<_output_size; i1++)
		{
			_dbz[t][i1] = 0.0;
			_dbr[t][i1] = 0.0;
			_dbh[t][i1] = 0.0;
		}
	}
}
//--------------------------------------------------------

void gru::clear_momentums()
{
	uint32_t t, i1, i2; //iterators
	for (t=0; t<_nb_timestep; t++)
	{
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_input_size; i2++)
			{
				_m_Wz[t][i1][i2] = 0.0;
				_m_Wr[t][i1][i2] = 0.0;
				_m_Wh[t][i1][i2] = 0.0;
				_v_Wz[t][i1][i2] = 0.0;
				_v_Wr[t][i1][i2] = 0.0;
				_v_Wh[t][i1][i2] = 0.0;
			}
		}
		
		for (i1=0; i1<_output_size; i1++)
		{
			for (i2=0; i2<_output_size; i2++)
			{
				_m_Uz[t][i1][i2] = 0.0;
				_m_Ur[t][i1][i2] = 0.0;
				_m_Uh[t][i1][i2] = 0.0;
				_v_Uz[t][i1][i2] = 0.0;
				_v_Ur[t][i1][i2] = 0.0;
				_v_Uh[t][i1][i2] = 0.0;
			}
		}
		for (i1=0; i1<_output_size; i1++)
		{
			_m_bz[t][i1] = 0.0;
			_m_br[t][i1] = 0.0;
			_m_bh[t][i1] = 0.0;
			_v_bz[t][i1] = 0.0;
			_v_br[t][i1] = 0.0;
			_v_bh[t][i1] = 0.0;
		}
	}
}
//---------------------------------------------------------------------

void gru::save(const char* filename)
{
	std::ofstream reportfile;
	
	reportfile.open(filename);

	// inputs
	std::string x;
	std::string dx;
	
	std::string errors;
	
	// gates
	std::string z_gate;
	std::string r_gate;
	std::string h_gate;
	std::string o_gate;
	std::string dz_gate;
	std::string dr_gate;
	std::string dh_gate;
	std::string do_gate;
	
	std::string state;
	std::string dstate;
	
	std::string out;
	std::string dout;
	std::string Dout;

	std::string bz;
	std::string br;
	std::string bh;
	std::string bo;
	std::string dbz;
	std::string dbr;
	std::string dbh;
	std::string dbo;
	std::string Wz;
	std::string Wr;
	std::string Wh;
	std::string Wo;
	std::string Uz;
	std::string Ur;
	std::string Uh;
	std::string Uo;
	std::string dWz;
	std::string dWr;
	std::string dWh;
	std::string dWo;
	std::string dUz;
	std::string dUr;
	std::string dUh;
	std::string dUo;


	uint32_t t, i1, i2; //iterators

	for (i1 = 0; i1 < _input_size; i1++){x+=("x[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _input_size; i1++){dx+=("dx[" + std::to_string(i1) + "], ");}
	
	for (i1 = 0; i1 < _output_size; i1++){z_gate +=("z_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){r_gate +=("r_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){h_gate +=("h_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){o_gate +=("o_gate[" + std::to_string(i1) + "], ");}

	for (i1 = 0; i1 < _output_size; i1++){ state  +=("state[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ out    +=("out[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dz_gate+=("dz_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dr_gate+=("dr_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dh_gate+=("dh_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ do_gate+=("do_gate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dstate +=("dstate[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dout   +=("dout[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ Dout   +=("Dout[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ errors +=("errors[" + std::to_string(i1) + "], ");}
	
	// Bias
	for (i1 = 0; i1 < _output_size; i1++){ bz+=("bz[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ br+=("br[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ bh+=("bh[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ bo+=("bo[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbz+=("dbz[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbr+=("dbr[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbh+=("dbh[" + std::to_string(i1) + "], ");}
	for (i1 = 0; i1 < _output_size; i1++){ dbo+=("dbo[" + std::to_string(i1) + "], ");}

	// W weights
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wz+=("Wz[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wr+=("Wr[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wh+=("Wh[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wo+=("Wo[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWz+=("dWz[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWr+=("dWr[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWh+=("dWh[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWo+=("dWo[" + std::to_string(i1) + "]" + "[" + std::to_string(i2) + "], ");}}
	
	
	// U weights
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uz+=("Uz[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ur+=("Ur[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uh+=("Uh[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uo+=("Uo[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUz+=("dUz[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUr+=("dUr[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUh+=("dUh[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}
	for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUo+=("dUo[" + std::to_string(i1) + "]" + "[" + std::to_string(i1) + "], ");}}

	reportfile << 
	"#Neuron, " <<
	x << dx <<
	z_gate << r_gate << h_gate << 
	out << 
	dz_gate << dr_gate << dh_gate <<
	dout << Dout << errors <<
	bz << br << bh << bo << 
	dbz << dbr << dbh << dbo << 
	Wz << Wr << Wh << 
	dWz << dWr << dWh << 
	Uz << Ur <<	Uh << 
	dUz << dUr << dUh <<
	std::endl;

	for (t = 0; t < _nb_timestep; t++)
	{
		x.clear(); dx.clear(); errors.clear();
		z_gate.clear(); r_gate.clear(); h_gate.clear(); o_gate.clear(); 
		dz_gate.clear(); dr_gate.clear(); dh_gate.clear(); do_gate.clear();
		state.clear(); dstate.clear(); out.clear(); dout.clear(); Dout.clear(); 
		bz.clear(); br.clear(); bh.clear(); bo.clear(); 
		dbz.clear(); dbr.clear(); dbh.clear(); dbo.clear(); 
		Wz.clear(); Wr.clear(); Wh.clear(); Wo.clear(); 
		Uz.clear(); Ur.clear();	Uh.clear(); Uo.clear(); 
		dWz.clear(); dWr.clear(); dWh.clear(); dWo.clear(); 
		dUz.clear(); dUr.clear(); dUh.clear(); dUo.clear();
 
		for (i1 = 0; i1 < _input_size; i1++){x+=(std::to_string(_x[t][i1]) + ", ");}
		for (i1 = 0; i1 < _input_size; i1++){dx+=(std::to_string( _dx[t][i1]) + ", ");}
		
		for (i1 = 0; i1 < _output_size; i1++){z_gate +=(std::to_string( _z_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){r_gate +=(std::to_string( _r_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){h_gate +=(std::to_string( _h_gate[t][i1]) + ", ");}

		for (i1 = 0; i1 < _output_size; i1++){ out    +=(std::to_string( _out[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dz_gate+=(std::to_string( _dz_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dr_gate+=(std::to_string( _dr_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dh_gate+=(std::to_string( _dh_gate[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dout   +=(std::to_string( _dout[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ Dout   +=(std::to_string( _Dout[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ errors +=(std::to_string( _errors[t][i1]) + ", ");}
		
		// Bias
		for (i1 = 0; i1 < _output_size; i1++){ bz+=(std::to_string( _bz[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ br+=(std::to_string( _br[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ bh+=(std::to_string( _bh[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbz+=(std::to_string( _dbz[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbr+=(std::to_string( _dbr[t][i1]) + ", ");}
		for (i1 = 0; i1 < _output_size; i1++){ dbh+=(std::to_string( _dbh[t][i1]) + ", ");}

		// W weights
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wz+=(std::to_string( _Wz[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wr+=(std::to_string( _Wr[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){Wh+=(std::to_string( _Wh[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWz+=(std::to_string( _dWz[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWr+=(std::to_string( _dWr[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _input_size; i2++){dWh+=(std::to_string( _dWh[t][i1][i2]) + ", ");}}
		
		
		// U weights
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uz+=(std::to_string( _Uz[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Ur+=(std::to_string( _Ur[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){Uh+=(std::to_string( _Uh[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUz+=(std::to_string( _dUz[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUr+=(std::to_string( _dUr[t][i1][i2]) + ", ");}}
		for (i1 = 0; i1 < _output_size; i1++){ for(i2 = 0; i2 < _output_size; i2++){dUh+=(std::to_string( _dUh[t][i1][i2]) + ", ");}}

		reportfile << 
		std::to_string(t) << ", " << 
		x << dx <<
		z_gate << r_gate << h_gate << 
		out << 
		dz_gate << dr_gate << dh_gate <<
		dout << Dout << errors <<
		bz << br << bh << bo << 
		dbz << dbr << dbh << dbo << 
		Wz << Wr << Wh << 
		dWz << dWr << dWh << 
		Uz << Ur <<	Uh << 
		dUz << dUr << dUh <<
		std::endl;
	}
	reportfile.close();
}
//----------------------------------------------------------------------------------
inline int gru::nbLines(std::string filename){
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


void gru::load(const char* grulogfilename, uint32_t input_size, uint32_t output_size)
{
	int nb_timesteps = nbLines(grulogfilename) - 1;
	int i1,i2;
	int t = 0;

	this->init(nb_timesteps, input_size, output_size);
	// stream for the log
    std::ifstream grulogfile; 
    // string to store each file line
    std::string line; 
    std::string delim;
	std::string number;
	grulogfile.open(grulogfilename);

	if (!grulogfile.is_open()) 
    {
        std::cerr << "There was a problem opening the GRU LOG input file " << grulogfilename << std::endl;
        exit(1);
    }
    else
    {
		std::getline(grulogfile, line); // read the first line only for column title
	}
    while(std::getline(grulogfile, line))
	{
		std::stringstream lineStream(line);
		lineStream >> number;
		for (i1 = 0; i1 < _input_size; i1++){lineStream >>  _x [ t ] [ i1 ];lineStream >> delim;}
		for (i1 = 0; i1 < _input_size; i1++){lineStream >>  _dx [ t ] [ i1 ];  lineStream >> delim;}
		
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _z_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _r_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _h_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _out [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dz_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dr_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dh_gate [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dout [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _Dout [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _errors [ t ] [ i1 ];  lineStream >> delim;}
		
		// Bias
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _bz [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _br [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _bh [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbz [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbr [ t ] [ i1 ];  lineStream >> delim;}
		for (i1 = 0; i1 < _output_size; i1++){lineStream >> _dbh [ t ] [ i1 ];  lineStream >> delim;}

		// W weights
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wz [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wr [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _Wh [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWz [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWr [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _input_size; i2++){lineStream >> _dWh [ t ] [ i1 ]  [ i2 ];  lineStream >> delim;}}
		
		
		// U weights
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Uz [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Ur [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _Uh [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUz [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUr [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}
		for (i1 = 0; i1 < _output_size; i1++){for(i2 = 0; i2 < _output_size; i2++){lineStream >> _dUh [ t ] [ i1 ]  [ i1 ];  lineStream >> delim;}}

		t++;
	}
	grulogfile.close();
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
inline float gru::sigmoid(float x)
{
	return (1.0 / (1.0 + exp(-x)));
}

//---------------------------------------------------------------------
inline float gru::dsigmoid(float x)
{
	return (x*(1.0 - x));

}

//---------------------------------------------------------------------
inline float gru::dtanh(float x)
{
	//float y = tanh(x);
	//return (1.0 - y*y);
	return (1.0-x*x);
}

//---------------------------------------------------------------------
float gru::randomize(float min, float max){
    float f = ((float) rand()) / RAND_MAX;
    return min + f * (max - min);
}
