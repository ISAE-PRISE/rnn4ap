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

#include <imu_log_handler.hh>

//---------------------------------------------------------------------
imu_log_handler::imu_log_handler()
{
	_nb_elem = 0;
	_end_of_file_timesteps= 20;
	// for randomizing
	std::srand ( unsigned ( std::time(0) ) );
}

//---------------------------------------------------------------------
imu_log_handler::~imu_log_handler()
{
	
}


//---------------------------------------------------------------------
void imu_log_handler::load_imu_log(const char *imulogfilename)
{
	// stream for the log
    std::ifstream imulogfile; 
    // string to store each file line
    std::string line; 
    char delim;
    // tmp data

    float acc_x= 0.0, acc_y= 0.0, acc_z= 0.0;
    float gyr_x= 0.0, gyr_y= 0.0, gyr_z= 0.0;
    float mag_x= 0.0, mag_y= 0.0, mag_z= 0.0;
    float phi=0.0, theta=0.0, psi=0.0;
        
    imulogfile.open(imulogfilename);
    
    if (!imulogfile.is_open()) 
    {
        std::cerr << "There was a problem opening the IMU LOG input file " << imulogfilename << std::endl;
        exit(1);
    }
    else
    {
		std::getline(imulogfile, line); // read the first line only for column title
	}
    while(std::getline(imulogfile, line))
	{
		_it_vector.push_back(_nb_elem);
		//_random_it_vector.push_back(_nb_elem);
		_nb_elem++;
		std::stringstream lineStream(line);
		 lineStream >> acc_x >> delim >> acc_y >> delim >> acc_z >> delim  
		>> gyr_x >> delim >> gyr_y >> delim >> gyr_z >> delim 
		>> mag_x >> delim >> mag_y >> delim >> mag_z >> delim
		>> phi >> delim >> theta >> delim >> psi;
		
		// put data in corresponding vector
		// inputs
		_acc_x.push_back(acc_x);
		_acc_y.push_back(acc_y);
		_acc_z.push_back(acc_z);
		_gyr_x.push_back(gyr_x);
		_gyr_y.push_back(gyr_y);
		_gyr_z.push_back(gyr_z);
		_mag_x.push_back(mag_x);
		_mag_y.push_back(mag_y);
		_mag_z.push_back(mag_z);
		// labels
		_phi.push_back(phi);
		_theta.push_back(theta);
		_psi.push_back(psi);
		
		_phi_mah.push_back(phi);
		_theta_mah.push_back(theta);
		_psi_mah.push_back(psi);
		
		_phi_mag.push_back(phi);
		_theta_mag.push_back(theta);
		_psi_mag.push_back(psi);
		
		_phi_mah2.push_back(phi);
		_theta_mah2.push_back(theta);
		_psi_mah2.push_back(psi);
		
		_phi_mag2.push_back(phi);
		_theta_mag2.push_back(theta);
		_psi_mag2.push_back(psi);
	}
	imulogfile.close();
	//std::cerr << "_nb_elem=" << _nb_elem << std::endl;
}

//---------------------------------------------------------------------
void imu_log_handler::randomize_training_dataset(){
	std::random_shuffle(_random_it_vector_train.begin(), _random_it_vector_train.end());
}

//---------------------------------------------------------------------
void imu_log_handler::create_split_and_create_random_dataset(float test_split_coeff)
{
	uint32_t i1, i2, split;
	_max_train_samples = 0;
	_max_test_samples = 0;
	_random_it_vector_train.clear();
	
	split = round(_it_vector.size()*(1.0-test_split_coeff));
	_test_start_index = split;
	
	for (i1 = 0; i1 < _test_start_index ; ++i1)
	{
		_random_it_vector_train.push_back(i1);
	}
	
	_max_train_samples = _random_it_vector_train.size();
	_max_test_samples = _it_vector.size()-_max_train_samples-_end_of_file_timesteps;

	for (i1 = _test_start_index; i1 < _test_start_index+_max_test_samples ; ++i1)
	{
		_it_vector_test.push_back(i1);
	}
}

//---------------------------------------------------------------------
std::vector<float> imu_log_handler::get_sensors(uint32_t it)
{
	unsigned int i1;
	unsigned int index_tmp;
	std::vector<float> input;

	input.push_back(_gyr_x[it]);
	input.push_back(_gyr_y[it]);
	input.push_back(_gyr_z[it]);
	input.push_back(_acc_x[it]);
	input.push_back(_acc_y[it]);
	input.push_back(_acc_z[it]);
	input.push_back(_mag_x[it]);
	input.push_back(_mag_y[it]);
	input.push_back(_mag_z[it]);
	
	return input;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_input_rdm_train(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> input;
	for(i1 = 0; i1 < _random_it_vector_train.size(); ++i1) 
	{
		std::vector<std::vector<float>> input_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> input_tmp_i3;
			index_tmp = _random_it_vector_train[i1] + i2;
			input_tmp_i3.push_back(_acc_x[index_tmp]);
			input_tmp_i3.push_back(_acc_y[index_tmp]);
			input_tmp_i3.push_back(_acc_z[index_tmp]);
			input_tmp_i3.push_back(_gyr_x[index_tmp]);
			input_tmp_i3.push_back(_gyr_y[index_tmp]);
			input_tmp_i3.push_back(_gyr_z[index_tmp]);
			input_tmp_i3.push_back(_mag_x[index_tmp]);
			input_tmp_i3.push_back(_mag_y[index_tmp]);
			input_tmp_i3.push_back(_mag_z[index_tmp]);
			input_tmp_i2.push_back(input_tmp_i3);
		}
		input.push_back(input_tmp_i2);
	}
	
	return input;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_target_rdm_train(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < _random_it_vector_train.size(); ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _random_it_vector_train[i1] + i2;
			target_tmp_i3.push_back(_phi[index_tmp]);
			target_tmp_i3.push_back(_theta[index_tmp]);
			target_tmp_i3.push_back(_psi[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	return target;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_target_phi_rdm_train(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < _random_it_vector_train.size(); ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _random_it_vector_train[i1] + i2;
			target_tmp_i3.push_back(_phi[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	return target;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_input_test(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> input;
	for(i1 = 0; i1 < _it_vector_test.size() ; ++i1) 
	{
		std::vector<std::vector<float>> input_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> input_tmp_i3;
			index_tmp = _it_vector_test[i1] + i2;
			input_tmp_i3.push_back(_acc_x[index_tmp]);
			input_tmp_i3.push_back(_acc_y[index_tmp]);
			input_tmp_i3.push_back(_acc_z[index_tmp]);
			input_tmp_i3.push_back(_gyr_x[index_tmp]);
			input_tmp_i3.push_back(_gyr_y[index_tmp]);
			input_tmp_i3.push_back(_gyr_z[index_tmp]);
			input_tmp_i3.push_back(_mag_x[index_tmp]);
			input_tmp_i3.push_back(_mag_y[index_tmp]);
			input_tmp_i3.push_back(_mag_z[index_tmp]);
			input_tmp_i2.push_back(input_tmp_i3);
		}
		input.push_back(input_tmp_i2);
	}
	
	return input;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_target_test(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < _it_vector_test.size() ; ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _it_vector_test[i1] + i2;
			target_tmp_i3.push_back(_phi[index_tmp]);
			target_tmp_i3.push_back(_theta[index_tmp]);
			target_tmp_i3.push_back(_psi[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	
	return target;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> imu_log_handler::get_target_phi_test(uint32_t nb_timestep)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < _it_vector_test.size() ; ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _it_vector_test[i1] + i2;
			target_tmp_i3.push_back(_phi[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	
	return target;
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_lstm_sgd_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_lstm_sgd_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_lstm_sgd_all_tdim(std::vector<std::vector<float>> nn_output_epoch)
{
	_lstm_sgd_all_tdim.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_lstm_sgd_phi(std::vector<std::vector<float>> nn_output_epoch)
{
	_lstm_sgd_phi.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_lstm_adam_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_lstm_adam_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_gru_sgd_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_gru_sgd_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_gru_adam_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_gru_adam_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_star_sgd_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_star_sgd_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_star_adam_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_star_adam_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_mgu_sgd_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_mgu_sgd_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_one_epoch_for_mgu_adam_all(std::vector<std::vector<float>> nn_output_epoch)
{
	_mgu_adam_all.push_back(nn_output_epoch);
}

//---------------------------------------------------------------------
void imu_log_handler::add_phi_mah(int iter, float val)
{
	_phi_mah[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_theta_mah(int iter, float val)
{
	_theta_mah[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_psi_mah(int iter, float val)
{
	_psi_mah[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_phi_mah2(int iter, float val)
{
	_phi_mah2[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_theta_mah2(int iter, float val)
{
	_theta_mah2[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_psi_mah2(int iter, float val)
{
	_psi_mah2[iter] = val;
}


//---------------------------------------------------------------------
void imu_log_handler::add_phi_mag(int iter, float val)
{
	_phi_mag[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_theta_mag(int iter, float val)
{
	_theta_mag[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_psi_mag(int iter, float val)
{
	_psi_mag[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_phi_mag2(int iter, float val)
{
	_phi_mag2[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_theta_mag2(int iter, float val)
{
	_theta_mag2[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::add_psi_mag2(int iter, float val)
{
	_psi_mag2[iter] = val;
}

//---------------------------------------------------------------------
void imu_log_handler::create_mag_mah_report(const char *reportfilename)
{
	unsigned int i1,i2;
	std::ofstream reportfile;
	reportfile.open(reportfilename);
	std::string ite_name = "sample_it,";
	reportfile << ite_name;
	std::string angle_name = "phi_ref,theta_ref,psi_ref,";
	reportfile << angle_name;
	std::string angle_name_mah = "phi_mah,theta_mah,psi_mah,";
	reportfile << angle_name_mah;
	std::string angle_name_mah2 = "phi_mah2,theta_mah2,psi_mah2,";
	reportfile << angle_name_mah2;
	
	std::string angle_name_mag = "phi_mag,theta_mag,psi_mag,";
	reportfile << angle_name_mag;
	std::string angle_name_mag2 = "phi_mag2,theta_mag2,psi_mag2\n";
	reportfile << angle_name_mag2;
	
	for(i1 = 0; i1 < _nb_elem ; ++i1) 
	{
		reportfile << i1 << ","; 
		reportfile << _phi[i1] << "," << _theta[i1] << "," << _psi[i1] << ",";
		reportfile << _phi_mah[i1] << "," << _theta_mah[i1] << "," << _psi_mah[i1] << ",";
		reportfile << _phi_mah2[i1] << "," << _theta_mah2[i1] << "," << _psi_mah2[i1] << ",";
		reportfile << _phi_mag[i1] << "," << _theta_mag[i1] << "," << _psi_mag[i1] << ",";
		reportfile << _phi_mag2[i1] << "," << _theta_mag2[i1] << "," << _psi_mag2[i1] << "\n";
	}
}


//---------------------------------------------------------------------
void imu_log_handler::create_report(const char *reportfilename)
{
	unsigned int i1,i2;
	std::ofstream reportfile;
	reportfile.open(reportfilename);
	std::string ite_name = "sample_it,";
	reportfile << ite_name;
	std::string angle_name = "phi_ref,theta_ref,psi_ref,";
	reportfile << angle_name;

	for (i2 = 0; i2 < _lstm_sgd_all.size(); i2++)
	{
		std::string lstm_name = "phi_lstm_sgd_all_epoch_" + std::to_string(i2+1) + ",theta_lstm_sgd_epoch_" + std::to_string(i2+1) + ",psi_lstm_sgd_all_epoch_" + std::to_string(i2+1) + ",";
		reportfile << lstm_name;
	}
	if (_gru_sgd_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _gru_sgd_all.size(); i2++)
		{
			std::string gru_name = "phi_gru_sgd_all_epoch_" + std::to_string(i2+1) + ",theta_gru_sgd_all_epoch_" + std::to_string(i2+1) + ",psi_gru_sgd_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << gru_name;
		}
	}
	if (_star_sgd_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _star_sgd_all.size(); i2++)
		{
			std::string star_name = "phi_star_sgd_all_epoch_" + std::to_string(i2+1) + ",theta_star_sgd_all_epoch_" + std::to_string(i2+1) + ",psi_star_sgd_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << star_name;
		}
	}
	if (_mgu_sgd_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _mgu_sgd_all.size(); i2++)
		{
			std::string mgu_name = "phi_mgu_sgd_all_epoch_" + std::to_string(i2+1) + ",theta_mgu_sgd_all_epoch_" + std::to_string(i2+1) + ",psi_mgu_sgd_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << mgu_name;
		}
	}
	if (_lstm_sgd_phi.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _lstm_sgd_phi.size(); i2++)
		{
			std::string mgu_name = "phi_lstm_sgd_only_epoch_" + std::to_string(i2+1) + ",";
			reportfile << mgu_name;
		}
	}
	if (_lstm_sgd_all_tdim.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _lstm_sgd_all_tdim.size(); i2++)
		{
			std::string lstm_name = "phi_lstm_sgd_all_tdim_epoch_" + std::to_string(i2+1) + ",theta_lstm_sgd_all_tdim_epoch_" + std::to_string(i2+1) + ",psi_lstm_sgd_all_tdim_epoch_" + std::to_string(i2+1) + ",";
			reportfile << lstm_name;
		}
	}
	if (_lstm_adam_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _lstm_adam_all.size(); i2++)
		{
			std::string lstm_name = "phi_lstm_adam_all_epoch_" + std::to_string(i2+1) + ",theta_lstm_adam_epoch_" + std::to_string(i2+1) + ",psi_lstm_adam_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << lstm_name;
		}
	}
	if (_gru_adam_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _gru_adam_all.size(); i2++)
		{
			std::string gru_name = "phi_gru_adam_all_epoch_" + std::to_string(i2+1) + ",theta_gru_adam_epoch_" + std::to_string(i2+1) + ",psi_gru_adam_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << gru_name;
		}
	}
	if (_star_adam_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _star_adam_all.size(); i2++)
		{
			std::string star_name = "phi_star_adam_all_epoch_" + std::to_string(i2+1) + ",theta_star_adam_epoch_" + std::to_string(i2+1) + ",psi_star_adam_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << star_name;
		}
	}
	if (_mgu_adam_all.size() == _lstm_sgd_all.size())
	{
		for (i2 = 0; i2 < _mgu_adam_all.size(); i2++)
		{
			std::string star_name = "phi_mgu_adam_all_epoch_" + std::to_string(i2+1) + ",theta_mgu_adam_epoch_" + std::to_string(i2+1) + ",psi_mgu_adam_all_epoch_" + std::to_string(i2+1) + ",";
			reportfile << star_name;
		}
	}
	std::string end_of_line = "id_in_original_file\n";
	reportfile << end_of_line;
	
	for(i1 = 0; i1 < _it_vector_test.size() ; ++i1) 
	{
		reportfile << i1 << ","; 
		reportfile << _phi[_it_vector_test[i1]] << "," << _theta[_it_vector_test[i1]] << "," << _psi[_it_vector_test[i1]] << ",";
		for (i2 = 0; i2 < _lstm_sgd_all.size(); i2++)
		{
			reportfile << _lstm_sgd_all[i2][i1][0] << "," << _lstm_sgd_all[i2][i1][1]  << "," << _lstm_sgd_all[i2][i1][2]  << ",";
		}
		if (_gru_sgd_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_sgd_all.size(); i2++)
			{
				reportfile << _gru_sgd_all[i2][i1][0] << "," << _gru_sgd_all[i2][i1][1]  << "," << _gru_sgd_all[i2][i1][2]  << ",";
			}
		}
		if (_star_sgd_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_sgd_all.size(); i2++)
			{
				reportfile << _star_sgd_all[i2][i1][0] << "," << _star_sgd_all[i2][i1][1]  << "," << _star_sgd_all[i2][i1][2]  << ",";
			}
		}
		if (_mgu_sgd_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_sgd_all.size(); i2++)
			{
				reportfile << _mgu_sgd_all[i2][i1][0] << "," << _mgu_sgd_all[i2][i1][1]  << "," << _mgu_sgd_all[i2][i1][2]  << ",";
			}
		}
		if (_lstm_sgd_phi.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_sgd_phi.size(); i2++)
			{
				reportfile << _lstm_sgd_phi[i2][i1][0] << ",";
			}
		}
		if (_lstm_sgd_all_tdim.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_sgd_all_tdim.size(); i2++)
			{
				reportfile << _lstm_sgd_all_tdim[i2][i1][0] << "," << _lstm_sgd_all_tdim[i2][i1][1]  << "," << _lstm_sgd_all_tdim[i2][i1][2]  << ",";
			}
		}
		if (_lstm_adam_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _lstm_adam_all.size(); i2++)
			{
				reportfile << _lstm_adam_all[i2][i1][0] << "," << _lstm_adam_all[i2][i1][1]  << "," << _lstm_adam_all[i2][i1][2]  << ",";
			}
		}
		if (_gru_adam_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _gru_adam_all.size(); i2++)
			{
				reportfile << _gru_adam_all[i2][i1][0] << "," << _gru_adam_all[i2][i1][1]  << "," << _gru_adam_all[i2][i1][2]  << ",";
			}
		}
		if (_star_adam_all.size() == _lstm_sgd_all.size())
		{
			for (i2 = 0; i2 < _star_adam_all.size(); i2++)
			{
				reportfile << _star_adam_all[i2][i1][0] << "," << _star_adam_all[i2][i1][1]  << "," << _star_adam_all[i2][i1][2]  << ",";
			}
		}
		if (_mgu_adam_all.size() == _mgu_sgd_all.size())
		{
			for (i2 = 0; i2 < _mgu_adam_all.size(); i2++)
			{
				reportfile << _mgu_adam_all[i2][i1][0] << "," << _mgu_adam_all[i2][i1][1]  << "," << _mgu_adam_all[i2][i1][2]  << ",";
			}
		}
		// End of line
		reportfile << _it_vector_test[i1] << "\n";
	}
	reportfile.close();
}

//---------------------------------------------------------------------
void imu_log_handler::create_loss_report(const char *reportfilename)
{
	unsigned int i1,i2;
	std::ofstream reportfile;
	reportfile.open(reportfilename);
	
	// SGD
	if (_loss_train_sgd_lstm_all.size()>0 && _loss_test_sgd_lstm_all.size()>0)
	{
		std::string lstm_name = "loss_train_sgd_lstm_all, loss_test_sgd_lstm_all,";
		reportfile << lstm_name;
	}
	if (_loss_train_sgd_gru_all.size()>0 && _loss_test_sgd_gru_all.size()>0)
	{
		std::string name = "loss_train_sgd_gru_all, loss_test_sgd_gru_all,";
		reportfile << name;
	}
	if (_loss_train_sgd_star_all.size()>0 && _loss_test_sgd_star_all.size()>0)
	{
		std::string name = "loss_train_sgd_star_all, loss_test_sgd_star_all,";
		reportfile << name;
	}
	if (_loss_train_sgd_mgu_all.size()>0 && _loss_test_sgd_mgu_all.size()>0)
	{
		std::string name = "loss_train_sgd_mgu_all, loss_test_sgd_mgu_all,";
		reportfile << name;
	}
	if (_loss_train_sgd_lstm_all_tdim.size()>0 && _loss_test_sgd_lstm_all_tdim.size()>0)
	{
		std::string lstm_name = "loss_train_sgd_lstm_all_tdim, loss_test_sgd_lstm_all_tdim,";
		reportfile << lstm_name;
	}
	// ADAM
	if (_loss_train_adam_lstm_all.size()>0 && _loss_test_adam_lstm_all.size()>0)
	{
		std::string lstm_name = "loss_train_adam_lstm_all, loss_test_adam_lstm_all,";
		reportfile << lstm_name;
	}
	if (_loss_train_adam_gru_all.size()>0 && _loss_test_adam_gru_all.size()>0)
	{
		std::string name = "loss_train_adam_gru_all, loss_test_adam_gru_all,";
		reportfile << name;
	}
	if (_loss_train_adam_star_all.size()>0 && _loss_test_adam_star_all.size()>0)
	{
		std::string name = "loss_train_adam_star_all, loss_test_adam_star_all,";
		reportfile << name;
	}
	if (_loss_train_adam_mgu_all.size()>0 && _loss_test_adam_mgu_all.size()>0)
	{
		std::string name = "loss_train_adam_mgu_all, loss_test_adam_mgu_all,";
		reportfile << name;
	}
	
	std::string ite_name = "epochs\n";
	reportfile << ite_name;
	
	for (i1 = 0; i1 < _loss_train_sgd_lstm_all.size(); i1++)
	{
		//SGD
		if (_loss_train_sgd_lstm_all.size()>0 && _loss_test_sgd_lstm_all.size()>0)
		{
			reportfile << _loss_train_sgd_lstm_all[i1] << "," << _loss_test_sgd_lstm_all[i1] << ",";
		}
		if (_loss_train_sgd_gru_all.size()>0 && _loss_test_sgd_gru_all.size()>0)
		{
			reportfile << _loss_train_sgd_gru_all[i1] << "," << _loss_test_sgd_gru_all[i1] << ",";
		}
		if (_loss_train_sgd_star_all.size()>0 && _loss_test_sgd_star_all.size()>0)
		{
			reportfile << _loss_train_sgd_star_all[i1] << "," << _loss_test_sgd_star_all[i1] << ",";
		}
		if (_loss_train_sgd_mgu_all.size()>0 && _loss_test_sgd_mgu_all.size()>0)
		{
			reportfile << _loss_train_sgd_mgu_all[i1] << "," << _loss_test_sgd_mgu_all[i1] << ",";
		}
		if (_loss_train_sgd_lstm_all_tdim.size()>0 && _loss_test_sgd_lstm_all_tdim.size()>0)
		{
			reportfile << _loss_train_sgd_lstm_all_tdim[i1] << "," << _loss_test_sgd_lstm_all_tdim[i1] << ",";
		}
		// ADAM
		if (_loss_train_adam_lstm_all.size()>0 && _loss_test_adam_lstm_all.size()>0)
		{
			reportfile << _loss_train_adam_lstm_all[i1] << "," << _loss_test_adam_lstm_all[i1] << ",";
		}
		if (_loss_train_adam_gru_all.size()>0 && _loss_test_adam_gru_all.size()>0)
		{
			reportfile << _loss_train_adam_gru_all[i1] << "," << _loss_test_adam_gru_all[i1] << ",";
		}
		if (_loss_train_adam_star_all.size()>0 && _loss_test_adam_star_all.size()>0)
		{
			reportfile << _loss_train_adam_star_all[i1] << "," << _loss_test_adam_star_all[i1] << ",";
		}
		if (_loss_train_adam_mgu_all.size()>0 && _loss_test_adam_mgu_all.size()>0)
		{
			reportfile << _loss_train_adam_mgu_all[i1] << "," << _loss_test_adam_mgu_all[i1] << ",";
		}
		reportfile << i1 << "\n";
	}

	reportfile.close();
}

//---------------------------------------------------------------------
void imu_log_handler::create_time_report(const char *reportfilename)
{
	unsigned int i1,i2;
	std::ofstream reportfile;
	reportfile.open(reportfilename);
	
	// SGD
	if (_time_train_sgd_lstm_all.size()>0 && _time_test_sgd_lstm_all.size()>0)
	{
		std::string name = "min_bp_sgd_lstm_all, max_bp_sgd_lstm_all, mean_bp_sgd_lstm_all,min_ff_sgd_lstm_all, max_ff_sgd_lstm_all, mean_ff_sgd_lstm_all,";
		reportfile << name;
	}
	if (_time_train_sgd_gru_all.size()>0 && _time_test_sgd_gru_all.size()>0)
	{
		std::string name = "min_bp_sgd_gru_all, max_bp_sgd_gru_all, mean_bp_sgd_gru_all,min_ff_sgd_gru_all, max_ff_sgd_gru_all, mean_ff_sgd_gru_all,";
		reportfile << name;
	}
	if (_time_train_sgd_star_all.size()>0 && _time_test_sgd_star_all.size()>0)
	{
		std::string name = "min_bp_sgd_star_all, max_bp_sgd_star_all, mean_bp_sgd_star_all,min_ff_sgd_star_all, max_ff_sgd_star_all, mean_ff_sgd_star_all,";
		reportfile << name;
	}
	if (_time_train_sgd_mgu_all.size()>0 && _time_test_sgd_mgu_all.size()>0)
	{
		std::string name = "min_bp_sgd_mgu_all, max_bp_sgd_mgu_all, mean_bp_sgd_mgu_all,min_ff_sgd_mgu_all, max_ff_sgd_mgu_all, mean_ff_sgd_mgu_all,";
		reportfile << name;
	}
	// ADAM
	if (_time_train_adam_lstm_all.size()>0 && _time_test_adam_lstm_all.size()>0)
	{
		std::string name = "min_bp_adam_lstm_all, max_bp_adam_lstm_all, mean_bp_adam_lstm_all,min_ff_adam_lstm_all, max_ff_adam_lstm_all, mean_ff_adam_lstm_all,";
		reportfile << name;
	}
	if (_time_train_adam_gru_all.size()>0 && _time_test_adam_gru_all.size()>0)
	{
		std::string name = "min_bp_adam_gru_all, max_bp_adam_gru_all, mean_bp_adam_gru_all,min_ff_adam_gru_all, max_ff_adam_gru_all, mean_ff_adam_gru_all,";
		reportfile << name;
	}
	if (_time_train_adam_star_all.size()>0 && _time_test_adam_star_all.size()>0)
	{
		std::string name = "min_bp_adam_star_all, max_bp_adam_star_all, mean_bp_adam_star_all,min_ff_adam_star_all, max_ff_adam_star_all, mean_ff_adam_star_all,";
		reportfile << name;
	}
	if (_time_train_adam_mgu_all.size()>0 && _time_test_adam_mgu_all.size()>0)
	{
		std::string name = "min_bp_adam_mgu_all, max_bp_adam_mgu_all, mean_bp_adam_mgu_all,min_ff_adam_mgu_all, max_ff_adam_mgu_all, mean_ff_adam_mgu_all,";
		reportfile << name;
	}
	
	std::string ite_name = "epochs\n";
	reportfile << ite_name;
	
	for (i1 = 0; i1 < _time_train_sgd_lstm_all.size(); i1++)
	{
		//SGD
		if (_time_train_sgd_lstm_all.size()>0 && _time_test_sgd_lstm_all.size()>0)
		{
			reportfile << _time_train_sgd_lstm_all[i1][0] << "," << _time_train_sgd_lstm_all[i1][1] << "," << _time_train_sgd_lstm_all[i1][2] << ",";
			reportfile << _time_test_sgd_lstm_all[i1][0] << "," << _time_test_sgd_lstm_all[i1][1] << "," << _time_test_sgd_lstm_all[i1][2] << ",";
		}
		if (_time_train_sgd_gru_all.size()>0 && _time_test_sgd_gru_all.size()>0)
		{
			reportfile << _time_train_sgd_gru_all[i1][0] << "," << _time_train_sgd_gru_all[i1][1] << "," << _time_train_sgd_gru_all[i1][2] << ",";
			reportfile << _time_test_sgd_gru_all[i1][0] << "," << _time_test_sgd_gru_all[i1][1] << "," << _time_test_sgd_gru_all[i1][2] << ",";
		}
		if (_time_train_sgd_star_all.size()>0 && _time_test_sgd_star_all.size()>0)
		{
			reportfile << _time_train_sgd_star_all[i1][0] << "," << _time_train_sgd_star_all[i1][1] << "," << _time_train_sgd_star_all[i1][2] << ",";
			reportfile << _time_test_sgd_star_all[i1][0] << "," << _time_test_sgd_star_all[i1][1] << "," << _time_test_sgd_star_all[i1][2] << ",";
		}
		if (_time_train_sgd_mgu_all.size()>0 && _time_test_sgd_mgu_all.size()>0)
		{
			reportfile << _time_train_sgd_mgu_all[i1][0] << "," << _time_train_sgd_mgu_all[i1][1] << "," << _time_train_sgd_mgu_all[i1][2] << ",";
			reportfile << _time_test_sgd_mgu_all[i1][0] << "," << _time_test_sgd_mgu_all[i1][1] << "," << _time_test_sgd_mgu_all[i1][2] << ",";
		}
		// ADAM
		if (_time_train_adam_lstm_all.size()>0 && _time_test_adam_lstm_all.size()>0)
		{
			reportfile << _time_train_adam_lstm_all[i1][0] << "," << _time_train_adam_lstm_all[i1][1] << "," << _time_train_adam_lstm_all[i1][2] << ",";
			reportfile << _time_test_adam_lstm_all[i1][0] << "," << _time_test_adam_lstm_all[i1][1] << "," << _time_test_adam_lstm_all[i1][2] << ",";
		}
		if (_time_train_adam_gru_all.size()>0 && _time_test_adam_gru_all.size()>0)
		{
			reportfile << _time_train_adam_gru_all[i1][0] << "," << _time_train_adam_gru_all[i1][1] << "," << _time_train_adam_gru_all[i1][2] << ",";
			reportfile << _time_test_adam_gru_all[i1][0] << "," << _time_test_adam_gru_all[i1][1] << "," << _time_test_adam_gru_all[i1][2] << ",";
		}
		if (_time_train_adam_star_all.size()>0 && _time_test_adam_star_all.size()>0)
		{
			reportfile << _time_train_adam_star_all[i1][0] << "," << _time_train_adam_star_all[i1][1] << "," << _time_train_adam_star_all[i1][2] << ",";
			reportfile << _time_test_adam_star_all[i1][0] << "," << _time_test_adam_star_all[i1][1] << "," << _time_test_adam_star_all[i1][2] << ",";
		}
		if (_time_train_adam_mgu_all.size()>0 && _time_test_adam_mgu_all.size()>0)
		{
			reportfile << _time_train_adam_mgu_all[i1][0] << "," << _time_train_adam_mgu_all[i1][1] << "," << _time_train_adam_mgu_all[i1][2] << ",";
			reportfile << _time_test_adam_mgu_all[i1][0] << "," << _time_test_adam_mgu_all[i1][1] << "," << _time_test_adam_mgu_all[i1][2] << ",";
		}
		reportfile << i1 << "\n";
	}

	reportfile.close();
}

