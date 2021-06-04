import os
import yaml

import matplotlib.pylab as plt
import numpy as np
import torch
import logging

from cellmincer.opto_ws import OptopatchBaseWorkspace
from cellmincer.opto_features import OptopatchGlobalFeatureExtractor

from scipy.signal import stft, istft
from sklearn.linear_model import LinearRegression

from abc import abstractmethod
from typing import Tuple

dtype = torch.float32


def preprocess(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    device = torch.device(config['device'])
    
    # load datasets
    dataset_names = config['datasets']['dataset_names']
    dataset_movie_paths = config['datasets']['dataset_movie_paths']
    dataset_params_paths = config['datasets']['dataset_params_paths']
    
    assert len(dataset_names) == len(dataset_movie_paths) == len(dataset_params_paths)
    assert all([os.path.exists(dataset_path) for dataset_path in dataset_paths])
    
    n_datasets = len(dataset_names)
    
    for i_dataset in range(n_datasets):
        params_path = os.path.join(dataset_params_paths[i_dataset])

        with open(params_path, 'r') as stream:
            try:
                params_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        ws_base = OptopatchBaseWorkspace.from_npz(dataset_movie_paths[i_dataset])
        
        
        # dejitter movie
        # estimate baseline (CCD dc offset)
        # this is a heuristic -- ideally, one needs to be given this information!
        baseline = np.min(ws_base.movie_txy[config['dejitter']['ignore_first_n_frames']:, :, :])
        print(f"Baseline CCD dc offset was estimated to be: {baseline:.3f}")

        log_movie_txy = np.log(np.maximum(ws_base.movie_txy - baseline, 1.))
        log_movie_mean_t = log_movie_txy.mean((-1, -2))

        if config['dejitter']['detrending_method'] in {'median', 'mean'}:

            log_movie_mean_trend_t = get_trend(
                log_movie_mean_t,
                config['dejitter']['detrending_order'],
                config['dejitter']['detrending_method'])

        elif config['dejitter']['detrending_method'] == 'stft':

            stft_f, stft_t, stft_Zxx = stft(
                log_movie_mean_t,
                boundary='constant',
                fs=config['dejitter']['sampling_rate'],
                nperseg=config['dejitter']['stft_nperseg'],
                noverlap=config['dejitter']['stft_noverlap'])

            jitter_freq_filter = 1. / (1 + np.exp(
                config['dejitter']['stft_lp_slope'] * (stft_f - config['dejitter']['stft_lp_cutoff'])))

            filtered_Zxx = stft_Zxx * jitter_freq_filter[:, None]
            _, filtered_log_movie_mean_t = istft(
                filtered_Zxx,
                fs=config['dejitter']['sampling_rate'],
                nperseg=config['dejitter']['stft_nperseg'],
                noverlap=config['dejitter']['stft_noverlap'])

            log_movie_mean_trend_t = filtered_log_movie_mean_t[:log_movie_mean_t.size]

        else:

            raise ValueError()

        log_jitter_factor_t = log_movie_mean_t - log_movie_mean_trend_t
        dejittered_movie_txy = np.exp(log_movie_txy - log_jitter_factor_t[:, None, None]) + baseline
        
        
        # estimate noise from dejittered movie
        movie_txy = dejittered_movie_txy

        slope_list = []
        intercept_list = []

        if noise_estimation_config['plot_example']:
            fig = plt.figure()
            ax = plt.gca()
            ax.set_xlabel('mean')
            ax.set_ylabel('variance')

        for i_bootstrap in range(noise_estimation_config['n_bootstrap']):

            # choose a random segment
            i_segment = np.random.randint(trim_config['n_segments'])
            t, trimmed_seg_txy = get_flanking_segments(movie_txy, i_segment, trim_config)

            # choose a random time
            i_t = np.random.randint(0, high=len(t) - noise_estimation_config['stationarity_window'])

            # calculate empirical mean and variance, assuming signal stationarity
            mu_empirical = np.mean(trimmed_seg_txy[
                i_t:(i_t + noise_estimation_config['stationarity_window']), ...], axis=0).flatten()
            var_empirical = np.var(trimmed_seg_txy[
                i_t:(i_t + noise_estimation_config['stationarity_window']), ...], axis=0, ddof=1).flatten()

            # perform linear regression
            reg = LinearRegression().fit(mu_empirical[:, None], var_empirical[:, None])
            slope_list.append(reg.coef_.item())
            intercept_list.append(reg.intercept_.item())

            if noise_estimation_config['plot_example']:
                fit_var = reg.predict(mu_empirical[:, None])
                ax.scatter(
                    mu_empirical[::noise_estimation_config['plot_subsample']],
                    var_empirical[::noise_estimation_config['plot_subsample']],
                    s=1,
                    alpha=0.1,
                    color='black')
                ax.plot(mu_empirical, fit_var, color='red', alpha=0.1)
        

def get_trend(
        series_t: np.ndarray,
        detrending_order: int,
        detrending_func: str) -> np.ndarray:
    assert detrending_order > 0
    assert detrending_func in {'mean', 'median'}
    detrending_func_map = {'mean': np.mean, 'median': np.median}
    detrending_func = detrending_func_map[detrending_func]
    
    # reflection pad in time
    padded_series_t = np.pad(
        array=series_t,
        pad_width=((detrending_order, detrending_order)),
        mode='reflect')
    
    trend_series_t = np.zeros_like(series_t)

    # calculate temporal moving average
    for i_t in range(series_t.size):
        trend_series_t[i_t] = detrending_func(padded_series_t[i_t:(i_t + 2 * detrending_order + 1)])

    return trend_series_t


def get_trimmed_segment(
        movie_txy: np.ndarray,
        i_stim: int,
        trim_config: dict,
        tranform_time: bool = True) -> np.ndarray:
    i_t_begin = trim_config['n_frames_total'] * i_stim + trim_config['trim_left']
    i_t_end = trim_config['n_frames_total'] * (i_stim + 1) - trim_config['trim_right']
    i_t_list = [i_t for i_t in range(i_t_begin, i_t_end)]
    if tranform_time:
        t = np.asarray([i_t - i_t_begin for i_t in i_t_list]) / trim_config['sampling_rate']
    else:
        t = i_t_list
    return t, movie_txy[i_t_begin:i_t_end, ...]

def get_flanking_segments(
        movie_txy: np.ndarray,
        i_stim: int,
        trim_config: dict) -> Tuple[np.ndarray, np.ndarray]:
    t_begin_left = (
        trim_config['n_frames_total'] * i_stim
        + trim_config['trim_left'])
    t_end_left = t_begin_left + trim_config['n_frames_fit_left']
    
    t_begin_right = (
        trim_config['n_frames_total'] * (i_stim + 1)
        - trim_config['trim_right']
        - trim_config['n_frames_fit_right'])
    t_end_right = t_begin_right + trim_config['n_frames_fit_right']
    
    i_t_list = (
        [i_t for i_t in range(t_begin_left, t_end_left)]
        + [i_t for i_t in range(t_begin_right, t_end_right)])
    
    t = np.asarray([i_t - t_begin_left for i_t in i_t_list]) / trim_config['sampling_rate']
    
    return t, movie_txy[i_t_list, ...]


class IntensityTrendModel:
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def get_baseline_txy(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ExponentialDecayIntensityTrendModel(IntensityTrendModel):
    def __init__(self,
                 t_fit: np.ndarray,
                 fit_seg_txy: np.ndarray,
                 init_unc_decay_rate: float,
                 device: torch.device,
                 dtype: torch.dtype):
        
        # initialize
        self.pos_trans = torch.nn.Softplus()
        self.unc_decay_rate_xy = torch.nn.Parameter(
            init_unc_decay_rate * torch.ones(
                fit_seg_txy.shape[1:], dtype=dtype, device=device))
        
        init_decay_rate = self.pos_trans(torch.tensor(init_unc_decay_rate)).item()
        before_stim_mean_xy = np.mean(fit_seg_txy[:len(t_fit)//2, ...], 0)
        after_stim_mean_xy = np.mean(fit_seg_txy[len(t_fit)//2:, ...], 0)
        t_0 = np.mean(t_fit[:len(t_fit)//2])
        t_1 = np.mean(t_fit[len(t_fit)//2:])
        
        a_xy = torch.tensor(
            (before_stim_mean_xy - after_stim_mean_xy) / (
                np.exp(-init_decay_rate * t_0) - np.exp(-init_decay_rate * t_1)),
            dtype=dtype, device=device)
        b_xy = (torch.tensor(before_stim_mean_xy, dtype=dtype, device=device)
                - np.exp(-init_decay_rate * t_0) * a_xy)
        
        self.a_xy = torch.nn.Parameter(a_xy)
        self.b_xy = torch.nn.Parameter(b_xy)
        
        
    def parameters(self):
        return [self.unc_decay_rate_xy, self.a_xy, self.b_xy]
    
    def get_baseline_txy(self, t: torch.Tensor) -> torch.Tensor:
        decay_rate_xy = self.pos_trans(self.unc_decay_rate_xy)
        return (self.a_xy[None, :, :] * torch.exp(- decay_rate_xy[None, :, :] * t[:, None, None]) +
                self.b_xy[None, :, :])

    
class PolynomialIntensityTrendModel(IntensityTrendModel):
    def __init__(self,
                 t_fit_torch: torch.Tensor,
                 fit_seg_txy_torch: torch.Tensor,
                 poly_order: int,
                 device: torch.device,
                 dtype: torch.dtype):
        assert poly_order >= 1
        self.n_series = torch.arange(0, poly_order + 1, device=device, dtype=torch.int64)

        # initialize to standard linear regression
        tn = t_fit_torch[:, None].pow(self.n_series)
        t_prec_nn = torch.mm(tn.t(), tn).inverse()
        an_xy = torch.einsum('mn,nt,txy->mxy', t_prec_nn, tn.t(), fit_seg_txy_torch).type(dtype)
        
        self.an_xy = torch.nn.Parameter(an_xy)
        
    def parameters(self):
        return [self.an_xy]
    
    def get_baseline_txy(self, t: torch.Tensor) -> torch.Tensor:
        tn = t[:, None].pow(self.n_series)
        return torch.einsum('tn,nxy->txy', tn, self.an_xy)
    
    def get_dd_baseline_txy(self, t: torch.Tensor) -> torch.Tensor:
        n = self.n_series.float()
        pref_n = n * (n - 1)
        dd_tn = pref_n[2:] * t[:, None].pow(self.n_series[:-2])
        return torch.einsum('tn,nxy->txy', dd_tn, self.an_xy[2:])