import os
import logging
import pprint
import time
import json

import matplotlib.pylab as plt
import numpy as np
import torch

from scipy.signal import stft, istft
from sklearn.linear_model import LinearRegression

from abc import abstractmethod
from typing import List, Tuple

from cellmincer.util import OptopatchBaseWorkspace, consts

class Preprocess:
    def __init__(
            self,
            params: dict):
        
        self.root_data_dir = params['root_data_dir']
        self.device = torch.device(params['device'])
        self.datasets = params['datasets']
        
        self.dejitter = params['dejitter']
        self.noise_estimation = params['noise_estimation']
        self.trim = params['trim']
        self.detrend = params['detrend']
        self.bfgs = params['bfgs']

    def run(self):
        
        assert all([os.path.exists(dataset['movie']) for dataset in self.datasets]), [os.path.exists(dataset['movie']) for dataset in self.datasets]

        logging.info('Creating data directories...')
        for i_dataset, dataset in enumerate(self.datasets):
            # appends config tag to data directory if tag exists
            data_dir = os.path.join(self.root_data_dir, dataset['name'])
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            logging.info(f'({i_dataset + 1}/{len(self.datasets)}) {data_dir}')

        logging.info('Preprocessing datasets...')
        for i_dataset, dataset in enumerate(self.datasets):
            logging.info(f'({i_dataset + 1}/{len(self.datasets)}) {dataset["name"]}')

            if dataset['movie'].endswith('.npy'):
                ws_base = OptopatchBaseWorkspace.from_npy(dataset['movie'], order=dataset['order'])
            elif dataset['movie'].endswith('.npz'):
                if 'key' in dataset:
                    ws_base = OptopatchBaseWorkspace.from_npz(dataset['movie'], order=dataset['order'], key=dataset['key'])
                else:
                    ws_base = OptopatchBaseWorkspace.from_npz(dataset['movie'], order=dataset['order'])
            elif dataset['movie'].endswith('.bin'):
                ws_base = OptopatchBaseWorkspace.from_bin_uint16(
                    dataset['movie'],
                    n_frames=dataset['params']['n_frames'],
                    width=dataset['params']['width'],
                    height=dataset['params']['height'],
                    order=dataset['order'])
            elif dataset['movie'].endswith('.tif'):
                ws_base = OptopatchBaseWorkspace.from_tiff(dataset['movie'], order=dataset['order'])
            else:
                logging.error('Unrecognized movie file format: options are .npy, .npz, .tiff')
                raise ValueError

            # dejitter movie
            # estimate baseline (CCD dc offset)
            # this is a heuristic -- ideally, one needs to be given this information!
            movie_txy = self.dejitter(
                movie_txy=ws_base.movie_txy,
                dataset=dataset)

            # estimate noise from dejittered movie
            noise_model_params = self.estimate_noise(
                movie_txy=movie_txy,
                dataset=dataset)

            # fit segment trends
            trimmed_segments_txy_list, mu_segments_txy_list = self.detrend(
                movie_txy=movie_txy,
                noise_model_params=noise_model_params,
                dataset=dataset)

            # save results to data directory
            data_dir = os.path.join(self.root_data_dir, dataset['name'])
            
            logging.info(f'writing output to {data_dir}/')

            trend_sub_movie_txy = np.concatenate([
                seg_txy - mu_txy
                for seg_txy, mu_txy in zip(trimmed_segments_txy_list, mu_segments_txy_list)],
                axis=0).astype(np.float32)
            trend_movie_txy = np.concatenate(mu_segments_txy_list, axis=0).astype(np.float32)

            output_file = os.path.join(data_dir, 'trend_subtracted.npy')
            np.save(output_file, trend_sub_movie_txy)

            output_file = os.path.join(data_dir, 'trend.npy')
            np.save(output_file, trend_movie_txy)

            output_file = os.path.join(data_dir, 'noise_params.json')
            with open(output_file, 'w') as f:
                json.dump(noise_model_params, f)

        logging.info('Preprocessing done.')
        
    def dejitter(
            self,
            movie_txy: np.ndarray,
            dataset: dict) -> np.ndarray:

        baseline = np.min(movie_txy[self.dejitter['ignore_first_n_frames']:, :, :])
        logging.info(f"baseline CCD dc offset estimate: {baseline:.3f}")

        log_movie_txy = np.log(np.maximum(movie_txy - baseline, 1.))
        log_movie_mean_t = log_movie_txy.mean((-1, -2))

        if self.dejitter['detrending_method'] in {'median', 'mean'}:
            log_movie_mean_trend_t = self.get_trend(
                log_movie_mean_t,
                self.dejitter['detrending_order'],
                self.dejitter['detrending_method'])

        elif self.dejitter['detrending_method'] == 'stft':
            stft_f, stft_t, stft_Zxx = stft(
                log_movie_mean_t,
                boundary='constant',
                fs=dataset['params']['sampling_rate'],
                nperseg=self.dejitter['stft_nperseg'],
                noverlap=self.dejitter['stft_noverlap'])

            jitter_freq_filter = 1. / (1 + np.exp(
                self.dejitter['stft_lp_slope'] * (stft_f - self.dejitter['stft_lp_cutoff'])))

            filtered_Zxx = stft_Zxx * jitter_freq_filter[:, None]
            _, filtered_log_movie_mean_t = istft(
                filtered_Zxx,
                fs=dataset['params']['sampling_rate'],
                nperseg=self.dejitter['stft_nperseg'],
                noverlap=self.dejitter['stft_noverlap'])

            log_movie_mean_trend_t = filtered_log_movie_mean_t[:log_movie_mean_t.size]

        else:
            raise ValueError()

        log_jitter_factor_t = log_movie_mean_t - log_movie_mean_trend_t
        dejittered_movie_txy = np.exp(log_movie_txy - log_jitter_factor_t[:, None, None]) + baseline
        
        if self.dejitter['show_diagnostic_plots']:
            fg_mask_xy = ws_base.corr_otsu_fg_pixel_mask_xy
            bg_mask_xy = ~fg_mask_xy

            # raw frame-to-frame log variations
            fg_raw_mean_t = np.mean(np.log(ws_base.movie_txy.reshape(ws_base.n_frames, -1)[
                self.dejitter['ignore_first_n_frames']:, fg_mask_xy.flatten()] - baseline), axis=-1)
            bg_raw_mean_t = np.mean(np.log(ws_base.movie_txy.reshape(ws_base.n_frames, -1)[
                self.dejitter['ignore_first_n_frames']:, bg_mask_xy.flatten()] - baseline), axis=-1)

            # de-jittered frame-to-frame log variations
            fg_dj_mean_t = np.mean(np.log(dejittered_movie_txy.reshape(ws_base.n_frames, -1)[
                self.dejitter['ignore_first_n_frames']:, fg_mask_xy.flatten()] - baseline), axis=-1)
            bg_dj_mean_t = np.mean(np.log(dejittered_movie_txy.reshape(ws_base.n_frames, -1)[
                self.dejitter['ignore_first_n_frames']:, bg_mask_xy.flatten()] - baseline), axis=-1)

            fig = plt.figure()
            ax = plt.gca()
            ax.hist(bg_raw_mean_t[1:] - bg_raw_mean_t[:-1], bins=200, range=(-0.01, 0.01), label='bg', alpha=0.5);
            ax.hist(fg_raw_mean_t[1:] - fg_raw_mean_t[:-1], bins=200, range=(-0.01, 0.01), label='fg', alpha=0.5);
            ax.set_title('BEFORE de-jittering')
            ax.set_xlabel('Frame-to-frame log intensity difference')
            ax.legend()
            
            fig.savefig(os.path.join(self.root_data_dir, dataset['name'], 'dejitter_before.png'))

            fig = plt.figure()
            ax = plt.gca()
            ax.hist(bg_dj_mean_t[1:] - bg_dj_mean_t[:-1], bins=200, range=(-0.01, 0.01), label='bg', alpha=0.5);
            ax.hist(fg_dj_mean_t[1:] - fg_dj_mean_t[:-1], bins=200, range=(-0.01, 0.01), label='fg', alpha=0.5);
            ax.set_title('AFTER de-jittering')
            ax.set_xlabel('Frame-to-frame log intensity difference')
            ax.legend()
            
            fig.savefig(os.path.join(self.root_data_dir, dataset['name'], 'dejitter_after.png'))

        return dejittered_movie_txy


    def estimate_noise(
            self,
            movie_txy: np.ndarray,
            dataset: dict) -> dict:

        slope_list = []
        intercept_list = []

        if self.noise_estimation['plot_example']:
            fig = plt.figure()
            ax = plt.gca()
            ax.set_xlabel('mean')
            ax.set_ylabel('variance')
            
            fig.savefig(os.path.join(self.root_data_dir, dataset['name'], 'noise_mean_var.png'))

        for i_bootstrap in range(self.noise_estimation['n_bootstrap']):

            # choose a random segment
            i_segment = np.random.randint(dataset['params']['n_segments'])
            t, trimmed_seg_txy = self.get_flanking_segments(movie_txy, i_segment, self.trim, dataset['params'])

            # choose a random time
            i_t = np.random.randint(0, high=len(t) - self.noise_estimation['stationarity_window'])

            # calculate empirical mean and variance, assuming signal stationarity
            mu_empirical = np.mean(trimmed_seg_txy[
                i_t:(i_t + self.noise_estimation['stationarity_window']), ...], axis=0).flatten()
            var_empirical = np.var(trimmed_seg_txy[
                i_t:(i_t + self.noise_estimation['stationarity_window']), ...], axis=0, ddof=1).flatten()

            # perform linear regression
            reg = LinearRegression().fit(mu_empirical[:, None], var_empirical[:, None])
            slope_list.append(reg.coef_.item())
            intercept_list.append(reg.intercept_.item())

            if self.noise_estimation['plot_example'] and i_bootstrap == 0:
                fit_var = reg.predict(mu_empirical[:, None])
                ax.scatter(
                    mu_empirical[::self.noise_estimation['plot_subsample']],
                    var_empirical[::self.noise_estimation['plot_subsample']],
                    s=1,
                    alpha=0.1,
                    color='black')
                ax.plot(mu_empirical, fit_var, color='red', alpha=0.1)
            
                fig.savefig(os.path.join(self.root_data_dir, dataset['name'], 'noise_reg.png'))

        alpha_median, alpha_std = np.median(slope_list), np.std(slope_list)
        beta_median, beta_std = np.median(intercept_list), np.std(intercept_list)

        # check that all variance is positive
        global_min_variance = np.inf
        for i_segment in range(dataset['params']['n_segments']):
            _, seg_txy = self.get_trimmed_segment(movie_txy, i_segment, self.trim, dataset['params'])
            min_obs_value_in_segment = np.min(seg_txy)
            min_variance = alpha_median * min_obs_value_in_segment + beta_median
            global_min_variance = min(global_min_variance, min_variance)
            logging.info(f'min variance in segment {i_segment}: {min_variance:.3f}')
            
            if min_variance > 0:
                raise ValueError('estimated negative variance; dataset may be too noisy')

        return {
            'alpha_median': alpha_median,
            'alpha_std': alpha_std,
            'beta_median': beta_median,
            'beta_std': beta_std,
            'global_min_variance': global_min_variance
        }


    def detrend(
            self,
            movie_txy: np.ndarray,
            noise_model_params: dict,
            dataset: dict) -> Tuple[List, List]:
        # trimmed segments of the movie
        trimmed_segments_txy_list = []

        # background activity fits
        mu_segments_txy_list = []

        for i_segment in range(dataset['params']['n_segments']):
            # get segment for fitting
            t_fit, fit_seg_txy = self.get_flanking_segments(movie_txy, i_segment, self.trim, dataset['params'])
            t_fit_torch = torch.tensor(t_fit, device=self.device, dtype=consts.DEFAULT_DTYPE)
            fit_seg_txy_torch = torch.tensor(fit_seg_txy, device=self.device, dtype=consts.DEFAULT_DTYPE)
            width, height = fit_seg_txy_torch.shape[1:]

            if self.detrend['trend_model'] == 'polynomial':
                trend_model = PolynomialIntensityTrendModel(
                    t_fit_torch=t_fit_torch,
                    fit_seg_txy_torch=fit_seg_txy_torch,
                    poly_order=self.detrend['poly_order'],
                    device=self.device,
                    dtype=consts.DEFAULT_DTYPE)
            elif self.detrend['trend_model'] == 'exponential':
                trend_model = ExponentialDecayIntensityTrendModel(
                    t_fit=t_fit,
                    fit_seg_txy=fit_seg_txy,
                    init_unc_decay_rate=self.detrend['init_unc_decay_rate'],
                    device=self.device,
                    dtype=consts.DEFAULT_DTYPE)
            else:
                raise ValueError()

            # fit 
            optim = torch.optim.LBFGS(trend_model.parameters(), **self.bfgs)

            def closure():
                optim.zero_grad()
                mu_txy = trend_model.get_baseline_txy(t_fit_torch)
                var_txy = torch.clamp(
                    noise_model_params['alpha_median'] * mu_txy + noise_model_params['beta_median'],
                    min=noise_model_params['global_min_variance'])
                loss_txy = 0.5 * (fit_seg_txy_torch - mu_txy).pow(2) / var_txy + 0.5 * var_txy.log()
                loss = loss_txy.sum()
                loss.backward()
                return loss

            for i_iter in range(self.detrend['max_iters_per_segment']):
                loss = optim.step(closure).item()

            logging.info(f'detrended segment {i_segment + 1}/{dataset["params"]["n_segments"]} | loss = {loss / (width * height * len(t_fit)):.6f}')

            t_trimmed, trimmed_seg_txy = self.get_trimmed_segment(movie_txy, i_segment, self.trim, dataset['params'])
            t_trimmed_torch = torch.tensor(t_trimmed, device=self.device, dtype=consts.DEFAULT_DTYPE)
            mu_txy = trend_model.get_baseline_txy(t_trimmed_torch).detach().cpu().numpy()

            if self.detrend['plot_segments']:
                fig = plt.figure()
                ax = plt.gca()
                ax.scatter(t_trimmed, np.mean(trimmed_seg_txy, axis=(-1, -2)), s=1)
                ax.scatter(t_trimmed, np.mean(mu_txy, axis=(-1, -2)), s=1)
                ax.set_title(f'segment {i_segment + 1}')
                
                fig.savefig(os.path.join(self.root_data_dir, dataset['name'], f'detrend_{i_segment + 1}.png'))

            # store
            trimmed_segments_txy_list.append(trimmed_seg_txy)
            mu_segments_txy_list.append(mu_txy)

        return trimmed_segments_txy_list, mu_segments_txy_list


    def get_trend(
            self,
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
            self,
            movie_txy: np.ndarray,
            i_stim: int,
            ds_params: dict,
            tranform_time: bool = True,) -> np.ndarray:
        i_t_begin = ds_params['n_frames_per_segment'] * i_stim + self.trim['trim_left']
        i_t_end = ds_params['n_frames_per_segment'] * (i_stim + 1) - self.trim['trim_right']
        i_t_list = [i_t for i_t in range(i_t_begin, i_t_end)]
        if tranform_time:
            t = np.asarray([i_t - i_t_begin for i_t in i_t_list]) / ds_params['sampling_rate']
        else:
            t = i_t_list
        return t, movie_txy[i_t_begin:i_t_end, ...]

    def get_flanking_segments(
            self,
            movie_txy: np.ndarray,
            i_stim: int,
            ds_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        t_begin_left = (
            ds_params['n_frames_per_segment'] * i_stim
            + self.trim['trim_left'])
        t_end_left = t_begin_left + self.trim['n_frames_fit_left']

        t_begin_right = (
            ds_params['n_frames_per_segment'] * (i_stim + 1)
            - self.trim['trim_right']
            - self.trim['n_frames_fit_right'])
        t_end_right = t_begin_right + self.trim['n_frames_fit_right']

        i_t_list = (
            [i_t for i_t in range(t_begin_left, t_end_left)]
            + [i_t for i_t in range(t_begin_right, t_end_right)])

        t = np.asarray([i_t - t_begin_left for i_t in i_t_list]) / ds_params['sampling_rate']

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
                fit_seg_txy.shape[1:], dtype=consts.DEFAULT_DTYPE, device=device))
        
        init_decay_rate = self.pos_trans(torch.tensor(init_unc_decay_rate)).item()
        before_stim_mean_xy = np.mean(fit_seg_txy[:len(t_fit)//2, ...], 0)
        after_stim_mean_xy = np.mean(fit_seg_txy[len(t_fit)//2:, ...], 0)
        t_0 = np.mean(t_fit[:len(t_fit)//2])
        t_1 = np.mean(t_fit[len(t_fit)//2:])
        
        a_xy = torch.tensor(
            (before_stim_mean_xy - after_stim_mean_xy) / (
                np.exp(-init_decay_rate * t_0) - np.exp(-init_decay_rate * t_1)),
            dtype=consts.DEFAULT_DTYPE, device=device)
        b_xy = (torch.tensor(before_stim_mean_xy, dtype=consts.DEFAULT_DTYPE, device=device)
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
        assert poly_order >= 0
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
