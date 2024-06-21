import os
import logging
import json
import pickle

import matplotlib.pylab as plt
import numpy as np
import torch

from scipy.signal import stft, istft
from sklearn.linear_model import LinearRegression

from abc import abstractmethod
from typing import List, Tuple

from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor, const

class Preprocess:
    def __init__(
            self,
            input_file: str,
            output_dir: str,
            manifest: dict,
            config: dict):
        '''
        Preprocessing CLI class.
        
        :param input_file: Path to raw dataset.
        :param output_dir: Path to directory for collecting dataset processing.
        :param manifest: Dictionary containing specifying parameters of dataset.
        :param config: Preprocessing config dictionary.
        '''
        self.ws_base = Preprocess.ws_base_from_input_manifest(
            input_file=input_file,
            manifest=manifest)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.n_frames_per_segment = manifest['n_frames_per_segment']
        self.n_segments = manifest['n_segments']
        self.sampling_rate = manifest['sampling_rate']
        self.infer_active_t_range = manifest.get('infer_active_t_range', False)
        self.stim = manifest.get('stim')
        
        self.dejitter_config = config['dejitter']
        self.ne_config = config['noise_estimation']
        self.trim = config['trim']
        self.detrend_config = config['detrend']
        self.bfgs = config['bfgs']
        self.device = torch.device(config['device'])
        self.feature_depth = config['feature_depth']
        
        if self.dejitter_config['show_diagnostic_plots'] or self.ne_config['plot_example'] or self.detrend_config['plot_segments']:
            self.plot_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(self.plot_dir, exist_ok=True)

    def run(self):
        movie_txy = self.ws_base.movie_txy
        
        # dejitter movie
        if self.dejitter_config['enabled']:
            movie_txy = self.dejitter(movie_txy)

        # estimate noise from dejittered movie
        noise_model_params = self.estimate_noise(movie_txy)

        # fit segment trends
        trimmed_segments_txy_list, mu_segments_txy_list = self.detrend(movie_txy, noise_model_params)

        trend_sub_movie_txy = np.concatenate([
            seg_txy - mu_txy
            for seg_txy, mu_txy in zip(trimmed_segments_txy_list, mu_segments_txy_list)],
            axis=0).astype(np.float32)
        trend_movie_txy = np.concatenate(mu_segments_txy_list, axis=0).astype(np.float32)

        # precompute global features
        feature_extractor = self.featurize(trend_sub_movie_txy)

        # save results to output directory
        logging.info(f'Writing output to {self.output_dir}')

        output_file = os.path.join(self.output_dir, 'trend_subtracted.npy')
        np.save(output_file, trend_sub_movie_txy)

        output_file = os.path.join(self.output_dir, 'trend.npy')
        np.save(output_file, trend_movie_txy)

        output_file = os.path.join(self.output_dir, 'noise_params.json')
        with open(output_file, 'w') as f:
            json.dump(noise_model_params, f)

        output_file = os.path.join(self.output_dir, 'features.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(feature_extractor.features, f)

        logging.info('Preprocessing done.')
        
    def dejitter(
            self,
            movie_txy: np.ndarray,
            plot_bins: int = 200,
            plot_range: Tuple[int] = (-0.01, 0.01)) -> np.ndarray:
        logging.info('Dejittering movie...')
        
        baseline = self.dejitter_config.get(
            'ccd_dc_offset',
            np.min(self.ws_base.movie_txy[self.dejitter_config['ignore_first_n_frames']:, :, :]))
        logging.info(f"baseline CCD dc offset: {baseline:.3f}")
        
        log_movie_txy = np.log(np.maximum(movie_txy - baseline + const.EPS, 1.))
        log_movie_mean_t = log_movie_txy.mean((-1, -2))

        if self.dejitter_config['detrending_method'] in {'median', 'mean'}:
            log_movie_mean_trend_t = self.get_trend(
                log_movie_mean_t,
                self.dejitter_config['detrending_order'],
                self.dejitter_config['detrending_method'])

        elif self.dejitter_config['detrending_method'] == 'stft':
            stft_f, stft_t, stft_Zxx = stft(
                log_movie_mean_t,
                boundary='constant',
                fs=self.sampling_rate,
                nperseg=self.dejitter_config['stft_nperseg'],
                noverlap=self.dejitter_config['stft_noverlap'])

            jitter_freq_filter = 1. / (1 + np.exp(
                self.dejitter_config['stft_lp_slope'] * (stft_f - self.dejitter_config['stft_lp_cutoff'])))

            filtered_Zxx = stft_Zxx * jitter_freq_filter[:, None]
            _, filtered_log_movie_mean_t = istft(
                filtered_Zxx,
                fs=self.sampling_rate,
                nperseg=self.dejitter_config['stft_nperseg'],
                noverlap=self.dejitter_config['stft_noverlap'])

            log_movie_mean_trend_t = filtered_log_movie_mean_t[:log_movie_mean_t.size]

        else:
            raise ValueError()

        log_jitter_factor_t = log_movie_mean_t - log_movie_mean_trend_t
        dejittered_movie_txy = np.exp(log_movie_txy - log_jitter_factor_t[:, None, None]) + baseline
        
        if self.dejitter_config['show_diagnostic_plots']:
            fg_mask_xy = self.ws_base.corr_otsu_fg_pixel_mask_xy
            bg_mask_xy = ~fg_mask_xy

            # raw frame-to-frame log variations
            fg_raw_mean_t = np.mean(np.log(movie_txy.reshape(self.ws_base.n_frames, -1)[
                self.dejitter_config['ignore_first_n_frames']:, fg_mask_xy.flatten()] - baseline + const.EPS), axis=-1)
            bg_raw_mean_t = np.mean(np.log(movie_txy.reshape(self.ws_base.n_frames, -1)[
                self.dejitter_config['ignore_first_n_frames']:, bg_mask_xy.flatten()] - baseline + const.EPS), axis=-1)

            # de-jittered frame-to-frame log variations
            fg_dj_mean_t = np.mean(np.log(dejittered_movie_txy.reshape(self.ws_base.n_frames, -1)[
                self.dejitter_config['ignore_first_n_frames']:, fg_mask_xy.flatten()] - baseline + const.EPS), axis=-1)
            bg_dj_mean_t = np.mean(np.log(dejittered_movie_txy.reshape(self.ws_base.n_frames, -1)[
                self.dejitter_config['ignore_first_n_frames']:, bg_mask_xy.flatten()] - baseline + const.EPS), axis=-1)

            fig = plt.figure()
            ax = plt.gca()
            ax.hist(bg_raw_mean_t[1:] - bg_raw_mean_t[:-1], bins=plot_bins, range=plot_range, label='bg', alpha=0.5)
            ax.hist(fg_raw_mean_t[1:] - fg_raw_mean_t[:-1], bins=plot_bins, range=plot_range, label='fg', alpha=0.5)
            ax.set_title('BEFORE de-jittering')
            ax.set_xlabel('Frame-to-frame log intensity difference')
            ax.legend()
            
            fig.savefig(os.path.join(self.plot_dir, 'dejitter_before.png'))

            fig = plt.figure()
            ax = plt.gca()
            ax.hist(bg_dj_mean_t[1:] - bg_dj_mean_t[:-1], bins=plot_bins, range=plot_range, label='bg', alpha=0.5)
            ax.hist(fg_dj_mean_t[1:] - fg_dj_mean_t[:-1], bins=plot_bins, range=plot_range, label='fg', alpha=0.5)
            ax.set_title('AFTER de-jittering')
            ax.set_xlabel('Frame-to-frame log intensity difference')
            ax.legend()
            
            fig.savefig(os.path.join(self.plot_dir, 'dejitter_after.png'))

        return dejittered_movie_txy


    def estimate_noise(self, movie_txy: np.ndarray) -> dict:

        if self.ne_config['plot_example']:
            fig = plt.figure()
            ax = plt.gca()
            ax.set_xlabel('mean')
            ax.set_ylabel('variance')

        logging.info('Estimating noise...')
        slope_list = []
        intercept_list = []

        min_var_empirical = np.inf
        
        for i_bootstrap in range(self.ne_config['n_bootstrap']):

            # choose a random segment
            i_segment = np.random.randint(self.n_segments)
            t, trimmed_seg_txy = self.get_flanking_segments(movie_txy, i_segment)

            # choose a random time
            i_t = np.random.randint(0, high=len(t) - self.ne_config['stationarity_window'])

            # calculate empirical mean and variance, assuming signal stationarity
            mu_empirical = np.mean(trimmed_seg_txy[
                i_t:(i_t + self.ne_config['stationarity_window']), ...], axis=0).flatten()
            var_empirical = np.var(trimmed_seg_txy[
                i_t:(i_t + self.ne_config['stationarity_window']), ...], axis=0, ddof=1).flatten()
            min_var_empirical = min(min_var_empirical, var_empirical.min())

            # perform linear regression
            reg = LinearRegression().fit(mu_empirical[:, None], var_empirical[:, None])
            slope_list.append(reg.coef_.item())
            intercept_list.append(reg.intercept_.item())

            if self.ne_config['plot_example'] and i_bootstrap == 0:
                fit_var = reg.predict(mu_empirical[:, None])
                ax.scatter(
                    mu_empirical[::self.ne_config['plot_subsample']],
                    var_empirical[::self.ne_config['plot_subsample']],
                    s=1,
                    alpha=0.1,
                    color='black')
                ax.plot(mu_empirical, fit_var, color='red', alpha=0.1)
            
                fig.savefig(os.path.join(self.plot_dir, 'noise_reg.png'))

        alpha_median, alpha_std = np.median(slope_list), np.std(slope_list)
        beta_median, beta_std = np.median(intercept_list), np.std(intercept_list)

        # ensure positive minimum variance
        for i_segment in range(self.n_segments):
            _, seg_txy = self.get_trimmed_segment(movie_txy, i_segment)
            global_min_variance = np.maximum(
                alpha_median * np.min(seg_txy) + beta_median,
                min_var_empirical).astype('float64')
            logging.info(f'min variance in segment {i_segment}: {global_min_variance:.3f}')

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
            noise_model_params: dict) -> Tuple[List, List]:
        
        logging.info('Detrending movie...')

        assert self.detrend_config['smoothing'] in ('none', 'median')

        if self.detrend_config['smoothing'] == 'median':
            fit_movie_txy = self.apply_median_smoothing(movie_txy)
        elif self.detrend_config['smoothing'] == 'none':
            fit_movie_txy = movie_txy
        
        # time coordinates of segments
        t_trimmed_list = []
        
        # trimmed segments of the movie
        trimmed_segments_txy_list = []

        # background activity fits
        mu_segments_txy_list = []
        
        for i_segment in range(self.n_segments):
            # get segment for fitting
            t_fit, fit_seg_txy = self.get_flanking_segments(fit_movie_txy, i_segment)

            t_fit_torch = torch.tensor(t_fit, device=self.device, dtype=const.DEFAULT_DTYPE)
            fit_seg_txy_torch = torch.tensor(fit_seg_txy, device=self.device, dtype=const.DEFAULT_DTYPE)
            width, height = fit_seg_txy_torch.shape[1:]

            if self.detrend_config['trend_model'] == 'polynomial':
                trend_model = PolynomialIntensityTrendModel(
                    t_fit_torch=t_fit_torch,
                    fit_seg_txy_torch=fit_seg_txy_torch,
                    poly_order=self.detrend_config['poly_order'],
                    device=self.device,
                    dtype=const.DEFAULT_DTYPE)
            elif self.detrend_config['trend_model'] == 'exponential':
                trend_model = ExponentialDecayIntensityTrendModel(
                    t_fit=t_fit,
                    fit_seg_txy=fit_seg_txy,
                    init_unc_decay_rate=self.detrend_config['init_unc_decay_rate'],
                    device=self.device,
                    dtype=const.DEFAULT_DTYPE)
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

            for i_iter in range(self.detrend_config['max_iters_per_segment']):
                loss = optim.step(closure).item()

            logging.info(f'detrended segment {i_segment + 1}/{self.n_segments} | loss = {loss / (width * height * len(t_fit)):.6f}')

            t_trimmed, trimmed_seg_txy = self.get_trimmed_segment(movie_txy, i_segment)
            t_trimmed_torch = torch.tensor(t_trimmed, device=self.device, dtype=const.DEFAULT_DTYPE)
            mu_txy = trend_model.get_baseline_txy(t_trimmed_torch).detach().cpu().numpy()
            
            if self.detrend_config['plot_segments']:
                fig = plt.figure()
                ax = plt.gca()
                ax.scatter(t_trimmed, np.mean(trimmed_seg_txy, axis=(-1, -2)), s=1, label='trimmed')
                ax.scatter(t_fit, np.mean(fit_seg_txy, axis=(-1, -2)), s=1, label='fit')
                ax.scatter(t_trimmed, np.mean(mu_txy, axis=(-1, -2)), s=1, label='trend')
                ax.legend()
                ax.set_title(f'segment {i_segment + 1}')

                fig.savefig(os.path.join(self.plot_dir, f'detrend_{i_segment + 1}.png'))

            # store
            t_trimmed_list.append(t_trimmed)
            trimmed_segments_txy_list.append(trimmed_seg_txy)
            mu_segments_txy_list.append(mu_txy)

        return trimmed_segments_txy_list, mu_segments_txy_list


    def featurize(self, movie_txy: np.ndarray) -> OptopatchGlobalFeatureExtractor:
        # compute active range mask if stim params provided
        active_mask = None
        if self.stim:
            active_mask = np.zeros(self.n_frames_per_segment * self.n_segments, dtype=bool)
            for i_seg in range(self.stim['segment_start'], self.stim['segment_end']):
                active_mask[i_seg * self.n_frames_per_segment + self.stim['frame_start']:
                     i_seg * self.n_frames_per_segment + self.stim['frame_end']] = 1
            active_mask = np.concatenate([
                active_mask[i_seg * self.n_frames_per_segment + self.trim['trim_left']:
                     (i_seg + 1) * self.n_frames_per_segment - self.trim['trim_right']]
                for i_seg in range(self.n_segments)])
        elif not self.infer_active_t_range:
            active_mask = np.ones((movie_txy.shape[0],), dtype=bool)

        logging.info('Extracting features...')
        feature_extractor = OptopatchGlobalFeatureExtractor(
            movie_txy=movie_txy,
            active_mask=active_mask,
            max_depth=self.feature_depth)

        return feature_extractor


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
            transform_time: bool = True) -> np.ndarray:
        i_t_begin = self.n_frames_per_segment * i_stim + self.trim['trim_left']
        i_t_end = self.n_frames_per_segment * (i_stim + 1) - self.trim['trim_right']
        i_t_list = [i_t for i_t in range(i_t_begin, i_t_end)]
        if transform_time:
            t = np.asarray([i_t - i_t_begin for i_t in i_t_list]) / self.sampling_rate
        else:
            t = i_t_list
        return t, movie_txy[i_t_begin:i_t_end, ...]

    def get_flanking_segments(
            self,
            movie_txy: np.ndarray,
            i_stim: int) -> Tuple[np.ndarray, np.ndarray]:
        t_begin_left = (
            self.n_frames_per_segment * i_stim
            + self.trim['trim_left'])
        t_end_left = t_begin_left + self.trim['n_frames_fit_left']

        t_begin_right = (
            self.n_frames_per_segment * (i_stim + 1)
            - self.trim['trim_right']
            - self.trim['n_frames_fit_right'])
        t_end_right = t_begin_right + self.trim['n_frames_fit_right']

        i_t_list = (
            [i_t for i_t in range(t_begin_left, t_end_left)]
            + [i_t for i_t in range(t_begin_right, t_end_right)])

        t = np.asarray([i_t - t_begin_left for i_t in i_t_list]) / self.sampling_rate

        return t, movie_txy[i_t_list, ...]
    
    def apply_median_smoothing(
            self,
            movie_txy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # force window length to be odd
        window_length = self.detrend_config['smoothing_window']
        if window_length % 2 == 0:
            window_length += 1
        pad_length = window_length // 2
        
        padded_movie_txy = np.pad(movie_txy, ((pad_length, pad_length), (0, 0), (0, 0)), 'edge')
        median_txy = np.stack([np.median(padded_movie_txy[i:i + window_length], axis=0) for i in range(movie_txy.shape[0])], axis=0)
        
        return median_txy
    
    @staticmethod
    def ws_base_from_input_manifest(
            input_file: str,
            manifest: dict) -> OptopatchBaseWorkspace:
        if input_file.endswith('.npy'):
            ws_base = OptopatchBaseWorkspace.from_npy(input_file, order=manifest['order'])
        elif input_file.endswith('.npz'):
            if 'key' in manifest:
                ws_base = OptopatchBaseWorkspace.from_npz(input_file, order=manifest['order'], key=manifest['key'])
            else:
                ws_base = OptopatchBaseWorkspace.from_npz(input_file, order=manifest['order'])
        elif input_file.endswith('.bin'):
            ws_base = OptopatchBaseWorkspace.from_bin_uint16(
                input_file,
                n_frames=manifest.get('n_frames', manifest['n_frames_per_segment'] * manifest['n_segments']),
                width=manifest['width'],
                height=manifest['height'],
                order=manifest['order'])
        elif input_file.endswith('.tif'):
            ws_base = OptopatchBaseWorkspace.from_tiff(input_file, order=manifest['order'])
        else:
            logging.error('Unrecognized movie file format: options are .npy, .npz, .bin, .tif')
            raise ValueError
            
        return ws_base

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
                fit_seg_txy.shape[1:], dtype=const.DEFAULT_DTYPE, device=device))
        
        init_decay_rate = self.pos_trans(torch.tensor(init_unc_decay_rate)).item()
        before_stim_mean_xy = np.mean(fit_seg_txy[:len(t_fit)//2, ...], 0)
        after_stim_mean_xy = np.mean(fit_seg_txy[len(t_fit)//2:, ...], 0)
        t_0 = np.mean(t_fit[:len(t_fit)//2])
        t_1 = np.mean(t_fit[len(t_fit)//2:])
        
        a_xy = torch.tensor(
            (before_stim_mean_xy - after_stim_mean_xy) / (
                np.exp(-init_decay_rate * t_0) - np.exp(-init_decay_rate * t_1)),
            dtype=const.DEFAULT_DTYPE, device=device)
        b_xy = (torch.tensor(before_stim_mean_xy, dtype=const.DEFAULT_DTYPE, device=device)
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
