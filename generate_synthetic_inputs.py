import numpy as np
import scipy.io as sio

# Map string keys to your PSF constructors
PSF_FUNCS = {
    "gaussian":          gaussian_psf,
    "kolmogorov":        kolmogorov_psf,
    "airy":              airy_psf,
    "moffat":            moffat_psf,
    "sinc":              sinc_psf,
    "lorentzian2":       lorentzian_squared_psf,
    "hermite":           hermite_psf,
    "parabolic":         parabolic_psf,
    "gabor":             gabor_psf,
    "delta":             delta_function_psf,
}

def generate_synthetic_inputs(
    mat_file: str,
    mat_key: str,
    psf_type: str,
    sigma: float,
    kernel_size: int,
    downsample_ratio: int,
    snr_spatial: float,
    num_msi_bands: int,
    snr_spectral: float,
    fwhm_factor: float,
    seed: int               = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # 1) reproducibility
    set_random_seed(seed)

    # 2) load raw HSI
    mat = sio.loadmat(mat_file)
    if mat_key not in mat:
        raise KeyError(f"Key '{mat_key}' not found in {mat_file}")
    raw_image = mat[mat_key].astype(np.float32)
    gt = normalize(raw_image)

    # 3) spectral degradation → HR MSI + SRF
    hr_msi, srf, _, _ = spectral_degradation(
        image=raw_image,
        SNR=snr_spectral,
        num_bands=num_msi_bands,
        fwhm_factor=fwhm_factor,
        user_srf=true_srf
    )

    # 4) build PSF & spatial degradation → LR HSI
    psf_fn = PSF_FUNCS.get(psf_type.lower())
    if psf_fn is None:
        raise ValueError(f"Unknown psf_type '{psf_type}'; choose one of {list(PSF_FUNCS)}")
    psf = psf_fn(sigma, kernel_size)

    lr_hsi = spatial_degradation(
        image=raw_image,
        psf=psf,
        downsample_ratio=downsample_ratio,
        SNR=snr_spatial
    )

    return gt, hr_msi, lr_hsi, srf