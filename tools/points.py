from tools.physics import (
    sample_spherical_points, sample_luminosity_function, sample_spectral_index,
    aberrate_points, boost_fluxes
)
from torch.types import Tensor
from torch import normal

def sample_points_with_flux(
        n_initial_points: int,
        luminosity_function_index: int = 2,
        mean_spectral_index: float = 0.8,
        sigma_spectral_index: float = 0.5
    ) -> tuple[Tensor]:
    '''
    :return: Tuple of longitudes, latitudes, fluxes and spectral indices.
    '''
    longitudes_deg, latitudes_deg = sample_spherical_points(
        n_points=n_initial_points
    )
    rest_fluxes = sample_luminosity_function(
        minimum_flux=1,
        index=luminosity_function_index,
        n_fluxes=n_initial_points
    )
    spectral_indices = sample_spectral_index(
        n_points=n_initial_points,
        mean_index=mean_spectral_index,
        sigma_index=sigma_spectral_index
    )
    return longitudes_deg, latitudes_deg, rest_fluxes, spectral_indices

def boost_points_with_flux(
        longitudes_deg: Tensor,
        latitudes_deg: Tensor,
        rest_fluxes: Tensor,
        spectral_indices: Tensor,
        observer_direction: tuple[float],
        observer_speed: float
    ) -> tuple[Tensor]:
    '''
    :return: Tuple of boosted longitudes, latitudes and fluxes.
    '''
    dipole_longitude, dipole_latitude = observer_direction

    boosted_longitudes_deg, boosted_latitudes_deg,\
    rest_source_to_dipole_angle = aberrate_points(
        rest_longitudes=longitudes_deg,
        rest_latitudes=latitudes_deg,
        observer_direction=(dipole_longitude, dipole_latitude),
        observer_speed=observer_speed
    )

    boosted_fluxes = boost_fluxes(
        fluxes=rest_fluxes,
        angle_to_source=rest_source_to_dipole_angle,
        observer_speed=observer_speed,
        spectral_index=spectral_indices
    )

    return boosted_longitudes_deg, boosted_latitudes_deg, boosted_fluxes

def flux_cut(
        minimum_flux: float,
        longitudes: Tensor,
        latitudes: Tensor,
        fluxes: Tensor
    ) -> tuple[Tensor]:
    cut = fluxes > minimum_flux
    return longitudes[cut], latitudes[cut], fluxes[cut]

def add_noise_to_fluxes(
        fluxes: Tensor,
        noise_model: float | str,
        noise_scaling_parameter: None | Tensor = None,
        **noise_model_kwargs
    ) -> Tensor:
    if type(noise_model) is float:
        return fluxes + normal(
            mean=0,
            std=noise_model * fluxes
        )
    elif callable(noise_model):
        assert type(noise_scaling_parameter) is not None, (
            'Flux percentage noise is a callable,'
            'pass Tensor to flux scaling paramater.'
        )
        return fluxes + noise_model(
            fluxes,
            noise_scaling_parameter,
            **noise_model_kwargs
        )
    else:
        raise TypeError('Type of flux percentage noise not recognised.')