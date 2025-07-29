import torch
from torch.types import Tensor
from astropy.modeling.rotations import (
    RotateCelestial2Native, RotateNative2Celestial
)
import astropy.units as u
import numpy as np
from numpy.typing import NDArray


def sample_spherical_points(n_points) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    longitudes_deg = 360 * np.random.rand(n_points)
    latitudes_deg = np.rad2deg(
        np.arcsin( 2 * np.random.rand(n_points) - 1)
    )
    return longitudes_deg, latitudes_deg


def sample_luminosity_function(
        index: float,
        minimum_flux: float,
        n_fluxes: int
    ) -> Tensor:
    return minimum_flux * (1 - torch.rand(size=(n_fluxes,))) ** (- 1 / index)


def sample_spectral_index(
        n_points: int,
        mean_index: float,
        sigma_index: float
    ) -> Tensor:
    return torch.normal(mean=mean_index, std=sigma_index, size=(n_points,))


def lorentz_factor(observer_speed: float) -> float: 
    return 1 /  np.sqrt(1 - observer_speed ** 2) 


def doppler_shift_factor(
        observer_speed: float,
        angle_to_source: NDArray,
    ) -> NDArray[np.float64]:
    angle_to_source = np.deg2rad(angle_to_source) # type: ignore
    gamma = lorentz_factor(observer_speed)
    return gamma * ( 1 + observer_speed * np.cos(angle_to_source) )


def boost_fluxes(
        fluxes: NDArray,
        angle_to_source: NDArray,
        observer_speed: float,
        spectral_index: float | NDArray
    ) -> NDArray[np.float64]:
    '''
    Returns S_nu * (delta ** (1 + alpha) ), i.e. special relativistic boosting
    of flux densities.

    :param angle_to_source: Angle between dipole vector and source in degrees
        in source rest frame.
    '''
    delta = doppler_shift_factor(observer_speed, angle_to_source)
    boost_factor = delta ** ( 1 + spectral_index )
    return fluxes * boost_factor


def boost_magnitudes(
        magnitudes: NDArray,
        angle_to_source: NDArray,
        observer_speed: float,
        spectral_index: float | NDArray,
    ) -> NDArray[np.float64]:
    '''
    Since m_nu = -2.5 log_10 (S_nu) + ZP and S'_nu = S_nu delta ** (1 + alpha),
    we can write the boosted magnitude as a function of function of rest frame
    magnitude:
    
    m'_nu = m_nu - 2.5 (1 + alpha) log_10 (delta). 
    '''
    delta = doppler_shift_factor(observer_speed, angle_to_source)
    return magnitudes - 2.5 * (1 + spectral_index) * np.log10(delta)


def native_to_dipole_frame(
        point_longitudes: NDArray,
        point_latitudes: NDArray,
        dipole_longitude: float,
        dipole_latitude: float,
        pole_longitude: float = 0.
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rotator = RotateCelestial2Native(
        lon=dipole_longitude * u.degree,
        lat=dipole_latitude * u.degree,
        lon_pole=pole_longitude * u.degree
    )
    new_long, new_lat = rotator(point_longitudes, point_latitudes)
    return new_long, new_lat


def dipole_to_native_frame(
        point_longitudes: NDArray[np.float32 | np.float64],
        point_latitudes: NDArray[np.float32 | np.float64],
        dipole_longitude: float,
        dipole_latitude: float,
        pole_longitude: float = 0.
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rotator = RotateNative2Celestial(
        lon=dipole_longitude * u.degree,
        lat=dipole_latitude * u.degree,
        lon_pole=pole_longitude * u.degree
    )
    new_long, new_lat = rotator(point_longitudes, point_latitudes)
    return new_long, new_lat


def aberrate_points(
        rest_longitudes: NDArray,
        rest_latitudes: NDArray,
        observer_direction: tuple[float, float],
        observer_speed: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Aberrate points by transforming into a frame with the dipole vector as the
    pole, boosting the latitude angle, then transforming back into the native frame.

    :param rest_longitudes: Source frame longitudes in degrees.
    :param rest_latitudes: Source frame latitudes in degrees.
    :param observer_direction: Dipole direction in degrees, format (long, lat).
    :param observer_speed: Observer speed in units of c.
    '''
    dipole_longitude, dipole_latitude = observer_direction
    
    dipole_frame_longitudes, dipole_frame_latitudes = native_to_dipole_frame(
        point_longitudes=rest_longitudes,
        point_latitudes=rest_latitudes,
        dipole_longitude=dipole_longitude,
        dipole_latitude=dipole_latitude
    )

    source_to_dipole_angle = 90. - dipole_frame_latitudes
    del dipole_frame_latitudes
    
    boosted_source_to_dipole_angle = compute_boosted_angles(
        source_frame_angles=source_to_dipole_angle,
        observer_speed=observer_speed
    )
    boosted_dipole_frame_latitudes = 90. - boosted_source_to_dipole_angle
    del boosted_source_to_dipole_angle

    boosted_longitudes, boosted_latitudes = dipole_to_native_frame(
        point_longitudes=dipole_frame_longitudes,
        point_latitudes=boosted_dipole_frame_latitudes,
        dipole_longitude=dipole_longitude,
        dipole_latitude=dipole_latitude
    )

    return boosted_longitudes, boosted_latitudes, source_to_dipole_angle


def compute_boosted_angles(
        source_frame_angles: NDArray,
        observer_speed: float
    ) -> NDArray[np.float64]:
    '''
    Given an angle between the direction of motion and the source
    in the source frame, find the boosted angle, corresponding to the
    angle perceived in the observer's frame.
    
    :param source_frame_angles: the angle in degrees between the
        direction of motion (i.e. the dipole vector) and the source.
    :param observer_speed: the speed (in units of c) of the observer.
    '''
    source_frame_angles = np.deg2rad(source_frame_angles)
    return np.rad2deg(
        np.arccos(
            (observer_speed + np.cos(source_frame_angles))
            / (observer_speed * np.cos(source_frame_angles) + 1)
        )
    )


def ellis_baldwin_amplitude(
        observer_speed: float,
        mean_spectral_index: float,
        luminosity_function_slope: float
    ) -> float:
    return ( 
        2 + luminosity_function_slope * ( 1 + mean_spectral_index )
    ) * observer_speed
