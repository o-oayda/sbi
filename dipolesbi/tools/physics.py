import torch
from torch.types import Tensor
from astropy.modeling.rotations import (
    RotateCelestial2Native, RotateNative2Celestial
)
import astropy.units as u


def sample_spherical_points(n_points) -> tuple[Tensor]:
    longitudes_deg = 360 * torch.rand(size=(n_points,))
    latitudes_deg = torch.rad2deg(
        torch.arcsin( 2 * torch.rand(size=(n_points,)) - 1)
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
    ) -> None:
    return torch.normal(mean=mean_index, std=sigma_index, size=(n_points,))


def lorentz_factor(observer_speed: float) -> float:
    return 1 / ( torch.sqrt(1 - torch.as_tensor(observer_speed) ** 2) )


def doppler_shift_factor(
        observer_speed: float,
        angle_to_source: float
    ) -> float:
    angle_to_source = torch.deg2rad(angle_to_source)
    gamma = lorentz_factor(observer_speed)
    return gamma * ( 1 + observer_speed * torch.cos(angle_to_source) )


def boost_fluxes(
        fluxes: Tensor,
        angle_to_source: Tensor,
        observer_speed: float,
        spectral_index: float | Tensor
    ) -> Tensor:
    '''
    :param angle_to_source: Angle between dipole vector and source in degrees
        in source rest frame.
    '''
    delta = doppler_shift_factor(observer_speed, angle_to_source)
    boost_factor = delta ** ( 1 + spectral_index )
    return fluxes * boost_factor


def boost_magnitudes(
        magnitudes: Tensor,
        angle_to_source: Tensor,
        observer_speed: float,
        spectral_index: float | Tensor
    ) -> Tensor:
    delta = doppler_shift_factor(observer_speed, angle_to_source)
    return magnitudes - 2.5 * (1 + spectral_index) * torch.log10(delta)


def native_to_dipole_frame(
        point_longitudes: Tensor,
        point_latitudes: Tensor,
        dipole_longitude: float,
        dipole_latitude: float,
        pole_longitude: float = 0.
    ) -> tuple[Tensor]:
    rotator = RotateCelestial2Native(
        lon=dipole_longitude * u.degree,
        lat=dipole_latitude * u.degree,
        lon_pole=pole_longitude * u.degree
    )
    new_long, new_lat = rotator(point_longitudes, point_latitudes)
    return torch.as_tensor(new_long), torch.as_tensor(new_lat)


def dipole_to_native_frame(
        point_longitudes: Tensor,
        point_latitudes: Tensor,
        dipole_longitude: float,
        dipole_latitude: float,
        pole_longitude: float = 0.
    ) -> tuple[Tensor]:
    rotator = RotateNative2Celestial(
        lon=dipole_longitude * u.degree,
        lat=dipole_latitude * u.degree,
        lon_pole=pole_longitude * u.degree
    )
    new_long, new_lat = rotator(point_longitudes, point_latitudes)
    return (
        torch.as_tensor(new_long, dtype=torch.float32),
        torch.as_tensor(new_lat, dtype=torch.float32)
    )


def aberrate_points(
        rest_longitudes: Tensor,
        rest_latitudes: Tensor,
        observer_direction: tuple[float],
        observer_speed: float
    ) -> tuple[Tensor]:
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
    
    boosted_source_to_dipole_angle = compute_boosted_angles(
        source_frame_angles=source_to_dipole_angle,
        observer_speed=observer_speed
    )
    boosted_dipole_frame_latitudes = 90. - boosted_source_to_dipole_angle

    boosted_longitudes, boosted_latitudes = dipole_to_native_frame(
        point_longitudes=dipole_frame_longitudes,
        point_latitudes=boosted_dipole_frame_latitudes,
        dipole_longitude=dipole_longitude,
        dipole_latitude=dipole_latitude
    )

    return boosted_longitudes, boosted_latitudes, source_to_dipole_angle


def compute_boosted_angles(
        source_frame_angles: Tensor,
        observer_speed: float
    ) -> Tensor:
    '''
    Given an angle between the direction of motion and the source
    in the source frame, find the boosted angle, corresponding to the
    angle perceived in the observer's frame.
    
    :param source_frame_angles: the angle in degrees between the
        direction of motion (i.e. the dipole vector) and the source.
    :param observer_speed: the speed (in units of c) of the observer.
    '''
    source_frame_angles = torch.deg2rad(source_frame_angles)
    return torch.rad2deg(
        torch.arccos(
            (observer_speed + torch.cos(source_frame_angles))
            / (observer_speed * torch.cos(source_frame_angles) + 1)
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