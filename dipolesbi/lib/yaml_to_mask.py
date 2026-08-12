"""Build a boolean HEALPix mask from a dipoleview mask YAML file.

This module is deliberately self-contained and has no dipoleview imports:
copy this single file next to a mask ``.yaml`` and it works on its own.
It needs ``numpy``, ``healpy``, ``astropy`` and ``pyyaml`` installed.

Usage
-----
    from yaml_to_mask import yaml_to_mask

    mask = yaml_to_mask('racs-low1_mask.yaml')          # bool array, npix long
    mask = yaml_to_mask('racs-low1_mask.yaml', nside=128)

or from the command line::

    python yaml_to_mask.py racs-low1_mask.yaml -o racs-low1_mask.npy

Convention
----------
``True`` = keep the pixel, ``False`` = masked out.  This matches the
``*_mask.npy`` files written by the dipoleview viewer.

The YAML file
-------------
``coordinates``  frame the map is pixelised in: ``galactic``, ``equatorial``
                 (``celestial``), ``ecliptic``, or ``ANY``.
``nside``        HEALPix nside, or ``ANY``.
``ordering``     ``RING`` or ``NESTED``, or ``ANY``.
``cuts``         inequalities; pixels **satisfying** them are masked out,
                 e.g. ``|b|<5`` removes the galactic plane.  Variables are
                 ``l``/``b`` (galactic), ``ra``/``dec`` (equatorial),
                 ``elon``/``elat`` (ecliptic) and ``lon``/``lat`` (whatever
                 frame the map is pixelised in).
``discs``        ``l=309.5, b=19.4, r=2.0`` — centre plus radius in degrees;
                 every pixel whose centre falls inside is masked out.
``pixels``       extra pixel indices masked by hand, i.e. ones that no cut
                 or disc accounts for.  Only meaningful together with a
                 fixed ``coordinates``, ``nside`` and ``ordering``.

Anything left as ``ANY`` is free for the caller to choose, because the mask
can be rebuilt at any resolution in any frame.  Cuts and discs carry their
own frame in the variable names, so they are converted to the pixelisation
frame automatically.  As soon as ``pixels`` is non-empty the three fields
are pinned, since raw pixel indices only mean something for one particular
pixelisation — passing a conflicting value then raises ``ValueError``.
"""

import re

import numpy as np
import healpy as hp
import yaml
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
import astropy.units as u

__all__ = ['yaml_to_mask']

ANY = 'ANY'

DEFAULT_COORDINATES = 'equatorial'
DEFAULT_NSIDE = 64
DEFAULT_ORDERING = 'RING'

_COORD_ALIASES = {
    'galactic': 'galactic', 'gal': 'galactic', 'g': 'galactic',
    'equatorial': 'equatorial', 'celestial': 'equatorial', 'icrs': 'equatorial',
    'c': 'equatorial', 'eq': 'equatorial', 'radec': 'equatorial',
    'j2000': 'equatorial', 'fk5': 'equatorial',
    'ecliptic': 'ecliptic', 'ecl': 'ecliptic', 'e': 'ecliptic',
}

_ORDER_ALIASES = {
    'ring': 'RING',
    'nest': 'NESTED', 'nested': 'NESTED',
}

# Which frame each expression variable is measured in.  'lon'/'lat' mean the
# pixelisation frame itself, whatever that happens to be.
_VAR_FRAME = {
    'l': 'galactic', 'b': 'galactic',
    'ra': 'equatorial', 'dec': 'equatorial',
    'elon': 'ecliptic', 'elat': 'ecliptic',
    'lon': None, 'lat': None,
}

# Longest names first so 'elon' is not read as 'lon', and 'dec' not as 'e'.
_VAR_ORDER = ['elon', 'elat', 'dec', 'lat', 'lon', 'ra', 'b', 'l']

_OPS = {
    '<': np.less, '<=': np.less_equal,
    '>': np.greater, '>=': np.greater_equal,
}

_RE_TWO_SIDED = re.compile(
    r'^(-?[\d.]+)\s*([<>]=?)\s*[a-z]+\s*([<>]=?)\s*(-?[\d.]+)$')
_RE_VAR_NUM = re.compile(r'^[a-z]+\s*([<>]=?)\s*(-?[\d.]+)$')
_RE_NUM_VAR = re.compile(r'^(-?[\d.]+)\s*([<>]=?)\s*[a-z]+$')
_RE_DISC = re.compile(
    r'^\s*([a-z]+)\s*=\s*(-?[\d.eE+]+)\s*,'
    r'\s*([a-z]+)\s*=\s*(-?[\d.eE+]+)\s*,'
    r'\s*r\s*=\s*([\d.eE+]+)\s*$')


# ---------------------------------------------------------------- normalise

def _norm_coordinates(value, what):
    """Map any spelling of a coordinate frame onto its canonical name."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() == ANY:
        return ANY
    try:
        return _COORD_ALIASES[text.lower()]
    except KeyError:
        raise ValueError(
            f'{what}: unknown coordinate system {value!r}. Use one of '
            "'galactic', 'equatorial' (or 'celestial'), 'ecliptic', or 'ANY'."
        ) from None


def _norm_ordering(value, what):
    """Map any spelling of a HEALPix ordering onto RING / NESTED."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() == ANY:
        return ANY
    try:
        return _ORDER_ALIASES[text.lower()]
    except KeyError:
        raise ValueError(
            f'{what}: unknown ordering {value!r}. Use RING, NESTED or ANY.'
        ) from None


def _norm_nside(value, what):
    """Validate an nside, passing ANY through untouched."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.upper() == ANY:
            return ANY
        try:
            value = int(text)
        except ValueError:
            raise ValueError(
                f'{what}: nside must be a power of two or ANY, got {value!r}.'
            ) from None
    nside = int(value)
    if nside <= 0 or nside & (nside - 1):
        raise ValueError(f'{what}: nside must be a power of two, got {nside}.')
    return nside


def _resolve(name, from_yaml, from_user, default):
    """Pick the value to use, refusing to override a pinned YAML entry."""
    if from_yaml == ANY or from_yaml is None:
        return default if from_user is None else from_user
    if from_user is not None and from_user != from_yaml:
        raise ValueError(
            f'{name}={from_user!r} contradicts the YAML file, which fixes '
            f'{name}={from_yaml!r}. This mask lists individual pixel indices, '
            f'which only make sense for one pixelisation, so {name} is not '
            f'yours to choose here. Drop the {name} argument (or pass '
            f'{name}={from_yaml!r}) to rebuild the mask as saved.'
        )
    return from_yaml


# ------------------------------------------------------------- pixel coords

def _skycoord(lon, lat, frame):
    """Build a SkyCoord from degrees in one of the three supported frames."""
    if frame == 'galactic':
        return SkyCoord(l=lon * u.deg, b=lat * u.deg, frame='galactic')
    if frame == 'equatorial':
        return SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs')
    if frame == 'ecliptic':
        return SkyCoord(lon=lon * u.deg, lat=lat * u.deg,
                        frame=BarycentricMeanEcliptic)
    raise ValueError(f'unknown coordinate system {frame!r}')


def _convert(lon, lat, frame_from, frame_to):
    """Convert degrees from one frame to another, returning (lon, lat)."""
    if frame_to is None or frame_to == frame_from:
        return lon, lat
    sky = _skycoord(lon, lat, frame_from)
    if frame_to == 'galactic':
        out = sky.galactic
        return out.l.deg, out.b.deg
    if frame_to == 'equatorial':
        out = sky.icrs
        return out.ra.deg, out.dec.deg
    if frame_to == 'ecliptic':
        out = sky.transform_to(BarycentricMeanEcliptic)
        return out.lon.deg, out.lat.deg
    raise ValueError(f'unknown coordinate system {frame_to!r}')


class _PixelFrames:
    """Pixel centres on demand in whichever frame a cut happens to use."""

    def __init__(self, nside, ordering, coordinates):
        npix = hp.nside2npix(nside)
        lon, lat = hp.pix2ang(nside, np.arange(npix),
                              nest=(ordering == 'NESTED'), lonlat=True)
        self.npix = npix
        self.coordinates = coordinates
        self._cache = {coordinates: (lon, lat), None: (lon, lat)}

    def values(self, variable):
        """The array of ``variable`` (e.g. 'dec') over all pixels."""
        frame = _VAR_FRAME[variable]
        if frame not in self._cache:
            base_lon, base_lat = self._cache[self.coordinates]
            self._cache[frame] = _convert(base_lon, base_lat,
                                          self.coordinates, frame)
        lon, lat = self._cache[frame]
        return lat if variable in ('b', 'dec', 'elat', 'lat') else lon


# ------------------------------------------------------------------- parsing

def _cut_variable(expr):
    """The coordinate variable a cut expression is written in."""
    for name in _VAR_ORDER:
        if re.search(r'(?<![a-z])' + name + r'(?![a-z])', expr):
            return name
    return None


def _cut_pixels(expr, pixels):
    """Indices of the pixels a single cut expression removes."""
    text = str(expr).strip()
    variable = _cut_variable(text.lower())
    if variable is None:
        raise ValueError(
            f'cut {expr!r}: no coordinate variable found. Use one of '
            "l, b, ra, dec, elon, elat, lon, lat."
        )
    values = pixels.values(variable)

    clean = text.lower()
    if '|' in clean:
        clean = re.sub(r'\|[^|]+\|', variable, clean)
        values = np.abs(values)

    match = _RE_TWO_SIDED.match(clean)
    if match:
        low, low_op, high_op, high = match.groups()
        keep = (_OPS[low_op](float(low), values)
                & _OPS[high_op](values, float(high)))
        return np.flatnonzero(keep)

    match = _RE_VAR_NUM.match(clean)
    if match:
        op, number = match.groups()
        return np.flatnonzero(_OPS[op](values, float(number)))

    match = _RE_NUM_VAR.match(clean)
    if match:
        number, op = match.groups()
        return np.flatnonzero(_OPS[op](float(number), values))

    raise ValueError(
        f'cut {expr!r}: could not be parsed. Expected something like '
        "'|b|<5', 'dec>47.3' or '60<ra<80'."
    )


def _parse_disc(entry):
    """Read ``l=309.5, b=19.4, r=2.0`` into (lon, lat, radius, frame)."""
    if isinstance(entry, dict):
        # Tolerate a mapping written out by hand.
        keys = {str(k).lower(): v for k, v in entry.items()}
        radius = keys.pop('r', keys.pop('radius', None))
        if radius is None or len(keys) != 2:
            raise ValueError(f'disc {entry!r}: expected two coordinates and r.')
        (name_a, val_a), (name_b, val_b) = keys.items()
        entry = f'{name_a}={val_a}, {name_b}={val_b}, r={radius}'

    match = _RE_DISC.match(str(entry).strip().lower())
    if match is None:
        raise ValueError(
            f'disc {entry!r}: could not be parsed. Expected something like '
            "'l=309.5, b=19.4, r=2.0'."
        )
    name_lon, value_lon, name_lat, value_lat, radius = match.groups()

    pairs = {('l', 'b'): 'galactic', ('ra', 'dec'): 'equatorial',
             ('elon', 'elat'): 'ecliptic', ('lon', 'lat'): None}
    frame = pairs.get((name_lon, name_lat), 'missing')
    if frame == 'missing':
        raise ValueError(
            f'disc {entry!r}: {name_lon}/{name_lat} is not a coordinate pair. '
            'Use l/b, ra/dec, elon/elat or lon/lat.'
        )
    return float(value_lon), float(value_lat), float(radius), frame


def _disc_pixels(entry, pixels):
    """Indices of the pixels a single disc removes."""
    lon, lat, radius, frame = _parse_disc(entry)
    if radius <= 0:
        raise ValueError(f'disc {entry!r}: radius must be positive.')
    if frame is None:
        frame = pixels.coordinates

    # Work in the pixelisation frame: move the centre there, then compare
    # against the pixel centres.  Angular separation is frame independent,
    # so this is the same disc however it was written down.
    centre_lon, centre_lat = _convert(np.atleast_1d(float(lon)),
                                      np.atleast_1d(float(lat)),
                                      frame, pixels.coordinates)
    pix_lon = pixels.values('lon')
    pix_lat = pixels.values('lat')

    centre = hp.ang2vec(float(centre_lon[0]), float(centre_lat[0]), lonlat=True)
    vectors = hp.ang2vec(pix_lon, pix_lat, lonlat=True)
    inside = vectors @ centre >= np.cos(np.deg2rad(radius))
    return np.flatnonzero(inside)


# -------------------------------------------------------------------- public

def yaml_to_mask(path, coordinates=None, nside=None, ordering=None):
    """Build a boolean HEALPix mask from a dipoleview mask YAML file.

    Parameters
    ----------
    path : str
        Path to the ``.yaml`` mask description.
    coordinates : str, optional
        Frame to pixelise the mask in: ``'galactic'``, ``'equatorial'``
        (``'celestial'``) or ``'ecliptic'``.  Only allowed when the file
        says ``ANY``; defaults to the file's value, or ``'equatorial'``.
    nside : int, optional
        HEALPix nside.  Only allowed when the file says ``ANY``; defaults
        to the file's value, or 64.
    ordering : str, optional
        ``'RING'`` or ``'NESTED'``.  Only allowed when the file says
        ``ANY``; defaults to the file's value, or ``'RING'``.

    Returns
    -------
    ndarray of bool
        Length ``12 * nside**2``.  ``True`` = keep, ``False`` = masked out.

    Raises
    ------
    ValueError
        If an argument contradicts a value the file has pinned down, or if
        a cut or disc cannot be parsed.
    """
    with open(path) as stream:
        spec = yaml.safe_load(stream) or {}
    if not isinstance(spec, dict):
        raise ValueError(f'{path}: expected a YAML mapping at the top level.')

    file_coordinates = _norm_coordinates(spec.get('coordinates', ANY), path)
    file_nside = _norm_nside(spec.get('nside', ANY), path)
    file_ordering = _norm_ordering(spec.get('ordering', ANY), path)

    coordinates = _resolve('coordinates', file_coordinates,
                           _norm_coordinates(coordinates, 'coordinates'),
                           DEFAULT_COORDINATES)
    nside = _resolve('nside', file_nside,
                     _norm_nside(nside, 'nside'), DEFAULT_NSIDE)
    ordering = _resolve('ordering', file_ordering,
                        _norm_ordering(ordering, 'ordering'), DEFAULT_ORDERING)

    pixels = _PixelFrames(nside, ordering, coordinates)
    mask = np.ones(pixels.npix, dtype=bool)

    for expr in spec.get('cuts') or []:
        mask[_cut_pixels(expr, pixels)] = False

    for entry in spec.get('discs') or []:
        mask[_disc_pixels(entry, pixels)] = False

    extra = spec.get('pixels') or []
    if isinstance(extra, str):
        extra = [int(part) for part in extra.replace(',', ' ').split()]
    extra = np.asarray(extra, dtype=np.int64)
    if extra.size:
        if file_nside == ANY or file_ordering == ANY or file_coordinates == ANY:
            raise ValueError(
                f'{path}: lists individual pixels, so coordinates, nside and '
                'ordering must all be given explicitly in the file.'
            )
        out_of_range = extra[(extra < 0) | (extra >= pixels.npix)]
        if out_of_range.size:
            raise ValueError(
                f'{path}: pixel indices out of range for nside={nside} '
                f'(npix={pixels.npix}), e.g. {out_of_range[0]}.'
            )
        mask[extra] = False

    return mask


def main(argv=None):
    """Command line entry point: write the mask out as a ``.npy`` file."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build a boolean HEALPix mask from a mask YAML file '
                    '(True = keep, False = masked out).')
    parser.add_argument('yaml_path', help='input .yaml mask description')
    parser.add_argument('-o', '--output', help='output .npy path')
    parser.add_argument('--coordinates', help="galactic / equatorial / ecliptic")
    parser.add_argument('--nside', type=int)
    parser.add_argument('--ordering', help='RING or NESTED')
    args = parser.parse_args(argv)

    mask = yaml_to_mask(args.yaml_path, coordinates=args.coordinates,
                        nside=args.nside, ordering=args.ordering)
    print(f'{mask.size} pixels: {int(mask.sum())} kept, '
          f'{int((~mask).sum())} masked '
          f'({100 * (~mask).mean():.1f}% of the sky)')
    if args.output:
        np.save(args.output, mask)
        print(f'Saved: {args.output}')
    return mask


if __name__ == '__main__':
    main()
