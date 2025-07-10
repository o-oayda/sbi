import os
from astropy.table import Table
import numpy as np
from scipy.constants import speed_of_light
import astropy.units as u
from scipy.spatial import cKDTree
from tqdm import tqdm
from numpy.typing import NDArray

class AlphaLookup:
    '''
    Adapted from lookup_alpha_catwise.py from Secrest+21.
    '''
    def __init__(self):
        self.lookup_table_path = 'dipolesbi/catwise/alpha_colors.fits'
        assert os.path.exists(self.lookup_table_path)
        self.AB_VEGA_OFFSET = 2.673
        self.SPEED_OF_LIGHT_ANGSTROMS_S = speed_of_light * 1e10
    
    def make_alpha(self,
            w1_magnitude: NDArray[np.float_],
            w12_color: NDArray[np.float_],
            no_check: bool = False
        ) -> Table:
        self.make_lookup_table()
        self.do_lookups(w12_color)
        self.check_magnitude(w1_magnitude, no_check=no_check)
        return self.process_for_table()
        
    def process_for_table(self) -> Table:
        out_table = Table()
        
        out_table['k'] = Table.Column(
            self.k,
            unit=u.erg / u.cm**2 / u.second / u.Hz
        )
        out_table['alpha_W1'] = self.alpha_W1
        out_table['nu_W1_iso'] = Table.Column(self.nu_W1_iso, unit=u.Hz)

        return out_table

    def make_lookup_table(self):
        self.lookup_table = Table.read(self.lookup_table_path)

        self.lookup_alpha = self.lookup_table['alpha'].data
        self.lookup_k_W1 = self.lookup_table['k_W1'].data # Flux conversion factor
        self.lookup_nu_W1_iso = self.lookup_table['nu_W1_iso'].data
        self.lookup_W1_W2 = self.lookup_table['W1_W2'].data
    
    def do_lookups(self, w12_color: np.ndarray):
        self.n_sources = len(w12_color)

        self.alpha_W1 = np.nan * np.empty( self.n_sources, dtype=float )
        self.k_W1 = np.nan * np.empty( self.n_sources, dtype=float )
        self.nu_W1_iso = np.nan * np.empty( self.n_sources, dtype=float )

        # Build a KDTree for efficient nearest-neighbor search
        tree = cKDTree(self.lookup_W1_W2.reshape(-1, 1))

        # Perform lookups for all sources
        _, indices = tree.query(w12_color.reshape(-1, 1), k=1)

        # Assign values based on nearest neighbors
        self.k_W1 = self.lookup_k_W1[indices]
        self.alpha_W1 = self.lookup_alpha[indices]
        self.nu_W1_iso = self.lookup_nu_W1_iso[indices]

    def check_magnitude(self, w1_magnitude: np.ndarray, no_check: bool = False):
        # Calculate k such that fnu = k * nu**alpha
        # We're using the Oke & Gunn / Fukugita AB magnitude, which has a
        # zeropoint of 48.60, so the AB - Vega offset for W1 is 2.673.

        W1_AB = w1_magnitude + self.AB_VEGA_OFFSET
        self.k = self.k_W1 * 10**( -W1_AB / 2.5 )

        # Double check to ensure that fnu = k * nu**alpha gives the right mag
        nu, Snu = self.get_passband('dipolesbi/catwise/RSR-W1.txt')
        W1_AB_check = np.empty(self.n_sources, dtype=float)
        
        if no_check:
            return None
        else:
            for i in tqdm(range(self.n_sources)):
                fnu = self.k[i] * nu**self.alpha_W1[i]
                W1_AB_check[i] = self.compute_synth_ABmag(nu, fnu, Snu)

            abs_dmag = np.abs(W1_AB_check - W1_AB)
            if abs_dmag.max() > 1e-12:
                print("WARNING: Measured and predicted magnitudes differ!")

    @staticmethod
    def closest(dx):
        return np.argmin(np.abs(dx))
    
    def get_passband(self, response_file):
        response_table = Table.read(response_file, format='ascii')
        response_table['nu'] = (
            self.SPEED_OF_LIGHT_ANGSTROMS_S / response_table['Angstrom']
        )

        # Frequency must be monotonically increasing and unique
        response_table.sort('nu')
        idx = np.unique(response_table['nu'].data, return_index=True)[1]
        response_table = response_table[idx]

        nu, S_nu = response_table['nu'], response_table['Fraction'].data

        return nu, S_nu
    
    def compute_synth_ABmag(self, nu, f_nu, S_nu, zp_AB=48.60):
        '''
        Returns a synthetic AB magnitude as per Fukugita et al.
        (1996, AJ, 111, 1748), Eq. 7.
        '''
        log_nu = np.log(nu)
        m = (
            -2.5 * np.log10(
                self.trapezoidal_integrate(f_nu * S_nu, log_nu)
                / self.trapezoidal_integrate(S_nu, log_nu)
            )
            - zp_AB
        )
        return m
    
    @staticmethod
    def trapezoidal_integrate(Fx_matrix, x):
        '''
        Function to integrate a matrix of functions with respect to x.
        Performs trapezoidal integration. Matrix have shape (m, n) where
        m is the length of x and n is the number of functions. Returns an
        array with shape (n,). If the matrix is a single array
        (single function), returns the same output as np.trapz.
        '''
        Fx_matrix = np.array(Fx_matrix, dtype=float)
        x = np.array(x, dtype=float)
        dx = np.diff(x)
        dFx_matrix = (Fx_matrix[:][1:] + Fx_matrix[:][:-1]) / 2

        return np.matmul(dx, dFx_matrix)