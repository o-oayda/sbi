from astropy.table import Table

class CatwiseSim:
    def __init__(self, nside: int = 64):
        self.nside = nside
        
    def load_catalogue(self):
        self.file_path = 'catwise/catwise2020_corr_w12-0p5_w1-17p0.fits'
        print('Loading CatWISE2020...')
        self.catalogue = Table.read(self.file_path)
        print('Finished loading CatWISE2020.')
    
    def create_colour_magnitude_distribution(self):
        pass