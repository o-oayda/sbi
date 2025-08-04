# run on machine with CUDA gpu
from dipolesbi.tools.inference import LikelihoodFreeInferer


inferer = LikelihoodFreeInferer()
inferer.load_posterior('based_posterior_catwise_0p5_17p0_studentst.pkl')
inferer.posterior.to('cpu')
inferer.save_posterior('based_posterior_catwise_0p5_17p0_studentst.pkl')
