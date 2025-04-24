from tools.inference import Inference

inference = Inference()
inference.run_sbi(sim_dir='sim1')
inference.save_posterior('based_posterior_sim1.pkl')