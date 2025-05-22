import torch
import numpy as np
from scipy import integrate
import pdb

def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # shape = z.shape
    # N = np.prod(shape[1:])
    return -1 / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1)) / (2 * sigma**2)

def ode_likelihood(x, 
                   score_model,
                   marginal_prob_std, 
                   diffusion_coeff,
                   start_t, 
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.
    
    Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the 
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the 
        forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
    """
    # print(x)
    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    # torch.manual_seed(42)
    epsilon = torch.randn_like(x)
    # print(epsilon)
    shape = x.shape
    shape1 = x[:,:-1].shape
        
    def divergence_eval(sample, time_steps, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        expected_res = 0
        N = 10
        with torch.enable_grad():
            sample.requires_grad_(True)
            for _ in range(N):
                new_epsilon = torch.randn_like(x) # epsilon #  
                score_e = torch.sum(score_model(sample, time_steps) * new_epsilon)
                grad_score_e = torch.autograd.grad(score_e, sample)[0]
                tmp_res = torch.sum(grad_score_e * new_epsilon, dim=(1))  
                expected_res += tmp_res
        return expected_res / N # torch.sum(grad_score_e * epsilon, dim=(1))    
    
    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
        # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        # print(t)
        time_steps = np.ones((shape[0],)) * t    
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = - 0.5 *  g**2 * score_eval_wrapper(sample, time_steps) # g * torch.randn_like(torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)).numpy().reshape((-1,)).astype(np.float64)#
        logp_grad = - 0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)
    ##################################################################################################################
    # def divergence_eval1(sample, time_steps, epsilon):      
    #     """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    #     expected_res = 0
    #     N = 100
    #     with torch.enable_grad():
    #         sample.requires_grad_(True)
    #         for _ in range(N):
    #             new_epsilon = torch.randn_like(x[:,:-1]) # epsilon[:,:-1] # 
    #             score_e = torch.sum(score_model1(sample, time_steps) * new_epsilon)
    #             grad_score_e = torch.autograd.grad(score_e, sample)[0]
    #             tmp_res = torch.sum(grad_score_e * new_epsilon, dim=(1))  
    #             expected_res += tmp_res
    #     return expected_res / N # torch.sum(grad_score_e * epsilon, dim=(1))    
    
    # def score_eval_wrapper1(sample, time_steps):
    #     """A wrapper for evaluating the score-based model for the black-box ODE solver."""
    #     sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape1)
    #     time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    #     with torch.no_grad():    
    #         score = score_model1(sample, time_steps)
    #     return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    # def divergence_eval_wrapper1(sample, time_steps):
    #     """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
    #     with torch.no_grad():
    #     # Obtain x(t) by solving the probability flow ODE.
    #         sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape1)
    #         time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    #         # Compute likelihood.
    #         div = divergence_eval1(sample, time_steps, epsilon)
    #         return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    # def ode_func1(t, x):
    #     """The ODE function for the black-box solver."""
    #     # print(t)
    #     time_steps = np.ones((shape1[0],)) * t    
    #     sample = x[:-shape1[0]]
    #     logp = x[-shape1[0]:]
    #     g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    #     sample_grad = - 0.5 *  g**2 * score_eval_wrapper1(sample, time_steps) # g * torch.randn_like(torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)).numpy().reshape((-1,)).astype(np.float64)#
    #     # if t < 2e-5:
    #     #     print(t)
    #     #     print(g)
    #     #     print(sample)
    #     #     print(sample_grad)
    #     logp_grad = 1.5 * g**2 * divergence_eval_wrapper1(sample, time_steps)
    #     return np.concatenate([sample_grad, logp_grad], axis=0)
    

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, start_t), init, rtol=1e-5, atol=1e-5, method='RK45', t_eval=np.linspace(eps, start_t, 100))  
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    # sigma_max = marginal_prob_std(1.)
    sigma_max = z.std()
    prior_logp = prior_likelihood(z, sigma_max)
    likelihood = prior_logp + delta_logp
    # prob = torch.exp(likelihood)

    # init_con = np.concatenate([x[:,:-1].cpu().numpy().reshape((-1,)), np.zeros((shape1[0],))], axis=0)
    # # init_con = np.concatenate([encoded.cpu().numpy().reshape((-1,)), np.zeros((shape1[0],))], axis=0)
    # # Black-box ODE solver
    # res_con = integrate.solve_ivp(ode_func1, (eps, 0.59), init_con, rtol=1e-5, atol=1e-5, method='RK45', t_eval=np.linspace(eps, 0.59, 100))  
    # zp_con = torch.tensor(res_con.y[:, -1], device=device)
    # z_con = zp_con[:-shape1[0]].reshape(shape1)
    # delta_logp_con = zp_con[-shape1[0]:].reshape(shape1[0])
    # sigma_max = marginal_prob_std(1.)
    # prior_logp_con = prior_likelihood(z_con, sigma_max)
    # likelihood_con = prior_logp_con + delta_logp_con

    real_p = np.exp(likelihood)
    # print(zp, zp_con)
    # pdb.set_trace()
    return real_p, prior_logp, delta_logp