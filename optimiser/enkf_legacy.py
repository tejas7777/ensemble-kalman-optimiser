import torch
import copy
import numpy as np
import math

'''
EKI algorithm provided in Haber et al, (https://arxiv.org/abs/1805.08034).
Modified to produce forward model on the entire batch
New Change
'''

class EnKFOriginal:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=10, gamma=1e-3, max_iterations=1, debug_mode=False, loss_type='mse'):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.parameters = list(model.parameters())
        self.theta = torch.cat([p.data.view(-1) for p in self.parameters])  #Flattened parameters
        self.shapes = [p.shape for p in self.parameters]  #For keeping track of original shapes
        self.cumulative_sizes = [0] + list(torch.cumsum(torch.tensor([p.numel() for p in self.parameters]), dim=0))
        self.debug_mode = debug_mode
        self.particles = None
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_decay = 1e-5

    '''
    The core optimiser step
    Input: Observations
    '''
    def step(self, dataset, num_output=None):
        for iteration in range(self.max_iterations):
            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            N = len(dataset)

            # Step [1] Draw K Particles
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  # Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  # Add the noise to the current parameter estimate

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            # Step [2] Iterate over the dataset rows
            Ln = None
            for x, y in dataset:
                current_params_unflattened = self.__unflatten_parameters(self.theta)
                with torch.no_grad():
                    F_current = self.__F(x, current_params_unflattened)

                Q = torch.zeros(y.shape[0], self.k)  # [batch_size, k]

                for i in range(self.k):
                    perturbed_params = particles[:, i]
                    perturbed_params_unflattened = self.__unflatten_parameters(perturbed_params)

                    # Evaluate the forward model on the perturbed parameters
                    with torch.no_grad():
                        F_perturbed = self.__F(x, perturbed_params_unflattened)

                    # Compute the difference
                    Q[:, i] = F_perturbed.squeeze() - F_current.squeeze()

                if self.debug_mode:
                    print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete for one data point")

                # Step [3] Construct the Hessian Matrix H_j = Q_j(transpose) x Q_j + Γ
                H_j = Q.T @ Q + self.gamma * torch.eye(self.k)
                H_inv = torch.inverse(H_j)

                if self.debug_mode:
                    print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed for one data point")

                # Step [4] Calculate the Gradient of loss function with respect to the current parameters
                gradient = self.__misfit_gradient(self.theta,x, y)
                gradient = gradient.view(-1, 1)

                if self.debug_mode:
                    print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed for one data point")

                # Step [5] Update the parameters
                adjustment = H_inv @ Q.T  # Shape [k, m]

                if Ln is None:
                    Ln = adjustment @ gradient
                else:
                    Ln = Ln + adjustment @ gradient

            #Perform line search to determine optimal learning rate
            final = self.Omega @ Ln
            final = final.view(-1)
            self.theta -= self.lr * final  # Now both are [n]

            # Update the actual model parameters
            self.__update_model_parameters(self.theta)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : parameter update completed for one data point")

    def __loss_gradient(self,values, obs):
        residuals = values - obs
        return residuals.view(-1,1)

    '''
    Forward Model
    Input: Parameters of the model
    Output: Predictions of the Shape [M_Out, Batch_Size]
    '''
    def __F(self, train, parameters):
        with torch.no_grad(): 
            for original_param, new_param in zip(self.model.parameters(), parameters):
                original_param.data.copy_(new_param.data)

            output =  self.model(train)
            return output

    '''
    Utitlity Method
    Input: Parameters of the model
    Output: Single Vector
    '''

    def __flatten_parameters(self, parameters):
        '''
        The weights from all the layers will be considered as a single vector
        '''
        return torch.cat([p.data.view(-1) for p in parameters])
    
    '''
    Utitlity Method
    Input: Single Vector
    Output: Parameters retaining the shape
    '''
    def __unflatten_parameters(self, flat_params):
        '''
        Regain the shape to so that we can use them to evaluate the model
        '''
        params_list = []
        start = 0
        for shape in self.shapes:
            num_elements = torch.prod(torch.tensor(shape))
            params_list.append(flat_params[start:start + num_elements].view(shape))
            start += num_elements
        return params_list

    '''
    Utitlity Method
    Input: Single Vector
    Output: Parameters retaining the shape
    '''
    def __update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            #param.grad = None
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements



    def __misfit_gradient(self,thetha, train, d_obs, loss_type='mse'):
        loss_mapper = {
            'mse': self.__mse_gradient,
            'cross_entropy': self.__cross_entropy_gradient,
        }

        return loss_mapper[loss_type](thetha, train, d_obs)
    
    def __mse_gradient(self,thetha, train, d_obs):
        #Forward pass to get model outputs
        t = self.__F(train, self.__unflatten_parameters(thetha))
        
        #compute residuals
        residuals = t - d_obs

        return residuals.view(-1, 1)
    
    def __cross_entropy_gradient(self, theta, train, d_obs, delta=1e-10):
        # Unflatten parameters
        params_unflattened = self.__unflatten_parameters(theta)
        
        # Forward pass
        predictions = self.__F(train, params_unflattened)
        num_classes = predictions.shape[1]
        d_obs = torch.nn.functional.one_hot(d_obs, num_classes=num_classes).float()
        
        # Compute gradient of cross-entropy loss with respect to predictions
        grad =  - d_obs / (predictions + delta)
        
        return grad.view(-1, 1)
    
    def __cross_entropy_kl_gradient(self, theta, train, d_obs, delta=1e-10, kl_weight=0.01):
        # Unflatten parameters
        params_unflattened = self.__unflatten_parameters(theta)
        
        # Forward pass
        predictions = self.__F(train, params_unflattened)
        num_classes = predictions.shape[1]
        d_obs = torch.nn.functional.one_hot(d_obs, num_classes=num_classes).float()
        
        # Compute gradient of cross-entropy loss with respect to predictions
        ce_grad = - d_obs / (predictions + delta)
        
        # Compute KL divergence
        kl_grad = torch.zeros_like(theta)
        for i, (param, shape) in enumerate(zip(params_unflattened, self.shapes)):
            # Assuming a standard normal prior
            mu = param[:shape[0]//2]
            log_var = param[shape[0]//2:]
            
            # Compute KL divergence
            kl = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
            
            # Compute KL gradient manually
            mu_grad = mu
            log_var_grad = 0.5 * (log_var.exp() - 1)
            
            kl_grad[self.cumulative_sizes[i]:self.cumulative_sizes[i+1]] = torch.cat([mu_grad, log_var_grad]).view(-1)
        
        # Reshape ce_grad to match the parameter space
        ce_grad_reshaped = torch.zeros_like(theta)
        ce_grad_flattened = ce_grad.view(-1)
        ce_grad_reshaped[:ce_grad_flattened.size(0)] = ce_grad_flattened
        
        # Combine gradients
        total_grad = ce_grad_reshaped + kl_weight * kl_grad
        
        return total_grad.view(-1, 1)




    def __simple_line_search(self, update, initial_lr,train, obs, reduction_factor=0.5, max_reductions=5):
        lr = initial_lr
        current_params_unflattened = self.__unflatten_parameters(self.theta)
        
        # Compute the initial predictions and loss directly
        current_predictions = self.__F(train, current_params_unflattened)
        current_loss = torch.mean((current_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

        for _ in range(max_reductions):
            new_theta = self.theta - lr * update
            new_predictions = self.__F(train, self.__unflatten_parameters(new_theta))
            new_loss = torch.mean((new_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

            if new_loss < current_loss:
                return lr
            
            lr -= lr*reduction_factor

        return lr






