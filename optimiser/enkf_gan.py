import torch
import copy

'''
Inspired by the EKI algorithm provided in Haber et al, (https://arxiv.org/abs/1805.08034).
Modified to produce forward model on the entire batch
New Change
'''

class EnKFGAN:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=10, gamma=1e-3, max_iterations=1, debug_mode=False, loss_type='cross_entropy'):
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

    '''
    The core optimiser step
    Input: Observations
    '''
    def step(self, train, obs):
        for iteration in range(self.max_iterations):
            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            # Step [1] Draw K Particles
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma
            particles = self.theta.unsqueeze(1) + self.Omega
            self.particles = particles

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            # Step [2] Evaluate the forward model using theta mean
            current_params_unflattened = self.__unflatten_parameters(self.theta)
            with torch.no_grad():
                F_current = self.__F(train, current_params_unflattened)

            # Flatten output for consistent handling
            F_current_flat = F_current.view(F_current.size(0), -1)
            
            Q = torch.zeros(F_current_flat.size(0), F_current_flat.size(1), self.k)
                
            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.__unflatten_parameters(perturbed_params)

                # Evaluate the forward model on the perturbed parameters
                with torch.no_grad():
                    F_perturbed = self.__F(train, perturbed_params_unflattened)

                # Flatten perturbed output
                F_perturbed_flat = F_perturbed.view(F_perturbed.size(0), -1)

                # Compute the difference
                Q[:, :, i] = F_perturbed_flat - F_current_flat

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete")

            # Step [3] Construct the Hessian Matrix
            Q = Q.view(-1, self.k)
            H_j = Q.T @ Q + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed")

            # Step [4] Calculate the Gradient of loss function
            gradient = self.__misfit_gradient(self.theta, train, obs, loss_type=self.loss_type)
            gradient = gradient.view(-1, 1)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed")

            # Step [5] Update the parameters
            adjustment = H_inv @ Q.T
            self.Omega = self.Omega.to(self.device)
            adjustment = adjustment.to(self.device)
            gradient = gradient.to(self.device)
            intermediate_result = adjustment @ gradient
            update = self.Omega @ intermediate_result
            update = update.view(-1)

            self.theta -= self.lr * update

            # Update the actual model parameters
            self.__update_model_parameters(self.theta)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : parameter update completed")

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
            'cross_entropy': self.__cross_entropy_gradient
        }

        return loss_mapper[loss_type](thetha, train, d_obs)
    
    def __mse_gradient(self, theta, train, d_obs):
        # Forward pass to get model outputs
        t = self.__F(train, self.__unflatten_parameters(theta))
        
        # Ensure t and d_obs have the same shape
        t_flat = t.view(t.size(0), -1)
        d_obs_flat = d_obs.view(d_obs.size(0), -1)
        
        # Compute residuals
        residuals = t_flat - d_obs_flat
        
        # Compute gradient
        gradient = residuals  # MSE gradient
        
        return gradient.view(-1, 1)
    
    def __cross_entropy_gradient(self, theta, train, d_obs, eps=1e-8):
        params_unflattened = self.__unflatten_parameters(theta)
        
        predictions = self.__F(train, params_unflattened)
        #print(f"Predictions shape: {predictions.shape}")
        #print(f"d_obs shape: {d_obs.shape}")
        
        predictions = predictions.view(predictions.size(0), -1)
        #print(f"Reshaped predictions shape: {predictions.shape}")
        
        predictions = torch.sigmoid(predictions)  # Apply sigmoid here
        predictions = torch.clamp(predictions, eps, 1 - eps)
        
        d_obs = d_obs.view(predictions.size(0), -1)
        #print(f"Reshaped d_obs shape: {d_obs.shape}")
        
        #grad =  - d_obs / (predictions + eps)

        grad = predictions - d_obs #/ (predictions * (1 - predictions) + eps)
        
        return grad.view(-1, 1)


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






