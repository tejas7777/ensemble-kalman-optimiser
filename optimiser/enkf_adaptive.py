import torch
import torch.nn.init as init
import math
import copy



'''
Adaptive EnKF by Dr. Marco Iglesias
'''

class EnKFAdaptive:
    def __init__(self, model, lr=0.5, sigma=0.01, k=10, gamma=1e-4, max_iterations=1, debug_mode=False, loss_type='mse'):
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
        self.iteration = 0
        self.J = k

    '''
    The core optimiser step
    Input: Observations
    '''
    def step(self,train, obs):
        if self.debug_mode:
            print(f"Iteration started")

        # Step [1] Draw K Particles
        if self.iteration == 0:
            U = torch.zeros((self.theta.numel(), self.J))  # [K, J]
            for j in range(self.J):
                perturbation = torch.empty_like(self.theta).unsqueeze(1)  # Add a dimension
                init.xavier_normal_(perturbation)
                U[:, j] = (self.theta.unsqueeze(1) + self.sigma * perturbation).squeeze(1)

            # init.xavier_normal_(U)
                
        else:
            # Introduce new perturbations in each iteration
            U = self.theta.unsqueeze(1) + self.sigma * torch.randn((self.theta.numel(), self.J))  # [K, J]

        if self.debug_mode:
            print(f"Drawing {self.J} Particles completed")

        # Step [2] Calculate the Mean
        u_mean = torch.mean(U, dim=1, keepdim=True)  # Shape: [K, 1]

        # Step [3] Compute Omega
        Omega = (1 / math.sqrt(self.J)) * (U - u_mean)  # Shape: [K, J]

        # Step [4] Evaluate the forward model on the network for each particle
        with torch.no_grad():
            mean_params_unflattened = self.__unflatten_parameters(u_mean.squeeze(1))
            F_current = self.__F(train, mean_params_unflattened)
            m, c = F_current.size()  # Batch size and number of classes
            V = torch.zeros(m, c, self.J)  # [batch_size, num_output, J]

            for i in range(self.J):
                perturbed_params = U[:, i]
                perturbed_params_unflattened = self.__unflatten_parameters(perturbed_params)
                F_perturbed = self.__F(train, perturbed_params_unflattened)
                V[:, :, i] = F_perturbed

            if self.debug_mode:
                print(f"Forward model evaluation complete")

        # Step [5] Calculate the mean and construct Q
        V_mean = torch.mean(V, dim=2, keepdim=True)  # Shape: [batch_size, num_output, 1]
        Q = (1 / math.sqrt(self.J)) * (V - V_mean)  # Shape: [batch_size, num_output, J]
        Q_vec = Q.view(-1, self.J)  # [m * c, J]

        # Step [7] Calculate the residuals for the mean particle
        residuals_mean = F_current - obs  # Shape: [batch_size, num_output]
        delta_v = residuals_mean.view(-1, 1)  # [MON, 1]

        # Step [8] Compute Alpha Star
        sum_squared_residuals = torch.sum(residuals_mean.pow(2))  # Scalar
        alpha = (1 / (m * c)) * (1 / self.J) * sum_squared_residuals

        if self.debug_mode:
            print(f"Q and Alpha Star inverse completed")

        # Step [9] Calculate Hessian
        H = Q_vec.t() @ Q_vec + self.gamma * torch.eye(self.J)  # [J, J]

        # Step [10] Calculate Update
        H_inv = torch.inverse(H)
        # Generate noise eta
        eta = torch.randn(delta_v.size())  # Shape: [MON, 1]
        eta = eta - eta.mean(dim=0)  # Subtract the mean
        # Update delta_v with noise
        delta_v_with_noise = delta_v #+ (1 / math.sqrt(alpha)) * eta  # [MON, 1]
        
        # Compute the update term
        update_term = H_inv @ Q_vec.t() @ delta_v_with_noise  # (J, MON) X (MON, 1) -> [J, 1]
        # Update U
        U = U - Omega @ update_term  # Shape: [K, J]

        self.theta = torch.mean(U,dim=1)

        self.U = U  # For next iteration

        self.iteration += 1

        if self.debug_mode:
            print(f"Parameter update completed")


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
    
    def __mse_gradient(self,thetha, train, d_obs):
        #Forward pass to get model outputs
        t = self.__F(train, self.__unflatten_parameters(thetha))
        
        #compute simple residuals
        residuals = t - d_obs

        return residuals.view(-1, 1)
    
    def __cross_entropy_gradient(self, F, theta, d_obs, delta=1e-10):
        # Unflatten parameters
        params_unflattened = self.__unflatten_parameters(theta)
        
        # Forward pass
        predictions = F(params_unflattened)
        num_classes = predictions.shape[1]
        d_obs = torch.nn.functional.one_hot(d_obs, num_classes=num_classes).float()
        
        # Compute gradient of cross-entropy loss with respect to predictions
        grad = - d_obs / (predictions + delta)
        
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






