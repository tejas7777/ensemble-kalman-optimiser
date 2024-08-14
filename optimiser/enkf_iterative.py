import torch
import math

class EnKFOptimizerIterative:
    def __init__(self, model, sigma=0.1, J=10, gamma=1e-3, max_iterations=1, debug_mode=False):
        self.model = model
        self.sigma = sigma
        self.J = J
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.parameters = list(model.parameters())
        self.theta = torch.cat([p.data.view(-1) for p in self.parameters])  # Flattened parameters
        self.shapes = [p.shape for p in self.parameters]  # For keeping track of original shapes
        self.debug_mode = debug_mode
        self.particles = None
        self.h0 = 8
        self.epsilon = 1e-10

    def flatten_parameters(self, parameters):
        '''
        The weights from all the layers will be considered as a single vector
        '''
        return torch.cat([p.data.view(-1) for p in parameters])

    def unflatten_parameters(self, flat_params):
        '''
        Here, we regain the shape so that we can use them to evaluate the model
        '''
        params_list = []
        start = 0
        for shape in self.shapes:
            num_elements = torch.prod(torch.tensor(shape))
            params_list.append(flat_params[start:start + num_elements].view(shape))
            start += num_elements
        return params_list

    def step(self, dataset, num_output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for iteration in range(self.max_iterations):
            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            N = len(dataset)

            # Step [1] Draw K Particles
            particles = self.theta.unsqueeze(1) + torch.randn((self.theta.numel(), self.J), device=device) * self.sigma
            self.particles = particles  # This is for retrieving

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.J} Particles completed")

            # Step [2] Compute mean of particles
            mean_particle = torch.mean(particles, dim=1)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Mean of Particles completed")

            # Step [3] Construct Omega
            Omega = (1 / math.sqrt(self.J)) * (particles - mean_particle.unsqueeze(1))

            # Step [4] Evaluate forward model for all particles and all data points
            F_outputs = []
            X_train = torch.stack([x for x, _ in dataset])
            for j in range(self.J):
                perturbed_params = particles[:, j]
                perturbed_params_unflattened = self.__unflatten_parameters(perturbed_params)
                with torch.no_grad():
                    F_output = self.__F(X_train, perturbed_params_unflattened)
                F_outputs.append(F_output)

            # Step [5] Iterate over the dataset
            Q_list = []
            L_list = []
            for n, (x, y) in enumerate(dataset):
                # Initialize v_n tensor of shape (num_output, self.J)
                v_n = torch.zeros((num_output, self.J), device=device)
                L_n = torch.zeros((num_output, self.J), device=device)

                # [i] Retrieve precomputed forward model outputs
                for j in range(self.J):
                    v_n[:, j] = F_outputs[j][n].squeeze().to(device)

                # [ii] Compute the mean of forward model
                v_mean = torch.mean(v_n, dim=1)

                # [iii] Assemble Q
                Q = (1 / math.sqrt(self.J)) * (v_n - v_mean.unsqueeze(1))
                Q_list.append(Q)

                # [iv] Compute gradient of the loss for all J particles
                for j in range(self.J):
                    L_n[:, j] = self.loss_gradient(v_n[:, j], y).squeeze().to(device)
                L_list.append(L_n)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Loss Computed")

            # Step [6] Construct C for each particle
            C = torch.zeros((self.theta.numel(), self.J), device=device)

            for j in range(self.J):
                for n in range(N):
                    Q_n = Q_list[n]
                    grad_L_n_j = L_list[n][:, j]
                    C[:, j] += Omega @ (Q_n.T @ grad_L_n_j) / N

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Frobenius Norm Computed")

            # Step [7] Compute the step size µ_i
            mu_i = self.h0 / (torch.norm(C, p='fro') + self.epsilon)

            # Step [8] Update each particle using µ_i
            particles -= mu_i * C

            # Set the new parameters in self.theta
            self.theta = torch.mean(particles, dim=1)

            #self.parameter_trajectory.append(self.theta.clone())

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Completed")

            

    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements

    def loss_gradient(self,values, obs):
        residuals = values - obs

        return residuals.view(-1,1)

    def simple_line_search(self, F, update, initial_lr, obs, x, reduction_factor=0.5, max_reductions=5):
        lr = initial_lr
        current_params_unflattened = self.unflatten_parameters(self.theta)

        # Compute the initial predictions and loss directly
        current_predictions = F(current_params_unflattened, x)
        current_loss = torch.mean((current_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

        for _ in range(max_reductions):
            new_theta = self.theta - lr * update
            new_predictions = F(self.unflatten_parameters(new_theta), x)
            new_loss = torch.mean((new_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

            if new_loss < current_loss:
                return lr

            lr *= reduction_factor

        return lr
    
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

    def __misfit_gradient(self,thetha, train, d_obs, loss_type='mse'):
        loss_mapper = {
            'mse': self.__mse_gradient,
            'cross_entropy': self.__cross_entropy_gradient,
            'cross_entropy_kl': self.__cross_entropy_kl_gradient
        }

        return loss_mapper[loss_type](thetha, train, d_obs)
    
    def __mse_gradient(self,thetha, train, d_obs):
        #Forward pass to get model outputs
        t = self.__F(train, self.__unflatten_parameters(thetha))
        
        #compute simple residuals
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
