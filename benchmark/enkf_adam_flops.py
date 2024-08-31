import torch
import torch.profiler
from model.dnn import DNN
from data.dataloader.michalewicz_function_dataset_loader import MichalewiczFunctionDataLoader
from train.batch_trainer.enkf_train import BatchTrainer as EnKFTrainer
from adam_train.batch_trainer.regression import BatchTrainer as AdamTrainer

def benchmark_flops(trainer_class, model, dataset_loader, params=None):
    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)

    # Get a single batch from the training loader
    batch = next(iter(trainer.train_loader))
    inputs, targets = batch

    # Profile the forward pass to compute FLOPs
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True  # Enable FLOPs counting
    ) as prof:
        # Forward pass
        outputs = model(inputs)
        # Calculate loss using the trainer's loss function
        loss = trainer.loss_function(outputs, targets)
        
        # Skip backward pass if using EnKF
        if isinstance(trainer, AdamTrainer):
            loss.backward()

    # Print the FLOPs result
    print(prof.key_averages().table(sort_by="flops", row_limit=10))

if __name__ == '__main__':
    # Setup DataLoader
    dataset_loader = MichalewiczFunctionDataLoader(num_samples=10000, dimension=10, m=10, noise_level=0.1, batch_size=10000)

    # Determine the input and output size from the feature dimensions
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)

    # Benchmark FLOPs for EnKFTrainer
    print("Benchmarking FLOPs for EnKFTrainer")
    benchmark_flops(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        params={"k": 500, "sigma": 0.001},
    )

    # Benchmark FLOPs for AdamTrainer
    print("Benchmarking FLOPs for AdamTrainer")
    benchmark_flops(
        trainer_class=AdamTrainer,
        model=model_adam,
        dataset_loader=dataset_loader,
    )
