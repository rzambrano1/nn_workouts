#!/usr/bin/env python3
"""
Example of a trining loop in the simple manually implemented MLP
"""
# Boilerplate
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Local modules
from nn import MLP

# Configuration composer. Keeps parameters separate from training code 
# In other words, decouples infrastructure from logic by avoiding harcoding 
# parameters into sections separated by multiple lines of code.
import hydra
from omegaconf import DictConfig, OmegaConf

# Experiment tracking.
import wandb

# Hyper-parameter optimization [bayesian optiization instead of grid/random search].
# The objective function for Optuna in Hydra's plug in is the training loop.
# Run HPO with $python train.py --multirun hydra/sweeper=optuna +hpo=optuna
import optuna

# Training loop
@hydra.main(version_base=None, config_path="conf", config_name="config")
def training_loop(cfg : DictConfig) -> None:

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity = cfg.logging.entity,
        # Set the wandb project where this run will be logged.
        project = cfg.logging.project,
        mode=cfg.logging.mode,
        notes=cfg.logging.notes,
        # Track hyperparameters and run metadata.
        # config = {
        #     "learning_rate": cfg.training.learning_rate,
        #     "architecture": cfg.architecture,
        #     "dataset": cfg.data.dataset,
        #     "epochs": cfg.training.epochs,
        # },
        config = OmegaConf.to_container(cfg, resolve=True) # This logs all of the commented out above automatically
    )

    # Creates an instance of a MLP with 3 inputs, two hidden layers with 4 neurons, and 1 output
    nn = MLP(cfg.model.n_inputs, [cfg.model.neurons_in_hidden_layer_1, cfg.model.neurons_in_hidden_layer_2, cfg.model.n_outputs]) 

    # Some features
    xs = cfg.data.features

    # Some labels
    ys = cfg.data.labels

    # An example training loop
    for epoch in range(cfg.training.epochs):

        print(f'--- Epoch: {epoch} ---')

        ypred = [nn(x) for x in xs]                                    # Forward pass
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys,ypred)])   # Calculate loss
        print(f'Loss = {loss.data:.4f}')
        if loss.data < cfg.training.early_stop_threshold:
            print('Early stopping...')
            break
        for p in nn.parameters():                                      # zero.grad - flush the gradients of the previous step
            p.grad = 0.0 
        loss.backward(verbose=False)                                   # Packpropagation step
        for p in nn.parameters():                                      # Optimization step [update weights]
            p.data += -cfg.training.learning_rate * p.grad

        # Log metrics to wandb.
        run.log({"loss": loss.data})

        if epoch % 100 == 0:
            y_true = [float(y) for y in ys]
            y_pred = [float(y_hat.data) for y_hat in ypred]

            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)

            print(f"MAE: {mae:.4f}")
            print(f"R2: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")

            wandb.log({
                "val/mae": mae,
                "val/r2": r2,
                "val/rmse": rmse
            })
    
    print(f'labels = {ys}')
    print(f'preds = {[round(float(y_hat.data), 4) for y_hat in ypred]}')

    # Finish the run and upload any remaining data.
    run.finish()

    return float(loss.data) # Returning metric Optuna is optimizing

if __name__ == "__main__":
    training_loop()