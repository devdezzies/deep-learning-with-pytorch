"""
contains various utility functions for pytorch model training and saving
"""
import torch 
from pathlib import Path 

def save_model(model: torch.nn.Module, 
              target_dir: str, 
              model_name: str):
    """Saves a pytorch model to a target directory

    Args:
        model: target pytorch model
        target_dir: string of target directory path to store the saved models 
        model_name: a filename for the saved model. Should be included either ".pth" or ".pt" as 
        the file extension.
    """
    # create target directory 
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path 
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    # save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
