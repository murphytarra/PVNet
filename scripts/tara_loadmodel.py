import argparse
import os

import torch

from pvnet.data.uk_regional_datamodule import DataModule
from pvnet.load_model import get_model_from_checkpoints


def main():
    """
    Main function to load a data batch and a checkpointed model,
    then get the model's output for that batch.
    """
    parser = argparse.ArgumentParser(
        description="Load data and a PVNet model to get a forecast."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="The full path to the data configuration YAML file.",
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="The full path to the model checkpoint directory.",
    )
    args = parser.parse_args()

    data_config_path = os.path.expanduser(args.config_path)
    checkpoint_path = os.path.expanduser(args.checkpoint_dir)

    if not os.path.exists(data_config_path):
        print(f"ERROR: Data config not found at: {data_config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint directory not found at: {checkpoint_path}")
        return

    print("\n" + "=" * 50)
    print(f"Initializing DataModule with config: {data_config_path}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print("=" * 50)

    try:
        model, model_config, _ = get_model_from_checkpoints([checkpoint_path])

        model.eval()
        print("Model loaded successfully.")

        datamodule = DataModule(
            configuration=data_config_path,
            batch_size=2,
            num_workers=0,
        )
        dataloader = datamodule.train_dataloader()
        print("Loading a batch...")
        batch = next(iter(dataloader))
        print("Data batch loaded successfully.")

        print("\nPassing data through the model...")
        with torch.no_grad():
            outputs = model(batch)

        print("\nSuccessfully received model output!")
        print("-" * 30)

        print(f"Shape of the model output tensor: {outputs.shape}")

        print("\nOutput for the first sample in the batch:")
        print(outputs[0])

    except Exception as e:
        print("\nAn error occurred. Full traceback:")
        raise e


if __name__ == "__main__":
    main()
