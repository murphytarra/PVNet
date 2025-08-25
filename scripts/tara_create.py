import argparse
import os

from pvnet.data.uk_regional_datamodule import DataModule


def main():
    """
    Main function to load a batch of data using a PVNet data configuration file
    provided as a command-line argument.
    """
    parser = argparse.ArgumentParser(
        description="Load a batch of data using a PVNet data configuration file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="The full path to the data configuration YAML file.",
    )
    args = parser.parse_args()

    data_config_path = os.path.expanduser(args.config_path)

    if not os.path.exists(data_config_path):
        print("=" * 60)
        print(f"ERROR: The file was not found at the expanded path: {data_config_path}")
        print("Please ensure the path is correct.")
        print("=" * 60)
        return

    print("\n" + "=" * 50)
    print(f"Initializing DataModule with config: {data_config_path}")
    print("=" * 50)

    try:
        datamodule = DataModule(
            configuration=data_config_path,
            batch_size=2,
            num_workers=0,
        )

        dataloader = datamodule.train_dataloader()
        print("Loading a batch... (This may take a moment depending on your data)")
        batch = next(iter(dataloader))

        ecmwf_data_tensor = batch["nwp"]["ecmwf"]["nwp"]

        print("\nSuccessfully loaded one batch!")
        print("-" * 30)

        print(f"Shape of the ECWMF tensor in the batch: {ecmwf_data_tensor.shape}")
        print("(Batch Size, Sequence Length, Channels, Height, Width)")

        print("\nData from the first sample, first timestep, first channel:")
        print(ecmwf_data_tensor[0, 0, 0, :, :])

    except Exception as e:
        print("\nAn error occurred. Full traceback:")
        raise e


if __name__ == "__main__":
    main()
