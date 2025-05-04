import torch
import pandas as pd
from utils.model import Model
from utils.config import EnConfig
from utils.data_loader import data_loader
import os


def generate_predictions(model_weights_path, config, output_csv):
    """
    Load a saved model's weights, generate predictions on the test dataset, and save them to a CSV file.

    Args:
        model_weights_path (str): Path to the saved model weights file (e.g., 'checkpoint/RH_acc_mosi_1_0.5678.pth')
        config (EnConfig): Configuration object matching the training setup
        output_csv (str): Path where the predictions CSV will be saved
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data
    _, test_loader, _ = data_loader(config)

    # Initialize the model with the provided configuration
    model = Model(config).to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # List to store predictions
    predictions = []

    # Generate predictions without computing gradients
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to the device
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)

            # Include context inputs if use_context is True
            if config.use_context:
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)

            # Forward pass through the model
            if config.use_context:
                outputs = model(
                    text_inputs, text_mask,
                    text_context_inputs, text_context_mask,
                    audio_inputs, audio_mask,
                    audio_context_inputs, audio_context_mask
                )
            else:
                outputs = model(
                    text_inputs, text_mask,
                    audio_inputs, audio_mask
                )

            # Extract multimodal predictions ('M')
            preds = outputs['M'].cpu()  # Shape: [batch_size, 1]

            # Collect predictions with video_id and clip_id
            for i in range(len(batch['video_id'])):
                predictions.append({
                    'video_id': batch['video_id'][i],  # String
                    'clip_id': batch['clip_id'][i],  # Integer (assuming df['clip_id'] is int)
                    'prediction': preds[i].item()  # Scalar float value
                })

    # Convert predictions to a DataFrame and save to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    # Path to the saved model weights (update this to your actual file path)
    model_weights_path = 'checkpoint/RH_acc_mosi_1_0.8734.pth'  # Example path, replace with your file

    # Configuration matching the training setup
    # Note: Adjust these parameters to match the config used when training the model
    config = EnConfig(
        batch_size=8,  # Adjust if different during training
        dataset_name='test',
        seed=1,  # Matches filename in your example
        num_hidden_layers=5,  # Default from run.py
        use_context=True,  # Default from run.py
        use_attnFusion=True,  # Default from run.py
        learning_rate = 5e-6,
    )

    # Output CSV file path
    output_csv = 'data/test/label.csv'  # Update to your desired output file name

    # Generate predictions and save to CSV
    generate_predictions(model_weights_path, config, output_csv)
