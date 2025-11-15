import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import sys
INPUT_FILENAME = r"C:\Users\Abdessamed\Music\my_file.wav"
OUTPUT_FILENAME = "cleaned_audio.wav"

# Set to True if you have a separate clip of just the noise
HAS_SEPARATE_NOISE_FILE = False
# Time in seconds to sample noise from the start of the audio
TIME_FOR_NOISE_ESTIMATE = 0.5
# 1. Load the noisy audio file
try:
    rate, data = wavfile.read(INPUT_FILENAME)
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILENAME}' not found. Please check the path.")
    sys.exit(1)  # Exit the script with an error code
# 2. Handle Data Conversion and Stereo-to-Mono
original_dtype = data.dtype
# Convert to float for processing and handle data scaling
if data.dtype != np.float64:
    # Scale data to the range [-1.0, 1.0] for processing
    # Use the original maximum value based on the file's data type
    data = data.astype(np.float64) / np.iinfo(data.dtype).max
# Check if the audio is stereo (has two channels/columns) and convert to mono
if data.ndim > 1:
    print("Stereo audio detected. Converting to mono by averaging channels.")
    # Convert to mono by taking the average across the channel axis (axis=1)
    data = np.mean(data, axis=1)
# 3. Determine the noise profile
if not HAS_SEPARATE_NOISE_FILE:
    # Calculate length of the noise segment
    noise_len = int(rate * TIME_FOR_NOISE_ESTIMATE)
    # Ensure noise_len does not exceed the total data length
    if noise_len >= len(data):
        print("Error: Audio file is too short to estimate noise from the start.")
        sys.exit(1)
    # Sample the first part of the audio as the noise profile
    noise_data = data[:noise_len]
else:
    # Placeholder for loading a separate noise clip
    noise_data = None
# 4. Perform noise reduction
print(f"Performing aggressive noise reduction on {len(data) / rate:.2f} seconds of audio...")
reduced_noise = nr.reduce_noise(
    y=data,
    sr=rate,
    y_noise=noise_data,
    stationary=True,
    # INCREASED for more aggressive noise removal (95%)
    prop_decrease=0.95
)
print("Reduction complete.")
# 5. Prepare data for saving
# Rescale the float data back to the original integer range
max_value = np.iinfo(original_dtype).max
scaled_data = (reduced_noise * max_value).astype(original_dtype)
# 6. Save the processed audio file
try:
    wavfile.write(OUTPUT_FILENAME, rate, scaled_data)
    print(f"\nSuccessfully cleaned audio saved to '{OUTPUT_FILENAME}'")
except Exception as e:
    print(f"Error saving file: {e}")