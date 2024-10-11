import librosa
import numpy as np
import torch
from models.modelfile import verhm
import random
import argparse
from scipy.signal import savgol_filter
import os, subprocess
import shlex

# Function to perform emotion prediction
def perform_emotion_prediction(args):
    result_dir = args.output_path  # Directory to store results
    os.makedirs(result_dir, exist_ok=True)  # Create the result directory if it doesn't exist
    eye_patterns = [  # Define eye patterns for visualization
        np.array([0.36537236, 0.950235724, 0.95593375, 0.916715622, 0.367256105, 0.119113259, 0.025357503]),
        np.array([0.234776169, 0.909951985, 0.944758058, 0.777862132, 0.191071674, 0.235437036, 0.089163929]),
        np.array([0.870040774, 0.949833691, 0.949418545, 0.695911646, 0.191071674, 0.072576277, 0.007108896]),
        np.array([0.000307991, 0.556701422, 0.952656746, 0.942345619, 0.425857186, 0.148335218, 0.017659493])
    ]
    emotion_model = verhm(args)  # Initialize the emotion prediction model
    emotion_model.load_state_dict(torch.load(args.pre_trained_model, map_location=torch.device(args.device)), strict=False)  # Load pre-trained model
    emotion_model = emotion_model.to(args.device)  # Move model to appropriate device (CPU or GPU)
    emotion_model.eval()  # Set model to evaluation mode
    input_voice = args.input_voice  # Path to input voice file
    input_file = os.path.splitext(os.path.basename(input_voice))[0]  # Extract file name without extension
    voice_data, sampling_rate = librosa.load(input_voice, sr=16000)  # Load voice data using librosa
    voice_tensor = torch.FloatTensor(voice_data).unsqueeze(0).to(args.device)  # Convert voice data to tensor
    level_tensor = torch.tensor([1]).to(args.device)  # Define level tensor
    person_tensor = torch.tensor([0]).to(args.device)  # Define person tensor
    emotion_prediction = emotion_model.predict(voice_tensor, level_tensor, person_tensor)  # Perform emotion prediction
    emotion_prediction = emotion_prediction.squeeze().detach().cpu().numpy()  # Convert prediction to numpy array
    if args.post_processing:  # If post-processing is enabled
        output = np.zeros_like(emotion_prediction)  # Initialize output array
        for i in range(emotion_prediction.shape[1]):  # Apply Savitzky-Golay filter to each emotion dimension
            output[:, i] = savgol_filter(emotion_prediction[:, i], 5, 2)
        output[:, 8:10] = 0  # Clear certain emotion dimensions
        i = random.randint(0, 60)  # Randomize initial index for eye patterns
        while i < output.shape[0] - 7:  # Apply eye patterns at random intervals
            eye_pattern = random.choice(eye_patterns)  # Select a random eye pattern
            output[i:i + 7, 8:10] = eye_pattern  # Apply eye pattern to output
            time1 = random.randint(60, 180)  # Randomize time interval before next eye pattern
            i += time1  # Increment index for next eye pattern
        np.save(os.path.join(result_dir, f"{input_file}.npy"), output)  # Save processed output
    else:
        np.save(os.path.join(result_dir, f"{input_file}.npy"), emotion_prediction)  # Save raw prediction

# Function to render output video
def render_output_video(args):
    voice_name = os.path.splitext(os.path.basename(args.input_voice))[0]  # Extract voice file name
    image_dir = os.path.join(args.output_path, voice_name)  # Directory to store images
    os.makedirs(image_dir, exist_ok=True)  # Create the image directory if it doesn't exist
    image_template = os.path.join(image_dir, "%d.png")  # Template for image filenames
    output_path = os.path.join(args.output_path, f"{voice_name}.mp4")  # Output video path
    blender_exe = args.b_path  # Path to Blender executable
    blender_script = "render/main.py"  # Path to Blender script
    blender_blend = "render/main.blend"  # Path to Blender blend file
    cmd = f'{blender_exe} -t 64 -b {blender_blend} -P {blender_script} -- "{args.output_path}" "{voice_name}"'  # Command to run Blender
    cmd = shlex.split(cmd)  # Split command for subprocess
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # Run Blender subprocess
    while p.poll() is None:  # While subprocess is running
        line = p.stdout.readline()  # Read subprocess output
        line = line.strip()  # Strip whitespace
        if line:  # If output line is not empty
            print(f'[{line}]')  # Print the output line
    # if p.returncode == 0:  # If subprocess exits successfully
    #     generate_value = verhm(args)
    #     generate_value.generate_value()
    # else:
    #     print('Subprogram failed')  # Print failure message
    generate_value = verhm(args)
    generate_value.generate_value()

    cmd = f'ffmpeg -r 30 -i "{image_template}" -i "{args.input_voice}" -pix_fmt yuv420p -strict -2 -s 512x768 "{output_path}" -y'  # Command to render video using ffmpeg
    subprocess.call(cmd, shell=True)  # Execute ffmpeg command

    cmd = f'rm -rf "{image_dir}"'  # Command to remove image directory
    subprocess.call(cmd, shell=True)  # Execute command to remove image directory

# Main function


import argparse

def main():
    # Create an argument parser with a description
    parser = argparse.ArgumentParser(
        description='Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring')

    # Define various command line arguments
    parser.add_argument("--input_voice", type=str, default="input_voice/hap.wav", 
                        help='path of the test data')  # Input voice file path
    parser.add_argument("--blendshapes", type=int, default=52, 
                        help='number of blendshapes:52')  # Number of blendshapes
    parser.add_argument("--features", type=int, default=832, 
                        help='number of feature dim')  # Number of feature dimensions
    parser.add_argument("--period", type=int, default=30, 
                        help='number of period')  # Number of periods
    parser.add_argument("--device", type=str, default="cuda", 
                        help='device')  # Device
    parser.add_argument("--output_path", type=str, default="output/", 
                        help='output folder')  # Output folder
    parser.add_argument("--max_seq_len", type=int, default=5000, 
                        help='max sequence length')  # Maximum sequence length
    parser.add_argument("--num_workers", type=int, default=0)  
    parser.add_argument("--batch_size", type=int, default=1)  
    parser.add_argument("--pre_trained_model", type=str, default="pretrain_model/verhm.pth", 
                        help='pre-trained models')  # Path to pre-trained model
    parser.add_argument("--post_processing", action="store_true", 
                        help='to use post processing')  # Whether to use post-processing
    parser.add_argument("--b_path", type=str, default="blender/blender", 
                        help='blender folder')  # Path to Blender folder

    args = parser.parse_args()  # Parse command line arguments

    perform_emotion_prediction(args)  # Perform emotion prediction
    render_output_video(args)  # Render output video

# Call the main function if this script is executed
if __name__ == "__main__":
    main()