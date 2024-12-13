import subprocess
import time

# Start the timer
start_time = time.time()

# Command to run the script via Git Bash
command = [
    'C:/Program Files/Git/bin/bash.exe',  # Path to Git Bash executable
    '--login', '-i', './run_experiments.sh',
    '--hparams', 'hparams/P300/BCIAUTP300/EEGNet.yaml', 
    '--data_folder', 'eeg_data', 
    '--output_folder', 'results/P300/BCIAUTP300/E1/', 
    '--nsbj', '15', 
    '--nsess', '7', 
    '--nruns', '1', 
    '--train_mode', 'leave-one-subject-out', 
    '--device', 'cpu'
]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Script output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    print("Error output:\n", e.stderr)


# End the timer
end_time = time.time()

# Calculate the time it took to run the script
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")