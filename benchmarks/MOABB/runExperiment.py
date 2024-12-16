import subprocess
import time
import os
import shutil

def run_experiment(command, output_time_file=None):
    """
    Runs the experiment with the specified command and saves the execution time to a file if provided.
    """
    start_time = time.time()
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Script output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print("Error output:\n", e.stderr)
        return None
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if output_time_file:
        with open(output_time_file, 'w') as file:
            file.write(f"Execution time: {execution_time:.2f} seconds\n")
        print(f"Execution time saved to {output_time_file}")
    



# Locate Git Bash dynamically
git_bash_path = shutil.which("bash")
if git_bash_path is None:
    raise FileNotFoundError("Git Bash not found. Ensure Git is installed and added to PATH.")

print(git_bash_path)

# Define the base command
base_command = [
    git_bash_path,  # Use the relative path
    '--login', '-i', './run_experiments.sh',
    '--data_folder', 'eeg_data',  
    '--nsbj', '15', 
    '--nsess', '7', 
    '--train_mode', 'leave-one-subject-out'
]

# Run with a single run first
single_run_command = base_command + ['--nruns', '1'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E0/']
run_experiment(single_run_command, output_time_file="single_run_time.txt")

# Run with 10 runs
full_run_command = base_command + ['--nruns', '10'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E0/']
run_experiment(full_run_command)

full_run_command = base_command + ['--nruns', '10'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet1.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E1/']
run_experiment(full_run_command)

full_run_command = base_command + ['--nruns', '10'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet2.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E2/']
run_experiment(full_run_command)

full_run_command = base_command + ['--nruns', '10'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet3.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E3/']
run_experiment(full_run_command)

full_run_command = base_command + ['--nruns', '10'] + ['--hparams', 'hparams/P300/BCIAUTP300/EEGNet4.yaml'] + ['--output_folder', 'results/P300/BCIAUTP300/E4/']
run_experiment(full_run_command)



