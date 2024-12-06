import os
import numpy as np
import onnx
import onnxruntime as ort


# The directory of your input and output data
input_data_dir1 = r"D:\Pangu-Weather-ReadyToGo\Test-Data\surface_data"
input_data_dir2 = r"D:\Pangu-Weather-ReadyToGo\Test-Data\upper_data"
output_data_dir = r"D:\Pangu-Weather-ReadyToGo\Test-Data\output_data"
model_24 = onnx.load("D:\Pangu-Weather-ReadyToGo\Pangu-Weather-main\models\pangu_weather_24.onnx")
file_name_surface='mslp-10mU-10mV-2mT-2024-08-01T00.npy'
file_name_upper="z-q-t-u-v-2024-08-01T00.npy"
# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
run_time=6
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession(r"D:\Pangu-Weather-ReadyToGo\Pangu-Weather-main\models\pangu_weather_24.onnx",
                                     sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir2, file_name_upper)).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir1, file_name_surface)).astype(np.float32)
# Run the inference session
input_24, input_surface_24 = input, input_surface
for i in range(run_time):
    output, output_surface = ort_session_24.run(None, {'input':input_24, 'input_surface':input_surface_24})
    input_24, input_surface_24 = output, output_surface
    # Save the results
    file_name_upper_out = f'z-q-t-u-v-2024-08-0{i+2}T00.npy'
    file_name_surface_out = f'mslp-10mU-10mV-2mT-2024-08-0{i+2}T00.npy'
    np.save(os.path.join(output_data_dir, f'{file_name_upper_out}'), output)
    np.save(os.path.join(output_data_dir, f'{file_name_surface_out}'), output_surface)
    print(f'2024-08-0{i+2}T00 run successfully')
  # Your can save the results here