import os
import numpy as np
import onnx
import onnxruntime as ort
#-------------以下是数据要求-----------
'''
surface: 4 surface variables (MSLP, U10, V10, T2M in the exact order)
upper:(Z, Q, T, U and V) 13levs
(1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa,
 250hPa, 200hPa, 150hPa, 100hPa and 50hPa
721*1440 latitude longitude
initial fields of at 12:00UTC, 2018/09/27.
Note that ndarray (.astype(np.float32)), not in double precision.
'''
# The directory of your input and output data
input_data_dir1 = r"D:\Pangu-Weather-ReadyToGo\Test-Data\surface_data"
input_data_dir2 = r"D:\Pangu-Weather-ReadyToGo\Test-Data\upper_data"
output_data_dir = r"D:\Pangu-Weather-ReadyToGo\Test-Data\output_data"
model_1 = onnx.load("D:\Pangu-Weather-ReadyToGo\Pangu-Weather-main\models\pangu_weather_1.onnx")
file_name_surface='mslp-10mU-10mV-2mT-2024-08-01T00.npy'
file_name_surface_out='mslp-10mU-10mV-2mT-2024-08-01T01.npy'
file_name_upper="z-q-t-u-v-2024-08-01T00.npy"
file_name_upper_out='z-q-t-u-v-2024-08-01T01.npy'
# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_1 = ort.InferenceSession(r"D:\Pangu-Weather-ReadyToGo\Pangu-Weather-main\models\pangu_weather_1.onnx",
                                     sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir2, file_name_upper)).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir1, file_name_surface)).astype(np.float32)
# Run the inference session
output, output_surface = ort_session_1.run(None, {'input':input, 'input_surface':input_surface})
# Save the results
np.save(os.path.join(output_data_dir, f'{file_name_upper_out}'), output)
np.save(os.path.join(output_data_dir, f'{file_name_surface_out}'), output_surface)

