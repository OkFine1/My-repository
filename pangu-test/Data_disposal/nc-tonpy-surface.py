import xarray as xr
import numpy as np
import os
df=xr.open_dataset(r"D:\Pangu-Weather-ReadyToGo\Test-Data\mslp-10mU-10mV-2mT-2024.08.00.nc")
time_list=np.array(df.valid_time)
#(MSLP, U10, V10, T2M in the exact order)
save_dir="D:\Pangu-Weather-ReadyToGo\Test-Data\surface_data"
for i in range(len(time_list)):
    time = time_list[i]
    msl = np.expand_dims(df.msl.loc[time, :, :], 0)
    u10 = np.expand_dims(df.u10.loc[time, :, :], 0)
    v10 = np.expand_dims(df.v10.loc[time, :, :], 0)
    t2m = np.expand_dims(df.t2m.loc[time, :, :], 0)
    res_array = np.concatenate([msl,u10, v10, t2m], axis=0)
    strname=f'mslp-10mU-10mV-2mT-{str(time)[:-16]}'+'.npy'
    save_path = os.path.join(save_dir, strname)
    np.save(save_path, res_array)
    print(strname,"saved successfully")

