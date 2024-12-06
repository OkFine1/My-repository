import xarray as xr
import numpy as np
import os
df=xr.open_dataset(r"D:\Pangu-Weather-ReadyToGo\Test-Data\Z-Q-T-U-V-13levs-2024.08.00.nc")
time_list=np.array(df.valid_time)
#(5,13,721,1440)
#(Z, Q, T, U and V in the exact order),
# 13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa,
# 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa in the exact order)
# print(df.pressure_level)
print(df)
save_dir=r"D:\Pangu-Weather-ReadyToGo\Test-Data\upper_data"
for i in range(len(time_list)):
    time = time_list[i]
    z = np.expand_dims(df.z.loc[time, :, :], 0)
    q = np.expand_dims(df.q.loc[time, :, :], 0)
    t = np.expand_dims(df.t.loc[time, :, :], 0)
    u = np.expand_dims(df.u.loc[time, :, :], 0)
    v = np.expand_dims(df.v.loc[time, :, :], 0)
    res_array = np.concatenate([z,q,t,u,v], axis=0)
    strname=f'z-q-t-u-v-{str(time)[:-16]}'+'.npy'
    save_path = os.path.join(save_dir, strname)
    np.save(save_path, res_array)
    print(strname,"saved successfully")
