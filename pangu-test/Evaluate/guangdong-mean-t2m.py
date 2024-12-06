import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

df=xr.open_dataset(r"D:\Pangu-Weather-ReadyToGo\Test-Data\mslp-10mU-10mV-2mT-2024.08.00.nc")
# u10 v10 t2m msl (valid_time, latitude, longitude) float32 71MB ...
var=df.t2m.loc['2024-08-02':'2024-08-07',25.5:20,109.5:117.5]
var_mean=np.mean(var,axis=(1,2))-273.15
time=var.valid_time
#ecmwf的预报
file_path=r"D:\Pangu-Weather-ReadyToGo\Test-Data\ECMWF_2024-08-fc.grib"
ds = xr.open_dataset(file_path,engine='cfgrib')
t2m_ecmwf=ds.t2m.loc['2024-08-01':'2024-08-06',25.5:20,109.5:117.5]
t2m_ecmwf=np.array(np.mean(t2m_ecmwf,axis=(1,2))-273.15)
# 设置文件夹路径
folder_path = r"D:\Pangu-Weather-ReadyToGo\Test-Data\output_nc\24h"  # 替换为你的文件夹路径

# 获取以 "mslp-10mU-10mV-2mT" 开头的所有 .nc 文件
file_list = glob.glob(os.path.join(folder_path, 'mslp-10mU-10mV-2mT*.nc'))#z-q-t-u-v*.nc
# 按照文件名中的日期部分排序（假设日期部分为文件名的第20-29位）
file_list.sort(key=lambda x: x.split('-')[-1].split('.')[0])
print(file_list)
combined_data=[]
# 逐个读取文件并提取数据
for file in file_list:
    nc_file=xr.open_dataset(file)
    #v_component_of_wind_10m u_component_of_wind_10m mean_sea_level_pressure
    t2m = nc_file.temperature_2m.loc[25.5:20,109.5:117.5]
    t2m=np.mean(t2m, axis=(0, 1))
    combined_data.append(t2m)
combined_data=np.array(combined_data)-273.15
# 计算误差的绝对值
error1 = np.abs(var_mean - combined_data)
error2 = np.abs(var_mean - t2m_ecmwf)

plt.figure(figsize=[8,4])
# 设置主图风格为Nature风格
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.frameon': False,
    'legend.fontsize': 10
})
plt.title('2m Temperature [$^o$C]', fontsize=12)
plt.ylim(27, 29.6)
plt.plot(time, var_mean, label='ERA5', c='k', lw=1.5)
plt.plot(time, combined_data, label='PanGu', c='red', lw=1.5)
plt.plot(time,t2m_ecmwf,label='EC', c='grey', lw=1.5)
plt.legend(loc='lower right')
# 创建嵌入的误差柱状图
inset_ax = plt.gca().inset_axes([0.08, 0.68, 0.25, 0.3])  # [x, y, width, height]
inset_ax.bar(np.arange(1,7,1),error1, color='red', alpha=1)
inset_ax.bar(np.arange(1,7,1),error2, color='k', alpha=0.5)
# # 对误差进行线性回归拟合
# slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(1,7,1), error)
# fit_line = slope * np.arange(1,7,1) + intercept
# inset_ax.plot(np.arange(1,7,1),fit_line,color='blue', lw=1.5)
# 设置误差图的标题和样式
inset_ax.set_title('Absolute Error', fontsize=6)
inset_ax.tick_params(axis='both', which='major', labelsize=8)
inset_ax.spines['top'].set_visible(False)
inset_ax.spines['right'].set_visible(False)
inset_ax.spines['bottom'].set_linewidth(1.0)
inset_ax.spines['left'].set_linewidth(1.0)
plt.show()