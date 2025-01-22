import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

max_steps = 150

def normalize(data, baseline):
    new_data = copy.deepcopy(data)

    for i in range(len(data)):
        new_data[i]['mean'] =( (data[i]['mean'] - baseline[i]['mean'])  )/ baseline[i]['std']

        new_data[i]['std'] = data[i]['std'] / baseline[i]['std'] / 5
    return new_data

def avg(data, idx=[0,1,2]):
    new_data = copy.deepcopy(data[0])
    for i in idx:
        new_data['mean'] += data[i]['mean']
        new_data['std'] += data[i]['std']
    new_data['mean'] /= len(idx)
    new_data['std'] /= len(idx)
    return new_data


baseline = [
    {'mean': np.array([2.5, 7.6, 12.0, 16.8, 21.2, 27.0, 31.3, 36.2, 37.8, 42.7]), 'std': np.array([2.03, 2.8, 3.35, 4.05, 4.16, 4.69, 4.74, 5.05, 4.89, 5.23])},
    {'mean': np.array([3.2, 8.9, 13.9, 19.5, 24.5, 31.0, 36.0, 41.4, 43.7, 49.1]), 'std': np.array([2.55, 3.64, 4.3, 5.14, 5.3, 5.9, 6.09, 6.48, 6.41, 6.76])},
    {'mean': np.array([3.8, 10.3, 15.8, 22.1, 27.9, 34.9, 40.5, 46.5, 49.5, 55.5]), 'std': np.array([2.85, 4.03, 4.85, 5.76, 5.94, 6.56, 6.7, 7.16, 7.09, 7.61])},
    {'mean': np.array([2.6, 7.4, 12.1, 16.7, 21.4, 26.3, 31.7, 35.8, 38.0, 42.8]), 'std': np.array([2.05, 2.9, 3.32, 4.06, 4.3, 4.65, 4.81, 4.98, 4.97, 5.28])},
    {'mean': np.array([3.2, 8.7, 13.9, 19.2, 24.6, 30.2, 36.2, 40.9, 43.7, 49.1]), 'std': np.array([2.55, 3.6, 4.21, 5.05, 5.38, 5.78, 6.0, 6.31, 6.4, 6.83])},
    {'mean': np.array([3.8, 10.0, 15.8, 21.7, 27.8, 34.0, 40.7, 45.9, 49.3, 55.4]), 'std': np.array([2.83, 4.0, 4.66, 5.53, 5.89, 6.27, 6.54, 6.92, 7.05, 7.48])}
]

diffuser = [
    {'mean': np.array([5.5, 13.0, 32.5, 54.5, 77.1, 85.2, 91.4, 101.8, 103.1, 108.1]), 'std': np.array([21.73, 28.8, 50.2, 61.08, 64.64, 62.58, 59.99, 57.34, 56.07, 53.12])},
    {'mean': np.array([13.8, 31.5, 64.4, 97.6, 120.1, 128.9, 133.5, 140.9, 142.5, 144.9]), 'std': np.array([39.11, 53.12, 66.9, 64.95, 53.93, 45.93, 40.35, 30.5, 27.32, 22.37])},
    {'mean': np.array([27.7, 54.0, 89.3, 123.9, 139.2, 143.2, 146.1, 148.7, 149.0, 149.2]), 'std': np.array([54.99, 65.65, 67.67, 52.21, 34.92, 27.3, 20.25, 11.7, 9.9, 8.76])},
    {'mean': np.array([4.3, 14.6, 34.2, 58.8, 79.6, 88.4, 96.8, 110.3, 112.0, 116.5]), 'std': np.array([17.44, 32.44, 51.92, 62.91, 64.88, 62.48, 59.44, 55.1, 53.58, 50.29])},
    {'mean': np.array([19.9, 41.0, 72.8, 106.4, 125.4, 134.5, 138.7, 145.0, 146.2, 147.0]), 'std': np.array([47.63, 60.45, 68.46, 62.4, 50.12, 40.41, 34.22, 22.94, 19.72, 17.18])},
    {'mean': np.array([31.7, 67.1, 101.2, 127.5, 139.3, 143.4, 146.0, 148.1, 148.8, 149.1]), 'std': np.array([58.46, 69.57, 65.47, 49.47, 35.0, 27.09, 20.68, 13.87, 10.94, 9.31])}
]

trans = [
    {'mean': np.array([2.4, 7.2, 11.8, 19.8, 47.1, 116.8, 149.5, 150.0, 150.0, 150.0]), 'std': np.array([1.77, 2.52, 6.83, 22.36, 52.7, 55.58, 7.48, 0.0, 0.0, 0.0])},
    {'mean': np.array([3.4, 9.1, 15.7, 30.9, 88.2, 148.7, 150.0, 150.0, 150.0, 150.0]), 'std': np.array([6.99, 10.48, 19.5, 39.91, 63.94, 12.55, 0.0, 0.0, 0.0, 0.0])},
    {'mean': np.array([4.6, 11.3, 20.3, 47.0, 111.1, 148.4, 149.3, 149.3, 149.3, 149.3]), 'std': np.array([10.63, 14.41, 26.06, 52.75, 58.03, 14.53, 10.61, 10.61, 10.61, 10.61])},
    {'mean': np.array([2.6, 7.4, 12.6, 22.3, 54.4, 126.5, 149.8, 150.0, 150.0, 150.0]), 'std': np.array([4.13, 5.52, 10.87, 27.99, 57.21, 49.32, 5.13, 1.51, 1.51, 1.51])},
    {'mean': np.array([2.6, 7.4, 12.6, 22.3, 54.4, 126.5, 149.8, 150.0, 150.0, 150.0]), 'std': np.array([4.13, 5.52, 10.87, 27.99, 57.21, 49.32, 5.13, 1.51, 1.51, 1.51])},
    {'mean': np.array([2.6, 7.4, 12.6, 22.3, 54.4, 126.5, 149.8, 150.0, 150.0, 150.0]), 'std': np.array([4.13, 5.52, 10.87, 27.99, 57.21, 49.32, 5.13, 1.51, 1.51, 1.51])}
]

# VLM
vlm = [
    {'mean': np.array([7.5, 17.3, 25.5, 31.1, 36.1, 40.8, 58.6, 73.3, 75.1, 85.3]), 'std': np.array([1.64, 2.12, 22.34, 24.05, 25.42, 24.47, 31.39, 41.13, 41.37, 44.88])},
    {'mean': np.array([32.1, 65.0, 78.1, 86.0, 98.3, 150.0, 150.0, 150.0, 150.0, 150.0]), 'std': np.array([68.15, 69.76, 64.3, 59.37, 56.76, 0.0, 0.0, 0.0, 0.0, 0.0])},
    {'mean': np.array([2.5, 7.2, 14.9, 26.6, 45.0, 49.0, 54.0, 61.7, 72.5, 75.8]), 'std': np.array([1.66, 2.29, 20.78, 35.5, 49.62, 48.93, 48.82, 46.01, 51.2, 49.88])},
    {'mean': np.array([2.5, 7.3, 23.6, 31.4, 51.3, 147.5, 147.5, 147.7, 148.4, 148.5]), 'std': np.array([1.65, 2.13, 28.81, 28.65, 41.41, 17.56, 17.14, 16.21, 13.3, 12.73])},
    {'mean': np.array([2.7, 8.2, 30.5, 46.0, 64.3, 150.0, 150.0, 150.0, 150.0, 150.0]), 'std': np.array([2.0, 11.51, 45.68, 52.54, 53.42, 0.0, 0.0, 0.0, 0.0, 0.0])},
    {'mean': np.array([2.6, 7.3, 18.5, 28.3, 52.7, 113.6, 114.6, 116.4, 150.0, 150.0]), 'std': np.array([1.65, 2.24, 29.63, 38.18, 51.8, 56.39, 54.75, 52.07, 0.0, 0.0])}
]

# BHD
bhd = [ 
    {'mean': np.array([4.6, 11.7, 27.3, 45.8, 63.5, 72.3, 79.3, 92.2, 94.0, 99.6]), 'std': np.array([18.03, 25.67, 44.53, 56.2, 61.31, 60.64, 58.87, 58.15, 56.92, 54.42])},
    {'mean': np.array([5.3, 13.6, 30.4, 51.2, 70.5, 78.6, 86.8, 99.7, 101.8, 107.8]), 'std': np.array([20.75, 30.38, 48.14, 59.26, 63.34, 61.8, 59.73, 57.55, 56.19, 53.18])}

]

chd = [
    {'mean': np.array([4.5, 12.1, 23.4, 40.2, 56.4, 65.7, 73.7, 84.9, 87.7, 94.2]), 'std': np.array([18.04, 26.76, 39.41, 51.88, 58.28, 58.59, 57.53, 57.6, 56.58, 54.52])},
    {'mean': np.array([4.5, 12.1, 23.4, 40.2, 56.4, 65.7, 73.7, 84.9, 87.7, 94.2]), 'std': np.array([18.04, 26.76, 39.41, 51.88, 58.28, 58.59, 57.53, 57.6, 56.58, 54.52])}
]


diffuser = normalize(diffuser, baseline)
trans = normalize(trans, baseline)
vlm = normalize(vlm, baseline)
bhd = normalize(bhd, baseline)
chd = normalize(chd, baseline)

diffuser_single = avg(diffuser, [0, 1, 2])
trans_single = avg(trans, [0, 1, 2])
vlm_single = avg(vlm, [0, 1, 2])
bhd_single = avg(bhd, [0])
chd_single = avg(chd, [0])

diffuser_multi = avg(diffuser, [3, 4, 5])
trans_multi = avg(trans, [3, 4, 5])
vlm_multi = avg(vlm, [3, 4, 5])
bhd_multi = avg(bhd, [1])
chd_multi = avg(chd, [1])


x = range(1, 11)

plt.figure()
plt.errorbar(x, diffuser_single['mean'], yerr=diffuser_single['std'], label='d1', color='blue', fmt='-o')
plt.errorbar(x, trans_single['mean'], yerr=trans_single['std'], label='t1', color='green', fmt='-o')
plt.errorbar(x, vlm_single['mean'], yerr=vlm_single['std'], label='vlm', color='red', fmt='-o')
plt.errorbar(x, bhd_single['mean'], yerr=bhd_single['std'], label='bhd', color='purple', fmt='-o')
plt.errorbar(x, chd_single['mean'], yerr=chd_single['std'], label='chd', color='orange', fmt='-o')
plt.legend()
plt.savefig('single.png')

plt.figure()
plt.errorbar(x, diffuser_multi['mean'], yerr=diffuser_multi['std'], label='d1', color='blue', fmt='-o')
plt.errorbar(x, trans_multi['mean'], yerr=trans_multi['std'], label='t1', color='green', fmt='-o')
plt.errorbar(x, vlm_multi['mean'], yerr=vlm_multi['std'], label='vlm', color='red', fmt='-o')
plt.errorbar(x, bhd_multi['mean'], yerr=bhd_multi['std'], label='bhd', color='purple', fmt='-o')
plt.errorbar(x, chd_multi['mean'], yerr=chd_multi['std'], label='chd', color='orange', fmt='-o')
plt.legend()
plt.savefig('multi.png')



