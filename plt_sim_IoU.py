import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 17}
matplotlib.rc('font', **font)
LineWidth = 2

# CLD vs threshold
threshold = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CLD_IoU_mean_case1 = [53.74, 53.39, 51.69, 51.90, 51.49, 50.34]
CLD_IoU_OC_case1 = [46.85, 47.20, 45.32, 44.45, 44.40, 42.18]
CLD_IoU_OD_case1 = [60.63, 59.58, 58.07, 59.35, 58.57, 58.51]

CLD_IoU_mean_case2 = [53.37, 52.99, 52.68, 51.65, 51.69, 50.94]
CLD_IoU_OC_case2 = [45.83, 44.74, 45.46, 43.85, 44.40, 43.28]
CLD_IoU_OD_case2 = [60.91, 61.23, 59.91, 59.73, 58.97, 58.61]

# CBS vs sampling times
sampling_times = [5, 10, 15, 20]

CBS_IoU_mean = [54.80, 54.57, 54.37, 53.68]
CBS_IoU_OC = [48.99, 48.89, 47.51, 47.99]
CBS_IoU_OD = [60.60, 60.26, 61.23, 59.38]

# CLD_CBS vs sampling times
# CLD_CBS_IoU_mean_case1_sampling_times = []
# CLD_CBS_IoU_OC_case1_sampling_times = []
# CLD_CBS_IoU_OD_case1_sampling_times = []

CLD_CBS_IoU_mean_case2_sampling_times = [61.87, 60.81, 61.19, 60.64]
CLD_CBS_IoU_OC_case2_sampling_times = [73.61, 72.56, 73.52, 72.37]
CLD_CBS_IoU_OD_case2_sampling_times = [50.13, 49.06, 48.85, 48.91]

# CLD_CBS vs threshold
CLD_CBS_IoU_mean_case1_threshold = []
CLD_CBS_IoU_OC_case1_threshold = []
CLD_CBS_IoU_OD_case1_threshold = []

CLD_CBS_IoU_mean_case2_threshold = []
CLD_CBS_IoU_OC_case2_threshold = []
CLD_CBS_IoU_OD_case2_threshold = []

# plot CLD
# plot_CLD_IoU_mean_case1 = plt.plot(threshold, CLD_IoU_mean_case1, "r-x", linewidth=LineWidth)
# plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
# plt.grid()
# plt.xlabel('threshold')
# plt.ylabel('IoU')
# plt.title('mean')
# plt.savefig("./vis/plot_CLD_IoU_mean_case1.png")

# plot_CLD_IoU_OD_case1 = plt.plot(threshold, CLD_IoU_OD_case1, "r-x", linewidth=LineWidth)
# plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
# plt.grid()
# plt.xlabel('threshold')
# plt.ylabel('IoU')
# plt.title('OD')
# plt.savefig("./vis/plot_CLD_IoU_OD_case1.png")

# plot_CLD_IoU_OC_case1 = plt.plot(threshold, CLD_IoU_OC_case1, "r-x", linewidth=LineWidth)
# plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
# plt.grid()
# plt.xlabel('threshold')
# plt.ylabel('IoU')
# plt.title('OC')
# plt.savefig("./vis/plot_CLD_IoU_OC_case1.png")

# plot CBS
# plot_CBS_IoU_mean = plt.plot(sampling_times, CBS_IoU_mean, "r-x", linewidth=LineWidth)
# plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
# plt.grid()
# plt.xlabel('times')
# plt.ylabel('IoU')
# plt.title('mean')
# plt.savefig("./vis/plot_CBS_IoU_mean.png")

# plot_CBS_IoU_OD = plt.plot(sampling_times, CBS_IoU_OD, "r-x", linewidth=LineWidth)
# plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
# plt.grid()
# plt.xlabel('times')
# plt.ylabel('IoU')
# plt.title('OD')
# plt.savefig("./vis/plot_CBS_IoU_OD.png")

plot_CBS_IoU_OC = plt.plot(sampling_times, CBS_IoU_OC, "r-x", linewidth=LineWidth)
plt.gcf().subplots_adjust(left=0.15, top=0.9, bottom=0.15, right=0.95)
plt.grid()
plt.xlabel('times')
plt.ylabel('IoU')
plt.title('OC')
plt.savefig("./vis/plot_CBS_IoU_OC.png")
