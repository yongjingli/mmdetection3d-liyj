import sys
import matplotlib.pyplot as plt

first_timestamp = -1
last_timestamp = -1
cpu_timestamps = []
memory_timestamps = []
cpu_usages = []
memory_usages = []

for line in sys.stdin:
	usage_detail = line.split()

	if usage_detail[0] == "cpu":
		if int(usage_detail[1]) < first_timestamp or first_timestamp == -1:
			first_timestamp = int(usage_detail[1])
		elif int(usage_detail[1]) > last_timestamp or last_timestamp == -1:
			last_timestamp = int(usage_detail[1])

		cpu_timestamps.append(int(usage_detail[1]))
		cpu_usages.append(float(usage_detail[7].replace('%','')))

	elif usage_detail[0] == "memory":
		if int(usage_detail[1]) < first_timestamp or first_timestamp == -1:
			first_timestamp = int(usage_detail[1])
		elif int(usage_detail[1]) > last_timestamp or last_timestamp == -1:
			last_timestamp = int(usage_detail[1])

		memory_timestamps.append(int(usage_detail[1]))
		memory_usages.append(float(usage_detail[7].replace('k',''))/1000.0)

	else:
		print "unkown profiler format"

memory_timestamps = [x-first_timestamp for x in memory_timestamps]
cpu_timestamps = [x-first_timestamp for x in cpu_timestamps]

plt.figure(figsize=(50, 50), dpi=80)

plt.subplot(2,1,1)
plt.plot(memory_timestamps, memory_usages)
plt.title("memory usage statistics")
plt.xlabel("time (second)")
plt.ylabel("memory usage (megabytes)")
plt.xlim(0, last_timestamp-first_timestamp)

plt.subplot(2,1,2)
plt.plot(cpu_timestamps, cpu_usages)
plt.title("cpu usage statistics")
plt.xlabel("time (second)")
plt.ylabel("cpu usage (percent)")
plt.xlim(0, last_timestamp-first_timestamp)

plt.show()

