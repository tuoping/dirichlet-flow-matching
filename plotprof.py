import psutil
import matplotlib.pyplot as plt

# Monitor memory usage
memory_usage = []
for _ in range(100):
    memory_usage.append(psutil.virtual_memory().used)

# Plot memory usage
plt.plot(memory_usage)
plt.xlabel('Time')
plt.ylabel('Memory Usage (bytes)')
plt.show()
plt.savefig("profiling.png")
