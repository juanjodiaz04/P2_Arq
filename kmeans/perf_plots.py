import matplotlib.pyplot as plt

# ==== Datos ====
threads = [32, 64, 128, 256, 512]   # número de hilos por bloque
ex_time_CPU = [2921.444, 2921.444, 2921.444, 2921.444, 2921.444]    # tiempo total algoritmo CPU(ms)
ex_time_GPU1 = [918.57, 910.16, 921.53, 915.09, 912.02]             # tiempo total algoritmo Versión Simple (ms)
ex_time_GPU2 = [482.92, 466.64, 477.98, 513.49, 544.66]             # tiempo total algoritmo Versión Optimizada (ms)

kernel_time_GPU1 = [533.42, 540.90, 552.64, 541.08, 542.41] # tiempo del kernel Versión Simple (ms)
kernel_time_GPU2 = [115.21, 79.44, 125.93, 146.98, 180.96] # tiempo del kernel Versión Optimizada (ms) 
# =================================

plt.figure(figsize=(10, 8))

# ---------------- SUBPLOT 1: tiempo total ----------------
plt.subplot(2, 1, 1)
plt.plot(threads, logex_time_CPU, marker='o')
plt.plot(threads, ex_time_GPU1, marker='o')
plt.plot(threads, ex_time_GPU2, marker='o')
plt.legend(["CPU", "GPU Versión Simple", "GPU Versión Optimizada"])
plt.title("Tiempo Total vs Hilos por Bloque")
plt.xlabel("Hilos por bloque")
plt.ylabel("Tiempo total (ms)")
plt.grid(True)

# ---------------- SUBPLOT 2: tiempo del kernel ----------------
plt.subplot(2, 1, 2)
plt.plot(threads, kernel_time_GPU1, marker='o')
plt.plot(threads, kernel_time_GPU2, marker='o')
plt.legend(["GPU Versión Simple", "GPU Versión Optimizada"])
plt.title("Tiempo del Kernel vs Hilos por Bloque")
plt.xlabel("Hilos por bloque")
plt.ylabel("Tiempo kernel (ms)")
plt.grid(True)

plt.tight_layout()
plt.show()
