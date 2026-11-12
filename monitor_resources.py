#!/usr/bin/env python3
"""
Скрипт для моніторингу ресурсів HLS News OCR пайплайну.
Відстежує CPU, RAM, GPU, Disk I/O та Network у реальному часі.
"""

import psutil
import time
import os
import csv
import subprocess
from datetime import datetime

# --- Налаштування ---
INTERVAL_SEC = 5  # Інтервал оновлення (секунди)
CSV_FILE = "resource_usage_log.csv"
TARGET_PROCESSES = ["hls_news_ocr.py", "ffmpeg", "ollama", "qdrant"]

# --- Ініціалізація GPU (NVIDIA) ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    print("⚠️ GPU (NVIDIA) не знайдено або pynvml не встановлено. Моніторинг GPU вимкнено.")

def get_process_stats():
    """Збирає статистику по цільових процесах."""
    stats = {name: {"cpu": 0.0, "ram_mb": 0.0, "pid": None} for name in TARGET_PROCESSES}
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            for target in TARGET_PROCESSES:
                if target in proc.info['name'] or target in cmdline:
                    # Для точності CPU потрібно викликати cpu_percent() двічі з інтервалом, 
                    # але для простоти беремо приблизне значення або 0.0 при першому запуску
                    cpu = proc.cpu_percent(interval=None)
                    ram_mb = proc.info['memory_info'].rss / (1024 * 1024) if proc.info['memory_info'] else 0
                    
                    stats[target]["cpu"] += cpu
                    stats[target]["ram_mb"] += ram_mb
                    stats[target]["pid"] = proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return stats

def get_gpu_stats():
    """Збирає статистику GPU."""
    if not GPU_AVAILABLE:
        return {"util": 0, "vram_used_mb": 0, "vram_total_mb": 0}
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "util": util,
            "vram_used_mb": mem_info.used / (1024 * 1024),
            "vram_total_mb": mem_info.total / (1024 * 1024)
        }
    except Exception:
        return {"util": 0, "vram_used_mb": 0, "vram_total_mb": 0}

def main():
    print(f"🚀 Запуск моніторингу (інтервал: {INTERVAL_SEC} сек). Натисніть Ctrl+C для зупинки.")
    print(f"📁 Дані зберігаються у: {os.path.abspath(CSV_FILE)}\n")

    # Ініціалізація CSV
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Total_CPU_%", "Total_RAM_MB", "Total_RAM_%",
            "Disk_Read_MB_s", "Disk_Write_MB_s", "Net_In_MB_s", "Net_Out_MB_s",
            "GPU_Util_%", "GPU_VRAM_MB", "OCR_CPU_%", "OCR_RAM_MB",
            "FFmpeg_CPU_%", "FFmpeg_RAM_MB", "Ollama_CPU_%", "Ollama_RAM_MB"
        ])

    # Початкові значення для розрахунку швидкості (I/O та Мережа)
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    
    # Перший виклик для ініціалізації лічильників CPU процесів
    get_process_stats()
    time.sleep(1)

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. Системні ресурси
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used_mb = ram.used / (1024 * 1024)
            ram_percent = ram.percent

            # 2. Дисковий I/O (різниця за INTERVAL_SEC)
            curr_disk = psutil.disk_io_counters()
            disk_read_mb = (curr_disk.read_bytes - prev_disk.read_bytes) / (1024 * 1024) / INTERVAL_SEC
            disk_write_mb = (curr_disk.write_bytes - prev_disk.write_bytes) / (1024 * 1024) / INTERVAL_SEC
            prev_disk = curr_disk

            # 3. Мережа (різниця за INTERVAL_SEC)
            curr_net = psutil.net_io_counters()
            net_in_mb = (curr_net.bytes_recv - prev_net.bytes_recv) / (1024 * 1024) / INTERVAL_SEC
            net_out_mb = (curr_net.bytes_sent - prev_net.bytes_sent) / (1024 * 1024) / INTERVAL_SEC
            prev_net = curr_net

            # 4. Процеси
            proc_stats = get_process_stats()

            # 5. GPU
            gpu_stats = get_gpu_stats()

            # --- Вивід у консоль (Dashboard) ---
            os.system('clear' if os.name == 'posix' else 'cls')
            print("="*60)
            print(f"🕒 Час: {timestamp} | Інтервал: {INTERVAL_SEC}с")
            print("-" * 60)
            print(f"💻 CPU: {cpu_percent:>5.1f}%  |  RAM: {ram_used_mb:>6.0f} MB ({ram_percent}%)")
            print(f"💾 Disk: Read: {disk_read_mb:>5.1f} MB/s | Write: {disk_write_mb:>5.1f} MB/s")
            print(f"🌐 Net:  In:  {net_in_mb:>5.1f} MB/s | Out:   {net_out_mb:>5.1f} MB/s")
            if GPU_AVAILABLE:
                print(f"🎮 GPU:  Util: {gpu_stats['util']:>3.0f}% | VRAM: {gpu_stats['vram_used_mb']:>5.0f} / {gpu_stats['vram_total_mb']:.0f} MB")
            print("-" * 60)
            print("📊 ПРОЦЕСИ:")
            for name, data in proc_stats.items():
                if data['pid'] is not None:
                    print(f"   {name:<12} | CPU: {data['cpu']:>5.1f}% | RAM: {data['ram_mb']:>6.0f} MB (PID: {data['pid']})")
                else:
                    print(f"   {name:<12} | Не запущено")
            print("="*60)
            print("Натисніть Ctrl+C для зупинки та збереження звіту.")

            # --- Запис у CSV ---
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, cpu_percent, ram_used_mb, ram_percent,
                    round(disk_read_mb, 2), round(disk_write_mb, 2), round(net_in_mb, 2), round(net_out_mb, 2),
                    gpu_stats['util'], round(gpu_stats['vram_used_mb'], 1),
                    proc_stats["hls_news_ocr.py"]["cpu"], proc_stats["hls_news_ocr.py"]["ram_mb"],
                    proc_stats["ffmpeg"]["cpu"], proc_stats["ffmpeg"]["ram_mb"],
                    proc_stats["ollama"]["cpu"], proc_stats["ollama"]["ram_mb"]
                ])

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n✅ Моніторинг зупинено. Дані збережено у", CSV_FILE)
        if GPU_AVAILABLE:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()