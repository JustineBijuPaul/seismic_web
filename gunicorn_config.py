# gunicorn.conf.py

import multiprocessing

bind = "0.0.0.0:8080"  # Bind to all IP addresses on port 8080
workers = multiprocessing.cpu_count() * 2 + 1  # Number of worker processes
threads = 2 
