# gunicorn.conf.py

import multiprocessing

bind = "0.0.0.0:8080"  # Bind to all IP addresses on port 8080
workers = multiprocessing.cpu_count() * 2 + 1  # Number of worker processes
threads = 2  # Number of threads per worker

# Logging settings
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Timeout settings
timeout = 120  # Worker timeout in seconds
graceful_timeout = 30  # Timeout for graceful worker restart
keepalive = 5  # Keep-alive connections timeout

# Other settings
worker_class = "sync"  # Use synchronous workers
capture_output = True  # Capture stdout and stderr

# Preload the application to reduce startup time
preload_app = True

# Additional settings
daemon = False  # Run Gunicorn in the foreground
pidfile = None  # Do not create a PID file
umask = 0  # Set the umask for the worker processes

# SSL settings (uncomment if needed)
# certfile = "/path/to/certfile.pem"
# keyfile = "/path/to/keyfile.pem"

# Additional environment variables (uncomment if needed)
# raw_env = [
#     "ENV_VAR_NAME=value",
#     "ANOTHER_ENV_VAR=value"
# ]
