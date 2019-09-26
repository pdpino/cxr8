from datetime import datetime
import time

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')