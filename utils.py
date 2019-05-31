import time


def timer(func):
    def wrapper(*args, **kwargs):
        initial_time = time.time()
        result = func(*args, **kwargs)
        final_time = time.time()
        print(f'Time taken: {final_time - initial_time}s')
        return result

    return wrapper
