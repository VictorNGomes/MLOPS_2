

class Pipeline:
    def __init__(self):
        self._functions = []

    def add_function(self, func):
        self._functions.append(func)
        return func

    def run(self):
        for func in self._functions:
            func()

pipeline = Pipeline()

def depends_on(*dependencies):
    def decorator(func):
        func.dependencies = dependencies
        pipeline.add_function(func)
        return func
    return decorator


