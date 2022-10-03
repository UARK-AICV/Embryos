class MetricTemplate():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, output, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
