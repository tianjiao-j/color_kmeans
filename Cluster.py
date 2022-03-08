class Cluster:
    center = []
    samples = []

    def __init__(self, samples):
        self.center = []
        self.samples = samples

    def get_samples(self):
        return self.samples
