from abc import ABC, abstractmethod


# Base search space class, that all others should inherit from
# The basic initialization requirements are a logger (default to print) for any information that must be recorded
# At bare minimum, each class shall implement the following methods:
# 1. Sample a completely random network
# 2. Sample a network to be profiled; must select a specific block in a specific place
# 3. Method to query the accuracy of architectures
# 4. A method that takes a specification and returns a string meaning what said specification means
# Additionally, each class with contain a "metrics" tuple, defining the information that can be retrieved for an
# architecture. The metrics tuple must at least contain "accuracy"
class BaseSpace(ABC):
    def __init__(self, logger=print):
        self.logger = logger

        self.metrics = ("accuracy",)

    # The random sample function shall take a parameter determining the number of random samples to perform
    # and shall return a list of architectures whose length is equal to that number
    @abstractmethod
    def random_sample(self, n=10):
        self.logger("Call to 'random_sample' of Abstract Class BaseSpace")
        self.logger("n =", n)
        return NotImplementedError

    @abstractmethod
    def block_sample(self, n=10, **kwargs):
        self.logger("Call to 'block_sample' of Abstract Class BaseSpace")
        self.logger("n =", n)
        self.logger("kwargs:{}".format(kwargs))
        return NotImplementedError

    @abstractmethod
    def accuracy_measure(self, architectures):
        self.logger("Call to 'accuracy_measure' of Abstract Class BaseSpace")
        self.logger("Architectures: {}".format(architectures))
        return NotImplementedError

    @abstractmethod
    def block_meaning(self, **kwargs):
        self.logger("Call to 'block_meaning' of Abstract Class BaseSpace")
        self.logger("kwargs:{}".format(kwargs))
        return NotImplementedError

    # This function fully evaluates a list of given architectures (or constructs them) using all metrics available
    # to the class.
    # It is not an abstract method as the function should be generalizable to all search spaces.
    def fully_evaluate_block(self, n=10, archs=None, **kwargs):
        if archs is None:
            archs = self.block_sample(n, **kwargs)

        results = {}
        for metric in self.metrics:
            metric_function = getattr(self, '%s_measure' % str.lower(metric)).__name__
            metric_results = eval("self.%s(archs)" % metric_function)
            results[metric] = metric_results

        return results, archs
