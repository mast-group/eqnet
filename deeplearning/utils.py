def check_hyperparameters(required_hyperparameters: set, given_hyperparameters: dict):
    included_hyperparameters = set(given_hyperparameters.keys())
    missing_hyperparameters = required_hyperparameters - included_hyperparameters
    assert len(missing_hyperparameters) == 0, "Hyperparameters are missing: " + str(missing_hyperparameters)

    extra_hyperparameters = included_hyperparameters - required_hyperparameters
    assert len(extra_hyperparameters) == 0, "Extra hyperparameters used: " + str(extra_hyperparameters)


class Bunch:
    """
    A simple class that can be instantiated with arbitrary attributes.
    Look at http://code.activestate.com/recipes/52308/ for more info
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
