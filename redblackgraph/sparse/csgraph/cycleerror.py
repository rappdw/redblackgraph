
class CycleError(ValueError):

    def __init__(self, *args, **kwargs): # real signature unknown
        super(CycleError, self).__init__(args, kwargs)
        self.vertex = kwargs['vertex']
