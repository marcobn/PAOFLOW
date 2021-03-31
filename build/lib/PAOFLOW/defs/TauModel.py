
class TauModel:

  def __init__ ( self, function=None, params=None, weight=1. ):

    self.function = function
    self.params = params
    self.weight = weight

  def evaluate ( self, temp, eigs ):
    return self.function(temp, eigs, self.params)
