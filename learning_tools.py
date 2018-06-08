
"""
LearningRate class to adjust learning rate
Base class keeps learning rate constant unless training error increases.
"""
class LearningRate(object):

    def __init__(self,
                 initialError, # Initial training error
                 initialRate = 1e-4, # Initial learning rate
                 lrDown = 0.1, # Learning rate multiplied by this factor if error increases
                 **kwargs):

        self.lrate = initialRate
        self.lastError = initialError
        self.lrDown = lrDown

    # Checks training error and decreases learning rate if error has increased
    def update(self,error,*args,**kwargs):

        if error < self.lastError:
            self.lastError = error
        else:
            self.lrate *= self.lrDown

# Learning rate decays as l0/(1+its/tau)
# Decreases l0 if training error increases
class DecayRate(LearningRate):

    def __init__(self,
                 initialError, # Initial training error
                 initialIts = 0, # Initial value of its
                 initialRate = 1e-4, # Initial learning rate
                 lrTau = 1, # Time constant of decay
                 lrDown = 0.1, # Multiplies initialRate if erorr increased
                 **kwargs):

         self.initialRate = initialRate
         self.its = initialIts
         self.lrTau = lrTau
         self.lrate = self.initialRate/(1+self.its/self.lrTau)
         self.lrDown = 0.1

         self.lastError = initialError

    # Reduces initialRate if error has increased. Updates learning rate.
    def update(self,error,*args,**kwargs):

         if error < self.lastError:
             self.its += 1
             self.lastError = error
         else:
             self.initialRate *= self.lrDown

         self.lrate = self.initialRate/(1+self.its/self.lrTau)

# Increases learning rate if error decreasing. Decreases learning rate if erorr increasing
class BoldDriver(LearningRate):

    def __init__(self,
                 initialError, # Initial error
                 initialRate = 1e-2, # Initial learning rate
                 learningUp = 5e-2, # Amount to increase rate if error decreasing
                 learningDown = 0.5, # Amount to decrease rate if error increasing
                 **kwargs):

        self.lrate = initialRate
        self.lUp = learningUp
        self.lDown = learningDown

        self.lastError = initialError

    # Increases or decreases rate if error is decreasing or increasing (respectively)
    def update(self,error,*args,**kwargs):

        if error < self.lastError:
            self.lrate += self.lUp*self.lrate
            self.lastError = error
        else:
            self.lrate *= self.lDown
