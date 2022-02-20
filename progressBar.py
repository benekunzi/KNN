class ProgressBar:
    def __init__(self, iterations: int):
        self.iterations = iterations
        self.divider = 20
        self.steps = iterations//self.divider
        self.progressStep = 0
        self.progressBarString = '[' + '_'*(self.divider+1) + ']'

    def progressBar(self, cit: int):
        if cit == 0:
            self.progressStep = 0
            self.progressBarString = '[' + '_'*(self.divider+1) + ']'
        elif cit >= self.steps*self.progressStep:
            self.progressStep += 1
            strList = list(self.progressBarString)
            strList[self.progressStep] = '#'
            self.progressBarString = ''.join(strList)
            print(f'\r{self.progressBarString}', end='\r')
            if cit == self.iterations:
                print()