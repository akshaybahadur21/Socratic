from src.Vision import Vision


class Socratic:
    def __init__(self):
        self.vision = Vision()

    def solve(self):
        self.vision.get_equation()


if __name__ == '__main__':
    socratic = Socratic()
    socratic.solve()
