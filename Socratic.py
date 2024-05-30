from src.Plotter import Plotter
from src.Vision import Vision
import argparse


class Socratic:
    def __init__(self):
        self.vision = Vision()
        self.plotter = Plotter()

    def solve(self, option):
        if option == "equation":
            self.vision.solve_equation()
        elif option == "plot":
            self.plotter.plot_equation()
        else:
            raise ValueError("Input can only be either Equation or Plot")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', "--option", required=True, help="equation or plot",
                        type=str)
    args = parser.parse_args()
    option = args.option

    socratic = Socratic()
    socratic.solve(option.lower())
