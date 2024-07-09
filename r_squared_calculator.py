
import numpy as np
from typing import Union, List

class RSquaredCalculator:
    """A class to calculate the R-squared (coefficient of determination) value."""

    @staticmethod
    def calculate(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the R-squared value.

        Args:
            y_true (Union[List[float], np.ndarray]): True values
            y_pred (Union[List[float], np.ndarray]): Predicted values

        Returns:
            float: R-squared value

        Raises:
            ValueError: If input arrays have different lengths or are empty
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_res = np.sum((y_true - y_pred)**2)

        if ss_tot == 0:
            return 0  # R-squared is undefined when ss_tot is zero

        r2 = 1 - (ss_res / ss_tot)
        return r2

def main():
    """Example usage of the RSquaredCalculator class."""
    y_true = [3, 5, 7, 9, 11]
    y_pred = [2.8, 4.9, 7.2, 8.7, 10.6]

    try:
        calculator = RSquaredCalculator()
        result = calculator.calculate(y_true, y_pred)
        print(f"R-squared: {result:.4f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

