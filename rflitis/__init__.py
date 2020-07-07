"""
Random Forest library for Python @ Litis lab.
-----------------------
RFLitis is a python library that implements the different
Random Forest methods that have been proposed by the
Machine Learning research team @ the LITIS lab.
The LITIS lab. (http://www.litislab.fr) is an information science &
technology research unit in Normandy, France. It involves three
institutions:
- University of Rouen Normandy (URN)
- University of Le Havre Normandy (ULHN)
- INSA Rouen Normandy (INSARN)
For the RFLitis lib to work you'll need to have installed:
- Scikit-learn (http://scikit-learn.org)
- NumPy (http://numpy.org)
Contributors:
    Simon BERNARD, simon.bernard@univ-rouen.fr
    Jules GIRARD, jules.girard@outlook.com
"""

from ._forestrk import ForestRKClassifier
from ._dynamic import DynamicRandomForestClassifier


__all__ = [
    "ForestRKClassifier",
    # "DynamicRandomForestClassifier"
]
