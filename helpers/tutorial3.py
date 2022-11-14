import pyomo.environ as pyo
from IPython.display import Markdown


def add_dual_variables(model):
    """Use this function before solving the model"""
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)


def display_dual_variables(model):
    """Use this function after solving the model"""
    display(Markdown("**Dual variables:**"))
    for c in model.component_objects(pyo.Constraint):
        display(Markdown(f"The dual variable associated with the {c.name} constraint is equal to ${model.dual[c]:.3f}$"))
