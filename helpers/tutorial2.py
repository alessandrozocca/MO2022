from IPython.display import HTML
import numpy as np
import sympy as sp

sp.var("x1 x2 x y t")

def plot_feasible_region(constraints, x_axis=None, y_axis=None, origin=(0,0), **kwargs):
    return sp.plot_implicit(sp.And(*constraints), x_axis, y_axis, axis_center=origin, line_color="lightblue", **kwargs)

def plot_objective(feasible_region_plot, objective_function, objective_values=None, **kwargs):
    if not hasattr(objective_values, "__iter__"):
        objective_values = [objective_values]
    x_axis = (feasible_region_plot.xlabel,) + feasible_region_plot.xlim
    y_axis = (feasible_region_plot.ylabel,) + feasible_region_plot.ylim

    show = kwargs.pop("show", True)
    line_color = kwargs.pop("line_color", "blue")

    if len(objective_function.free_symbols | feasible_region_plot[0].expr.free_symbols) == 1:
        objective_plots = [sp.plot(objective_function, x_axis, ylim=y_axis[1:], 
                                      line_color="blue", show=False, **kwargs)[0]]
        if objective_values[0] is not None:
            objective_plots += [sp.plot_implicit(sp.Eq(y, v), x_axis, y_axis, 
                                      line_color=line_color, show=False, **kwargs)[0]
                                for v in objective_values]
    else:
        assert objective_values[0] is not None, "You should provide one or more objective values to plot"
        objective_plots = [sp.plot_implicit(sp.Eq(objective_function, v), x_axis, y_axis, 
                                            line_color=line_color, show=False, **kwargs)[0] 
                          for v in objective_values]
       
    feasible_region_plot.extend(objective_plots)
    if show:
        feasible_region_plot.show()