# from visdom import Visdom
# import numpy as np

# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}

#     def plot(self, var_name, split_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=var_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)



from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=var_name,
                    xlabel='Epochs',
                    ylabel=var_name
                )
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                update='append',
                name=split_name
            )

    # def plot2(self, var_name, split_name, x, y):
    #     if var_name not in self.plots:
    #         self.plots[var_name] = self.viz.line(
    #             X=np.array([x, x]),
    #             Y=np.array([y, y]),
    #             env=self.env,
    #             opts=dict(
    #                 legend=[split_name],
    #                 title=var_name,
    #                 xlabel='Epochs',
    #                 ylabel=var_name
    #             )
    #         )
    #     else:
    #         self.viz.scatter(
    #             X=np.array([x]),
    #             Y=np.array([y]),
    #             env=self.env,
    #             win=self.plots[var_name],
    #             update='append',
    #             name=split_name
    #         )