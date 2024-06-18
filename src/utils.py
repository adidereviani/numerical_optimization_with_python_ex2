import matplotlib.pyplot as plt
from matplotlib import cm
from tests.examples import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_function_values(dict_a=None, dict_b=None, constrained=False, title=None):
    if not constrained:
        dict_list = [dict_a, dict_b]
        plt.figure(figsize=(12, 7))
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Function values', fontsize=14)
        plt.title(f'Comparison of Gradient and Newton Methods: {title} function', fontsize=18)
        x_len = abs(len(dict_a['Function value list']) - len(dict_b['Function value list'])) * 0.5
        x_len_ratio = max(len(dict_a['Function value list']), len(dict_b['Function value list'])) / min(len(dict_a['Function value list']), len(dict_b['Function value list']))
        plt.scatter(x=[1], y=[dict_a['Function value list'][0]], linewidth=3, color="red")
        for func_list in dict_list:
            plt.plot(np.arange(1, len(func_list['Function value list'])+1), func_list['Function value list'], label=func_list['Minimization method'])
        if x_len_ratio > 100:
            plt.xscale('log')
            plt.xlim(left=1)
        else:
            plt.xticks(np.arange(0, x_len + 1, 5))
            plt.xlim(left=1, right=x_len)
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 7))
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Function values', fontsize=14)
        plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=18)
        x_len = len(dict_a['Function value list'])
        plt.scatter(x=[1], y=[dict_a['Function value list'][0]], linewidth=3, color="red")
        plt.plot(np.arange(1, len(dict_a['Function value list']) + 1), dict_a['Function value list'], label='Optimization path')
        plt.xticks(np.arange(0, x_len + 1, 1))
        plt.xlim(left=1, right=x_len)
        plt.legend()
        plt.show()

def plot_contour_2d(f, lag=None, dict_a=None, dict_b=None, constrained=False, title=None):

    if not constrained:
        dict_list = [dict_a, dict_b]
        location = []
        for i in dict_list:
            for col in range(i['Function location list'].shape[1]):
                location.append(np.linalg.norm(i['Function location list'][:, col]))
        maximum_location_norm = max(location)
        x_y = np.linspace(-maximum_location_norm, maximum_location_norm, 40)

        x_coordinates, y_coordinates = np.meshgrid(x_y, x_y)
        z = f.func(np.vstack([np.array(x_coordinates).reshape(-1), np.array(y_coordinates).reshape(-1)]))
        z_coordinates = z.reshape(x_coordinates.shape)
        x1_dict_a, x2_dict_a, z_dict_a = dict_a['Function location list'][0], dict_a['Function location list'][1],  dict_a['Function value list']
        x1_dict_b, x2_dict_b, z_dict_b = dict_b['Function location list'][0], dict_b['Function location list'][1], dict_b['Function value list']
        plt.figure(figsize=(14, 9))
        plt.plot(x1_dict_a, x2_dict_a, label='Gradient descent')
        plt.plot(x1_dict_b, x2_dict_b, label='Newton descent')
        plt.contour(x_coordinates, y_coordinates, z_coordinates, 80)
        plt.xlabel('X values', fontsize=12)
        plt.ylabel('Y values', fontsize=12)
        plt.title(f'Comparison of Gradient and Newton with wolfe condition, Methods: {title} function', fontsize=16)
        plt.legend()
        plt.show()
    else:
        location = []
        location_x = np.hstack(dict_a['Function location list'])
        for vec in location_x.T:
            location.append(np.linalg.norm(vec))
        maximum_location_norm = max(location)
        x_y = [np.linspace(-maximum_location_norm, maximum_location_norm, 40)] * location_x.shape[0]
        x_coordinates, y_coordinates = np.meshgrid(*x_y)
        z = lag.feasible_coordinates(np.vstack([np.array(x_coordinates).reshape(1, -1), np.array(y_coordinates).reshape(1, -1)]))
        z_coordinates = z.reshape(x_coordinates.shape)
        x1_dict_a, x2_dict_a, z_dict_a = location_x[0], location_x[1],  dict_a['Function value list']
        plt.figure(figsize=(12, 7))
        plt.plot(x1_dict_a, x2_dict_a, label='Optimization path')
        plt.contour(x_coordinates, y_coordinates, z_coordinates, 60)
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Y values', fontsize=14)
        plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=16)
        plt.legend()
        plt.show()


def plot_contour_3d(f, lag=None, dict_a=None, dict_b=None, constrained=False, title=None):
    if not constrained:
        dict_list = [dict_a, dict_b]
        location = []

        for i in dict_list:
            for col in range(i['Function location list'].shape[1]):
                location.append(np.linalg.norm(i['Function location list'][:, col]))
        maximum_location_norm = max(location)
        x_y = np.linspace(-maximum_location_norm, maximum_location_norm, 40)

        x_coordinates, y_coordinates = np.meshgrid(x_y, x_y)
        z = f.func(np.array([np.array(x_coordinates).reshape(-1), np.array(y_coordinates).reshape(-1)]))
        z_coordinates = z.reshape(x_coordinates.shape)
        x1_dict_a, x2_dict_a, z_dict_a = dict_a['Function location list'][0], dict_a['Function location list'][1],  dict_a['Function value list']
        x1_dict_b, x2_dict_b, z_dict_b = dict_b['Function location list'][0], dict_b['Function location list'][1], dict_b['Function value list']
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(projection='3d')
        ax.plot(x1_dict_a, x2_dict_a, z_dict_a, label='Gradient descent')
        ax.plot(x1_dict_b, x2_dict_b, z_dict_b, label='Newton descent')
        surf = ax.plot_surface(x_coordinates, y_coordinates, z_coordinates, cmap=cm.pink, alpha=0.5, linewidth=0, antialiased=False)
        ax.set_zlim(math.floor(z_coordinates.min()), math.ceil(z.max()))
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Y values', fontsize=14)
        plt.title(f'Comparison of Gradient and Newton with wolfe condition, Methods: {title} function', fontsize=16)
        plt.legend()
        plt.show()
    else:
        if dict_a['Function location list'][0].shape[0] == 2:
            location = []
            location_x = np.hstack(dict_a['Function location list'])

            for j in location_x.T:
                location.append(np.linalg.norm(j))
            maximum_location_norm = max(location)
            x_y = [np.linspace(-maximum_location_norm, maximum_location_norm, 40)] * location_x.shape[0]
            x_coordinates, y_coordinates = np.meshgrid(*x_y)
            if dict_a['Function value list'][0] > dict_a['Function value list'][1]:
                d = 1
            else:
                d = -1
            z = lag.feasible_coordinates(np.vstack([np.array(x_coordinates).reshape(1, -1), np.array(y_coordinates).reshape(1, -1)]))
            z_coordinates = z.reshape(x_coordinates.shape) * d
            x1_dict_a, x2_dict_a, z_dict_a = location_x[0], location_x[1], dict_a['Function value list']
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(projection='3d')
            ax.plot(x1_dict_a, x2_dict_a, z_dict_a, label='Optimization path')
            surf = ax.plot_surface(x_coordinates, y_coordinates, z_coordinates, cmap=cm.pink, alpha=0.5, linewidth=0, antialiased=False)
            min_z, max_z = np.nanmin(z_coordinates), np.nanmax(z_coordinates)
            ax.set_zlim(math.floor(min_z), math.ceil(max_z))
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.xlabel('X values', fontsize=14)
            plt.ylabel('Y values', fontsize=14)
            plt.title(f'Newton method with wolfe condition, constrained {title} function', fontsize=15)
            plt.legend()
            plt.show()
        else:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.set_title("3D plot of " + 'qp')
            lag_ineq = lag.ineq_constraints.value_list
            grad_list = []
            for val in lag_ineq:
                grad_list.append(val[1])
            grad_list = -np.hstack(grad_list)
            ineqs = [[tuple(row) for row in grad_list]]
            poly_3d_collection = Poly3DCollection(ineqs, alpha=0.7, edgecolors="r")
            ax.add_collection3d(poly_3d_collection)
            location = np.hstack(dict_a['Function location list'])
            x1 = location[0]
            x2 = location[1]
            x3 = location[2]
            ax.plot(x1, x2, x3, color='r', marker=".", linestyle="-", label='Optimization path')
            ax.set_zlabel("Z values")
            plt.xlabel("X values")
            plt.ylabel("Y values")
            plt.legend()
            plt.show()