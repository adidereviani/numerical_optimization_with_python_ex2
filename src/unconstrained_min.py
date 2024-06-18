from tests.examples import *
class Minimization:

    def __init__(self, obj_tol=1e-12):
        self.function_dict = {}
        self.obj_tol = obj_tol

    def minimizer_func(self, f, x0, obj_tol=1e-12, param_tol=1e-8, hessian=False, wolfe_cond_backtracking=False, title=None, print_status=True):
        max_condition = 1e+5
        step_length = 1e-1
        f_prev = []
        x_prev = []
        x = np.array(x0).copy().reshape(-1, 1)
        f.hessian = hessian
        status = False
        if hessian:
            func_name = 'Newton descent'
            max_iter = f.max_iter_hessian
            f_x, g_x, h_x = f.func(x)
            x_prev.append(x.copy())
            f_prev.append(f_x)
            if print_status:
                print("Iteration number 0:\nFunction location is: {}\nFunction value is: {}".format(x.T, f_x))
                print('-' * 100)
            for i in range(1, max_iter+1):
                condition = np.linalg.cond(h_x)
                if condition > max_condition:
                    if print_status:
                        print("\nInversion condition number exceeds maximum condition number")
                    status = False
                    break
                if wolfe_cond_backtracking:
                    alpha = Minimization.wolfe_cond_backtracking(f=f, x=x, hessian=hessian)
                else:
                    alpha = 1
                direction = alpha * np.linalg.solve(h_x, -g_x)
                x += direction

                f_x, g_x, h_x = f.func(x)

                x_prev.append(x.copy())
                f_prev.append(f_x)
                if len(f_prev) and len(x_prev) > 1:
                    function_values_diff = f_prev[-2] - f_prev[-1]
                    function_location_diff = np.linalg.norm(x_prev[-2] - x_prev[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        if print_status:
                            print('\nProcess status: Successfully reached the numerical tolerance for termination')
                        x_prev.pop()
                        f_prev.pop()
                        status = True
                        break
                if abs(f_x) >= 1e+6:
                    if print_status:
                        print('\nThe functions value exceeds acceptable limits')
                    x_prev.pop()
                    f_prev.pop()
                    status = False
                    break
                if print_status:
                    print("Iteration number {}:\nFunction location is: {}\nFunction value is: {}".format(i, x.T, f_x))
                    print('-' * 100)

        else:
            func_name = 'Gradient descent'
            max_iter = f.max_iter
            f_x, g_x = f.func(x)
            x_prev.append(x)
            f_prev.append(f_x)
            if print_status:
                print("Iteration number 0:\nFunction location is: {}\nFunction value is: {}".format(x.T, f_x))
                print('-' * 100)
            for i in range(1, max_iter+1):
                if wolfe_cond_backtracking:
                    alpha = Minimization.wolfe_cond_backtracking(f=f, x=x, step_len=step_length, hessian=hessian)
                else:
                    alpha = 1
                x = x.copy()
                x -= alpha * step_length * g_x
                f_x, g_x = f.func(x)
                x_prev.append(x)
                f_prev.append(f_x)
                if abs(f_x) >= 1e+4:
                    if print_status:
                        print('\nThe functions value exceeds acceptable limits')
                    x_prev.pop()
                    f_prev.pop()
                    status = False
                    break
                if len(x_prev) and len(x_prev) > 1:
                    function_values_diff = f_prev[-2]-f_prev[-1]
                    function_location_diff = np.linalg.norm(x_prev[-2]-x_prev[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        if print_status:
                            print('\nProcess status: Successfully reached the numerical tolerance for termination')
                        x_prev.pop()
                        f_prev.pop()
                        status = True
                        break
                if print_status:
                    print("Iteration number {}:\nFunction location is: {}\nFunction value is: {}".format(i, x.T, f_x))
                    print('-' * 100)
        self.function_dict = {'Minimization method': func_name, 'Function location list': np.hstack(x_prev), 'Function value list': f_prev}
        if print_status:
            print(f'{func_name}: {title} function\nIteration number {len(f_prev)-1}: \nFinal function location is {x_prev[-1].T}\nFinal function value is {f_prev[-1]}\nStatus is {status}\n')
        return x_prev[-1], f_prev[-1], status

    @staticmethod
    def wolfe_cond_backtracking(f, x, step_len=None, alpha=1, wcc=0.01, bc=0.5, hessian=False):
        if hessian:
            f_x_1, g_x_1, h_x_1 = f.func(x)
            direction = np.linalg.solve(h_x_1, -g_x_1)
            while f.func(x + alpha * direction)[0] > f_x_1 + wcc * alpha * g_x_1.T.dot(direction):
                alpha *= bc
            return alpha
        else:
            f_x_1, g_x_1 = f.func(x)
            while f.func(x - alpha * step_len * g_x_1)[0] > f_x_1 + wcc * alpha * step_len * g_x_1.T.dot(-g_x_1):
                alpha *= bc
            return alpha
