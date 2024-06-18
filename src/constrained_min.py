from src.unconstrained_min import *
from tests.examples import *

class Minimizer_function_constrained:
    def __init__(self, mu=10):
        self.function_dict = None
        self.nu = None
        self.x_dim = None
        self.mu = mu
        self.func_dict = {}
        self.lag = None

    def interior_pt(self, f_0, x0, ineq_constraints=None, eq_constraints_mat=None, eq_constraints_rhs=None, max_outer_iterations=None, obj_tol=1e-12, param_tol=1e-8, max_m=False, d=1):
        if max_m:
            d = -1
        if eq_constraints_mat is not None:
            self.nu = np.zeros(shape=(eq_constraints_mat.shape[0], 1))
        else:
            self.nu = np.zeros(shape=(0, 1))

        self.x_dim = len(x0)

        f_prev = []
        x_prev = []
        nu_prev = []

        x0 = np.array(x0).reshape(-1, 1)
        f_x0 = f_0.func(x0)[0]
        x_prev.append(x0)
        nu_prev.append(self.nu)
        f_prev.append(f_x0)

        print(f'Outer iteration number {0}:\nFunction location is: {x0.T}\nFunction value is: {f_x0*d}')
        print('-' * 45)

        lagrangian_func = Lagrangian_function(f_0=f_0, x_0=x0, ineq_constraints=ineq_constraints, eq_constraints_mat=eq_constraints_mat, eq_constraints_rhs=eq_constraints_rhs, t=1)
        self.lag = lagrangian_func
        x0 = np.vstack([x0, self.nu]).reshape(-1, 1)
        outer_iteration = 0

        while outer_iteration < max_outer_iterations:
            l_min = Minimization()
            x_nu = l_min.minimizer_func(f=lagrangian_func, x0=x0, hessian=True, print_status=False, wolfe_cond_backtracking=True)[0]
            x0 = x_nu
            x = x0[:self.x_dim]
            self.nu = x0[self.x_dim:]
            f_value = f_0.func(x)[0]
            # Update all lists
            x_prev.append(x)
            nu_prev.append(self.nu)
            f_prev.append(f_value)

            if len(f_prev) and len(x_prev) > 1:
                function_values_diff = f_prev[-2] - f_prev[-1]
                function_location_diff = np.linalg.norm(x_prev[-2] - x_prev[-1])
                if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                    print('\nProces status: Achieved numeric tolerance for successful termination')
                    break
            if ineq_constraints.m / lagrangian_func.t < obj_tol:
                print('\nProces status: Achieved numeric tolerance for successful termination')
                break
            # Update t
            lagrangian_func.t *= self.mu
            print(f'Outer iteration number {outer_iteration+1}:\nFunction location is: {x.T}\nFunction value is: {f_value*d}')
            print('-' * 45)
            outer_iteration += 1
        self.function_dict = {'Function location list': x_prev, 'Function value list': d*np.array(f_prev)}
        return x_prev[-1], d*f_prev[-1], nu_prev[-1]

class Lagrangian_function:

    def __init__(self, f_0, x_0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, max_iter_hessian=100, t=1):  # Max iter hessian = Max inner iterations
        self.max_iter_hessian = max_iter_hessian
        self.f_0 = f_0
        self.ineq_constraints = ineq_constraints
        if eq_constraints_mat is None:
            self.A = None
            self.b = None
        else:
            self.A = np.array(eq_constraints_mat)
            self.b = np.array(eq_constraints_rhs).reshape(-1, 1)
        self.x_0 = x_0
        self.hessian = True
        self.x_dim = len(x_0)
        self.t = t

    def log_barrier_function(self, x):
        value_list = []
        grad_list = []
        hessian_list = []

        ineq_func = self.ineq_constraints
        ineq_func.constraints_values(x)
        ineq_func_list = ineq_func.value_list
        for val in ineq_func_list:
            if val[0] > -1e-12:
                return np.inf, None, None
            log_barrier_val = -math.log(-val[0])
            log_barrier_grad = val[1] / val[0]
            log_barrier_hessian = val[2] * val[0] - log_barrier_grad.dot(log_barrier_grad.T)
            value_list.append(log_barrier_val)
            grad_list.append(log_barrier_grad)
            hessian_list.append(log_barrier_hessian)
        return sum(value_list), np.sum(grad_list, axis=0), np.sum(hessian_list, axis=0)

    def func(self, x):
        nu_dim = x.size - self.x_dim
        x, x_nu = x[:self.x_dim], x[self.x_dim:]
        f_x, g_x, h_x = self.f_0.func(x)
        log_barrier = self.log_barrier_function(x)

        if not np.isfinite(log_barrier[0]):
            return np.inf, None, None
        lagrangian_value = self.t * f_x + log_barrier[0]
        if self.A is not None:
            lagrangian_value += x_nu.dot((self.A.dot(x) - self.b)).reshape(-1)
            if lagrangian_value.size == 1:
                lagrangian_value = lagrangian_value.item()

            lagrangian_grad = np.vstack([self.t * g_x - log_barrier[1], (self.A.dot(x) - self.b)])
            top_lagrangian_hessian = np.hstack([self.t * h_x - log_barrier[2], self.A.T])
            bottom_lagrangian_hessian = np.hstack([self.A, np.zeros(shape=(nu_dim, nu_dim))])
            lagrangian_hessian = np.vstack([top_lagrangian_hessian, bottom_lagrangian_hessian])
        else:
            lagrangian_grad = self.t * g_x - log_barrier[1]
            lagrangian_hessian = self.t * h_x - log_barrier[2]

        return lagrangian_value, lagrangian_grad, lagrangian_hessian

    def feasible_coordinates(self, cord):
        res = np.full(fill_value=True, shape=(cord.shape[1],))
        ineq_con = self.ineq_constraints
        ineq_con.constraints_values(x=cord)
        ineq_list = ineq_con.value_list
        for ineq in ineq_list:
            val = ineq[0]
            res &= val <= 0
        z = self.f_0.func(cord)
        z[~res] = np.nan
        return z