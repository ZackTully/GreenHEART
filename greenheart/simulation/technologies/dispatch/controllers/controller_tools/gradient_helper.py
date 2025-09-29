import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import scipy
from colorama import Fore, Style

class GradientHelper:
    def __init__(self, mpc):
        self.mpc = mpc
        np.set_printoptions(linewidth=200, suppress=True, precision=4)



    def get_mpc_attrs(self):
        # This method cannot be run before the mpc optimization has been set up
        self.opti = self.mpc.opti
        self.opt_vars = self.mpc.opt_vars
        self.opt_params = self.mpc.opt_params
        
        self.obj_weights = self.mpc.weights
        self.obj_terms_uw = self.mpc.obj_terms_uw
        self.terms = self.mpc.term_keys


        for attr in self.mpc.__dir__():
            if attr.endswith("_label"):
                setattr(self, attr, getattr(self.mpc, attr))



    def check_initial_values(self, opti):


        tol = 1e-6

        # Get all (scalarised) dual variables as a symbolic column vector.
        lam_g = opti.value(opti.lam_g, opti.initial())

        g_init = opti.value(opti.g, opti.initial())
        lbg_init = opti.value(opti.lbg, opti.initial())
        ubg_init = opti.value(opti.ubg, opti.initial())


        lb_bool = g_init >= lbg_init - tol
        ub_bool = g_init <= ubg_init + tol


        # np.all(g_init >= lbg_init - tol)
        # np.all(g_init <= ubg_init + tol)

        np.where(~ub_bool)[0].astype(int)
        np.where(~lb_bool)[0].astype(int)


        for i in range(g_init.shape[0]):
            if not lb_bool[i]:
                opti.g[i]
                opti.lbg[i]



                pass
            if not ub_bool[i]:
                pass




        []


    def cast_numpy(self, arr):
        if isinstance(arr, scipy.sparse.spmatrix):
            arr = arr.toarray()
        return arr

    def print_banner(self, title):
        print(Fore.GREEN + "=" * 100)
        print(" " * int(100/2 - len(title)/2) +  title)
        print("=" * 100)
        print("" + Style.RESET_ALL)


    def print_with_labels(self, arr, labels, format=None):
        print("")
        max_label = np.max([len(lab) for lab in labels])

        if format is None:
            arr_rows = arr.__str__().split("\n")
        elif format == "objective":
            arr_rows = [f"{v:.9f}".rjust(30) for v in arr[:, 0]]

        for i, lab, in enumerate(labels):
            arr_rows[i] = labels[i].ljust(max_label+2, " ") + Fore.BLUE + arr_rows[i] + Style.RESET_ALL
        print("\n".join(arr_rows))



    def print_opt_var_values(self, sol):

        self.print_banner("Optimization variables")

        sol_forecast = sol.value(self.opt_params["dex"])
        sol_curtail = sol.value(self.opt_vars["gridcurtail"])
        sol_uct = sol.value(self.opt_vars["uct"])
        sol_usp = sol.value(self.opt_vars["usp"])
        sol_x = sol.value(self.opt_vars["x"])
        sol_yex = sol.value(self.opt_vars["yex"])

        self.print_with_labels(sol_forecast, ["forecast"])
        self.print_with_labels(sol_curtail, ["curtail"])
        self.print_with_labels(sol_uct, self.mct_label)
        self.print_with_labels(sol_usp, self.msp_label)
        self.print_with_labels(sol_x, self.n_label)
        self.print_with_labels(sol_yex, self.pex_label)


    def print_objective_values(self, sol, terms= None):

        self.print_banner("Unweighted Obj. Values (per step)")
        if terms is None:
            values = np.atleast_2d([sol.value(val)/self.mpc.horizon for val in self.obj_terms_uw.values()]).T
            labels = list(self.obj_terms_uw.keys())
        else:
            obj_terms = {k:v for k,v in self.obj_terms_uw.items() if k in terms}
            values = np.atleast_2d([sol.value(val)/self.mpc.horizon for val in obj_terms.values()]).T
            labels = list(obj_terms.keys())

        self.print_with_labels(values, labels, format="objective")

    def print_objective_trajectory_values(self, sol, terms=None, weighted = False):
        if weighted:
            self.print_banner("Weighted Obj. Values (trajectory)")
        else:
            self.print_banner("Unweighted Obj. Values (trajectory)")


        obj_traj = self.mpc.objective_manager.obj_terms_uw_traj
        if terms is not None:
            obj_traj = {k:v for k, v in obj_traj.items() if k in terms}

        if weighted:
            for key in obj_traj.keys():
                obj_traj[key] = [val* self.mpc.objective_manager.weights[key] for val in obj_traj[key]]


        values = np.atleast_2d([[sol.value(v) for v in vals] for vals in obj_traj.values()])
        labels = list(obj_traj.keys())
        self.print_with_labels(values, labels)

    

    def print_objective_jacobian(self, sol, terms=None):
        def jac_f(sol, var ,f):
            jac_var = self.cast_numpy(sol.value(ca.jacobian(f, var))).reshape(var.T.shape).T
            return jac_var


        for obj_key in self.obj_terms_uw.keys():
            if (terms is not None) and (obj_key in terms):

                self.print_banner(obj_key)

                self.print_with_labels(jac_f(sol, self.opt_vars["gridcurtail"], self.obj_terms_uw[obj_key]), ["curtail"])
                self.print_with_labels(jac_f(sol, self.opt_vars["uct"], self.obj_terms_uw[obj_key]), self.mct_label)
                self.print_with_labels(jac_f(sol, self.opt_vars["usp"], self.obj_terms_uw[obj_key]), self.msp_label)
                self.print_with_labels(jac_f(sol, self.opt_vars["x"], self.obj_terms_uw[obj_key]), self.n_label)
                self.print_with_labels(jac_f(sol, self.opt_vars["yex"], self.obj_terms_uw[obj_key]), self.pex_label)

     


    def check_gradients(self, sol):

        self.print_opt_var_values(sol)
        self.print_objective_values(sol)
        self.print_objective_jacobian(sol)



        if False:





            if not hasattr(self, "opt_vars"):
                self.get_mpc_attrs()

            def cast_numpy(arr):
                if isinstance(arr, scipy.sparse.spmatrix):
                    arr = arr.toarray()
                return arr

            def jac_f(sol, var):
                jac_var = cast_numpy(sol.value(ca.jacobian(self.opti.f, var))).reshape(var.T.shape).T
                return jac_var

            def jac_g(sol, var):
                jac_var = cast_numpy(sol.value(ca.jacobian(self.opti.g, var))).reshape((self.opti.g.shape[0], var.shape[1], var.shape[0]))
                return jac_var

            def jac_obj(sol, var, f):
                jac_var = cast_numpy(sol.value(ca.jacobian(f, var))).reshape(var.T.shape).T
                return jac_var

            def print_with_labels(jac, labels):

                print("")
                np.set_printoptions(linewidth=200, suppress=True, precision=4)
                max_label = np.max([len(lab) for lab in labels])

                jac_rows = jac.__str__().split("\n")

                for i, lab, in enumerate(labels):
                    jac_rows[i] = labels[i].ljust(max_label+2, " ") + jac_rows[i]

                print("\n".join(jac_rows))

            def print_constraints(sol, var):
                jac_g_var = jac_g(sol, var)

                for i in range(sol.opti.g.shape[0]):
                    out_str = "" 
                    out_str += str(sol.value(sol.opti.lbg[i])).ljust(20, " ")

                    expr = str(sol.opti.g[i])

                    out_str += "<=     " +  expr + "     <="

                    out_str += str(sol.value(sol.opti.ubg[i])).rjust(20, " ")

                    print(out_str)

                pass

            sol_curtail = sol.value(self.opt_vars["gridcurtail"])
            sol_uct = sol.value(self.opt_vars["uct"])
            sol_usp = sol.value(self.opt_vars["usp"])
            sol_x = sol.value(self.opt_vars["x"])
            sol_yex = sol.value(self.opt_vars["yex"])

            jac_curtail = jac_f(sol, self.opt_vars["gridcurtail"])            
            jac_uct = jac_f(sol, self.opt_vars["uct"])
            jac_usp = jac_f(sol, self.opt_vars["usp"])
            jac_x = jac_f(sol, self.opt_vars["x"])
            jac_yex = jac_f(sol, self.opt_vars["yex"])

            jac_g_uct = jac_g(sol, self.opt_vars["uct"])
            jac_g_usp = jac_g(sol, self.opt_vars["usp"])
            jac_g_x = jac_g(sol, self.opt_vars["x"])
            jac_g_yex = jac_g(sol, self.opt_vars["yex"])

            if print_jacs:

                print_with_labels(sol_curtail, ["curtail"])
                print_with_labels(sol_uct, self.mct_label)
                print_with_labels(sol_usp, self.msp_label)
                print_with_labels(sol_x, self.n_label)
                print_with_labels(sol_yex, self.pex_label)

                print_with_labels(jac_uct, self.mct_label)
                print_with_labels(jac_usp, self.msp_label)
                print_with_labels(jac_x, self.n_label)
                print_with_labels(jac_yex, self.pex_label)

                for obj_key in self.obj_terms_uw.keys():
                    print("")
                    print("===============================================================")
                    print(obj_key)
                    print("===============================================================")

                    print_with_labels(jac_obj(sol, self.opt_vars["gridcurtail"], self.obj_terms_uw[obj_key]), ["curtail"])
                    print_with_labels(jac_obj(sol, self.opt_vars["uct"], self.obj_terms_uw[obj_key]), self.mct_label)
                    print_with_labels(jac_obj(sol, self.opt_vars["usp"], self.obj_terms_uw[obj_key]), self.msp_label)
                    print_with_labels(jac_obj(sol, self.opt_vars["x"], self.obj_terms_uw[obj_key]), self.n_label)
                    print_with_labels(jac_obj(sol, self.opt_vars["yex"], self.obj_terms_uw[obj_key]), self.pex_label)

            try:
                jac_uct = sol.value(ca.jacobian(self.opti.f, self.opt_vars["uct"]))
                jac_usp = sol.value(ca.jacobian(self.opti.f, self.opt_vars["usp"]))
                jac_x = sol.value(ca.jacobian(self.opti.f, self.opt_vars["x"]))
                jac_yex = sol.value(ca.jacobian(self.opti.f, self.opt_vars["yex"]))
                # jac_yco = sol.value(ca.jacobian(self.opti.f, self.opt_vars["uct"]))

                jac = sol.value(ca.jacobian(self.opti.f, self.opti.x)).toarray()[0]
                # jac = self.opti.debug.value(ca.jacobian(self.opti.debug.f, self.opti.debug.x)).toarray()[0]
                assert (np.abs(jac) < 1).any()
                # True
            except:
                np.set_printoptions(linewidth=200, suppress=True, precision=4)

                uc_slice = slice(0, self.mct * self.horizon)
                us_slice = slice(
                    self.mct * self.horizon, (self.mct + self.msp) * self.horizon
                )
                x_slice = slice(
                    (self.mct + self.msp) * self.horizon,
                    (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
                )
                ys_slice = slice(
                    (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
                    (self.mct + self.msp) * self.horizon
                    + self.n * (self.horizon + 1)
                    + self.pse * self.horizon,
                )

                jac_uc = np.reshape(jac[uc_slice], (self.horizon, self.mc))
                jac_us = np.reshape(jac[us_slice], (self.horizon, self.ms))
                jac_x = np.reshape(jac[x_slice], (self.horizon + 1, self.n))
                jac_ys = np.reshape(jac[ys_slice], (self.horizon, self.pse))

                self.print_block_matrices(
                    mat=[[jac_uc, jac_us, jac_x[0 : self.horizon, :], jac_ys]],
                    in_labels=["jac uc", "jac us", "jac x", "jac yex"],
                    out_labels=[f"step {i}" for i in range(self.horizon)],
                )

                []

                # self.opti.debug.value_parameters()
                # self.opti.debug.value_variables()
                # self.opti.debug.stats()
                # self.opti.debug.arg()
                # self.opti.debug.constraints()
                # self.opti.debug.show_infeasibilities()