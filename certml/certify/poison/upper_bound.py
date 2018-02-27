"""Certify Machine Learning Pipeline"""

import numpy as np
import cvxpy as cvx
from certml.utils.data import get_projection_matrix
from certml.utils.cvx import cvx_dot


class UpperBound:

    def __init__(self, pipeline, norm_sq_constraint=None, max_iter=None, num_iter_to_throw_out=None,
                 learning_rate=None, init_w=None, init_b=None, verbose=True, print_interval=500):
        self.norm_sq_constraint = norm_sq_constraint
        self.max_iter = max_iter
        self.num_iter_to_throw_out = num_iter_to_throw_out
        self.learning_rate = learning_rate
        self.init_w = init_w
        self.init_b = init_b

        self.verbose = verbose
        self.print_interval = print_interval

        self.pipeline = pipeline

        # TODO Actually get these values from the pipeline!
        self.x = None
        self.y = None
        self.class_map = None
        self.loss = None
        self.loss_grad = None

        self.params = None
        self.centroids = None
        self.centroid_vec = None
        self.sphere_radii = None
        self.slab_radii = None

        self.minimizer = Minimizer()  # TODO Sphere or slab?

    def certify(self, epsilons):
        pass

    def cert_rda(self, epsilon):
        """Online Certification Algorithm using Regularized Dual Averaging"""

        # Initialize Variables as 0
        x_bs = np.zeros((self.max_iter, self.x.shape[1]))
        y_bs = np.zeros(self.max_iter)
        sum_w = np.zeros(self.x.shape[1])

        ##########
        # Line 1 #
        ##########

        # Initialize Sum of Gradients (z)
        sum_of_grads_w_sq = np.ones(self.x.shape[1])
        sum_of_grads_w = np.zeros(self.x.shape[1])
        sum_of_grads_b = 0

        # Initialize Upper Bound (U*)
        best_upper_bound = 10000

        # Initialize ??? (\lambda)
        current_lambda = 1 / self.learning_rate

        # Initialize Model (\theta)
        w = self.init_w
        b = self.init_b

        ##########
        # Line 2 #
        ##########

        for iter_idx in range(self.max_iter):

            #####################
            # Line 5 (1st Half) #
            #####################

            # Calculate gradient of loss
            grad_w, grad_b = self.loss_grad(self.x, self.y, w=w, b=b)

            if self.verbose:
                if iter_idx % self.print_interval == 0:
                    print("At iter %s:" % iter_idx)

            ##########
            # Line 3 #
            ##########

            # Find the attack point that maximizes loss function.
            # We do not know which class gives the maximum loss.
            # Pick the class with the worse (more negative) margin.
            worst_margin = None
            for y_b in set(self.y):

                class_idx = self.class_map[y_b]
                x_b = self.minimizer.minimize_over_feasible_set(
                    y_b,
                    w,
                    self.centroids[class_idx, :],
                    self.centroid_vec,
                    self.sphere_radii[class_idx],
                    self.slab_radii[class_idx])

                margin = y_b * (w.dot(x_b) + b)
                if (worst_margin is None) or (margin < worst_margin):
                    worst_margin = margin
                    worst_y_b = y_b
                    worst_x_b = x_b

            #####################
            # Line 5 (2nd Half) #
            #####################

            # Take the gradient with respect to that y
            if worst_margin < 1:
                grad_w -= epsilon * worst_y_b * worst_x_b
                grad_b -= epsilon * worst_y_b

            #####################
            # Line 4 (2nd Half) #
            #####################

            # Loss due to malicious data
            bad_loss = self.loss(worst_x_b, worst_y_b, w=w, b=b)

            # Store iterate to construct matching lower bound
            x_bs[iter_idx, :] = worst_x_b
            y_bs[iter_idx] = worst_y_b

            #####################
            # Line 4 (1st Half) #
            #####################

            # Loss due to clean data
            good_loss = self.loss(self.x, self.y, w=w, b=b)
            params_norm_sq = (np.linalg.norm(w) ** 2 + b ** 2)

            # Total Loss of the Poisoned Dataset
            total_loss = good_loss + epsilon * bad_loss

            if best_upper_bound > total_loss:
                best_upper_bound = total_loss
                best_upper_good_loss = good_loss
                best_upper_bad_loss = bad_loss
                best_upper_params_norm_sq = params_norm_sq
                best_upper_good_acc = np.mean((self.y * (self.x.dot(w) + b)) > 0)
                if worst_margin > 0:
                    best_upper_bad_acc = 1.0
                else:
                    best_upper_bad_acc = 0.0

            if self.verbose:
                if iter_idx % self.print_interval == 0:
                    print("  Bad margin (%s)         : %s" % (worst_y_b, worst_margin))
                    print("  Bad loss (%s)           : %s" % (worst_y_b, bad_loss))
                    print("  Good loss               : %s" % good_loss)
                    print("  Total loss              : %s" % total_loss)
                    print("  Sq norm of params_bias  : %s" % params_norm_sq)
                    print("  Grad w norm             : %s" % np.linalg.norm(grad_w))

            ##########
            # Line 6 #
            ##########

            # Update Gradient (z[t] = z[t-1] - g[t])
            sum_of_grads_w -= grad_w
            sum_of_grads_b -= grad_b

            # Update ?? (\lambda[t] = max(...))
            candidate_lambda = np.sqrt(np.linalg.norm(sum_of_grads_w) ** 2 + sum_of_grads_b ** 2) / np.sqrt(
                self.norm_sq_constraint)
            if candidate_lambda > current_lambda:
                current_lambda = candidate_lambda

                # Update Model (\theta[t] = z[t] / \lambda[t])
            w = sum_of_grads_w / current_lambda
            b = sum_of_grads_b / current_lambda

        print('Optimization run for %s iterations' % self.max_iter)

        print('Final upper bound:')
        print("  Total loss              : %s" % best_upper_bound)
        print('')

        # Determine Lower Bound from Candidate Attack Points
        '''
        X_modified, Y_modified, idx_train, idx_poison = sample_lower_bound_attack(
            X_train, Y_train,
            x_bs, y_bs,
            epsilon,
            num_iter_to_throw_out)
        '''


class Minimizer(object):
    """ CVX Minimizer / Data Maximizer

    This class determines the attack point the maximises the
    loss constrained by the feasible set. This is the first
    line of the loop in Algorithm 1.

    For the sphere defense this is the optimization problem:

    .. math
        max_x &  1 - y w^T x
        s.t.  &  || x - c ||2 < r

    For the slab defense this is the optimization problem:

    .. math
        max_x & 1 - y w^T x
        s.t.  &  |<x - c, c_vec>| < r

    where
        c :      class centroid
        c_vec :  vector between the two class centroids
    """

    def __init__(self, use_sphere=True, use_slab=True):
        """ Minimizer CVX Minimizer / Data Maximizer

        Parameters
        ----------
        use_sphere : bool
            Use sphere projection?
        use_slab : bool
            Use slab projection?
        """

        # TODO This currently does not change for different loss function!
        d = 3

        self.cvx_x = cvx.Variable(d)
        self.cvx_y = cvx.Parameter(1)
        self.cvx_w = cvx.Parameter(d)
        self.cvx_centroid = cvx.Parameter(d)
        self.cvx_centroid_vec = cvx.Parameter(d)
        self.cvx_sphere_radius = cvx.Parameter(1)
        self.cvx_slab_radius = cvx.Parameter(1)

        self.cvx_x_c = self.cvx_x - self.cvx_centroid

        self.constraints = []
        if use_sphere:
            self.constraints.append(cvx.norm(self.cvx_x_c, 2) < self.cvx_sphere_radius)
        if use_slab:
            self.constraints.append(cvx.abs(cvx_dot(self.cvx_centroid_vec, self.cvx_x_c)) < self.cvx_slab_radius)

        self.objective = cvx.Maximize(1 - self.cvx_y * cvx_dot(self.cvx_w, self.cvx_x))

        self.prob = cvx.Problem(self.objective, self.constraints)

    def minimize_over_feasible_set(self, y, w, centroid, centroid_vec, sphere_radius, slab_radius,
                                   verbose=False):
        """ Minimize over Feasible Set

        Includes both sphere and slab.

        Parameters
        ----------
        y : int
            Input label
        w : np.ndarray of shape (dimensions,)
            Coefficient
        centroid : np.ndarray of shape (classes, dimensions)
            Centroid for each class
        centroid_vec : np.ndarray of shape (dimensions,)
            Vector between the two centroids
        sphere_radius : np.ndarray of shape (classes,)
            Sphere radius for each class
        slab_radius : np.ndarray of shape (classes,)
            Slab radius for each class
        verbose : bool
            CVX verbose

        Returns
        -------
        x : np.ndarray
            Optimal x;
        """
        P = get_projection_matrix(w, centroid, centroid_vec)

        self.cvx_y.value = y
        self.cvx_w.value = P.dot(w.reshape(-1))
        self.cvx_centroid.value = P.dot(centroid.reshape(-1))
        self.cvx_centroid_vec.value = P.dot(centroid_vec.reshape(-1))
        self.cvx_sphere_radius.value = sphere_radius
        self.cvx_slab_radius.value = slab_radius

        self.prob.solve(verbose=verbose)

        x_opt = np.array(self.cvx_x.value).reshape(-1)

        return x_opt.dot(P)
