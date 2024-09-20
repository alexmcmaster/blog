class AttitudeEstimatorEKF:
    
    def __init__(self, css_orientations, sc_moment_of_inertia):
        self.css_ors = css_orientations
        self.sc_moi = sc_moment_of_inertia
    
    @staticmethod
    def _W_matrix(w):
        wx, wy, wz = w
        return np.array((
            (0,  -wx, -wy, -wz),
            (wx,  0,   wz, -wy),
            (wy, -wz,  0,   wx),
            (wz,  wy, -wx,  0 )
        ))
    
    @staticmethod
    def numerical_jacobian(f, m, n, x, u=None, epsilon=1e-2):
        jacobian = np.zeros((m, n))
        perturb = np.eye(n) * epsilon
        for i in range(n):
            if u is None:
                jacobian[:, i] = (f(x + perturb[:, i]) - f(x - perturb[:, i])) / (2 * epsilon)
            else:
                jacobian[:, i] = (f(x + perturb[:, i], u) - f(x - perturb[:, i], u)) / (2 * epsilon)
        return jacobian

    def _process_model(self, x, dt, u):
        q, w = x[:4], x[4:]
        x_hat = np.zeros(7)

        # Quaternion dynamics.
        q_dot = 0.5 * self._W_matrix(w) @ q
        x_hat[:4] = q + q_dot * dt
        x_hat[:4] /= np.linalg.norm(x_hat[:4])

        # Angular velocity dynamics.
        w_dot = np.linalg.inv(self.sc_moi) @ (np.cross(-w, self.sc_moi @ w))
        x_hat[4:] = w + w_dot * dt

        return x_hat

    def _process_model_jacobian(self, x, dt, u):
        q, w = x[:4], x[4:]
        
        F = np.zeros((7, 7))

        # Jacobian of quaternion kinematics.
        dq_dq = np.eye(4) + 0.5 * self._W_matrix(w) * dt

        # Jacobian of quaternion w.r.t. angular rates.
        dq_dw = 0.5 * dt * np.array([
            [-q[1], -q[2], -q[3]],
            [ q[0],  q[3], -q[2]],
            [-q[3],  q[0],  q[1]],
            [ q[2], -q[1],  q[0]]
        ])

        # Jacobian of angular rate dynamics.
        Ix, Iy, Iz = np.diag(self.sc_moi)
        wx, wy, wz = w
        dw_dw = np.zeros((3, 3))
        dw_dw[0, 0] = 1
        dw_dw[0, 1] = (Iy - Iz) / Ix * wz * dt
        dw_dw[0, 2] = (Iy - Iz) / Ix * wy * dt
        dw_dw[1, 0] = (Iz - Ix) / Iy * wz * dt
        dw_dw[1, 1] = 1
        dw_dw[1, 2] = (Iz - Ix) / Iy * wx * dt
        dw_dw[2, 0] = (Ix - Iy) / Iz * wy * dt
        dw_dw[2, 1] = (Ix - Iy) / Iz * wx * dt
        dw_dw[2, 2] = 1

        # Assemble full Jacobian.
        F[:4, :4] = dq_dq
        F[:4, 4:] = dq_dw
        F[4:, 4:] = dw_dw

        return F

    def _measurement_model(self, x, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed):
        q, w = x[:4], x[4:]
        
        h = np.zeros(9)
        
        body_dcm_est = quat2dcm(q)

        # CSS measurement model.        
        sun_vector_relative = sun_vector_inertial - sc_position_inertial
        sun_vector_body = (body_dcm_est @ sun_vector_relative) / np.linalg.norm(sun_vector_relative)
        sun_distance = np.linalg.norm(sun_vector_inertial)
        for i in range(6):
            # See https://hanspeterschaub.info/basilisk/_downloads/5a5aa3cb20faf38a4d8da52afc25a9b6/Basilisk-CoarseSunSensor-20170803.pdf
            gamma_hat = np.dot(self.css_ors[i], sun_vector_body)
            gamma_k = gamma_hat  # NOTE: assuming kelly factor of 0
            gamma_li = gamma_k * (AU2m(1) ** 2) / (sun_distance ** 2)  # NOTE: assuming no eclipse
            gamma_clean = gamma_li  # NOTE: assuming scaling factor is 1
            h[i] = np.max((0, gamma_clean))
        
        # TAM measurement model.
        mag_field_body = body_dcm_est @ mag_field_earth_fixed
        h[6:] = mag_field_body
        
        return h

    def _measurement_model_jacobian(self, x, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed):
        q, w = x[:4], x[4:]
        H = np.zeros((9, 7))

        q0, q1, q2, q3 = q
        body_dcm_est = quat2dcm(q)
        sun_vector_relative = sun_vector_inertial - sc_position_inertial
        sun_vector_body = (body_dcm_est @ sun_vector_relative) / np.linalg.norm(sun_vector_relative)
        Bx, By, Bz = mag_field_earth_fixed
        
        # Ensure input vectors are unit vectors
        #sun_vector_ref = sun_vector_ref / np.linalg.norm(sun_vector_ref)
        #mag_field_ref = mag_field_ref / np.linalg.norm(mag_field_ref)

        # Compute DCM and its derivative
        #dcm = quat2dcm(q)
        #d_dcm = d_quat2dcm(q)

        # CSS Jacobian
        #H_css = np.zeros((6, 7))
        #sun_vector_body = dcm @ sun_vector_ref
        #for i in range(6):
        #    if np.dot(self.css_ors[i], sun_vector_body) > 0:
        #        for j in range(4):
        #            H_css[i, j] = np.dot(self.css_ors[i], d_dcm[:,:,j] @ sun_vector_ref)

        # TAM Jacobian
        #H_tam = np.zeros((3, 7))
        #for j in range(4):
        #    H_tam[:, j] = d_dcm[:,:,j] @ mag_field_ref

        # Combine Jacobians
        #H = np.vstack([H_css, H_tam])
        
        # Jacobian of CSS measurements w.r.t. quaternion.
        dR_dq0 = 2 * np.array([[ q0,  q3, -q2],
                               [-q3,  q0,  q1],
                               [ q2, -q1,  q0]])

        dR_dq1 = 2 * np.array([[ q1,  q2,  q3],
                               [ q2, -q1, -q0],
                               [ q3,  q0, -q1]])

        dR_dq2 = 2 * np.array([[-q2,  q1,  q0],
                               [ q1,  q2,  q3],
                               [-q0,  q3, -q2]])

        dR_dq3 = 2 * np.array([[-q3, -q0,  q1],
                               [ q0, -q3,  q2],
                               [ q1,  q2,  q3]])
        for i in range(6):
            dh_dq0 = np.max((0, np.dot(self.css_ors[i], dR_dq0 @ sun_vector_body)))
            dh_dq1 = np.max((0, np.dot(self.css_ors[i], dR_dq1 @ sun_vector_body)))
            dh_dq2 = np.max((0, np.dot(self.css_ors[i], dR_dq2 @ sun_vector_body)))
            dh_dq3 = np.max((0, np.dot(self.css_ors[i], dR_dq3 @ sun_vector_body)))
            H[i, :4] = (dh_dq0, dh_dq1, dh_dq2, dh_dq3)
        
        # Jacobian of TAM measurements w.r.t. quaternion.
        H[6:, :4] = 2 * np.array([
            [ q0*Bx + q3*By - q2*Bz,  q1*Bx + q2*By + q3*Bz, -q2*Bx + q1*By - q0*Bz, -q3*Bx + q0*By + q1*Bz],
            [-q3*Bx + q0*By + q1*Bz,  q2*Bx - q1*By + q0*Bz,  q1*Bx + q2*By + q3*Bz,  q0*Bx + q3*By - q2*Bz],
            [ q2*Bx - q1*By + q0*Bz,  q3*Bx - q0*By - q1*Bz,  q0*Bx + q3*By - q2*Bz, -q1*Bx - q2*By - q3*Bz]
        ])

        return H

    def _prediction_step(self, dt, x, P, u):
        # Linearize process model about x[k-1].
        #F = self._process_model_jacobian(x, dt, u)
        F = self.numerical_jacobian(
            functools.partial(self._process_model, dt=dt, u=None), 7, 7, x
        )
        
        # Advance state using full, nonlinear process model.
        x = self._process_model(x, dt, u)
        x[:4] /= np.linalg.norm(x[:4])
        
        # Advance state covariance using linearized process model F.
        P = F @ P @ F.T + Q
        
        return x, P
    
    def _correction_step(self, dt, x, P, z, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed):
        # Linearize measurement model about predicted x.
        #H = self._measurement_model_jacobian(x, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed)
        H = self.numerical_jacobian(
            functools.partial(self._measurement_model, sc_position_inertial=sc_position_inertial,
                                                       sun_vector_inertial=sun_vector_inertial,
                                                       mag_field_earth_fixed=mag_field_earth_fixed),
            9, 7, x
        )
        
        # Calculate measurement residual using full, nonlinear measurement model.
        h = self._measurement_model(x, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed)
        y = z - h
        
        # Use Jacobian for all further calculations.
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        x[:4] /= np.linalg.norm(x[:4])
        P = (np.eye(len(P)) - K @ H) @ P
        return x, P, y, h
        
    def estimate(self, dt, x, P, u, z, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed):
        x, P = self._prediction_step(dt, x, P, u)
        x, P, y, h = self._correction_step(dt, x, P, z, sc_position_inertial, sun_vector_inertial, mag_field_earth_fixed)
        return x, P, y, h
    
    def simulate(self, t, x_init, P_init, Q, R, controls, measurements,
                 sc_positions_inertial, sun_vectors_inertial, mag_fields_earth_fixed):
        
        # Set up containers to store estimates.
        estimates = np.zeros((len(t), len(x_init)))
        estimates_cov = np.zeros((len(t), len(x_init)))
        estimates[0] = x_init
        estimates_cov[0] = np.diag(P_init)
        
        # Handy for troubleshooting and analysis.
        residuals = np.zeros((len(t), measurements.shape[1]))
        measurements_est = np.zeros((len(t), measurements.shape[1]))
        
        # Initialize state and covariance.
        x, P = x_init, P_init
        
        # Simulate each time step and store the results.
        for k in tqdm(range(1, len(t))):
            dt = t[k] - t[k-1]
            u = controls[k]
            z = measurements[k]
            sc_position_inertial = sc_positions_inertial[k]
            sun_vector_inertial = sun_vectors_inertial[k]
            mag_field_earth_fixed = mag_fields_earth_fixed[k]
            x, P, y, h = self.estimate(dt, x, P, u, z,
                                       sc_position_inertial,
                                       sun_vector_inertial,
                                       mag_field_earth_fixed)
            estimates[k] = x
            estimates_cov[k] = np.diag(P)
            residuals[k] = y
            measurements_est[k] = h
        
        return estimates, estimates_cov, residuals, measurements_est
    
    def plot(self, t, ground_truth, estimates, estimates_cov, measurements, measurements_est, residuals,
             start_idx=0, end_idx=-1):
        t = t[start_idx:end_idx]
        ground_truth = ground_truth[start_idx:end_idx]
        estimates = estimates[start_idx:end_idx]
        estimates_cov = estimates_cov[start_idx:end_idx]
        measurements = measurements[start_idx:end_idx]
        measurements_est = measurements_est[start_idx:end_idx]
        residuals = residuals[start_idx:end_idx]

        plt.figure(figsize=(15, 30))
        plt.rc("font", size=20)
        
        plt.subplot(3, 1, 1)
        att_error_degrees = attitude_error(estimates[:, :4], ground_truth[:, :4])
        plt.plot(t, att_error_degrees)
        plt.xlabel("Time (s)")
        plt.ylabel("Attitude Error (degrees)")
        
        plt.subplot(3, 1, 2)
        for i, (label, color) in enumerate((("wx", "red"), ("wy", "green"), ("wz", "blue"))):
            plt.plot(t, estimates[:, 4+i] - ground_truth[:, 4+i], label=f"{label} error", color=color)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity Error (m/s)")
        
        plt.subplot(3, 1, 3)
        for i, label in enumerate(("css+x", "css-x", "css+y", "css-y", "css+z", "css-z", "mag_x", "mag_y", "mag_z")):
            plt.plot(t, residuals[:, i], label=f"{label} (res.)")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Measurement Residual")
        
        
ads = AttitudeEstimatorEKF(css_orientations=css_vectors, sc_moment_of_inertia=moi)
