import sys
import numpy as np
from calculate_optical_flow import *

def compute_camera_feed_output(image, view):
    
    rgb, optical_flow_derivatives, uv = compute_flow(image, view)

    lab = cv2.cvtColor((rgb/rgb.max()*255).astype(np.uint8), cv2.COLOR_RGB2LAB)

    def apply_complex(a):
        return np.complex(a[0], a[1])
    def apply_complex_conjugate(a):
        return np.complex(a[0], -a[1])

    # flow derivative
    du_dv_complex_value = np.apply_along_axis(apply_complex, 2, optical_flow_derivatives)
    uv_complex = np.apply_along_axis(apply_complex, 2, lab[:,:,:2])
    uv_complex_conjugate = np.apply_along_axis(apply_complex_conjugate, 2, lab[:,:,:2])

    fl_ci = du_dv_complex_value / uv_complex_conjugate
    amplitude = fl_ci.imag
    real_part = fl_ci.real
    phase = np.angle(uv_complex)
    temporal = np.angle(fl_ci)

    return amplitude, real_part, phase, temporal, optical_flow_derivatives, rgb
