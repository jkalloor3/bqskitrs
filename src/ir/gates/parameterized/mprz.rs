use crate::ir::gates::utils::{rot_z, get_indices};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use ndarray_linalg::SVD;
use crate::squaremat::Matmul;

fn svd(matrix: ArrayViewMut2<c64>) -> (Array2<c64>, Array2<c64>) {
    let result = matrix.svd(true, true);
    let actual_result = match result {
        Ok(res)  => res,
        Err(_res) => panic!("Problem svding the matrix: {:?}", matrix),
    };

    // Safety: u/vt are the same size since matrix is a square matrix with sides of size `size`
    (actual_result.0.unwrap(), actual_result.2.unwrap())
}


/// A gate representing a multiplexed Z rotation one 1 qubit
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MPRZGate {
    size: usize,
    dim: usize,
    target_qudit: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MPRZGate {
    pub fn new(size: usize, target_qudit: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = base.pow((size - 1) as u32) as usize;
        MPRZGate {
            size,
            dim,
            target_qudit,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for MPRZGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, _params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let mut arr: Array2<c64> = Array2::zeros((self.dim, self.dim));
        let mut i: usize = 0;
        for param in _params {
            let block = rot_z(*param, None);
            let (x1, x2) = get_indices(i, self.target_qudit, self.size);
            arr[[x1, x1]] = block[[0, 0]];
            arr[[x2, x2]] = block[[1, 1]];
            arr[[x2, x1]] = block[[1, 0]];
            arr[[x1, x2]] = block[[0, 1]];
            i += 1;
        }
        let (u, vt) = svd(arr.view_mut());
        u.matmul(vt.view())
    }
}

impl Gradient for MPRZGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let orig_utry = self.get_utry(_params, _const_gates);
        let mut grad: Array3<c64> = Array3::zeros((_params.len(), self.dim, self.dim));

        for (i, &param) in _params.iter().enumerate() {
            let dpos = c64::new(0.0, 1.0) * (c64::new(0.0, 1.0) * (param / 2.0)).exp() / 2.0;
            let dneg = -c64::new(0.0, 1.0) * (c64::new(0.0, -1.0) * (param / 2.0)).exp() / 2.0;

            let (x1, x2) = get_indices(i, self.target_qudit, self.size);

            let mut matrix = orig_utry.clone();

            matrix[[x1, x1]] = dpos;
            matrix[[x2, x2]] = dneg;

            for j in 0..self.dim {
                for k in 0..self.dim {
                    grad[[i, j, k]] = matrix[[j, k]];
                }
            }
        }
        grad
    }

    fn get_utry_and_grad(
        &self,
        _params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let utry = self.get_utry(_params, _const_gates);
        let grad = self.get_grad(_params, _const_gates);
        (utry, grad)
    }
}

impl Size for MPRZGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MPRZGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 0;
        while i < self.dim {
            let (x1, x2) = get_indices(i, self.target_qudit, self.size);
            let real = env_matrix[[x2, x2]].re;
            let imag = env_matrix[[x2, x2]].im;
            // Get angle of angle -theta/2
            let b = (imag / real).atan();
            let real_2 = env_matrix[[x1, x1]].re;
            let imag_2 = env_matrix[[x1, x1]].im;
            // Get angle of theta/2
            let a = (imag_2 / real_2).atan();
            let theta = a - b;
            thetas.push(theta);
            i = i + 2;
        }
        thetas
    }
}