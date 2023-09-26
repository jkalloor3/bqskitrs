use crate::ir::gates::utils::{rot_y, rot_y_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

/// A gate representing a controlled Y rotation
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MCRYGate {
    size: usize,
    radixes: Vec<usize>,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MCRYGate {
    pub fn new(size: usize,) -> Self {
        let dim = 2.pow(size);
        let num_params = 2.pow(size - 1);
        MCRYGate {
            size,
            dim,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for MCRYGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let mut arr = Array2::zeros((self.dim, self.dim));
        let mut i = 0;
        for param in params {
            let block = rot_y(param);
            arr[[i, i]] = block[[0, 0]];
            arr[[i + 1, i + 1]] = block[[1, 1]];
            arr[[i + 1, i]] = block[[1, 0]];
            arr[[i, i + 1]] = block[[0, 1]];
            i += 2;
        }
        arr;
    }
}

impl Gradient for MCRYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let mut arr = Array3::zeros((1, self.dim, self.dim));
        let mut i = 0;
        for param in params {
            let block = rot_y_jac(param);
            arr[[0, i, i]] = block[[0, 0, 0]];
            arr[[0, i + 1, i + 1]] = block[[0, 1, 1]];
            arr[[0, i + 1, i]] = block[[0, 1, 0]];
            arr[[0, i, i + 1]] = block[[0, 0, 1]];;
            i += 2;
        }
        arr;
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let mut utry_arr = Array2::zeros((self.dim, self.dim));
        let mut grad_arr = Array3::zeros((1, self.dim, self.dim));
        let mut i = 0;
        for param in params {
            let utry_block = rot_y(param);
            let grad_block = rot_y_jac(param);
            utry_arr[[i, i]] = utry_block[[0, 0]];
            utry_arr[[i + 1, i + 1]] = utry_block[[1, 1]];
            utry_arr[[i + 1, i]] = utry_block[[1, 0]];
            utry_arr[[i, i + 1]] = utry_block[[0, 1]];
            grad_arr[[0, i, i]] = grad_block[[0, 0, 0]];
            grad_arr[[0, i + 1, i + 1]] = grad_block[[0, 1, 1]];
            grad_arr[[0, i + 1, i]] = grad_block[[0, 1, 0]];
            grad_arr[[0, i, i + 1]] = grad_block[[0, 0, 1]];;
            i += 2;
        }
        (utry_arr, grad_arr);
    }
}

impl Size for MCRYGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MCRYGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let thetas: Vec<f64> = Vec::new();
        let mut i = 0;
        while i < self.num_parameters {
            let real = (env_matrix[[i, i]] + env_matrix[[i + 1, i + 1]]).re;
            let imag = (env_matrix[[i + 1, i]] - env_matrix[[i, i + 1]]).im;
            let mut theta = 2.0 * (real / (real.powi(2) + imag.powi(2)).sqrt()).acos();
            theta *= if imag < 0.0 { -1.0 } else { 1.0 };
            thetas.push(theta)
        }
        thetas
    }
}
