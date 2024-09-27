use crate::ir::gates::utils::rot_y;
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use crate::squaremat::Matmul;

use ndarray_linalg::SVD;

fn svd(matrix: ArrayViewMut2<c64>) -> (Array2<c64>, Array2<c64>) {
    let result = matrix.svd(true, true);
    let actual_result = match result {
        Ok(res)  => res,
        Err(_res) => panic!("Problem svding the matrix: {:?}", matrix),
    };

    // Safety: u/vt are the same size since matrix is a square matrix with 
    // sides of size `size`
    (actual_result.0.unwrap(), actual_result.2.unwrap())
}


/// A gate representing a multiplexed Y rotation on 1 qubit
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MPRYGate {
    size: usize,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MPRYGate {
    pub fn new(size: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = base.pow((size - 1) as u32) as usize;
        MPRYGate {
            size,
            dim,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for MPRYGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let mut arr: Array2<c64> = Array2::zeros((self.dim, self.dim));
        let mut i: usize = 0;
        for param in params {
            let block = rot_y(*param);
            arr[[i, i]] = block[[0, 0]];
            arr[[i + 1, i + 1]] = block[[1, 1]];
            arr[[i + 1, i]] = block[[1, 0]];
            arr[[i, i + 1]] = block[[0, 1]];
            i += 2;
        }
        let (u, vt) = svd(arr.view_mut());
        u.matmul(vt.view())
    }
}

impl Gradient for MPRYGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        unimplemented!()
    }

    fn get_utry_and_grad(
        &self,
        _params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        unimplemented!()
    }
}

impl Size for MPRYGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MPRYGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 0;
        // Each variable is indepent, so you can optimize 2
        while i < self.dim {
            let a = (env_matrix[[i, i]] + env_matrix[[i + 1, i + 1]]).re;
            let b = (env_matrix[[i + 1, i]] - env_matrix[[i, i + 1]]).re;
            let mut theta = 2.0 * (a / (a.powi(2) + b.powi(2)).sqrt()).acos();
            theta *= if b > 0.0 { -1.0 } else { 1.0 };
            thetas.push(theta);
            i = i + 2;
        }
        thetas
    }
}
