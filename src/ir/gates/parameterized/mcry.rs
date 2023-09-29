use crate::ir::gates::utils::{rot_y, rot_y_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use crate::squaremat::Matmul;

use ndarray_linalg::SVD;

fn svd(matrix: ArrayViewMut2<c64>) -> (Array2<c64>, Array2<c64>) {
    // let size = matrix.shape()[0];
    // let layout = MatrixLayout::C {
    //     row: size as i32,
    //     lda: size as i32,
    // };
    let result = matrix.svd(true, true).unwrap();
    // Safety: u/vt are the same size since matrix is a square matrix with sides of size `size`
    (result.0.unwrap(), result.2.unwrap())
    // unsafe {
    //     (
    //         Array2::from_shape_vec_unchecked((size, size), result.U.unwrap()),
    //         Array2::from_shape_vec_unchecked((size, size), result.VT.unwrap()),
    //     )
    // }
}


/// A gate representing a multiplexed Y rotation on 1 qubit
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MCRYGate {
    size: usize,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MCRYGate {
    pub fn new(size: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = base.pow((size - 1) as u32) as usize;
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

impl Gradient for MCRYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        unimplemented!()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        unimplemented!()
    }
}

impl Size for MCRYGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MCRYGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        // println!("{} days", 31);
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 0;
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
