use crate::ir::gates::utils::{rot_z, rot_z_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use ndarray_linalg::SVD;
use std::f64::consts::PI;
use crate::squaremat::Matmul;

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


/// A gate representing a multiplexed Z rotation one 1 qubit
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MCRZGate {
    size: usize,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MCRZGate {
    pub fn new(size: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = base.pow((size - 1) as u32) as usize;
        MCRZGate {
            size,
            dim,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for MCRZGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let mut arr = Array2::zeros((self.dim, self.dim));
        let mut i: usize = 0;
        for param in params.into_iter() {
            let block = rot_z(*param, None);
            arr[[i, i]] = block[[0, 0]];
            arr[[i + 1, i + 1]] = block[[1, 1]];
            i += 2;
        }
        let (u, vt) = svd(arr.view_mut());
        u.matmul(vt.view())
    }
}

impl Gradient for MCRZGate {
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

impl Size for MCRZGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MCRZGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 0;
        while i < self.dim {
            let real = env_matrix[[i + 1, i + 1]].re;
            let imag = env_matrix[[i + 1, i + 1]].im;
            // Get angle of angle -theta/2
            let b = (imag / real).atan();
            let real_2 = env_matrix[[i, i]].re;
            let imag_2 = env_matrix[[i, i]].im;
            // Get angle of theta/2
            let a = (imag_2 / real_2).atan();
            let mut theta = a - b;
            thetas.push(theta);
            i = i + 2;
        }
        thetas
    }
}

// sudo docker run -it -e OPENBLAS_ARGS="DYNAMIC_ARCH=1" -v $(pwd):/io -t bqskitrs_docker
// /bin/maturin build  --release --features=openblas --compatibility=manylinux2014