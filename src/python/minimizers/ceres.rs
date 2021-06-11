use pyo3::{exceptions::PyTypeError, prelude::*, types::PyTuple};

use crate::minimizers::{CeresJacSolver, ResidualFunction};

use numpy::{PyArray1, PyArray2};

use crate::minimizers::Minimizer;

#[pyclass(name = "LeastSquares_Jac_SolverNative", module = "bqskitrs")]
struct PyCeresJacSolver {
    #[pyo3(get)]
    distance_metric: String,
    num_threads: usize,
    ftol: f64,
    gtol: f64,
}

#[pymethods]
impl PyCeresJacSolver {
    #[new]
    fn new(num_threads: Option<usize>, ftol: Option<f64>, gtol: Option<f64>) -> Self {
        let threads = if let Some(threads) = num_threads {
            threads
        } else {
            1
        };
        let ftol = if let Some(ftol) = ftol {
            ftol
        } else {
            1e-6 // Ceres documented default
        };
        let gtol = if let Some(gtol) = gtol {
            gtol
        } else {
            1e-10 // Ceres documented default
        };
        Self {
            distance_metric: String::from("Residuals"),
            num_threads: threads,
            ftol,
            gtol,
        }
    }

    fn minimize(&self, py: Python, cost_fn: PyObject, x0: PyObject) -> PyResult<Py<PyArray1<f64>>> {
        let x0_rust = x0.extract::<Vec<f64>>(py)?;
        let solv = CeresJacSolver::new(self.num_threads, self.ftol, self.gtol);
        let cost_fun = match cost_fn.extract::<ResidualFunction>(py) {
            Ok(fun) => Ok(fun),
            Err(err) => Err(PyTypeError::new_err(err.to_string())),
        }?;
        let x = solv.minimize(cost_fun, x0_rust);
        Ok(PyArray1::from_vec(py, x).to_owned())
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let num_threads = PyTuple::new(py, &[slf.num_threads]).into_py(py);
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, num_threads))
    }
}
