#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


template<typename T>
void assign(T *dest, const T *src, size_t length) {
  for (size_t i = 0; i < length; i++) {
    dest[i] = src[i];
  }
}

void matmul(
  float *dest, 
  const float *a, 
  const float *b, 
  size_t in_dim, 
  size_t hidden_dim,
  size_t out_dim
) {
  for (size_t i = 0; i < in_dim; i++) {
    for (size_t j = 0; j < out_dim; j++) {
      dest[i * out_dim + j] = 0;

      for (size_t k = 0; k < hidden_dim; k++) {
        dest[i * out_dim + j] += a[i*hidden_dim + k] * b[k*out_dim + j];
      }
    }
  }
}

void matmul(float *dest, const float *src, float val, size_t length) {
  for (size_t i = 0; i < length; i++) {
    dest[i] = src[i] * val;
  }
}

void subtract(float *dest, const float *a, const float *b, size_t row, size_t col) {
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      dest[i * col + j] = a[i * col + j] - b[i * col + j];
    }
  }
}

void exp(float *dest, const float *src, size_t length) {
  for (size_t i = 0; i < length; i++) {
    dest[i] = exp(src[i]);
  }
}

void row_norm(float *dest, const float *src, size_t row, size_t col) {
  for (size_t i = 0; i < row; i++) {
    float row_sum = 0;

    // compute row sum
    for (size_t j = 0; j < col; j++) {
      row_sum += src[i * col + j];
    }

    // normalize
    for (size_t j = 0; j < col; j++) {
      dest[i * col + j] = src[i * col + j] / row_sum;
    }
  }
}

void transpose(float *dest, const float *src, size_t row, size_t col) {
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      dest[j * row + i] = src[i * col + j];
    }
  }
}

void clear(float *dest, size_t length) {
  for (size_t i = 0; i < length; i++) {
    dest[i] = 0;
  }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    auto X_i = new float[batch * n];
    auto X_i_T = new float[batch * n];
    auto y_i = new unsigned char[batch];
    auto h = new float[batch * k];
    auto h_exp = new float[batch * k];
    auto Z = new float[batch * k];
    auto I_y = new float[batch * k];
    auto l_grad = new float[batch * k];
    auto theta_grad = new float[n * k];

    for (size_t i = 0; i < m; i += batch) {
      assign(X_i, X + (i * n), batch * n); // batch x n
      assign(y_i, y + i, batch); // batch x 1

      matmul(h, X_i, theta, batch, n, k); // batch x k
      exp(h_exp, h, batch * k); // batch x k

      row_norm(Z, h_exp, batch, k); // batch x k

      clear(I_y, batch * k);
      for (size_t j = 0; j < batch; j++) {
        I_y[j*k + y_i[j]] = 1;
      }

      subtract(l_grad, Z, I_y, batch, k); // batch x k
      transpose(X_i_T, X_i, batch, n); // n x batch

      matmul(theta_grad, X_i_T, l_grad, n, batch, k); // n x k
      matmul(theta_grad, theta_grad, lr / batch, n * k); // n x k
      
      subtract(theta, theta, theta_grad, n, k); // n x k
    }

    delete[] X_i;
    delete[] y_i;
    delete[] h;
    delete[] h_exp;
    delete[] Z;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
