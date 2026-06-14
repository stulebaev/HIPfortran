// hipcc hipblas_sgemm.cpp -lhipblas -o hipblas_sgemm
// export HSA_OVERRIDE_GFX_VERSION=10.3.0
// ./hipblas_sgemm
//
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// Macro for checking HIP runtime errors
#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Macro for checking hipBLAS errors
#define CHECK_HIPBLAS_ERROR(status) \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLAS Error code: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 3;
    const int N = 3;
    const int K = 3;

    // Define scalars for alpha and beta
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Host matrices (Column-major format)
    std::vector<float> h_A = {1.0f, 4.0f, 7.0f,  2.0f, 5.0f, 8.0f,  3.0f, 6.0f, 9.0f}; // 3x3
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f}; // 3x3 Identity
    std::vector<float> h_C(M * N, 0.0f);                                               // Result matrix

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on the GPU
    CHECK_HIP_ERROR(hipMalloc(&d_A, M * K * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, K * N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, M * N * sizeof(float)));

    // Copy matrices from Host to Device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    // Initialize hipBLAS handle
    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    // Perform SGEMM: C = alpha * A * B + beta * C
    // hipBLAS assumes column-major matrix storage layout by default
    CHECK_HIPBLAS_ERROR(hipblasSgemm(
        handle,
        HIPBLAS_OP_N, // Operation type for Matrix A (No transpose)
        HIPBLAS_OP_N, // Operation type for Matrix B (No transpose)
        M,            // Number of rows in A and C
        N,            // Number of columns in B and C
        K,            // Number of columns in A, rows in B
        &alpha,       // Scalar multiplier for A * B
        d_A,          // Device pointer to matrix A
        M,            // Leading dimension of A (LDA)
        d_B,          // Device pointer to matrix B
        K,            // Leading dimension of B (LDB)
        &beta,        // Scalar multiplier for C
        d_C,          // Device pointer to matrix C
        M             // Leading dimension of C (LDC)
    ));

    // Copy the result matrix C back to Host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Print the output matrix
    std::cout << "Result Matrix C (3x3 column-major array printed as a grid):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i + j * M] << " ";
        }
        std::cout << "\n";
    }

    // Clean up resources
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    return 0;
}
