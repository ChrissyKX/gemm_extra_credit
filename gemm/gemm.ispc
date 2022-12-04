
task void gemm_task(uniform int m, uniform int n, uniform int k,
          uniform int n_num_block, uniform int kk, uniform int bsize,
                        uniform double A[],
                        uniform double B[],
                        uniform double C[],
                        uniform double alpha,
                        uniform double beta)
{    
     uniform int jj = ((taskIndex / m) % n_num_block) * bsize;
     uniform int jj_end = min(n, jj + bsize);
     uniform int kk_end = min(k, kk + bsize);

     uniform int i = taskIndex % m;

     foreach (j = jj ... jj_end) {
         double sum = beta * C[i * k + j];
         for (int _k = kk; _k < kk_end; _k++) {
             sum += alpha * A[i * k + _k] * B[_k * n + j];
         }
         C[i * k + j] = sum;
     }
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) {
	// YOUR IMPLEMENTATION HERE

    uniform int BSIZE = 64;  // 64B cache line can hold 8 doubles
    uniform int bsize = BSIZE;
    uniform int k_num_block = (k + bsize  - 1) / bsize;
    uniform int n_num_block = (n + bsize - 1) / bsize;
    uniform int num_task = m * n_num_block;

    for (uniform int kk = 0; kk < k; kk += bsize) {
        launch[num_task] gemm_task(m, n, k, n_num_block, kk, bsize, A, B, C, alpha, beta);
        sync;
    }

}
