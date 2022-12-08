task void gemm_task(uniform int m, uniform int n, uniform int k,
          uniform int n_num_block, uniform int kk, uniform int bwidth, uniform int bheight,
                        uniform double A[],
                        uniform double B[],
                        uniform double C[],
                        uniform double alpha,
                        uniform double beta)
{    
     uniform int jj = ((taskIndex) % n_num_block) * bwidth;
     uniform int jj_end = min(n, jj + bwidth);
     uniform int kk_end = min(k, kk + bheight);

     for (uniform int i = 0; i < m; i++) {
         for (uniform int j = jj; j < jj_end; j += programCount) {
             int _j = j + programIndex;
             if (_j < jj_end) {
                 double sum = beta * C[i * n + _j];
                 for (uniform int _k = kk; _k < kk_end; _k++) {
                 sum += alpha * A[i * k + _k] * B[_k * n + _j];
                 }
                 C[i * n + _j] = sum;
             }
          }
     }
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) {
	// YOUR IMPLEMENTATION HERE
    uniform int bheight = 128;
    uniform int bwidth = 8;  // set to SIMD width
    //uniform int MIN_HT = 64;

    //if (bheight < MIN_HT) {
       //bheight = MIN_HT;
    //} 
    
    //print("bwidth: %d bheight: %d\n", bwidth, bheight);

    uniform int k_num_block = (k + bheight  - 1) / bheight;
    uniform int n_num_block = (n + bwidth - 1) / bwidth;
    uniform int num_task = n_num_block;

    for (uniform int kk = 0; kk < k; kk += bheight) {
        launch[num_task] gemm_task(m, n, k, n_num_block, kk, bwidth, bheight, A, B, C, alpha, beta);
        sync;

    }

}
