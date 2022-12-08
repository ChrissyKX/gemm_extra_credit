task void gemm_task(uniform int m, uniform int n, uniform int k,
          uniform int n_num_block, uniform int kk, uniform int bwidth, uniform int bheight,
                        uniform double A[],
                        uniform double B[],
                        uniform double C[],
                        uniform double alpha,
                        uniform double beta)
{    
     uniform int jj = ((taskIndex / m) % n_num_block) * bwidth;
     uniform int jj_end = min(n, jj + bwidth);
     uniform int kk_end = min(k, kk + bheight);

     uniform int i = taskIndex % m;

     foreach (j = jj ... jj_end) {
         double sum = beta * C[i * n + j];
         for (int _k = kk; _k < kk_end; _k++) {
             sum += alpha * A[i * k + _k] * B[_k * n + j];
         }
         C[i * n + j] = sum;
     }
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) {
	// YOUR IMPLEMENTATION HERE
    uniform int CLINE = 8;  // cache line size in terms of oudbles
    uniform int CORE_COUNT = 4;
    uniform int INTERLEAVE_FACTOR = 1;
    uniform int BRATE = CORE_COUNT * INTERLEAVE_FACTOR * CLINE;
    uniform int MAX_BWIDTH = 128;
    uniform int MAX_BHEIGHT = 128;
    uniform int MIN_BSIZE = 4096;
    // uniform int BSIZE = 64;  // 64B cache line can hold 8 doubles, don't go under 8 doubles   

    uniform int bwidth = n / BRATE * CLINE;
    if (bwidth == 0) {
       bwidth = CLINE;
    } else if (bwidth > MAX_BWIDTH) {
       bwidth = MAX_BWIDTH;
    }
    uniform int bheight = bwidth * 2;
    if (bheight < MIN_BSIZE / bwidth) {
       bheight = MIN_BSIZE / bwidth;
    }
    if (bheight > MAX_BHEIGHT) {
       bheight = MAX_BHEIGHT;
    } 
    
    print("bwidth: %d bheight: %d\n", bwidth, bheight);
    bwidth = 32;
    bheight = 256;
    
    uniform int k_num_block = (k + bheight  - 1) / bheight;
    uniform int n_num_block = (n + bwidth - 1) / bwidth;
    uniform int num_task = m * n_num_block;

    for (uniform int kk = 0; kk < k; kk += bheight) {
        double start = CycleTimer::currentSeconds();
        launch[num_task] gemm_task(m, n, k, n_num_block, kk, bwidth, bheight, A, B, C, alpha, beta);
        sync;
        double elapse = CycleTimer::currentSeconds() - start;
        print("time one launch: %\n", elapse);
    }

}
