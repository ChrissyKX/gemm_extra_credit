task void gemm_task(uniform int m, uniform int n, uniform int k,
          uniform int m_num_block, uniform int n_num_block, uniform int kk,
          uniform int bwidth, uniform int bheight,
                        uniform double A[],
                        uniform double B[],
                        uniform double C[],
                        uniform double alpha,
                        uniform double beta)
{    
     uniform int ii = (taskIndex % m_num_block) * bwidth;
     uniform int jj = ((taskIndex / m_num_block) % n_num_block) * bwidth;
     uniform int ii_end = min(m, ii + bwidth);
     uniform int jj_end = min(n, jj + bwidth);
     uniform int kk_end = min(k, kk + bheight);

     for (uniform int i = ii; i < ii_end; i++) {
         foreach (_j = jj... jj_end) {
             double sum = beta * C[i * n + _j];
             for (uniform int _k = kk; _k < kk_end; _k++) {
                 sum += alpha * A[i * k + _k] * B[_k * n + _j];
             }
             C[i * n + _j] = sum;
          
          }
     }
}

void _gemm(uniform int m, uniform int n, uniform int k,  
                        uniform double A[],
                        uniform double B[],
                        uniform double C[],
                        uniform double alpha,
                        uniform double beta)
{
    uniform int bsize = 32;
    uniform int _i, _j,ii, jj, _i_end, _j_end;
    uniform int _k, kk, _k_end;
    double sum;
    uniform int aw = k;
    uniform int bw = n;
    uniform int cw = n;

    foreach (_i = 0 ... m, _j = 0 ... n) {
            // print("Elem before mul is % , beta is % \n", C[_i * cw + _j], beta);
            C[_i * cw + _j] = beta * C[_i * cw + _j]; // initialize to C * beta
            // print("Elem after mul is % \n", C[_i * cw + _j]);
        // }
    }   

    for (kk = 0; kk < aw; kk += bsize) {
        for (jj = 0; jj < bw; jj += bsize) {
            for (ii = 0; ii < m; ii+= bsize) {   // for each row in A
                // for (_i = ii; _i < ii + bsize; _i++) {
                    // for (_j = jj; _j < jj + bsize; _j++) {
                // 
                _i_end = min(m, ii+bsize);
                _j_end = min(n, jj+bsize);
                _k_end = min(k, kk+bsize);
                foreach (_i = ii ... _i_end, _j = jj ... _j_end) {
                        sum = C[_i * cw + _j];
                        for (_k = kk; _k < _k_end; _k++) {
                            sum += alpha * A[_i * aw + _k] * B[_k * bw + _j];
                        }
                        C[_i * cw + _j] =  sum;
                }  
            }
        }
    }
}


export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) {

    if (k <= 64) { // do without launch
        _gemm(m, n, k, A, B, C, alpha, beta);
        return;
    }

    uniform int MAX_NUMD = 2048;
    uniform int bwidth = n / 8;  // 8 is hyperthread count
    if (bwidth > 128) {
       bwidth = 128;
    }
    uniform int bheight = MAX_NUMD / bwidth;
    if (bheight < 1) {
       bheight = 1;
    }
    
    //print("bwidth: %d bheight: %d\n", bwidth, bheight);

    uniform int m_num_block = (m + bwidth  - 1) / bwidth;
    uniform int n_num_block = (n + bwidth - 1) / bwidth;
    uniform int num_task = n_num_block * m_num_block;

    for (uniform int kk = 0; kk < k; kk += bheight) {
        launch[num_task] gemm_task(m, n, k, m_num_block, n_num_block, kk, bwidth, bheight, A, B, C, alpha, beta);
        sync;
    }
}


