# -*- coding: utf-8 -*-

void
pack_view(int n, int nrv, int ncv,
          const ${dtype} *__restrict__ v,
          const int *__restrict__ vix,
          const int *__restrict__ vcstri,
          const int *__restrict__ vrstri,
          ${dtype} *__restrict__  pmat)
{
    if (ncv == 1)
        for (int i = 0; i < n; i++)
            pmat[i] = v[vix[i]];
    else if (nrv == 1)
        for (int i = 0; i < n; i++)
            for (int c = 0; c < ncv; c++)
                pmat[c*n + i] = v[vix[i] + vcstri[i]*c];
    else
        for (int i = 0; i < n; i++)
            for (int r = 0; r < nrv; r++)
                for (int c = 0; c < ncv; c++)
                    pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r
                                                + vcstri[i]*c];
}
