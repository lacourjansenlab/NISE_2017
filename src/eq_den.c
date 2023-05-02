#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include "omp.h"
#include "types.h"
#include "NISE_subs.h"
#include "propagate.h"
#include "absorption.h"
//#include "luminescence.h"
#include "1DFFT.h"
#include "project.h"
#include "eq_den.h"


/*Here we defined the equilibrium density operator*/
void eq_den(float *Hamiltonian_i, float *rho_l, int N, t_non *non){
  int index;
  float *H,*e,*e_1;
  float *rho_r;
  //float *rho_l;
  float *diag_sum;  /*Here is store the diagonal element for one segment*/
  N=non->singles;
  H=(float *)calloc(N*N,sizeof(float));
  e=(float *)calloc(N,sizeof(float));
  e_1=(float *)calloc(N,sizeof(float));
  rho_r=(float *)calloc(N*N,sizeof(float));
  //rho_l=(float *)calloc(N*N,sizeof(float));
  int a,b,c,pro_dim;
  int site_num,site_num_1,site_num_2,seg_num;/*site num is the number of the site, which used to make sure the index number later*/

  float kBT=non->temperature*0.6950859; // Here 0.6950859 is the K_b in the unit of cm-1
  /*Here I have one question, the input1D file does not contain temperature, 
  where we should include it */
  
  float u,i_u; 
  /* projection */
  pro_dim=project_dim(non);
  diag_sum = (float *)calloc(pro_dim,sizeof(float));

  /* Do projection and make sure the segment number equal to the segment number in the projection file.*/   
  if (non->Npsites==non->singles){
    zero_coupling(Hamiltonian_i,non);
  } else {
    printf("Segment number and the projection number are different");
    exit(1);
  }

  // Build Hamiltonian, convert the triangular Hamiltonian to the  square Hamiltonian
  for (a=0;a<N;a++){
    /* Fill diagonal elements */
    H[a+N*a]=Hamiltonian_i[a+N*a-(a*(a+1))/2]; 
    /* Fill couplings */
    for (b=a+1;b<N;b++){
      H[a+N*b]=Hamiltonian_i[b+N*a-(a*(a+1))/2];
      H[b+N*a]=Hamiltonian_i[b+N*a-(a*(a+1))/2];
    }
  }
/*Here we diagonalize the square Hamiltonian, the H is replaced by the eigenvector in eigen basis, e is the eigenvalue */

  diagonalizeLPD(H,e,N);
 
  // Exponentiate [u=exp(-H*kBT)]
  for (a=0;a<N;a++){
    e_1[a]=exp(-e[a]*kBT);
    u=u+e_1[a];
  }
  /*Here calculate the inverse of u*/
  i_u=1.0/u;

  // Transform to site basis, H*u*H
  /*Here we first calculate the right side u*H, which output rho_r */
  for (a=0;a<N;a++){
    for (b=0;b<N;b++){
      rho_r[b+a*N]+=H[b+a*N]*e_1[b]*i_u;
    }
  }  
  /*Secondly, we calculate the left side H*u_r, which output rho_l */
  for (a=0;a<N;a++){
    for (b=0;b<N;b++){
      for (c=0;c<N;c++){
        rho_l[a+c*N]+=H[b+a*N]*rho_r[b+c*N];
      }
    }
  }
/* The weights in the site basis were calculated, 
Here we should not combine the two loop in one loop as it would results in the heary calcuation.*/

/*Here we need to normalize rho, exp(-KB*H)/[Trexp(-KB*H)] within every segment*/
/*We first sum the diagonal element in one segment*/
  for (site_num=0;site_num<non->singles;site_num++){
    seg_num=non->psites[site_num];
    diag_sum[seg_num]+= rho_l[site_num*N+site_num];
  }

    /*Secondly, we normalize the density matrix within one segment*/
    //for (seg_num=0;seg_num<pro_dim;seg_num++){
  for (site_num_1=0;site_num_1<non->singles;site_num_1++){
    seg_num=non->psites[site_num_1];
    for (site_num_2=0;site_num_2<non->singles;site_num_2++){
      if (seg_num==non->psites[site_num_2]){
        rho_l[site_num_1*N+site_num_2]=rho_l[site_num_2*N+site_num_1]=rho_l[site_num_1*N+site_num_2]/diag_sum[seg_num];
      }
    }
  }
  

  free(H);
  free(e_1);
  free(e);
  free(rho_r);
  //free(rho_l);
  return ;
}

// Multiply a real matrix on a real vector (vr,vi)
void matrix_on_real_vector(float *mat,float *vr,int N){
    float *xr;
    float *xi;
    int a,b;
    xr = (float *)calloc(N * N, sizeof(float));
    //xi = (float *)calloc(N * N, sizeof(float));
    // Multiply
    for (a=0;a<N;a++){
        for (b=0;b<N;b++){
            xr[a]+=mat[a+b*N]*vr[b];
	    //xi[a]+=c[a+b*N]*vi[b];
	}
    }
    // Copy back
    copyvec(xr,vr,N);
    //copyvec(xi,vi,N);
    free(xr);
    //free(xi);
}


/* Multiply with double exciton dipole mu_ef on single states */
void dipole_double_CG2DES(t_non* non, float* dipole, float* cr, float* ci, float* fr, float* fi) {
    int N;
    int i, j, k, index;
    int seg_num;
    N = non->singles * (non->singles + 1) / 2;
    for (i = 0; i < N; i++) fr[i] = 0, fi[i] = 0;

    for (i = 0; i < non->singles; i++) {
       seg_num=non->psites[i];
        for (j = i + 1; j < non->singles; j++) {
            index = Sindex(i, j, non->singles);
            if (seg_num==non->psites[j]){
              fr[index] += dipole[i] * cr[j];
              fi[index] += dipole[i] * ci[j];
              fr[index] += dipole[j] * cr[i];
              fi[index] += dipole[j] * ci[i];
              } else {
              fr[index] =0;
              fi[index] =0;
              fr[index] =0;
              fi[index] =0;

          }
        }
    }
    return;
}

/* Multiply with double exciton dipole mu_ef on double states */
void dipole_double_inverse_CG2DES(t_non* non, float* dipole, float* cr, float* ci, float* fr, float* fi) {
    int N;
    int i, j, k, index;
    int seg_num;
    N = non->singles * (non->singles + 1) / 2;
    for (i = 0; i < non->singles; i++) fr[i] = 0, fi[i] = 0;

    for (i = 0; i < non->singles; i++) {
        seg_num=non->psites[i];
        for (j = i + 1; j < non->singles; j++) {
            index = Sindex(i, j, non->singles);
            if (seg_num==non->psites[j]){
            fr[j] += dipole[i] * cr[index];
            fi[j] += dipole[i] * ci[index];
            fr[i] += dipole[j] * cr[index];
            fi[i] += dipole[j] * ci[index];
            } else {
            fr[index] =0;
            fi[index] =0;
            fr[index] =0;
            fi[index] =0;
            } 
        }
    }
    return;
}


// Diagonalize real nonsymmetric matrix. Output complex eigenvalues, left and right eigenvectors.
void diagonalize_real_nonsym(float* K, float* eig_re, float* eig_im, float* evecL, float* evecR, float* ivecL, float* ivecR, int N) {
    int INFO, lwork;
    float *work, *Kcopy;
    int i, j;
    float *pivot;

    /* Diagonalization*/
    /* Find lwork for diagonalization */
    lwork = -1;
    work = (float *)calloc(1, sizeof(float));
    sgeev_("V", "V", &N, Kcopy, &N, eig_re, eig_im, evecL, &N, evecR, &N, work, &lwork, &INFO);
    lwork = work[0];
    free(work);
    work = (float *)calloc(lwork, sizeof(float));
    Kcopy = (float *)calloc(N * N, sizeof(float));
    /* Copy matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            Kcopy[i * N + j] = K[i * N + j];
        }
    }
    /* Do diagonalization*/
    sgeev_("V", "V", &N, Kcopy, &N, eig_re, eig_im, evecL, &N, evecR, &N, work, &lwork, &INFO);
    if (INFO != 0) {
        printf("Something went wrong trying to diagonalize a matrix...\nExit code %d\n",INFO);
        exit(0);
    }
    free(work);

    /* Copy matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            ivecL[i * N + j] = evecL[i * N + j];
            ivecR[i * N + j] = evecR[i * N + j];
        }
    }

    /* Inverse right eigenvectors*/
    pivot = (float *)calloc(N,sizeof(float));
    sgetrf_(&N, &N, ivecR, &N, pivot, &INFO); //LU factorization
    if (INFO != 0) {
        printf("Something went wrong trying to factorize right eigenvector matrix...\nExit code %d\n",INFO);
        exit(0);
    }    
    lwork = -1; 
    work = (float *)calloc(1, sizeof(float));
    sgetri_(&N, ivecR, &N, pivot, work, &lwork, &INFO); //Find lwork for diagonalization
    lwork = work[0];
    free(work);
    work = (float *)calloc(lwork, sizeof(float));
    sgetri_(&N, ivecR, &N, pivot, work, &lwork, &INFO); //Do inversion
    if (INFO != 0) {
        printf("Something went wrong trying to inverse right eigenvector matrix...\nExit code %d\n",INFO);
        exit(0);
    }
    free(work), free(pivot);

    /* Inverse left eigenvectors*/
    pivot = (float *)calloc(N,sizeof(float));
    sgetrf_(&N, &N, ivecL, &N, pivot, &INFO); //LU factorization
    if (INFO != 0) {
        printf("Something went wrong trying to factorize left eigenvector matrix...\nExit code %d\n",INFO);
        exit(0);
    }    
    lwork = -1; 
    work = (float *)calloc(1, sizeof(float));
    sgetri_(&N, ivecL, &N, pivot, work, &lwork, &INFO); // Find lwork for diagonalization
    lwork = work[0];
    free(work);
    work = (float *)calloc(lwork, sizeof(float));
    sgetri_(&N, ivecL, &N, pivot, work, &lwork, &INFO); //Do inversion
    if (INFO != 0) {
        printf("Something went wrong trying to inverse left eigenvector matrix...\nExit code %d\n",INFO);
        exit(0);
    }

    /* Free space */
    free(Kcopy), free(work), free(pivot);
    return;
}