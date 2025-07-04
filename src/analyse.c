#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
//#include <fftw3.h>
//#include "omp.h"
#include "types.h"
#include "NISE_subs.h"
#include "read_trajectory.h"
#include "propagate.h"
#include "analyse.h"

/* This function calculates various properties of the exciton wavefunctions */
/* Such as the delocalization length in specific spectral regions and */
/* density matrices characterizing the exciton states. The average */
/* Hamiltonian and correlations functions are also evaluated. */
void analyse(t_non *non){
  /* Initialize variables */
  float *average_frequency;
  float *average_coupling;
  float *average_H;
  float avall,flucall;
  float *fluctuation;
  float *Jfluctuation;
  float *mu_eg,*Hamil_i_e,*H,*e;
  float *mu_xyz;
  float participation_ratio;
  float local_participation_ratio;
  float spectral_participation_ratio;
  float local_spectral_participation_ratio;
  float *cEig,*dip2,*cDOS;
  float *rho,*local_rho,*spec_rho,*rho2,*rho4;
  // float *mu_x, *mu_y,*mu_z;
  float *dipeb;
  /* Aid arrays */
  float *vecr,*veci,*vecr_old,*veci_old;

  /* Floats */
  float shift1;
  float x;
  float normal;

  /* File handles */
  FILE *H_traj,*mu_traj,*PDF_traj;
  FILE *C_traj;
  FILE *outone,*log;
  FILE *Cfile;

  /* Integers */
  int nn2,N;
  int itime,N_samples;
  int samples;
  int ti,tj,i,j;
  int t1,fft;
  int elements;
  int Nsam;
  int counts;
  int cl,Ncl;
  int xx;

  /* Time parameters */
  time_t time_now,time_old,time_0;
  /* Initialize time */
  time(&time_now);
  time(&time_0);
  shift1=(non->max1+non->min1)/2;
  printf("Frequency shift %f.\n",shift1);
  non->shifte=shift1;

  /* Allocate memory */
  N=non->singles;
  nn2=non->singles*(non->singles+1)/2;
  Hamil_i_e=(float *)calloc(nn2,sizeof(float));
  average_H=(float *)calloc(nn2,sizeof(float));
  H=(float *)calloc(N*N,sizeof(float));
  e=(float *)calloc(N,sizeof(float));
  average_frequency=(float *)calloc(N,sizeof(float));
  average_coupling=(float *)calloc(N,sizeof(float)); // Coupling strength = sum of all couplings for one molecule
  fluctuation=(float *)calloc(N,sizeof(float));
  Jfluctuation=(float *)calloc(N,sizeof(float));
  cEig=(float *)calloc(N,sizeof(float));
  cDOS=(float *)calloc(N,sizeof(float));
  dip2=(float *)calloc(N,sizeof(float));
  rho=(float *)calloc(N*N,sizeof(float));
  rho2=(float *)calloc(N*N,sizeof(float));
  rho4=(float *)calloc(N*N,sizeof(float));
  local_rho=(float *)calloc(N*N,sizeof(float));
  spec_rho=(float *)calloc(N*N,sizeof(float));
  mu_xyz=(float *)calloc(non->singles*3,sizeof(float));

  /* Open Trajectory files */
  H_traj=fopen(non->energyFName,"rb");
  if (H_traj==NULL){
    printf("Hamiltonian file not found!\n");
    exit(1);
  }

  mu_traj=fopen(non->dipoleFName,"rb");
  if (mu_traj==NULL){
    printf("Dipole file %s not found!\n",non->dipoleFName);
    exit(1);
  }

  if (string_in_array(non->technique,(char*[]){"AnalyseFull","AnalyzeFull"},2)){
    outone=fopen("ExcitonStats.dat","w");
    if (outone==NULL){
      printf("Problem encountered opening AnalyseFull.dat for writing.\n");
      printf("Disk full or write protected?\n");
      exit(1);
    }
    fprintf(outone,"# Exciton Energy & mu_x & my_y & mu_z & mu^2\n");
    fclose(outone);
  }

  /* Open file with cluster information if appicable */
  if (non->cluster!=-1){
    Cfile=fopen("Cluster.bin","rb");
    if (Cfile==NULL){
      printf("Cluster option was activated but no Cluster.bin file provided.\n");
      printf("Please, provide cluster file or remove Cluster keyword from\n");
      printf("input file.\n");
      exit(0);
    }
    Ncl=0; // Counter for snapshots calculated
  }


  itime=0;
  // Do calculation
  N_samples=(non->length-1)/non->sample+1;
  if (N_samples>0) {
    printf("Making %d samples!\n",N_samples);
  } else {
    printf("Insufficient data to calculate spectrum.\n");
    printf("Please, lower max times or provide longer\n");
    printf("trajectory.\n");
    exit(1);
  }

  if (non->end==0) non->end=N_samples;
  Nsam=non->end-non->begin;
  if (non->end>N_samples){
    printf(RED "Endpoint larger than number of samples was specified.\n" RESET);
    printf(RED "Endpoint was %d but cannot be larger than %d.\n" RESET,non->end,N_samples);
    exit(0);
  }

  log=fopen("NISE.log","a");
  fprintf(log,"Begin sample: %d, End sample: %d.\n",non->begin,non->end);
  fclose(log);

  /* Initialize the values of various parameters */
  participation_ratio=0;
  local_participation_ratio=0;
  spectral_participation_ratio=0;
  local_spectral_participation_ratio=0;
  avall=0;
  flucall=0;
  counts=0;
  Ncl=0;

  /* Read coupling, this is done if the coupling and transition-dipoles are *
   * time-independent and only one snapshot is stored */
  read_coupling(non,C_traj,mu_traj,Hamil_i_e,mu_xyz);

  // Loop over samples first time
  for (samples=non->begin;samples<non->end;samples++){

    ti=samples*non->sample;
    if (non->cluster!=-1){
      if (read_cluster(non,ti,&cl,Cfile)!=1){
        printf("Cluster trajectory file to short, could not fill buffer!!!\n");
        printf("ITIME %d\n",ti);
        exit(1);
      }
      // Configuration belong to cluster
      if (non->cluster==cl){
         Ncl++;
      }
    }
    if (non->cluster==-1 || non->cluster==cl){

    /* Read Hamiltonian */
      read_Hamiltonian(non,Hamil_i_e,H_traj,ti);

      build_diag_H(Hamil_i_e,H,e,N);

      participation_ratio+=calc_participation_ratio(N,H);
      local_participation_ratio+=calc_local_participation_ratio(N,H,non->min1,non->max1,e,non->shifte);
      spectral_participation_ratio+=calc_spectral_participation_ratio(N,H);
      local_spectral_participation_ratio+=calc_local_spectral_participation_ratio(N,H,non->min1,non->max1,e,non->shifte);

      /* Call subroutines for finding varius density matrices */
      find_dipole_mag(non,dip2,samples,mu_traj,H,mu_xyz,e);
      calc_densitymatrix(non,rho,rho2,rho4,local_rho,spec_rho,H,e,dip2);
      counts=find_cEig(cEig,cDOS,dip2,H,e,N,non->min1,non->max1,counts,non->shifte);
    /* Find Averages */
      for (i=0;i<non->singles;i++){
        average_frequency[i]+=Hamil_i_e[Sindex(i,i,N)];
        avall+=Hamil_i_e[Sindex(i,i,N)];
        for (j=0;j<non->singles;j++){
          if (j>=i){
            average_H[Sindex(i,j,N)]+=Hamil_i_e[Sindex(i,j,N)];
          }
          if (j!=i){
            average_coupling[i]+=Hamil_i_e[Sindex(i,j,N)];
          }
        }
      }     
    }
  }
  
  /* Adjust number of calculated samples if clusters were used */
  if (Ncl>0) Nsam=Ncl;
  /* Normalize average_frequencies */
  for (i=0;i<non->singles;i++){
    average_frequency[i]/=Nsam;
    average_coupling[i]/=Nsam;
    for(j=i;j<non->singles;j++){
      average_H[Sindex(i,j,non->singles)]/=Nsam;
    }
  }
  avall/=(Nsam*non->singles);   

  // Loop over samples second time
  for (samples=non->begin;samples<non->end;samples++){
    ti=samples*non->sample;
    if (non->cluster!=-1){
      if (read_cluster(non,ti,&cl,Cfile)!=1){
        printf("Cluster trajectory file to short, could not fill buffer!!!\n");
        printf("ITIME %d\n",ti);
        exit(1);
      }      
    }

    // Include frame if configuration belong to cluster or no clusters are used
    if (non->cluster==-1 || non->cluster==cl){

      /* Read Hamiltonian */
      read_Hamiltonian(non,Hamil_i_e,H_traj,ti);

      // Find standard deviation for frequencies
      for (i=0;i<non->singles;i++){
        x=(Hamil_i_e[Sindex(i,i,N)]-average_frequency[i]);
        fluctuation[i]+=x*x;
        x=(Hamil_i_e[Sindex(i,i,N)]-avall);
        flucall+=x*x;
        x=0;
        for (j=0;j<non->singles;j++){
          if (i!=j) x+=Hamil_i_e[Sindex(i,j,N)];
        }
        x=x-average_coupling[i];
        Jfluctuation[i]+=x*x;
      }
    }
  }
  /* Normalize fluctuations and take square root */
  for (i=0;i<non->singles;i++){
    fluctuation[i]/=Nsam;
    fluctuation[i]=sqrt(fluctuation[i]);
    Jfluctuation[i]/=Nsam;
    Jfluctuation[i]=sqrt(Jfluctuation[i]);
  }
  flucall/=(Nsam*non->singles);   
  flucall=sqrt(flucall);

  /* Write Average Hamiltonian in GROASC format */
  outone=fopen("Av_Hamiltonian.dat","w");
    if (outone==NULL){
    printf("Problem encountered opening Analyse.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }
  fprintf(outone,"0 ");
  for (i=0;i<non->singles;i++){
    for (j=i;j<non->singles;j++){
      if (i==j) average_H[Sindex(i,j,non->singles)]+=non->shifte;
      fprintf(outone,"%f ",average_H[Sindex(i,j,non->singles)]);
    }
  }
  fprintf(outone,"\n");
  fclose(outone);

  outone=fopen("Analyse.dat","w");
  if (outone==NULL){
    printf("Problem encountered opening Analyse.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }

  if (counts==0){
    printf(YELLOW "\nNo exciton states were found in the defined frequency range!\n");
    printf("Range dependent properties will be zero in the output.\n\n" RESET);
    counts=1;
  }

  /* Normalize delocalization parameters */
  participation_ratio/=(non->singles*Nsam);
  local_participation_ratio/=counts;
  spectral_participation_ratio/=(non->singles*Nsam);
  local_spectral_participation_ratio/=counts;
  printf("===================================\n");
  printf("Result of Hamiltonian analysis:\n");
  printf("Delocalization size according to\n");
  printf("Thouless, Phys. Rep. 13:93 (1974)\n");
  printf("for full Hamiltonian:\n");
  printf("R=%f\n",participation_ratio);
  printf("for selected frequency range\n");
  printf("(%f to %f) cm-1:\n",non->min1,non->max1);
  printf("R=%f\n",local_participation_ratio);
  printf("Delocalization size according to\n");
  printf("the Manhattan Exciton Size\n");
  printf("for full Hamiltonian:\n");
  printf("S=%f\n",spectral_participation_ratio);
  printf("for selected frequency range\n");
  printf("(%f to %f) cm-1:\n",non->min1,non->max1);
  printf("S=%f\n",local_spectral_participation_ratio);
  printf("Average site frequency %f cm-1.\n",avall+non->shifte);
  printf("Overall standard deviation of site\n");
  printf("frequencies from overall average:\n");
  printf("%f cm-1.\n",flucall);
  printf("===================================\n");
  printf("\n");

  /* Print output to file */
  fprintf(outone,"# Using frequency range %f to %f cm-1\n",non->min1,non->max1);
  fprintf(outone,"# Site AvFreq.  SDFreq     AvJ      SDJ      cEig         cDOS      2lambda\n");
  for (i=0;i<non->singles;i++){
    fprintf(outone,"%d %f %f %f %F %e %e %f\n",i,average_frequency[i]+non->shifte,
      fluctuation[i],average_coupling[i],Jfluctuation[i],cEig[i]/counts,
      cDOS[i]/counts,fluctuation[i]*fluctuation[i]/k_B/non->temperature);
  }
  fclose(outone);

  outone=fopen("LocalDensityMatrix.dat","w");
  if (outone==NULL){
    printf("Problem encountered opening LocalDensityMatrix.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }

  normal=0; /* Normalize Density Matrix */
  for (i=0;i<non->singles;i++) normal+=local_rho[i+non->singles*i];
  if (normal==0) normal=1; // No exciton states in range
  for (i=0;i<non->singles;i++){
    for (j=0;j<non->singles;j++){
      fprintf(outone,"%e ",local_rho[i+non->singles*j]/normal);
    }
    fprintf(outone,"\n");
  }
  fclose(outone);

  outone=fopen("SpectralDensityMatrix.dat","w");
  if (outone==NULL){
    printf("Problem encountered opening SpectralDensityMatrix.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }
  normal=0; /* Normalize Density Matrix */
  for (i=0;i<non->singles;i++) normal+=spec_rho[i+non->singles*i];
  if (normal==0) normal=1; // No exciton states in range
  for (i=0;i<non->singles;i++){
    for (j=0;j<non->singles;j++){
      fprintf(outone,"%e ",spec_rho[i+non->singles*j]/normal);
    }
    fprintf(outone,"\n");
  }
  fclose(outone);

  outone=fopen("AbsoluteDensityMatrix.dat","w");
  if (outone==NULL){
    printf("Problem encountered opening AbsoluteDensityMatrix.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }

  normal=0; /* Normalize Density Matrix */
  for (i=0;i<non->singles;i++) normal+=rho2[i+non->singles*i];
  if (normal==0) normal=1; // No exciton states in range
  for (i=0;i<non->singles;i++){
    for (j=0;j<non->singles;j++){
      rho2[i+non->singles*j]=rho2[i+non->singles*j]/normal;
      fprintf(outone,"%e ",rho2[i+non->singles*j]);
    }
    fprintf(outone,"\n");
  }
  fclose(outone);
  /* Do clustering with Absolute Density Matrix */
  cluster(non,rho2);


  outone=fopen("ParticipationRatioMatrix.dat","w");
  if (outone==NULL){
    printf("Problem encountered opening ParticipationRatioMatrix.dat for writing.\n");
    printf("Disk full or write protected?\n");
    exit(1);
  }
  
  normal=(non->singles*Nsam); 
  for (i=0;i<non->singles;i++){
    for (j=0;j<non->singles;j++){
      fprintf(outone,"%e ",rho4[i+non->singles*j]/normal);
    }
    fprintf(outone,"\n");
  }
  fclose(outone);

  free(average_frequency);
  free(fluctuation);
  free(average_coupling);
  free(Jfluctuation);
  free(Hamil_i_e);
  free(mu_xyz);
  free(average_H);
  free(cEig);
  free(cDOS);
  free(H);
  free(e);
  free(dip2);
  free(rho);
  free(rho2);
  free(rho4);
  free(local_rho);
  free(spec_rho);
  
  fclose(mu_traj),fclose(H_traj);
  if (non->cluster!=-1){
    fclose(Cfile);
  } 

  // The calculation is finished
  log=fopen("NISE.log","a");
  fprintf(log,"Finished Calculating Response!\n");
  fprintf(log,"Writing to file!\n");  
  fclose(log);

  printf("----------------------------------------------\n");
  printf(" Analyse calculation succesfully completed\n");
  printf("----------------------------------------------\n\n");

  return;
}	

/* Calculate Thouless Partition  Ratio */
float calc_participation_ratio(int N,float *H){
  int i,j,a,b;
  float inter,parti;

  parti=0;
  for (i=0;i<N;i++){
    inter=0;
    /* Loop over sites */
    for (j=0;j<N;j++){
      inter+=H[i+N*j]*H[i+N*j]*H[i+N*j]*H[i+N*j];
    }
    parti+=1.0/inter;
  }

  return parti;
}

/* Find Participation Ratio for states in given energy range */
float calc_local_participation_ratio(int N,float *H,float min,float max,float *e,float shift){
  int i,j,a,b;
  float inter,parti;

  parti=0;
  for (i=0;i<N;i++){
    inter=0;
    /* Loop over sites */
    if (e[i]>min-shift && e[i]<max-shift){
      for (j=0;j<N;j++){
          inter+=H[i+N*j]*H[i+N*j]*H[i+N*j]*H[i+N*j];
      }
      parti+=1.0/inter;
    }
  }
 
  return parti;
}

/* Calculate Spectral Partition  Ratio */
float calc_spectral_participation_ratio(int N,float *H){
  int i,j,a,b;
  float inter,parti;

  parti=0;
  for (i=0;i<N;i++){
    inter=0;
    /* Loop over sites */
    for (j=0;j<N;j++){
      inter+=fabs(H[i+N*j]);
    }
    parti+=inter*inter;
  }

  return parti;
}

/* Find Spectral Participation Ratio for states in given energy range */
float calc_local_spectral_participation_ratio(int N,float *H,float min,float max,float *e,float shift){
  int i,j,a,b;
  float inter,parti;

  parti=0;
  for (i=0;i<N;i++){
    inter=0;
    /* Loop over sites */
    if (e[i]>min-shift && e[i]<max-shift){
      for (j=0;j<N;j++){
          inter+=fabs(H[i+N*j]);
      }
      parti+=inter*inter;
    }
  }
 
  return parti;
}

/* The subroutine determine the spectrally weighted and normal weigth */
/* of the sites to the eigenstates in the given spectral range. */
int find_cEig(float *cEig,float *cDOS,float *dip2,float *H,float *e,int N,float min,float max,int counts,float shift){
  int i,j;
  
  /* Loop over eigenstates */
  for (i=0;i<N;i++){
    /* Only for eigenstates in given range */
    if (e[i]>min-shift && e[i]<max-shift){
      counts++;
      /* Loop over sites */
      for (j=0;j<N;j++){
	      cEig[j]+=dip2[i]*H[i+N*j]*H[i+N*j];
	      cDOS[j]+=H[i+N*j]*H[i+N*j];
      }
    }
  }
  return counts;
}  

/* Find dipole magnitude for the eigenstates (mu squared) */
void find_dipole_mag(t_non *non,float *dip2,int step,FILE *mu_traj,float *H,float *mu_xyz,float *e){
  float *dip, *dipeb;
  int i,j,x,N;
  FILE *outone;

  N=non->singles;
  dip=(float *)calloc(N,sizeof(float));
  dipeb=(float *)calloc(3*N,sizeof(float));

  /* Clear the dipole square vector */
  /* as values from previous frame are there */
  clearvec(dip2,N);

  for (x=0;x<3;x++){
      /* Read mu(tj) */
      read_dipole(non,mu_traj,dip,mu_xyz,x,step);

    
    /* Transform to eigen basis */
    for (i=0;i<N;i++){
      dipeb[i+x*N]=0;
      for (j=0;j<N;j++){
          dipeb[i+x*N]+=H[i+j*N]*dip[j]; // i is eigen state, j site
      }
    }
    /* Add to exciton dipole square */
    for (i=0;i<N;i++){
      dip2[i]+=dipeb[i+x*N]*dipeb[i+x*N];
    }
  }

  /* Write Exciton Energy and dipole info to file */
  if (string_in_array(non->technique,(char*[]){"AnalyseFull","AnalyzeFull"},2)){
    outone=fopen("ExcitonStats.dat","a");
    if (outone==NULL){
      printf("Problem encountered opening AnalyseFull.dat for writing.\n");
      printf("Disk full or write protected?\n");
      exit(1);
    }
    for (i=0;i<non->singles;i++){
      fprintf(outone,"%f %f %f %f %f\n",e[i]+non->shifte,dipeb[i],dipeb[i+N],dipeb[i+2*N],dip2[i]);
    }  
    //fprintf(outone,"\n");
    fclose(outone);
  }


  free(dip);
  free(dipeb);
  return;
}

/* Calculate full density matrix and density matrix is specific frequency
 * range in the site basis */
void calc_densitymatrix(t_non *non,float *rho,float *rho2,float *rho4,float *local_rho,float *spec_rho,float *H,float* e,float *dip2){
  int i,j,k,N;
  float shift,d;
  float max,min;

  N=non->singles;
  shift=non->shifte;
  min=non->min1;
  max=non->max1;
  /* Loop over eigenstates */
  for (i=0;i<N;i++){
    /* Loop over sites */
    for (j=0;j<N;j++){
      for (k=0;k<N;k++){
        /* Find density matrix element */
        d=H[i+j*N]*H[i+k*N];
        rho[j+k*N]+=d;
	      rho4[j+k*N]+=d*d;
	      /* Only include the eigenstate if it is within the *
	       * given spectral region */
        if (e[i]>min-shift && e[i]<max-shift){
	        rho2[j+k*N]+=fabs(d);
          local_rho[j+k*N]+=d;
          spec_rho[j+k*N]+=d*dip2[i];
        }
      }
    }
  }
  return;        
}

/* Define Segments depending on a clustering of the absolute density matrix */
void cluster(t_non *non,float *rho){
  int *segments;
  int i,j,k;
  int N;
  int n_seg;
  float norm;
  FILE *handle;
  N=non->singles;

  segments=(int *)calloc(N,sizeof(int));

  /* Assign all sites their own segment as a start */
  for (i=0;i<N;i++){
    segments[i]=i;
  }

  /* Normalize absolute value density matrix */
  norm=rho[0];
  for (i=0;i<N;i++){
    for (j=0;j<N;j++){
	    rho[i+j*N]=rho[i+j*N]/norm;
    }
  }

  /* Run over all possible segments */
  for (i=0;i<N;i++){
    /* Run over all later sites */
    for (j=i+1;j<N;j++){
      /* Test if two sites belong to the same segment */
	    if (rho[i+j*N]>non->thres){
        /* If so merge segments to the one with lowest index */
        if (segments[j]<segments[i]){
		      for (k=0;k<N;k++){
	          if (segments[k]==segments[i] && k!=i){
              segments[k]=segments[j];
		        }
		      }
		      segments[i]=segments[j];
	      } else {
          for (k=0;k<N;k++){
            if (segments[k]==segments[j] && k!=j){
              segments[k]=segments[i];
            }
          }
		      segments[j]=segments[i];
	      }
	    }
    }
  }

  /* Reduce segments numbers */
  n_seg=0;
  for (i=0;i<N;i++){
    if (segments[i]>n_seg){
	    /* Update segment numbers */
	    n_seg=n_seg+1;
	    for (j=i+1;j<N;j++){
        if (segments[j]==segments[i]){
		      segments[j]=n_seg;
        }
	    }
	    segments[i]=n_seg;
    }
  }

  printf("Using segmentation threshold (Threshold) %f.\n",non->thres);
  printf("on the Absolute Value Density Matrix calculated in the selected\n");
  printf("frequency range: (%f to %f) cm-1:\n",non->min1,non->max1);
  printf("Identified %d segments\n\n",n_seg+1);  

  handle=fopen("Segments.dat","w");
  fprintf(handle,"%d\n",N);
  /* Run over all sites */
  for (i=0;i<N;i++){
    fprintf(handle,"%d ",segments[i]);	   
  }
  fclose(handle);

  free(segments);
  return;
}

