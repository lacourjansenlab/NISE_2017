#ifndef _1DFFT_
#define _1DFFT_
void Lineshape_FFT(t_non *non);
void ONE_DFFT(t_non *non);
void do_1DFFT(t_non *non,char fname[],float *re_S_1,float *im_S_1,int samples);
#endif // _1DFFT_
