#ifndef _calc_CG_2DES_
#define _calc_CG_2DES_

void call_final_CG_2DES(
  t_non *non,float *P_DA,int pro_dim,float *re_doorway,float *im_doorway,
  float *re_window_SE, float *im_window_SE,float *re_window_GB, float *im_window_GB,
  float *re_window_EA, float *im_window_EA,float *re_2DES , float *im_2DES);
void calc_CG_2DES(t_non *non);
int CG_index(t_non *non,int seg_num,int alpha,int beta,int t1);
void write_response_to_file(t_non *non,char fname[],float *im,float *re,int tmax);
void read_doorway_window_from_file(t_non *non,char fname[],float *im,float *re,int tmax);
void CG_2DES_doorway(t_non *non,float *re_doorway,float *im_doorway);
void CG_2DES_P_DA(t_non *non,float *P_DA, int N);
void CG_2DES_window_GB(t_non *non,float *re_window_GB,float *im_window_GB);
void CG_2DES_window_SE(t_non *non,float *re_window_SE,float *im_window_SE);
void CG_2DES_window_EA(t_non *non,float *re_window_EA,float *im_window_EA);
void call_final_CG_2DES(t_non *non,float *P_DA,int pro_dim,
    float *re_doorway,float *im_doorway,float *re_window_SE, float *im_window_SE,
    float *re_window_GB, float *im_window_GB,float *re_window_EA, float *im_window_EA,
    float *re_2DES , float *im_2DES);
void CG_full_2DES_segments(t_non *non,float *re_doorway,float *im_doorway,
    float *re_window_SE,float *im_window_SE,float *re_window_GB, float *im_window_GB,
    float *re_window_EA,float *im_window_EA,float *P_DA,int N, char *waittime,int wfile);
#endif /* _calc_CG_2DES_ */     
