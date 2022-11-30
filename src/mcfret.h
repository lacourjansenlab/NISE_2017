#ifndef _MCFRET_
#define _MCFRET_
void mcfret(t_non *non);
void mcfret_autodetect(t_non *non, float treshold);
void mcfret_response_function(t_non *non, float *re_S_1, float *im_S_1,int emission);
void mcfret_coupling(t_non *non);
void mcfret_rate(t_non *non);
void mcfret_validate(t_non *non);
void mcfret_analyse(t_non *non);

#endif /* _MCFRET_ */
