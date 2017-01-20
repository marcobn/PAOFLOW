/**********************************************************************
 * cints.c  C implementation of simple math functions in pyutil.
 *
 * The equations herein are based upon
 * 'Gaussian Expansion Methods for Molecular Orbitals.' H. Taketa,
 * S. Huzinaga, and K. O-ohata. H. Phys. Soc. Japan, 21, 2313, 1966.
 * [THO paper].
 *
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
 **********************************************************************/


#include "Python.h"
#include "cints.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Not required for MSVC since the code is included below
#if defined(_WIN32) && !defined(_MSC_VER)
double lgamma(double x);
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ITMAX 100
#define EPS 3.0e-7
#define FPMIN 1.0e-30
#define SMALL 0.00000001

// lgamma not included in ANSI standard and so not available in MSVC
#if defined(_MSC_VER)
double lgamma(double z) {
    double c[7];
    double x,y ,tmp, ser, v;
    int i;

    if (z<=0) return 0;

    c[0]=2.5066282746310005;
    c[1]=76.18009172947146;
    c[2]=-86.50532032941677;
    c[3]=24.01409824083091;
    c[4]=-1.231739572450155;
    c[5]=0.1208650973866179e-2;
    c[6]=-0.5395239384953e-5;
   
    x   = z;
    y   = x;
    tmp = x+5.5;
    tmp = (x+0.5)*log(tmp)-tmp;
    ser = 1.000000000190015;
    for (i=1; i<7; i++) {
        y   += 1.0;
        ser += c[i]/y;
        }
    v = tmp+log(c[0]*ser/x);
    return v;
    }
#endif

static double fB(int i, int l1, int l2, double px, double ax, double bx, 
		 int r, double g){
  return binomial_prefactor(i,l1,l2,px-ax,px-bx)*Bfunc(i,r,g);
}

static double Bfunc(int i, int r, double g){
  return fact_ratio2(i,r)*pow(4*g,r-i);
}

static double contr_coulomb(int lena, double *aexps, double *acoefs,
			    double *anorms, double xa, double ya, double za,
			    int la, int ma, int na, 
			    int lenb, double *bexps, double *bcoefs,
			    double *bnorms, double xb, double yb, double zb,
			    int lb, int mb, int nb, 
			    int lenc, double *cexps, double *ccoefs,
			    double *cnorms, double xc, double yc, double zc,
			    int lc, int mc, int nc, 
			    int lend, double *dexps, double *dcoefs,
			    double *dnorms, double xd, double yd, double zd,
			    int ld, int md, int nd){

  int i,j,k,l;
  double Jij = 0.,incr=0.;

  for (i=0; i<lena; i++)
    for (j=0; j<lenb; j++)
      for (k=0; k<lenc; k++)
	for (l=0; l<lend; l++){
	  incr = coulomb_repulsion(xa,ya,za,anorms[i],la,ma,na,aexps[i],
			      xb,yb,zb,bnorms[j],lb,mb,nb,bexps[j],
			      xc,yc,zc,cnorms[k],lc,mc,nc,cexps[k],
			      xd,yd,zd,dnorms[l],ld,md,nd,dexps[l]);
	  
	  Jij += acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*incr;
	}
  return Jij;
}

/*Luis*/
static double contr_coulomb_v3(int lena, double *aexps, double *acoefs,
			    double *anorms, double xa, double ya, double za,
			    double *la, double *ma, double *na, 
			    int lenb, double *bexps, double *bcoefs,
			    double *bnorms, double xb, double yb, double zb,
			    double *lb, double *mb, double *nb, 
			    int lenc, double *cexps, double *ccoefs,
			    double *cnorms, double xc, double yc, double zc,
			    double *lc, double *mc, double *nc, 
			    int lend, double *dexps, double *dcoefs,
			    double *dnorms, double xd, double yd, double zd,
			    double *ld, double *md, double *nd){

  int i,j,k,l;
  double Jij = 0.,incr=0.;
  /*printf("point 3\n");*/
  for (i=0; i<lena; i++)
    for (j=0; j<lenb; j++)
      for (k=0; k<lenc; k++)
	for (l=0; l<lend; l++){
          //printf("contr_coulomb_v3 la=%f ma=%f na=%f\n",la[i],ma[i],na[i]);
	  incr = coulomb_repulsion(
                              xa,ya,za,anorms[i],(int) la[i],(int) ma[i],(int) na[i],aexps[i],
			      xb,yb,zb,bnorms[j],(int) lb[j],(int) mb[j],(int) nb[j],bexps[j],
			      xc,yc,zc,cnorms[k],(int) lc[k],(int) mc[k],(int) nc[k],cexps[k],
			      xd,yd,zd,dnorms[l],(int) ld[l],(int) md[l],(int) nd[l],dexps[l]);
	  
	  Jij += acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*incr;
	}
  return Jij;
}

static double coulomb_repulsion(double xa, double ya, double za, double norma,
				int la, int ma, int na, double alphaa,
				double xb, double yb, double zb, double normb,
				int lb, int mb, int nb, double alphab,
				double xc, double yc, double zc, double normc,
				int lc, int mc, int nc, double alphac,
				double xd, double yd, double zd, double normd,
				int ld, int md, int nd, double alphad){

  double rab2, rcd2,rpq2,xp,yp,zp,xq,yq,zq,gamma1,gamma2,delta,sum;
  double *Bx, *By, *Bz;
  int I,J,K;
  rab2 = dist2(xa,ya,za,xb,yb,zb);
  rcd2 = dist2(xc,yc,zc,xd,yd,zd);
  xp = product_center_1D(alphaa,xa,alphab,xb);
  yp = product_center_1D(alphaa,ya,alphab,yb);
  zp = product_center_1D(alphaa,za,alphab,zb);
  xq = product_center_1D(alphac,xc,alphad,xd);
  yq = product_center_1D(alphac,yc,alphad,yd);
  zq = product_center_1D(alphac,zc,alphad,zd);
  rpq2 = dist2(xp,yp,zp,xq,yq,zq);
  gamma1 = alphaa+alphab;
  gamma2 = alphac+alphad;
  delta = (1./gamma1+1./gamma2)/4.;

  Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta);
  By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta);
  Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta);

  sum = 0.;
  for (I=0; I<la+lb+lc+ld+1;I++)
    for (J=0; J<ma+mb+mc+md+1;J++)
      for (K=0; K<na+nb+nc+nd+1;K++)
	sum += Bx[I]*By[J]*Bz[K]*Fgamma(I+J+K,0.25*rpq2/delta);

  free(Bx);
  free(By);
  free(Bz);  
  
  return 2.*pow(M_PI,2.5)/(gamma1*gamma2*sqrt(gamma1+gamma2))
    *exp(-alphaa*alphab*rab2/gamma1) 
    *exp(-alphac*alphad*rcd2/gamma2)*sum*norma*normb*normc*normd;
}

static double *B_array(int l1, int l2, int l3, int l4, double p, double a,
		double b, double q, double c, double d,
		double g1, double g2, double delta){
  int Imax,i1,i2,r1,r2,u,I,i;
  double *B;
  Imax = l1+l2+l3+l4+1;
  B = (double *)malloc(Imax*sizeof(double));
  for (i=0; i<Imax; i++) B[i] = 0.;

  for (i1=0; i1<l1+l2+1; i1++)
    for (i2=0; i2<l3+l4+1; i2++)
      for (r1=0; r1<i1/2+1; r1++)
	for (r2=0; r2<i2/2+1; r2++)
	  for (u=0; u<(i1+i2)/2-r1-r2+1; u++){
	    I = i1+i2-2*(r1+r2)-u;
	    B[I] = B[I] + B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,
				 p,a,b,q,c,d,g1,g2,delta);
	  }

  return B;
}

static double B_term(int i1, int i2, int r1, int r2, int u, int l1, int l2,
	      int l3, int l4, double Px, double Ax, double Bx,
	      double Qx, double Cx, double Dx, double gamma1,
	      double gamma2, double delta){
  /* THO eq. 2.22 */
  return fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1)
    *pow(-1,i2)*fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2)
    *pow(-1,u)*fact_ratio2(i1+i2-2*(r1+r2),u)
    *pow(Qx-Px,i1+i2-2*(r1+r2)-2*u)
    /pow(delta,i1+i2-2*(r1+r2)-u);
}


static double kinetic(double alpha1, int l1, int m1, int n1,
	       double xa, double ya, double za,
	       double alpha2, int l2, int m2, int n2,
	       double xb, double yb, double zb){

  double term0,term1,term2;
  term0 = alpha2*(2*(l2+m2+n2)+3)*
    overlap(alpha1,l1,m1,n1,xa,ya,za,
		   alpha2,l2,m2,n2,xb,yb,zb);
  term1 = -2*pow(alpha2,2)*
    (overlap(alpha1,l1,m1,n1,xa,ya,za,
		    alpha2,l2+2,m2,n2,xb,yb,zb)
     + overlap(alpha1,l1,m1,n1,xa,ya,za,
		      alpha2,l2,m2+2,n2,xb,yb,zb)
     + overlap(alpha1,l1,m1,n1,xa,ya,za,
		      alpha2,l2,m2,n2+2,xb,yb,zb));
  term2 = -0.5*(l2*(l2-1)*overlap(alpha1,l1,m1,n1,xa,ya,za,
					 alpha2,l2-2,m2,n2,xb,yb,zb) +
		m2*(m2-1)*overlap(alpha1,l1,m1,n1,xa,ya,za,
					 alpha2,l2,m2-2,n2,xb,yb,zb) +
		n2*(n2-1)*overlap(alpha1,l1,m1,n1,xa,ya,za,
					 alpha2,l2,m2,n2-2,xb,yb,zb));
  return term0+term1+term2;
}

static double overlap(double alpha1, int l1, int m1, int n1,
		      double xa, double ya, double za,
		      double alpha2, int l2, int m2, int n2,
		      double xb, double yb, double zb){
  /*Taken from THO eq. 2.12*/
  double rab2,gamma,xp,yp,zp,pre,wx,wy,wz;

  rab2 = dist2(xa,ya,za,xb,yb,zb);
  gamma = alpha1+alpha2;
  xp = product_center_1D(alpha1,xa,alpha2,xb);
  yp = product_center_1D(alpha1,ya,alpha2,yb);
  zp = product_center_1D(alpha1,za,alpha2,zb);

  pre = pow(M_PI/gamma,1.5)*exp(-alpha1*alpha2*rab2/gamma);

  wx = overlap_1D(l1,l2,xp-xa,xp-xb,gamma);
  wy = overlap_1D(m1,m2,yp-ya,yp-yb,gamma);
  wz = overlap_1D(n1,n2,zp-za,zp-zb,gamma);
  return pre*wx*wy*wz;
}

static double overlap_1D(int l1, int l2, double PAx,
			 double PBx, double gamma){
  /*Taken from THO eq. 2.12*/
  int i;
  double sum;
  sum = 0.;
  for (i=0; i<(1+floor(0.5*(l1+l2))); i++)
    sum += binomial_prefactor(2*i,l1,l2,PAx,PBx)* 
      fact2(2*i-1)/pow(2*gamma,i);
  return sum;
}
    
static double nuclear_attraction(double x1, double y1, double z1, double norm1,
				 int l1, int m1, int n1, double alpha1,
				 double x2, double y2, double z2, double norm2,
				 int l2, int m2, int n2, double alpha2,
				 double x3, double y3, double z3){
  int I,J,K;
  double gamma,xp,yp,zp,sum,rab2,rcp2;
  double *Ax,*Ay,*Az;

  gamma = alpha1+alpha2;

  xp = product_center_1D(alpha1,x1,alpha2,x2);
  yp = product_center_1D(alpha1,y1,alpha2,y2);
  zp = product_center_1D(alpha1,z1,alpha2,z2);

  rab2 = dist2(x1,y1,z1,x2,y2,z2);
  rcp2 = dist2(x3,y3,z3,xp,yp,zp);

  Ax = A_array(l1,l2,xp-x1,xp-x2,xp-x3,gamma);
  Ay = A_array(m1,m2,yp-y1,yp-y2,yp-y3,gamma);
  Az = A_array(n1,n2,zp-z1,zp-z2,zp-z3,gamma);

  sum = 0.;
  for (I=0; I<l1+l2+1; I++)
    for (J=0; J<m1+m2+1; J++)
      for (K=0; K<n1+n2+1; K++)
	sum += Ax[I]*Ay[J]*Az[K]*Fgamma(I+J+K,rcp2*gamma);

  free(Ax);
  free(Ay);
  free(Az);
  return -norm1*norm2*
    2*M_PI/gamma*exp(-alpha1*alpha2*rab2/gamma)*sum;
}
    
static double A_term(int i, int r, int u, int l1, int l2,
		     double PAx, double PBx, double CPx, double gamma){
  /* THO eq. 2.18 */
  return pow(-1,i)*binomial_prefactor(i,l1,l2,PAx,PBx)*
    pow(-1,u)*fact(i)*pow(CPx,i-2*r-2*u)*
    pow(0.25/gamma,r+u)/fact(r)/fact(u)/fact(i-2*r-2*u);
}

static double *A_array(int l1, int l2, double PA, double PB,
		double CP, double g){
  /* THO eq. 2.18 and 3.1 */
  int Imax,i,r,u,I;
  double *A;

  Imax = l1+l2+1;
  A = (double *)malloc(Imax*sizeof(double));
  for (i=0; i<Imax; i++) A[i] = 0.;
  for (i=0; i<Imax; i++)
    for (r=0; r<floor(i/2)+1;r++)
      for (u=0; u<floor((i-2*r)/2.)+1; u++){
	I = i-2*r-u;
	A[I] += A_term(i,r,u,l1,l2,PA,PB,CP,g);
      }
  return A;
}


static int fact(int n){
  if (n <= 1) return 1;
  return n*fact(n-1);
}

static int fact2(int n){ /* double factorial function = 1*3*5*...*n */
  if (n <= 1) return 1;
  return n*fact2(n-2);
}

static double dist2(double x1, double y1, double z1,
		    double x2, double y2, double z2){
  return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
}
static double dist(double x1, double y1, double z1,
		   double x2, double y2, double z2){
  return sqrt(dist2(x1,y1,z1,x2,y2,z2));
}

static double binomial_prefactor(int s, int ia, int ib, double xpa, double xpb){
  int t;
  double sum=0.;
  for (t=0; t<s+1; t++)
    if ((s-ia <= t) && (t <= ib)) 
      sum += binomial(ia,s-t)*binomial(ib,t)*pow(xpa,ia-s+t)*pow(xpb,ib-t);
  return sum;
} 

static int binomial(int a, int b){return fact(a)/(fact(b)*fact(a-b));}

static double Fgamma(double m, double x){
  double val;
  if (fabs(x) < SMALL) x = SMALL;
  val = gamm_inc(m+0.5,x);
  /* if (val < SMALL) return 0.; */ /* Gives a bug for D orbitals. */
  return 0.5*pow(x,-m-0.5)*val; 
}

static double gamm_inc(double a, double x){ /* Taken from NR routine gammap */
  double gamser,gammcf,gln;
  
  assert (x >= 0.);
  assert (a > 0.);
  if (x < (a+1.0)) {
    gser(&gamser,a,x,&gln);
    return exp(gln)*gamser;
  } else {
    gcf(&gammcf,a,x,&gln);
    return exp(gln)*(1.0-gammcf);
  }
}
 
static void gser(double *gamser, double a, double x, double *gln){
  int n;
  double sum,del,ap;

  *gln=lgamma(a);
  if (x <= 0.0) {
    assert(x>=0.);
    *gamser=0.0;
    return;
  } else {
    ap=a;
    del=sum=1.0/a;
    for (n=1;n<=ITMAX;n++) {
      ++ap;
      del *= x/ap;
      sum += del;
      if (fabs(del) < fabs(sum)*EPS) {
	*gamser=sum*exp(-x+a*log(x)-(*gln));
	return;
      }
    }
    printf("a too large, ITMAX too small in routine gser");
    return;
  }
}
 
static void gcf(double *gammcf, double a, double x, double *gln){
  int i;
  double an,b,c,d,del,h;
  
  *gln=lgamma(a);
  b=x+1.0-a;
  c=1.0/FPMIN;
  d=1.0/b;
  h=d;
  for (i=1;i<=ITMAX;i++) {
    an = -i*(i-a);
    b += 2.0;
    d=an*d+b;
    if (fabs(d) < FPMIN) d=FPMIN;
    c=b+an/c;
    if (fabs(c) < FPMIN) c=FPMIN;
    d=1.0/d;
    del=d*c;
    h *= del;
    if (fabs(del-1.0) < EPS) break;
  }
  assert(i<=ITMAX);
  *gammcf=exp(-x+a*log(x)-(*gln))*h;
}

static int ijkl2intindex(int i, int j, int k, int l){
  int tmp,ij,kl;
  if (i<j) return ijkl2intindex(j,i,k,l);
  if (k<l) return ijkl2intindex(i,j,l,k);
  ij = i*(i+1)/2+j;
  kl = k*(k+1)/2+l;
  if (ij<kl){
    tmp = ij;
    ij = kl;
    kl = tmp;
  }
  return ij*(ij+1)/2+kl;
}

static int ijkl2intindex_old(int i, int j, int k, int l){
  int tmp,ij,kl;
  if (i<j){
    tmp = i;
    i = j;
    j = tmp;
  }
  if (k<l){
    tmp = k;
    k = l;
    l = tmp;
  }
  ij = i*(i+1)/2+j;
  kl = k*(k+1)/2+l;
  if (ij<kl){
    tmp = ij;
    ij = kl;
    kl = tmp;
  }
  return ij*(ij+1)/2+kl;
}

static int fact_ratio2(int a, int b){ return fact(a)/fact(b)/fact(a-2*b); }

static double product_center_1D(double alphaa, double xa, 
			 double alphab, double xb){
  return (alphaa*xa+alphab*xb)/(alphaa+alphab);
}

static double three_center_1D(double xi, int ai, double alphai,
			      double xj, int aj, double alphaj,
			      double xk, int ak, double alphak){

  double gamma, dx, px, xpi,xpj,xpk,intgl;
  int q,r,s,n;
  
  gamma = alphai+alphaj+alphak;
  dx = exp(-alphai*alphaj*pow(xi-xj,2)/gamma) *
    exp(-alphai*alphak*pow(xi-xk,2)/gamma) *
    exp(-alphaj*alphak*pow(xj-xk,2)/gamma);
  px = (alphai*xi+alphaj*xj+alphak*xk)/gamma;
    
  xpi = px-xi;
  xpj = px-xj;
  xpk = px-xk;
  intgl = 0;
  for (q=0; q<ai+1; q++){
    for (r=0; r<aj+1; r++){
      for (s=0; s<ak+1; s++){
	n = (q+r+s)/2;
	if ((q+r+s)%2 == 0) {
	  intgl += binomial(ai,q)*binomial(aj,r)*binomial(ak,s)*
	    pow(xpi,ai-q)*pow(xpj,aj-r)*pow(xpk,ak-s)*
	    fact2(2*n-1)/pow(2*gamma,n)*sqrt(M_PI/gamma);
	}
      }
    }
  }
  return dx*intgl;
}


/* work is the work space for the various exponents, contraction */
/*  coefficients, etc., used by the contracted code. Decided to  */
/*  allocate this all at once rather than doing mallocs/frees. */
/*  Only speeds things up a little, but every little bit counts, I guess. */
#define MAX_PRIMS_PER_CONT (10)
double work[12*MAX_PRIMS_PER_CONT];
double work_luis[24*MAX_PRIMS_PER_CONT]; /*Luis*/

static PyObject *fact_wrap(PyObject *self,PyObject *args){
  int ok = 0, n=0;
  ok = PyArg_ParseTuple(args,"i",&n);
  if (!ok) return NULL;
  return Py_BuildValue("i",fact(n));
}
static PyObject *fact2_wrap(PyObject *self,PyObject *args){
  int ok = 0, n=0;
  ok = PyArg_ParseTuple(args,"i",&n);
  if (!ok) return NULL;
  return Py_BuildValue("i",fact2(n));
}
static PyObject *dist2_wrap(PyObject *self,PyObject *args){
  int ok = 0;
  double x1,y1,z1,x2,y2,z2;
  PyObject *A, *B;
  ok = PyArg_ParseTuple(args,"OO",&A,&B);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(A,"ddd",&x1,&y1,&z1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&x2,&y2,&z2);
  if (!ok) return NULL;
  return Py_BuildValue("d",dist2(x1,y1,z1,x2,y2,z2));
}
static PyObject *dist_wrap(PyObject *self,PyObject *args){
  int ok = 0;
  double x1,y1,z1,x2,y2,z2;
  PyObject *A, *B;
  ok = PyArg_ParseTuple(args,"OO",&A,&B);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(A,"ddd",&x1,&y1,&z1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&x2,&y2,&z2);
  if (!ok) return NULL;
  return Py_BuildValue("d",dist(x1,y1,z1,x2,y2,z2));
}
static PyObject *binomial_wrap(PyObject *self,PyObject *args){
  int ok = 0, ia=0, ib=0;
  ok = PyArg_ParseTuple(args,"ii",&ia,&ib);
  if (!ok) return NULL;
  return Py_BuildValue("i",binomial(ia,ib));
}
static PyObject *binomial_prefactor_wrap(PyObject *self,PyObject *args){
  int ok = 0, s=0, ia=0, ib=0;
  double xpa=0.,xpb=0.;
  ok = PyArg_ParseTuple(args,"iiidd",&s,&ia,&ib,&xpa,&xpb);
  if (!ok) return NULL;
  return Py_BuildValue("i",binomial_prefactor(s,ia,ib,xpa,xpb));
}
static PyObject *Fgamma_wrap(PyObject *self,PyObject *args){
  int ok = 0;
  double m=0.,x=0.;
  ok = PyArg_ParseTuple(args,"dd",&m,&x);
  if (!ok) return NULL;
  return Py_BuildValue("d",Fgamma(m,x));
}
static PyObject *ijkl2intindex_wrap(PyObject *self,PyObject *args){
  int ok = 0,i,j,k,l;
  ok = PyArg_ParseTuple(args,"iiii",&i,&j,&k,&l);
  if (!ok) return NULL;
  return Py_BuildValue("i",ijkl2intindex(i,j,k,l));
}
static PyObject *fB_wrap(PyObject *self,PyObject *args){
  int ok = 0,i,l1,l2,r;
  double px,ax,bx,g;
  ok = PyArg_ParseTuple(args,"iiidddid",&i,&l1,&l2,&px,&ax,&bx,&r,&g);
  if (!ok) return NULL;
  return Py_BuildValue("d",fB(i,l1,l2,px,ax,bx,r,g));
}
static PyObject *fact_ratio2_wrap(PyObject *self,PyObject *args){
  int ok = 0,a,b;
  ok = PyArg_ParseTuple(args,"ii",&a,&b);
  if (!ok) return NULL;
  return Py_BuildValue("i",fact_ratio2(a,b));
}

static PyObject *contr_coulomb_wrap(PyObject *self,PyObject *args){
  int ok=0;
  double xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd;
  int la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd;
  int lena,lenb,lenc,lend;
  PyObject *aexps_obj,*acoefs_obj,*anorms_obj,
    *bexps_obj,*bcoefs_obj,*bnorms_obj,
    *cexps_obj,*ccoefs_obj,*cnorms_obj,
    *dexps_obj,*dcoefs_obj,*dnorms_obj,
    *xyza_obj,*lmna_obj,*xyzb_obj,*lmnb_obj,
    *xyzc_obj,*lmnc_obj,*xyzd_obj,*lmnd_obj;
  double *aexps,*acoefs,*anorms,
    *bexps,*bcoefs,*bnorms,
    *cexps,*ccoefs,*cnorms,
    *dexps,*dcoefs,*dnorms;
  int i;
  double Jij=0; /* return value */

  ok = PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOOOOOO",
			&aexps_obj,&acoefs_obj,&anorms_obj,&xyza_obj,&lmna_obj,
			&bexps_obj,&bcoefs_obj,&bnorms_obj,&xyzb_obj,&lmnb_obj,
			&cexps_obj,&ccoefs_obj,&cnorms_obj,&xyzc_obj,&lmnc_obj,
			&dexps_obj,&dcoefs_obj,&dnorms_obj,&xyzd_obj,&lmnd_obj);
  if (!ok) return NULL;


  ok=PyArg_ParseTuple(xyza_obj,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzb_obj,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzc_obj,"ddd",&xc,&yc,&zc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzd_obj,"ddd",&xd,&yd,&zd);
  if (!ok) return NULL;

  ok=PyArg_ParseTuple(lmna_obj,"iii",&la,&ma,&na);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnb_obj,"iii",&lb,&mb,&nb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnc_obj,"iii",&lc,&mc,&nc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnd_obj,"iii",&ld,&md,&nd);
  if (!ok) return NULL;

  /* Test that each is a sequence: */
  if (!PySequence_Check(aexps_obj)) return NULL;
  if (!PySequence_Check(acoefs_obj)) return NULL;
  if (!PySequence_Check(anorms_obj)) return NULL;
  if (!PySequence_Check(bexps_obj)) return NULL;
  if (!PySequence_Check(bcoefs_obj)) return NULL;
  if (!PySequence_Check(bnorms_obj)) return NULL;
  if (!PySequence_Check(cexps_obj)) return NULL;
  if (!PySequence_Check(ccoefs_obj)) return NULL;
  if (!PySequence_Check(cnorms_obj)) return NULL;
  if (!PySequence_Check(dexps_obj)) return NULL;
  if (!PySequence_Check(dcoefs_obj)) return NULL;
  if (!PySequence_Check(dnorms_obj)) return NULL;

  /* Get the length of each sequence */
  lena = PySequence_Size(aexps_obj);
  if (lena<0) return NULL;
  if (lena != PySequence_Size(acoefs_obj)) return NULL;
  if (lena != PySequence_Size(anorms_obj)) return NULL;
  lenb = PySequence_Size(bexps_obj);
  if (lenb<0) return NULL;
  if (lenb != PySequence_Size(bcoefs_obj)) return NULL;
  if (lenb != PySequence_Size(bnorms_obj)) return NULL;
  lenc = PySequence_Size(cexps_obj);
  if (lenc<0) return NULL;
  if (lenc != PySequence_Size(ccoefs_obj)) return NULL;
  if (lenc != PySequence_Size(cnorms_obj)) return NULL;
  lend = PySequence_Size(dexps_obj);
  if (lend<0) return NULL;
  if (lend != PySequence_Size(dcoefs_obj)) return NULL;
  if (lend != PySequence_Size(dnorms_obj)) return NULL;

  /* Allocate the space for each array */
  if (lena+lenb+lenc+lend > 4*MAX_PRIMS_PER_CONT) return NULL;
  aexps = work;
  acoefs = aexps + lena;
  anorms = acoefs + lena;
  bexps = anorms + lena;
  bcoefs = bexps + lenb;
  bnorms = bcoefs + lenb;
  cexps = bnorms + lenb;
  ccoefs = cexps + lenc;
  cnorms = ccoefs + lenc;
  dexps = cnorms + lenc;
  dcoefs = dexps + lend;
  dnorms = dcoefs + lend;

  /* Unpack all of the lengths: */
  for (i=0; i<lena; i++){
    aexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(aexps_obj,i));
    acoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(acoefs_obj,i));
    anorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(anorms_obj,i));
  }
  for (i=0; i<lenb; i++){
    bexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bexps_obj,i));
    bcoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bcoefs_obj,i));
    bnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bnorms_obj,i));
  }
  for (i=0; i<lenc; i++){
    cexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(cexps_obj,i));
    ccoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(ccoefs_obj,i));
    cnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(cnorms_obj,i));
  }
  for (i=0; i<lend; i++){
    dexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dexps_obj,i));
    dcoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dcoefs_obj,i));
    dnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dnorms_obj,i));
  }
  Jij = contr_coulomb(lena,aexps,acoefs,anorms,xa,ya,za,la,ma,na,
		      lenb,bexps,bcoefs,bnorms,xb,yb,zb,lb,mb,nb,
		      lenc,cexps,ccoefs,cnorms,xc,yc,zc,lc,mc,nc,
		      lend,dexps,dcoefs,dnorms,xd,yd,zd,ld,md,nd);

  return Py_BuildValue("d", Jij);
}

/*Luis*/
static PyObject *contr_coulomb_v3_wrap(PyObject *self,PyObject *args){
  int ok=0;
  double xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd;
  /*int la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd;*/
  int lena,lenb,lenc,lend;
  PyObject *aexps_obj,*acoefs_obj,*anorms_obj,
    *bexps_obj,*bcoefs_obj,*bnorms_obj,
    *cexps_obj,*ccoefs_obj,*cnorms_obj,
    *dexps_obj,*dcoefs_obj,*dnorms_obj,
    *xyza_obj,*la_obj,*ma_obj,*na_obj,
    *xyzb_obj,*lb_obj,*mb_obj,*nb_obj,
    *xyzc_obj,*lc_obj,*mc_obj,*nc_obj,
    *xyzd_obj,*ld_obj,*md_obj,*nd_obj;
  double *aexps,*acoefs,*anorms,*la,*ma,*na,
         *bexps,*bcoefs,*bnorms,*lb,*mb,*nb,
         *cexps,*ccoefs,*cnorms,*lc,*mc,*nc,
         *dexps,*dcoefs,*dnorms,*ld,*md,*nd;
  int i;
  double Jij=0; /* return value */
  
  /*Luis*/
  /*printf("point 1\n");*/
  ok = PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOOOOOOOOOOOOOO",
			&aexps_obj,&acoefs_obj,&anorms_obj,&xyza_obj,&la_obj,&ma_obj,&na_obj,
			&bexps_obj,&bcoefs_obj,&bnorms_obj,&xyzb_obj,&lb_obj,&mb_obj,&nb_obj,
			&cexps_obj,&ccoefs_obj,&cnorms_obj,&xyzc_obj,&lc_obj,&mc_obj,&nc_obj,
			&dexps_obj,&dcoefs_obj,&dnorms_obj,&xyzd_obj,&ld_obj,&md_obj,&nd_obj);
  if (!ok) return NULL;
  /*printf("point 2\n");*/

  ok=PyArg_ParseTuple(xyza_obj,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzb_obj,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzc_obj,"ddd",&xc,&yc,&zc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(xyzd_obj,"ddd",&xd,&yd,&zd);
  if (!ok) return NULL;

  //printf("xa = %f; xb = %f; xc = %f\n",xa,ya,za);

  /*Luis
  ok=PyArg_ParseTuple(lmna_obj,"iii",&la,&ma,&na);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnb_obj,"iii",&lb,&mb,&nb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnc_obj,"iii",&lc,&mc,&nc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(lmnd_obj,"iii",&ld,&md,&nd);
  if (!ok) return NULL;
  */

  /* Test that each is a sequence: */
  if (!PySequence_Check(aexps_obj)) return NULL;
  if (!PySequence_Check(acoefs_obj)) return NULL;
  if (!PySequence_Check(anorms_obj)) return NULL;
  if (!PySequence_Check(bexps_obj)) return NULL;
  if (!PySequence_Check(bcoefs_obj)) return NULL;
  if (!PySequence_Check(bnorms_obj)) return NULL;
  if (!PySequence_Check(cexps_obj)) return NULL;
  if (!PySequence_Check(ccoefs_obj)) return NULL;
  if (!PySequence_Check(cnorms_obj)) return NULL;
  if (!PySequence_Check(dexps_obj)) return NULL;
  if (!PySequence_Check(dcoefs_obj)) return NULL;
  if (!PySequence_Check(dnorms_obj)) return NULL;
    /*Luis*/
 
  if (!PySequence_Check(la_obj)) return NULL;
  if (!PySequence_Check(lb_obj)) return NULL;
  if (!PySequence_Check(lc_obj)) return NULL;
  if (!PySequence_Check(ld_obj)) return NULL;
  if (!PySequence_Check(ma_obj)) return NULL;
  if (!PySequence_Check(mb_obj)) return NULL;
  if (!PySequence_Check(mc_obj)) return NULL;
  if (!PySequence_Check(md_obj)) return NULL;
  if (!PySequence_Check(na_obj)) return NULL;
  if (!PySequence_Check(nb_obj)) return NULL;
  if (!PySequence_Check(nc_obj)) return NULL;
  if (!PySequence_Check(nd_obj)) return NULL;

  /* Get the length of each sequence */
  lena = PySequence_Size(aexps_obj);
  if (lena<0) return NULL;
  if (lena != PySequence_Size(acoefs_obj)) return NULL;
  if (lena != PySequence_Size(anorms_obj)) return NULL;
  if (lena != PySequence_Size(la_obj)) return NULL;/*Luis*/
  if (lena != PySequence_Size(ma_obj)) return NULL;/*Luis*/
  if (lena != PySequence_Size(na_obj)) return NULL;/*Luis*/
  lenb = PySequence_Size(bexps_obj);
  if (lenb<0) return NULL;
  if (lenb != PySequence_Size(bcoefs_obj)) return NULL;
  if (lenb != PySequence_Size(bnorms_obj)) return NULL;
  if (lenb != PySequence_Size(lb_obj)) return NULL;/*Luis*/
  if (lenb != PySequence_Size(mb_obj)) return NULL;/*Luis*/
  if (lenb != PySequence_Size(nb_obj)) return NULL;/*Luis*/
  lenc = PySequence_Size(cexps_obj);
  if (lenc<0) return NULL;
  if (lenc != PySequence_Size(ccoefs_obj)) return NULL;
  if (lenc != PySequence_Size(cnorms_obj)) return NULL;
  if (lenc != PySequence_Size(lc_obj)) return NULL;/*Luis*/
  if (lenc != PySequence_Size(mc_obj)) return NULL;/*Luis*/
  if (lenc != PySequence_Size(nc_obj)) return NULL;/*Luis*/
  lend = PySequence_Size(dexps_obj);
  if (lend<0) return NULL;
  if (lend != PySequence_Size(dcoefs_obj)) return NULL;
  if (lend != PySequence_Size(dnorms_obj)) return NULL;
  if (lend != PySequence_Size(ld_obj)) return NULL;/*Luis*/
  if (lend != PySequence_Size(md_obj)) return NULL;/*Luis*/
  if (lend != PySequence_Size(nd_obj)) return NULL;/*Luis*/

  /* Allocate the space for each array */
   /*Luis*/
  if (lena+lenb+lenc+lend > 4*MAX_PRIMS_PER_CONT) return NULL;
  //printf("allocating spaces\n");
  aexps  = work_luis;
  acoefs = aexps + lena;
  anorms = acoefs + lena;
  la     = anorms + lena;
  ma     = la + lena;
  na     = ma + lena;
  bexps  = na + lena;
  bcoefs = bexps + lenb;
  bnorms = bcoefs + lenb;
  lb     = bnorms + lenb;
  mb     = lb + lenb;
  nb     = mb + lenb;
  cexps  = nb + lenb;
  ccoefs = cexps + lenc;
  cnorms = ccoefs + lenc;
  lc     = cnorms + lenc;
  mc     = lc + lenc;
  nc     = mc + lenc;
  dexps  = nc + lenc;
  dcoefs = dexps + lend;
  dnorms = dcoefs + lend;
  ld     = dnorms + lend;
  md     = ld + lend;
  nd     = md + lend;

  /* Unpack all of the lengths: */
  for (i=0; i<lena; i++){
    aexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(aexps_obj,i));
    acoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(acoefs_obj,i));
    anorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(anorms_obj,i));
    la[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(la_obj,i)); /*Luis*/
    ma[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(ma_obj,i)); /*Luis*/
    na[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(na_obj,i)); /*Luis*/
    //printf("i=%i la[i]=%f ma[i]=%f na[i]=%f\n",i,la[i],ma[i],na[i]);
  }
  for (i=0; i<lenb; i++){
    bexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bexps_obj,i));
    bcoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bcoefs_obj,i));
    bnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(bnorms_obj,i));
    lb[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(lb_obj,i)); /*Luis*/
    mb[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(mb_obj,i)); /*Luis*/
    nb[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(nb_obj,i)); /*Luis*/
    //printf("i=%i lb[i]=%f mb[i]=%f nb[i]=%f\n",i,lb[i],mb[i],nb[i]);
  }
  for (i=0; i<lenc; i++){
    cexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(cexps_obj,i));
    ccoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(ccoefs_obj,i));
    cnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(cnorms_obj,i));
    lc[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(lc_obj,i)); /*Luis*/
    mc[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(mc_obj,i)); /*Luis*/
    nc[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(nc_obj,i)); /*Luis*/
    //printf("i=%i lc[i]=%f mc[i]=%f nc[i]=%f\n",i,lc[i],mc[i],nc[i]);
  }
  for (i=0; i<lend; i++){
    dexps[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dexps_obj,i));
    dcoefs[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dcoefs_obj,i));
    dnorms[i] = PyFloat_AS_DOUBLE(PySequence_GetItem(dnorms_obj,i));
    ld[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(ld_obj,i)); /*Luis*/
    md[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(md_obj,i)); /*Luis*/
    nd[i]   = PyFloat_AS_DOUBLE(PySequence_GetItem(nd_obj,i)); /*Luis*/
    //printf("i=%i ld[i]=%f md[i]=%f nd[i]=%f\n",i,ld[i],md[i],nd[i]);
  }
  /*Luis*/
  Jij = contr_coulomb_v3(lena,aexps,acoefs,anorms,xa,ya,za,la,ma,na,
		         lenb,bexps,bcoefs,bnorms,xb,yb,zb,lb,mb,nb,
		         lenc,cexps,ccoefs,cnorms,xc,yc,zc,lc,mc,nc,
		         lend,dexps,dcoefs,dnorms,xd,yd,zd,ld,md,nd);

  return Py_BuildValue("d", Jij);
}
static PyObject *contr_nuke_vec_wrap(PyObject *self,PyObject *args){

  /* This turned out to be slower than multiple calls to 
     nuclear_attraction_vec. I'm leaving the code in place
     for reference.
  */
  PyObject *aexps, *acoefs, *anorms, *aorigin, *apowers,
    *bexps, *bcoefs, *bnorms, *borigin, *bpowers,
    *xc, *yc, *zc, *wc, *qc;
  int ok;
  double xa,ya,za,xb,yb,zb;
  int la,ma,na,lb,mb,nb;
  int nprima, nprimb, ncenters;
  int i,j,k;
  double anormi,aexpi,acoefi, bnormj,bexpj,bcoefj,xck,yck,zck,wck,qck;
  double incr=0, Vnij=0;
    

  ok = PyArg_ParseTuple(args,"OOOOOOOOOOOOOOO",
			&aexps,&acoefs,&anorms,&aorigin,&apowers,
			&bexps,&bcoefs,&bnorms,&borigin,&bpowers,
			&xc, &yc, &zc, &wc, &qc);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(aorigin,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(borigin,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(apowers,"iii",&la,&ma,&na);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(bpowers,"iii",&lb,&mb,&nb);
  if (!ok) return NULL;

  nprima = PySequence_Size(aexps);
  if (nprima<0) return NULL;
  if (nprima != PySequence_Size(acoefs)) return NULL;
  if (nprima != PySequence_Size(anorms)) return NULL;
  
  nprimb = PySequence_Size(bexps);
  if (nprimb<0) return NULL;
  if (nprimb != PySequence_Size(bcoefs)) return NULL;
  if (nprimb != PySequence_Size(bnorms)) return NULL;

  ncenters = PySequence_Size(xc);
  if (ncenters < 0) return NULL;
  if (ncenters != PySequence_Size(yc)) return NULL;
  if (ncenters != PySequence_Size(zc)) return NULL;
  if (ncenters != PySequence_Size(wc)) return NULL;
  if (ncenters != PySequence_Size(qc)) return NULL;
  
  for (k=0; k<ncenters; k++){
    xck = PyFloat_AS_DOUBLE(PySequence_GetItem(xc,k));
    yck = PyFloat_AS_DOUBLE(PySequence_GetItem(yc,k));
    zck = PyFloat_AS_DOUBLE(PySequence_GetItem(zc,k));
    wck = PyFloat_AS_DOUBLE(PySequence_GetItem(wc,k));
    qck = PyFloat_AS_DOUBLE(PySequence_GetItem(qc,k));
    for (i=0; i<nprima; i++){
      anormi = PyFloat_AS_DOUBLE(PySequence_GetItem(anorms,i));
      aexpi = PyFloat_AS_DOUBLE(PySequence_GetItem(aexps,i));
      acoefi = PyFloat_AS_DOUBLE(PySequence_GetItem(acoefs,i));
      for (j=0; j<nprimb; j++){
	bnormj = PyFloat_AS_DOUBLE(PySequence_GetItem(bnorms,j));
	bexpj = PyFloat_AS_DOUBLE(PySequence_GetItem(bexps,j));
	bcoefj = PyFloat_AS_DOUBLE(PySequence_GetItem(bcoefs,j));
	incr = nuclear_attraction(xa,ya,za,anormi,la,ma,na,aexpi,
				  xb,yb,zb,bnormj,lb,mb,nb,bexpj,
				  xck,yck,zck);
	Vnij += acoefi*bcoefj*wck*qck*incr;
      }
    }
  }
  return Py_BuildValue("d",Vnij);
  
}

static PyObject *coulomb_repulsion_wrap(PyObject *self,PyObject *args){
  int ok=0;
  double norma,alphaa,normb,alphab,normc,alphac,normd,alphad,
    xa,ya,za,xb,yb,zb,xc,yc,zc,xd,yd,zd;
  int la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd;
  PyObject *A,*B,*C,*D,*powa,*powb,*powc,*powd;

  ok=PyArg_ParseTuple(args,"OdOdOdOdOdOdOdOd",&A,&norma,&powa,&alphaa,
		      &B,&normb,&powb,&alphab,&C,&normc,&powc,&alphac,
		      &D,&normd,&powd,&alphad);
  if (!ok) return NULL;

  ok=PyArg_ParseTuple(A,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(B,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(C,"ddd",&xc,&yc,&zc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(D,"ddd",&xd,&yd,&zd);
  if (!ok) return NULL;

  ok=PyArg_ParseTuple(powa,"iii",&la,&ma,&na);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(powb,"iii",&lb,&mb,&nb);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(powc,"iii",&lc,&mc,&nc);
  if (!ok) return NULL;
  ok=PyArg_ParseTuple(powd,"iii",&ld,&md,&nd);
  if (!ok) return NULL;
  
  return Py_BuildValue("d", 
    coulomb_repulsion(xa,ya,za,norma,la,ma,na,alphaa,
		      xb,yb,zb,normb,lb,mb,nb,alphab,
		      xc,yc,zc,normc,lc,mc,nc,alphac,
		      xd,yd,zd,normd,ld,md,nd,alphad));
}

static PyObject *kinetic_wrap(PyObject *self,PyObject *args){
  int ok=0,l1,m1,n1,l2,m2,n2;
  double xa,ya,za,xb,yb,zb,alpha1,alpha2;
  PyObject *A,*B,*powa,*powb;

  ok = PyArg_ParseTuple(args,"dOOdOO",&alpha1,&powa,&A,&alpha2,
			&powb,&B);

  if (!ok) return NULL;

  ok = PyArg_ParseTuple(powa,"iii",&l1,&m1,&n1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powb,"iii",&l2,&m2,&n2);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(A,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;

  return Py_BuildValue("d",
		       kinetic(alpha1,l1,m1,n1,xa,ya,za,
			       alpha2,l2,m2,n2,xb,yb,zb));
}

static PyObject *overlap_wrap(PyObject *self,PyObject *args){
  int ok=0,l1,m1,n1,l2,m2,n2;
  double xa,ya,za,xb,yb,zb,alpha1,alpha2;
  PyObject *A,*B,*powa,*powb;

  ok = PyArg_ParseTuple(args,"dOOdOO",&alpha1,&powa,&A,&alpha2,
			&powb,&B);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(powa,"iii",&l1,&m1,&n1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powb,"iii",&l2,&m2,&n2);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(A,"ddd",&xa,&ya,&za);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&xb,&yb,&zb);
  if (!ok) return NULL;

  return Py_BuildValue("d",
		       overlap(alpha1,l1,m1,n1,xa,ya,za,
				      alpha2,l2,m2,n2,xb,yb,zb));
}

static PyObject *nuclear_attraction_wrap(PyObject *self,PyObject *args){
  int ok=0,l1,m1,n1,l2,m2,n2;
  double x1,y1,z1,x2,y2,z2,x3,y3,z3,norm1,alpha1,norm2,alpha2;
  PyObject *A,*B,*C,*powa,*powb;

  ok = PyArg_ParseTuple(args,"OdOdOdOdO",&A,&norm1,&powa,&alpha1,
			&B,&norm2,&powb,&alpha2,&C);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(A,"ddd",&x1,&y1,&z1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&x2,&y2,&z2);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(C,"ddd",&x3,&y3,&z3);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powa,"iii",&l1,&m1,&n1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powb,"iii",&l2,&m2,&n2);
  if (!ok) return NULL;

  return Py_BuildValue("d",
     nuclear_attraction(x1,y1,z1,norm1,l1,m1,n1,alpha1,
			x2,y2,z2,norm2,l2,m2,n2,alpha2,
			x3,y3,z3));
}

static PyObject *three_center_1D_wrap(PyObject *self,PyObject *args){

  double xi,xj,xk,alphai,alphaj,alphak;
  int ai,aj,ak;
  int ok=0;

  ok = PyArg_ParseTuple(args,"diddiddid",&xi,&ai,&alphai,
			&xj,&aj,&alphaj,&xk,&ak,&alphak);
  if (!ok) return NULL;
  return Py_BuildValue("d",
		       three_center_1D(xi,ai,alphai,
				       xj,aj,alphaj,xk,ak,alphak));
}

static PyObject *nuclear_attraction_vec_wrap(PyObject *self,PyObject *args){
  int ok=0,l1,m1,n1,l2,m2,n2;
  double x1,y1,z1,x2,y2,z2,norm1,alpha1,norm2,alpha2;
  PyObject *A,*B,*xc_obj, *yc_obj, *zc_obj, *wc_obj, *qc_obj, 
    *powa, *powb;
  double retval = 0,wc,qc,xc,yc,zc;
  int veclength = 0, i;

  ok = PyArg_ParseTuple(args,"OdOdOdOdOOOOO",&A,&norm1,&powa,&alpha1,
			&B,&norm2,&powb,&alpha2,
			&xc_obj, &yc_obj, &zc_obj, &wc_obj, &qc_obj);
  if (!ok) return NULL;

  ok = PyArg_ParseTuple(A,"ddd",&x1,&y1,&z1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(B,"ddd",&x2,&y2,&z2);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powa,"iii",&l1,&m1,&n1);
  if (!ok) return NULL;
  ok = PyArg_ParseTuple(powb,"iii",&l2,&m2,&n2);
  if (!ok) return NULL;

  if (!PySequence_Check(xc_obj)) return NULL;
  if (!PySequence_Check(yc_obj)) return NULL;
  if (!PySequence_Check(zc_obj)) return NULL;
  if (!PySequence_Check(wc_obj)) return NULL;
  if (!PySequence_Check(qc_obj)) return NULL;
  

  veclength = PySequence_Size(xc_obj);
  if (veclength<0) return NULL;
  if (veclength != PySequence_Size(yc_obj)) return NULL;
  if (veclength != PySequence_Size(zc_obj)) return NULL;
  if (veclength != PySequence_Size(wc_obj)) return NULL;
  if (veclength != PySequence_Size(qc_obj)) return NULL;


  for (i=0; i<veclength; i++){
    xc = PyFloat_AS_DOUBLE(PySequence_GetItem(xc_obj,i));
    yc = PyFloat_AS_DOUBLE(PySequence_GetItem(yc_obj,i));
    zc = PyFloat_AS_DOUBLE(PySequence_GetItem(zc_obj,i));
    wc = PyFloat_AS_DOUBLE(PySequence_GetItem(wc_obj,i));
    qc = PyFloat_AS_DOUBLE(PySequence_GetItem(qc_obj,i));

    retval += wc*qc*nuclear_attraction(x1,y1,z1,norm1,l1,m1,n1,alpha1,
				     x2,y2,z2,norm2,l2,m2,n2,alpha2,
				     xc,yc,zc);
  }

  return Py_BuildValue("d",retval);
}



/* Python interface */
static PyMethodDef cints_methods[] = {
  {"fact",fact_wrap,METH_VARARGS},
  {"fact2",fact2_wrap,METH_VARARGS},
  {"dist2",dist2_wrap,METH_VARARGS},
  {"dist",dist_wrap,METH_VARARGS},
  {"binomial",binomial_wrap,METH_VARARGS},
  {"binomial_prefactor",binomial_prefactor_wrap,METH_VARARGS},
  {"Fgamma",Fgamma_wrap,METH_VARARGS},
  {"ijkl2intindex",ijkl2intindex_wrap,METH_VARARGS},
  {"fB",fB_wrap,METH_VARARGS},
  {"fact_ratio2",fact_ratio2_wrap,METH_VARARGS},
  {"contr_coulomb",contr_coulomb_wrap,METH_VARARGS},
  {"contr_coulomb_v3",contr_coulomb_v3_wrap,METH_VARARGS}, /*Luis*/
  {"coulomb_repulsion",coulomb_repulsion_wrap,METH_VARARGS},
  {"kinetic",kinetic_wrap,METH_VARARGS}, 
  {"overlap",overlap_wrap,METH_VARARGS}, 
  {"nuclear_attraction",nuclear_attraction_wrap,METH_VARARGS},
  {"nuclear_attraction_vec",nuclear_attraction_vec_wrap,METH_VARARGS},
  {"contr_nuke_vec",contr_nuke_vec_wrap,METH_VARARGS},
  {"three_center_1D",three_center_1D_wrap,METH_VARARGS},
  {NULL,NULL} /* Sentinel */
};

static void module_init(char* name)
{
  (void) Py_InitModule(name,cints_methods);
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
#if defined(PYQUANTE_FULLY_QUALIFIED_MODULE_NAME)
void initpyquante_cints_ext(){module_init("pyquante_cints_ext");}
#else
void initcints(){module_init("cints");}
#endif

#undef ITMAX
#undef EPS
#undef FPMIN
