/*************************************************************************
 *
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
 **************************************************************************/
/* My routines */
#ifdef _MSC_VER
double lgamma(double);
#endif

static double fB(int i, int l1, int l2, double px, double ax, double bx, 
	  int r, double g);
static double Bfunc(int i, int r, double g);
static double contr_coulomb(int ia, double *aexps, double *acoefs, double *anorms,
			    double xa, double ya, double za, int la, int ma, int na, 
			    int ib, double *bexps, double *bcoefs, double *bnorms,
			    double xb, double yb, double zb, int lb, int mb, int nb, 
			    int ic, double *cexps, double *ccoefs, double *cnorms,
			    double xc, double yc, double zc, int lc, int mc, int nc, 
			    int id, double *dexps, double *dcoefs, double *dnorms,
			    double xd, double yd, double zd, int ld, int md, int nd);

/*Luis*/
static double contr_coulomb_v3(int ia, double *aexps, double *acoefs, double *anorms,
			    double xa, double ya, double za, double *la, double *ma, double *na, 
			    int ib, double *bexps, double *bcoefs, double *bnorms,
			    double xb, double yb, double zb, double *lb, double *mb, double *nb, 
			    int ic, double *cexps, double *ccoefs, double *cnorms,
			    double xc, double yc, double zc, double *lc, double *mc, double *nc, 
			    int id, double *dexps, double *dcoefs, double *dnorms,
			    double xd, double yd, double zd, double *ld, double *md, double *nd);

static double coulomb_repulsion(double xa, double ya, double za, double norma,
				int la, int ma, int na, double alphaa,
				double xb, double yb, double zb, double normb,
				int lb, int mb, int nb, double alphab,
				double xc, double yc, double zc, double normc,
				int lc, int mc, int nc, double alphac,
				double xd, double yd, double zd, double normd,
				int ld, int md, int nd, double alphad);

static double *B_array(int l1, int l2, int l3, int l4, double p, double a,
		double b, double q, double c, double d,
		double g1, double g2, double delta);

static double B_term(int i1, int i2, int r1, int r2, int u, int l1, int l2,
		     int l3, int l4, double Px, double Ax, double Bx,
		     double Qx, double Cx, double Dx, double gamma1,
		     double gamma2, double delta);
static double kinetic(double alpha1, int l1, int m1, int n1,
		      double xa, double ya, double za,
		      double alpha2, int l2, int m2, int n2,
		      double xb, double yb, double zb);
static double overlap(double alpha1, int l1, int m1, int n1,
		      double xa, double ya, double za,
		      double alpha2, int l2, int m2, int n2,
		      double xb, double yb, double zb);
static double overlap_1D(int l1, int l2, double PAx,
			 double PBx, double gamma);
static double nuclear_attraction(double x1, double y1, double z1, double norm1,
				 int l1, int m1, int n1, double alpha1,
				 double x2, double y2, double z2, double norm2,
				 int l2, int m2, int n2, double alpha2,
				 double x3, double y3, double z3);
static double A_term(int i, int r, int u, int l1, int l2,
		     double PAx, double PBx, double CPx, double gamma);
static double *A_array(int l1, int l2, double PA, double PB,
		       double CP, double g);

static int fact(int n);
static int fact2(int n);
static double dist2(double x1, double y1, double z1, 
		    double x2, double y2, double z2);
static double dist(double x1, double y1, double z1, 
		   double x2, double y2, double z2);
static double binomial_prefactor(int s, int ia, int ib, double xpa, double xpb);
static int binomial(int a, int b);

static double Fgamma(double m, double x);
static double gamm_inc(double a, double x);

static int ijkl2intindex(int i, int j, int k, int l);

static int fact_ratio2(int a, int b);

static double product_center_1D(double alphaa, double xa, 
			 double alphab, double xb);

static double three_center_1D(double xi, int ai, double alphai,
			      double xj, int aj, double alphaj,
			      double xk, int ak, double alphak);

/* Routines from Numerical Recipes */
static void gser(double *gamser, double a, double x, double *gln);
static void gcf(double *gammcf, double a, double x, double *gln);

/* Wrappers */
static PyObject *fact_wrap(PyObject *self,PyObject *args);
static PyObject *fact2_wrap(PyObject *self,PyObject *args);
static PyObject *dist2_wrap(PyObject *self,PyObject *args);
static PyObject *dist_wrap(PyObject *self,PyObject *args);
static PyObject *dist_wrap(PyObject *self,PyObject *args);
static PyObject *binomial_prefactor_wrap(PyObject *self,PyObject *args);
static PyObject *Fgamma_wrap(PyObject *self,PyObject *args);
static PyObject *ijkl2intindex_wrap(PyObject *self,PyObject *args);
static PyObject *fB_wrap(PyObject *self,PyObject *args);
static PyObject *fact_ratio2_wrap(PyObject *self,PyObject *args);
static PyObject *contr_coulomb_wrap(PyObject *self,PyObject *args);
static PyObject *contr_coulomb_v3_wrap(PyObject *self,PyObject *args); /*Luis*/
static PyObject *coulomb_repulsion_wrap(PyObject *self,PyObject *args);
static PyObject *kinetic_wrap(PyObject *self,PyObject *args);
static PyObject *overlap_wrap(PyObject *self,PyObject *args);
static PyObject *nuclear_attraction_wrap(PyObject *self,PyObject *args);
static PyObject *nuclear_attraction_vec_wrap(PyObject *self,PyObject *args);
static PyObject *three_center_1D_wrap(PyObject *self,PyObject *args);

