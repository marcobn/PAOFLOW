from math import sqrt,pow,log,exp,pi,floor


def ijkl2intindex(i,j,k,l):
    "Indexing into the get2ints long array"
    if i<j: i,j = j,i
    if k<l: k,l = l,k
    ij = i*(i+1)/2+j
    kl = k*(k+1)/2+l
    if ij < kl: ij,kl = kl,ij
    return ij*(ij+1)/2+kl


def contr_coulomb_v2(aexps,acoefs,anorms,xyza,powa,
                     bexps,bcoefs,bnorms,xyzb,powb,
                     cexps,ccoefs,cnorms,xyzc,powc,
                     dexps,dcoefs,dnorms,xyzd,powd):

    Jij = 0.
    for i in xrange(len(aexps)):
        for j in xrange(len(bexps)):
            for k in xrange(len(cexps)):
                for l in xrange(len(dexps)):
                    incr = coulomb_repulsion(xyza,anorms[i],powa[i],aexps[i],
                                             xyzb,bnorms[j],powb[j],bexps[j],
                                             xyzc,cnorms[k],powc[k],cexps[k],
                                             xyzd,dnorms[l],powd[l],dexps[l])
                    Jij = Jij + acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*incr
    return Jij


def contr_coulomb_v3(aexps,acoefs,anorms,xyza,powax,poway,powaz,
                     bexps,bcoefs,bnorms,xyzb,powbx,powby,powbz,
                     cexps,ccoefs,cnorms,xyzc,powcx,powcy,powcz,
                     dexps,dcoefs,dnorms,xyzd,powdx,powdy,powdz):

    Jij = 0.
    for i in xrange(len(aexps)):
        for j in xrange(len(bexps)):
            for k in xrange(len(cexps)):
                for l in xrange(len(dexps)):
                    incr = coulomb_repulsion(xyza,anorms[i],(powax[i],poway[i],powaz[i]),aexps[i],
                                             xyzb,bnorms[j],(powax[j],poway[j],powaz[j]),bexps[j],
                                             xyzc,cnorms[k],(powax[k],poway[k],powaz[k]),cexps[k],
                                             xyzd,dnorms[l],(powax[l],poway[l],powaz[l]),dexps[l])
                    Jij = Jij + acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*incr
    return Jij

##################################################################################################


def coulomb_repulsion((xa,ya,za),norma,(la,ma,na),alphaa,
                      (xb,yb,zb),normb,(lb,mb,nb),alphab,
                      (xc,yc,zc),normc,(lc,mc,nc),alphac,
                      (xd,yd,zd),normd,(ld,md,nd),alphad):

    rab2 = dist2((xa,ya,za),(xb,yb,zb)) #needs dist2
    rcd2 = dist2((xc,yc,zc),(xd,yd,zd)) #needs dist2
    xp,yp,zp = gaussian_product_center(alphaa,(xa,ya,za),alphab,(xb,yb,zb)) #needs gaussian_product_center
    xq,yq,zq = gaussian_product_center(alphac,(xc,yc,zc),alphad,(xd,yd,zd)) #needs gaussian_product_center
    rpq2 = dist2((xp,yp,zp),(xq,yq,zq)) #needs dist2
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = 0.25*(1/gamma1+1/gamma2)

    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta) #needs B_array
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta) #needs B_array
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta) #needs B_array

    sum = 0.
    for I in xrange(la+lb+lc+ld+1):
        for J in xrange(ma+mb+mc+md+1):
            for K in xrange(na+nb+nc+nd+1):
                sum = sum + Bx[I]*By[J]*Bz[K]*Fgamma(I+J+K,0.25*rpq2/delta) #needs Fgamma

    return 2*pow(pi,2.5)/(gamma1*gamma2*sqrt(gamma1+gamma2)) \
           *exp(-alphaa*alphab*rab2/gamma1) \
           *exp(-alphac*alphad*rcd2/gamma2)*sum*norma*normb*normc*normd



#########################################################################################

def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    Imax = l1+l2+l3+l4+1
    B = [0]*Imax
    for i1 in xrange(l1+l2+1):
        for i2 in xrange(l3+l4+1):
            for r1 in xrange(i1/2+1):
                for r2 in xrange(i2/2+1):
                    for u in xrange((i1+i2)/2-r1-r2+1):
                        I = i1+i2-2*(r1+r2)-u
                        B[I] = B[I] + B_term(i1,i2,r1,r2,u,l1,l2,l3,l4, 
                                             p,a,b,q,c,d,g1,g2,delta)   #needs B_term
    return B

def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    "THO eq. 2.22"
    return fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1)*pow(-1,i2)*fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2)*pow(-1,u)*fact_ratio2(i1+i2-2*(r1+r2),u)*pow(Qx-Px,i1+i2-2*(r1+r2)-2*u)/pow(delta,i1+i2-2*(r1+r2)-u)#needs fact_ratio2

######################################################################################


def fB(i,l1,l2,P,A,B,r,g):
    return binomial_prefactor(i,l1,l2,P-A,P-B)*B0(i,r,g)


def binomial_prefactor(s,ia,ib,xpa,xpb):
    "From Augspurger and Dykstra"
    sum = 0
    for t in xrange(s+1):
        if s-ia <= t <= ib:
                sum = sum + binomial(ia,s-t)*binomial(ib,t)* \
                  pow(xpa,ia-s+t)*pow(xpb,ib-t)
    return sum


def binomial(a,b):
    "Binomial coefficient"
    return fact(a)/fact(b)/fact(a-b)
######################################################################################
def B0(i,r,g): return fact_ratio2(i,r)*pow(4*g,r-i)

######################################################################################

def fact_ratio2(a,b): return fact(a)/fact(b)/fact(a-2*b) #needs fact

def fact(i): # needs nothing extra
    "Normal factorial"
    val = 1
    while (i>1):
        val = i*val
        i = i-1
    return val


###########################################################################################


def dist2(A,B): #needs nothing extra
    return pow(A[0]-B[0],2)+pow(A[1]-B[1],2)+pow(A[2]-B[2],2)

def gaussian_product_center(alpha1,A,alpha2,B): #needs nothing extra
    gamma = alpha1+alpha2
    return (alpha1*A[0]+alpha2*B[0])/gamma,(alpha1*A[1]+alpha2*B[1])/gamma,(alpha1*A[2]+alpha2*B[2])/gamma




######################################################################################################################
def Fgamma(m,x):
    "Incomplete gamma function"
    SMALL=0.00000001
    x = max(abs(x),SMALL)
    val = gamm_inc(m+0.5,x) #needs gamma_inc
    return 0.5*pow(x,-m-0.5)*val;


def gamm_inc(a,x):
    "Incomple gamma function \gamma; computed from NumRec routine gammp."
    gammap,gln = gammp(a,x) #needs gammp
    return exp(gln)*gammap


def gammp(a,x):
    "Returns the incomplete gamma function P(a;x). NumRec sect 6.2."
    assert (x > 0 and a >= 0), "Invalid arguments in routine gammp"

    if x < (a+1.0): 
        gamser,gln = _gser(a,x) #needs _gser
        return gamser,gln

    gammcf,gln = _gcf(a,x) #needs _gcf
    return 1.0-gammcf ,gln 


######################################################################################################################
def _gser(a,x):
    "Series representation of Gamma. NumRec sect 6.1."
    ITMAX=100
    EPS=3.e-7

    gln=gammln(a) #needs gammln
    assert(x>=0),'x < 0 in gser'
    if x == 0 : return 0,gln

    ap = a
    delt = sum = 1./a
    for i in xrange(ITMAX):
        ap=ap+1.
        delt=delt*x/ap
        sum=sum+delt
        if abs(delt) < abs(sum)*EPS: break
    else:
        print 'a too large, ITMAX too small in gser'
    gamser=sum*exp(-x+a*log(x)-gln)
    return gamser,gln

def _gcf(a,x):
    "Continued fraction representation of Gamma. NumRec sect 6.1"
    ITMAX=100
    EPS=3.e-7
    FPMIN=1.e-30

    gln=gammln(a) #needs gammln
    b=x+1.-a
    c=1./FPMIN
    d=1./b
    h=d
    for i in xrange(1,ITMAX+1):
        an=-i*(i-a)
        b=b+2.
        d=an*d+b
        if abs(d) < FPMIN: d=FPMIN
        c=b+an/c
        if abs(c) < FPMIN: c=FPMIN
        d=1./d
        delt=d*c
        h=h*delt
        if abs(delt-1.) < EPS: break
    else:
        print 'a too large, ITMAX too small in gcf'
    gammcf=exp(-x+a*log(x)-gln)*h
    return gammcf,gln



def gammln(x): #needs nothing extra
    "Numerical recipes, section 6.1"
    cof = [76.18009172947146,-86.50532032941677,
           24.01409824083091,-1.231739572450155,
           0.1208650973866179e-2,-0.5395239384953e-5]
    y=x
    tmp=x+5.5
    tmp = tmp - (x+0.5)*log(tmp)
    ser=1.000000000190015 # don't you just love these numbers?!
    for j in xrange(6):
        y = y+1
        ser = ser+cof[j]/y
    return -tmp+log(2.5066282746310005*ser/x);


####################################################################################################################


