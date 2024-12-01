/*
                       Copyright (c) 1995 by:
        Leif Laaksonen , Centre for Scientific Computing , ESPOO, FINLAND
            Confidential unpublished property of Leif Laaksonen
                        All rights reserved


      This is a program to convert the output using the "cube"
      command from the Gaussian program to a plot format recognized
      by gOpenMol.




      Run this program on the output from GaussianXX using the
      'cube' keyword in GaussianXX.

      This program produces a coordinate file and a plot file, 
      that can be read into SCARECROW or gOpenMol.

      The running of this program is quite obvious (see later in this
      file).

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 This program converts the Gaussian output using the 'cube'
 command to a form understandable to gopenmol.
 Usage:
 gcube2plt -iinput.cube -ooutput.plt
 Options:  -mXXX , where XXX is the molecular orbital number to be
                   placed in the plot file
           -p      prevent the output of the coordinate file
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


      If you need any help please feel free to contact:

      Leif.Laaksonen@csc.fi


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Taken from the;
                 Gaussian 94 User's Reference Manual
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

> Cube keyword


DESCRIPTION

The Cube properties keyword can be used to evaluate molecular orbitals, the 
electrostatic potential, the electron density, density gradient, the norm of 
the density gradient, and Laplacian of the density over a 3 dimensional grid 
(cube) of points. By default, Cube evaluates the electron density (corresponding to
the Density option). Which density is used is controlled by the Density keyword; 
use Density=Current to evaluate the cube over the density from a correlated or 
Cl-Singles wavefunction instead of the default Hartree-Fock density.

Note that only one of the available quantities can be evaluated within any one job 
step. Save the checkpoint file (using %Chk), and include Guess=(Reod,Only) 
Density=Checkpoint in the route section of a subsequent job (or job step) in 
order to evaluate a different quantity without repeating any of the other steps
 of the calculation.

Gaussian 94 provides reasonable defaults for grids, so Cube no longer requires 
that the cube be specified by the user. However, the output filename must still
always be provided (see below).

Alternatively, Cube may be given a parameter specifying the number of points to
use per "side" (the default is 80). For example, Cube=100 specifies a grid of 
1,000,000 points ( 100 ), evenly distributed over the rectangular grid generated
by the program (which is not necessarily a cube). In addition, the input
format used by earlier versions of Gaussian is still supported; Cube=Cards
indicates that a grid will be input. It may be used to specify a grid of
arbitrary size and shape.

The files create by Cube can be manipulated using the cubman utility, described
in chapter 5.

Note that Pop=None will inhibit cube file creation.

INPUT FORMAT

When the user elects to provide it, the grid information is read from the input
stream. The first line-required for all Cube jobs-gives a file name for the cube
file. Subsequent lines, which are included only with Cube=Cards, must conform to
format (15,3F12.6), according to the following syntax:

Output-file-name              Required in all Cube jobs.
IFlag, X0, Y0, Z0             Output unit number and initial point.
N1, X1, Y1, Z1                Number of points and step-size in the X-direction.
N2, X2, Y2, Z2                Number of points and step-size in the Y-direction.
N3, X3, Y3, Z3                Number of points and step-size in the Z-direction.

If IFlag is positive, the output file is unformatted; if it is negative, 
the output file is formatted. If N1<O the input cube coordinates are assumed 
to be in Bohr, otherwise, they are interpreted as Angstroms (|N1| is used as 
the number of X-direction points in any case). Note that the three axes are 
used exactly as specified; they are not orthogonalized, so the grid need not 
be rectangular.

If the Orbitals option is selected, the cube filename (or cube filename and 
cube specification input) is immediately followed by a list of the orbitals 
to evaluate, in free-format, terminated by a blank line. In addition to 
numbers for the orbitals (with alpha orbitals numbered starting at N+l), the
following abbreviations can appear in the list:

HOMO             The highest occupied molecular orbital
LUMO             The lowest unoccupied molecular orbital
OCCA             All occupied (alpha) orbitals
OCCB             All beta occupied orbitals for UHF
ALL              All orbitals
VALENCE          All occupied non-core orbitals
VIRTUALS         All virtual orbitals


OUTPUT FILE FORMATS

Using the default input to Cube produces an unformatted output file (you can 
use the cubman utility to convert it to a formatted version if you so desire; 
see chapter 5). When the Cards option is specified, then the IFlag parameter's
sign determines the output file type. If IFlag>0, the output is unformatted. If
IFlag<0, the output is formatted. All values in the cube file are in atomic units,
regardless of the input units.

For density and potential grids, unformatted files have one row per record 
(i.e., N1 * N2 records each of length N3). For formatted output, each row is 
written out in format (6E13.5). In this case, if N3 is not a multiple of six,
then there may be blank space in some lines.

The norm of the density gradient is also a scalar (i.e., one value per point),
and is written out in the same manner. Density+gradient grids are similar, but
with two writes for each row (of lengths N3 and 3*N3). Density+gradient+Laplacian
grids have 3 writes per row (of lengths N3, 3*N3, and N3).

For example, for a density cube, the output file looks like this:

NAtoms, X-Origin, Y-Origin, Z-Origin
N1, X1, Y1, Z1              # of increments in the slowest running direction
N2, X2, Y2, Z2
N3, X3, Y3, Z3              # of increments in the fastest running direction
IA1, Chgl, X1, Y1, Z1       Atomic number, charge, and coordinates of thefirst atom
...
IAn, Chgn, Xn, Yn, Zn       Atomic number, charge, and coordinates of the last atom

(N1 * N2) records, each of length N3  Values of the density at each point in the grid

Note that a separate write is used for each record.


For molecular orbital output, NAtoms will be less than zero, and an additional 
record follows the data for the final atom (in format lOI5 if the file is formatted):

NMO,  ( MO ( I ), I = 1, NMO )                      Number of MOs and their numbers

If NMO orbitals were evaluated, then each record is NMo * N3 long and has the 
values for all orbitals at each point together.

READING CUBE FILES WITH FORTRAN PROGRAMS

If one wishes to read the values of the density or potential back into an 
array dimensioned X(N3,N2,N1) code like the following Fortran loop may be used:

      Do 10 I1 = 1,N1
      Do 10 I2 = 1,N2
         Read(n,'(6E13.5)') (X(I3,I2,I1),I3=1,N3)
10    Continue

where n is the unit number corresponding to the cube file.

If the origin is (X0,Y0,Z0), and the increments (X1,Y1,Z1), then point 
(I1,I2,I3) has the coordinates:

X-co0rdinate: X0+(I1-1)*X1+(I2-1)*X2+(I3-1)* X3
Y-coordinate: Y0+(I1-1)*Y1+(I2-1)*Y2+(I3-1)* Y3
Z-coordinate: Z0+(I1-1)*Z1+(I2-1)*z2+(I3-1)* Z3

The output is similar if the gradient or gradient and Laplacian of the charge 
density are also requested, except that in these cases there are two or three 
records, respectively, written for each pair of I1, I2 values. Thus, if the 
density, gradient, and Laplacian are to be read into arrays D(N3,N2,N1),
G(3,N3,N2,N1), RL(N3,N2,N1) from a formatted output file, a correct set of 
Fortran loops would be:

      Do 10 I1 = 1, N1
      Do 10 I2 = 1, N2
        Read(n,'(6F13.5)') (D(I3,I2,I1),I3=1,N3)
        Read(n,'(6F13.5)') ((G(IXYZ,I3,I2,I1),IXYZ=1,3), I3=1,N3)
        Read(n,'(6F13.5)') (RL(I3,I2,I1),I3=1,N3)
10    Continue

where again n is the unit number corresponding to the cube file.


OPTIONS

Density      Compute just the density values. This is the default.

Potential    Compute the electrostatic potential at each point.

Gradient     Compute the density and gradient.

Laplacian    Compute the density, gradient, and Laplacian ofthe density 
             ((nabla)2(rho)).

NormGradient Compute the norm of the density gradient at each point.

Orbitals     Compute the values of one or more molecular orbitals at each point.

FrozenCore   Remove the SCF core density. This is the default for the density,
             and is not allowed for the potential.

Full         Evaluate the density including all electrons.

Total        Use the total density. This is the default.

Alpha        Use only the alpha spin density.

Beta         Use only the beta spin density.

Spin         Use the spin density (difference between alpha and beta densities).

Cards        Read grid specification from the input stream (as described above).

Example 1: Produce the total electron density.

x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
#p rhf/6-31g* 5d test geom=modela  cube=(density,read) FormCheck=OptCart

Gaussian Test Job 257:
Density cube

0,1
o h f

input.cube
  -51        -2.0        -2.0        -1.0
   40         0.1         0.0         0.0
   40         0.0         0.1         0.0
   20         0.0         0.0         0.1
	   
x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-


Example 2: Produce the data to plot molecular orbitals

x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
#p rhf/6-31g* 5d test geom=modela  cube=(orbitals) FormCheck=OptCart

Gaussian Test Job 257:
Density cube

0,1
o h f

input.cube
  -51        -2.0        -2.0        -1.0
   40         0.1         0.0         0.0
   40         0.0         0.1         0.0
   20         0.0         0.0         0.1
ALL

x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-


!!!!!!!!!!!!!!!!!!O B S E R V E!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
The cube data has to be given as an orthogonal x-, y-, z-coordinate system
in such a way that the x-axis comes first and the z-axis is given as the 
last one. This means that x is the slowest running coordinate and z is the
fastes running coordinate.

This program produces also a coordinate file in the CHARMM 'crd' format
which can be read by gOpenMol or SCARECROW.

!!!!!!!!!!!!!!!!!!O B S E R V E!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


*/
#include "c_defs.h"

#include <stdio.h>
#include <ctype.h>
#include <sys/types.h>
#ifdef HAVE_MALLOC
#   include <malloc.h>
#endif
#include <stdlib.h>
#include <string.h>

#define BUFF_LEN        256
#define BOHR_RADIUS     0.52917715  /* conversion constant */
#define GAUSSIAN_TYPE   200
#define MAX_TITLE_LINES   5
#define Rabs(a)    ( ( a ) > 0.0 ? (a) : -(a))
#define SMALL 1.e-05

#define FWRITE(value_p , size)    { Items = \
                                 fwrite(value_p, size , 1 , Output_p);\
                                 if(Items < 1) {\
                     printf("?ERROR - in writing contour file (*)\n");\
                     return(1);}}

#define FWRITEN(value_p , num , size) { Items = \
                                fwrite(value_p, size , num , Output_p);\
                   if(Items < 1) {\
                     printf("?ERROR - in writing contour file (**)\n");\
                     return(1);}}

/* ................................................................... */
char *Usage = 
"This program converts the Gaussian output using the 'cube'\n\
 command to a form understandable to gopenmol.\n\
 Usage:\n\
 gcube2plt -iinput.file -ooutput.file\n\
 Options:  -mXXX , where XXX is the molecular orbital number to be\n\
                   placed in the plot file\n\
           -p      prevent the output of the coordinate file\n\
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\
 The 'Cube' input for GaussianXX has to be defined as an orthogonal\n\
 coordinate system, where the x coordinate is the slowest running and\n\
 and z the fastest running coordinate\n";
/* ................................................................... */

   char *AtomSymbols={"\
Ac  Ag  Al  Am  As  Au  B   Ba  Be  Bi  Br  C   Ca  Cd  \
Ce  Cl  Co  Cr  Cs  Cu  D   Dy  Er  Eu  F   Fe  Ga  Gd  \
Ge  H   Hf  Hg  Ho  I   In  Ir  K   La  Li  Lu  Mg  Mn  \
Mo  N   Na  Nb  Nd  Ni  Np  O   Os  P   Pa  Pb  Pd  Pm  \
Po  Pr  Pt  Pu  Ra  Rb  Re  Rh  Ru  S   Sb  Sc  Se  Si  \
Sm  Sn  Sr  Ta  Tb  Tc  Te  Th  Ti  Tl  Tm  U   V   W   \
Y   Yb  Zn  Zr  He  Ne  Ar  Kr  Xe  Rn  "};


    int AtomSymbol_p[] =
       { 89 , 47 , 13 , 95 , 33 , 79 , 5  , 56 , 4  , 83 , 35 , 6  , 20 , 48 ,
         58 , 17 , 27 , 24 , 55 , 29 , 0  , 66 , 68 , 63 , 9  , 26 , 31 , 64 ,
         32 , 1  , 72 , 80 , 67 , 53 , 49 , 77 , 19 , 57 , 3  , 71 , 12 , 25 ,
         42 , 7  , 11 , 41 , 60 , 28 , 93 , 8  , 76 , 15 , 91 , 82 , 46 , 61 ,
         84 , 59 , 78 , 94 , 88 , 37 , 75 , 45 , 44 , 16 , 51 , 55 , 34 , 14 ,
         62 , 50 , 38 , 73 , 73 , 65 , 43 , 90 , 22 , 81 , 69 , 92 , 23 , 74 ,
         39 , 70 , 30 , 40 , 2  , 10 , 18 , 36 , 54 , 86};


/* input */
char TitleText[MAX_TITLE_LINES][BUFF_LEN];
int TitleLines;                /* actual number of title lines */
int Natoms;                    /* number of atoms */
float Xorig,Yorig,Zorig;       /* x-, y- and z- origin */
int N1;                        /* # if incs in the slowest running direct */
float N1X1,N1Y1,N1Z1;          /* direction */
int N2;
float N2X1,N2Y1,N2Z1;          /* direction */
int N3;                        /* # if incs in the fastest running direct */
float N3X1,N3Y1,N3Z1;          /* direction */
int *IA;                       /* atomic number arraypointer */
float *Chgn;                   /* charge array pointer */
float *XC,*YC,*ZC;             /* coordinate pointers  */
float *Data;                   /* data */
int    MolecularOrbitals = 0;  /* switch to handle molecular orbitals */
int   *MolecularOrbital_p;     /* index pointer */
int    MolecularOrbitalsDefined; /* orbitals defined in file */
int    MolecularOrbital2Plot;
int    ToSave;                    /* index into the orbital array */

/* output */
float Xmax,Ymax,Zmax;
int   TypeOfData = GAUSSIAN_TYPE;
int   ProduceCoordinateFile = 1;

char InputFile[BUFF_LEN];
char OutputFile[BUFF_LEN];
char CoordinateFile[BUFF_LEN];

/* functions */
void MakeOutputFileName(char *);
int  ReadInputData(void);
int  WriteInputData(void);
int  WriteCoordinateFile(void);

/* externals */
extern char *Number2Name(int);

/**************************************************************************/
long int F77_FUNC(gcube2plt,GCUBE2PLT) ( const char * filename, const int * length )
/**************************************************************************/
{
    int i;

    printf("**********************************************************\n");
    printf("gCube2Plt conversion  ( .cube --> .plt )                \n\n");

    for ( i = 0; ( i < *length ) && ( i < 248 ); i ++ ) {
        InputFile[ i ] = filename[ i ];
        OutputFile[ i ] = filename[ i ];
        CoordinateFile[ i ] = filename[ i ];
    }
    InputFile[ i ] = '\0';
    OutputFile[ i ] = '\0';
    CoordinateFile[ i ] = '\0';

    strncat( InputFile , ".cube" , 5 ) ;
    strncat( OutputFile , ".plt" , 4 ) ;
    strncat( CoordinateFile , ".crd" , 4 ) ;

    printf("Input file:      '%s'\n",InputFile);
    printf("Output file (plot file):  '%s'\n",OutputFile);
    printf("Coordinate file (in CHARMM 'crd' format): '%s'\n\n",CoordinateFile);

/*  Process the data ... */

    if( ReadInputData() ) {
       printf("$ERROR - can't read input data\n");
       exit(1);
    }

    WriteCoordinateFile();
    WriteInputData();

    printf("Job done ...\n");
    printf("**********************************************************\n");
    fflush( stdout ) ;

    return 0L;
}


/**************************************************************************/
int ReadInputData()
/**************************************************************************/
{
    FILE *Input_p;
    char  Text[BUFF_LEN];
    int   i,j,k,l,ijk,ijkl;
    int   IsInteger;
    char  Temp1[BUFF_LEN];
    char  Temp2[BUFF_LEN];
    char  Temp3[BUFF_LEN];
    char  Temp4[BUFF_LEN];
    int   Hit;
    int   tatoms;
    float txorig;
    float tyorig;
    float tzorig;

    Input_p = fopen(InputFile , "r");
    if(Input_p == NULL) {
      printf("$ERROR - can't open input file '%s'\n",InputFile);
      return(1);
    }

   /* @@@ printf("\nTitle in file (job title):\n"); */
/* first comes an unknow number of title lines (MAX lines is 5) */
    TitleLines = 0;
    for(i = 0 ; i < MAX_TITLE_LINES ; i++) {
       fgets(Text,BUFF_LEN,Input_p);
       sscanf(Text,"%s %s %s %s",Temp1,Temp2,Temp3,Temp4);
/* if Temp1 is an integer and Temp2,temp3,temp4 are floats then the title
   lines are all read */
   IsInteger = 1;
   for(j = 0 ; j < strlen(Temp1) ; j++) {
      if(isalpha(Temp1[j])) {
         IsInteger = 0;
         break;
      }
   }
   if(!IsInteger) {
       printf("%s",Text);
       strncpy(TitleText[i],Text,BUFF_LEN);
       TitleText[i][strlen(TitleText[i]) - 1] = '\0';
       TitleLines++;}
   else
       break;
   }

/* now starts the data ... */
   sscanf(Text,"%d %f %f %f",&Natoms,&Xorig,&Yorig,&Zorig);
    Xorig *= BOHR_RADIUS;
     Yorig *= BOHR_RADIUS;
      Zorig *= BOHR_RADIUS;
    printf("Number of atoms: %d, x-, y-, z-origin (in Angstrom): %f,%f,%f\n",
            abs(Natoms),Xorig,Yorig,Zorig);

   MolecularOrbitals                = 0;
   if(Natoms < 0) MolecularOrbitals = 1;

   Natoms = abs(Natoms);

   fscanf(Input_p,"%d %f %f %f",&N1,&N1X1,&N1Y1,&N1Z1);
    N1X1 *= BOHR_RADIUS;
     N1Y1 *= BOHR_RADIUS;
      N1Z1 *= BOHR_RADIUS;
    /* @@@ printf("Number of points: %d, in direction (x,y,z) %f %f %f\n",
            N1,N1X1,N1Y1,N1Z1); */

/* check that this is a 'pure' x-coordinate */
   if(Rabs(N1X1) < SMALL) {
     printf("$ERROR - most likely your step in the x-direction (%f) is too small\n",N1X1);
     exit(20);}
   if(Rabs(N1Y1) > SMALL || Rabs(N1Z1) > SMALL) {
     printf("$ERROR - first input has to be pure x-axis (y: %f , z: %f)\n",
            N1Y1,N1Z1);
     exit(21);}

   fscanf(Input_p,"%d %f %f %f",&N2,&N2X1,&N2Y1,&N2Z1);
    N2X1 *= BOHR_RADIUS;
     N2Y1 *= BOHR_RADIUS;
      N2Z1 *= BOHR_RADIUS;
    /* @@@ printf("Number of points: %d, in direction (x,y,z) %f %f %f\n",
            N2,N2X1,N2Y1,N2Z1); */

/* check that this is a 'pure' y-coordinate */
   if(Rabs(N2Y1) < SMALL) {
     printf("$ERROR - most likely your step in the y-direction (%f) is too small\n",N2Y1);
     exit(22);}
   if(Rabs(N2X1) > SMALL || Rabs(N2Z1) > SMALL) {
     printf("$ERROR - second input has to be pure y-axis (x: %f , z: %f)\n",
            N2X1,N2Z1);
     exit(23);}

   fscanf(Input_p,"%d %f %f %f",&N3,&N3X1,&N3Y1,&N3Z1);
    N3X1 *= BOHR_RADIUS;
     N3Y1 *= BOHR_RADIUS;
      N3Z1 *= BOHR_RADIUS;
   /* @@@ printf("Number of points: %d, in direction (x,y,z) %f %f %f\n",
            N3,N3X1,N3Y1,N3Z1); */

/* check that this is a 'pure' z-coordinate */
   if(Rabs(N3Z1) < SMALL) {
     printf("$ERROR - most likely your step in the z-direction (%f) is too small\n",N3Z1);
     exit(24);}
   if(Rabs(N3X1) > SMALL || Rabs(N3Y1) > SMALL) {
     printf("$ERROR - last input has to be pure z-axis (x: %f , y: %f)\n",
            N3X1,N3Y1);
     exit(25);}

   IA = (int *)malloc(sizeof(int) * Natoms);
     if(IA == NULL) exit(10);
   Chgn = (float *)malloc(sizeof(float) * Natoms);
     if(Chgn == NULL) exit(11);
   XC = (float *)malloc(sizeof(float) * Natoms);
     if(XC == NULL) exit(12);
   YC = (float *)malloc(sizeof(float) * Natoms);
     if(YC == NULL) exit(13);
   ZC = (float *)malloc(sizeof(float) * Natoms);
     if(ZC == NULL) exit(14);

/* atoms ... */
   /* @@@ printf("Atoms...\n"); */
   for(i = 0 ; i < Natoms ; i++)  {
    /* @@@ fscanf(Input_p,"%d %f %f %f %f",&IA[i],&Chgn[i],&XC[i],&YC[i],&ZC[i]); */
    fscanf(Input_p,"%d %e %e %e %e",&IA[i],&Chgn[i],&XC[i],&YC[i],&ZC[i]); 
     XC[i] *= BOHR_RADIUS;
      YC[i] *= BOHR_RADIUS;
       ZC[i] *= BOHR_RADIUS;
    /* @@@ printf("Atomic number: %d, charge: %f, coord (x,y,z): %f %f %f\n",
            IA[i],Chgn[i],XC[i],YC[i],ZC[i]); */

   }

   if(MolecularOrbitals) {    /* 1 */
/* molecular orbitals to handle ... */
   fscanf(Input_p,"%d",&MolecularOrbitalsDefined);

   MolecularOrbital_p = (int *)malloc(sizeof(int) * MolecularOrbitalsDefined);
   if(MolecularOrbital_p == NULL) exit(15);

   for(i = 0 ; i < MolecularOrbitalsDefined ; i++) {
       fscanf(Input_p,"%d",&MolecularOrbital_p[i]);
   }

   Hit = 0;
   for(i = 0 ; i < MolecularOrbitalsDefined ; i++) {
       if(MolecularOrbital_p[i] == MolecularOrbital2Plot) {
       Hit = MolecularOrbital2Plot;
       ToSave = i;
       break;}
   }

   if(!Hit) {
   printf("$ERROR - Specified orbital nr: %d is not in the file\n",
          MolecularOrbital2Plot);
   exit(16);}

   Data = (float *)malloc(sizeof(float) * N1 * N2 * N3 * 
                                          MolecularOrbitalsDefined);
     if(Data == NULL) exit(17);

   for(i = 0   ; i < N1 ; i++)  { /* 2 */
    for(j = 0  ; j < N2 ; j++)  { /* 3 */
     for(k = 0 ; k < N3 ; k++)  { /* 4 */
      for(l = 0; l < MolecularOrbitalsDefined ; l++) { /* 5 */

     ijkl = i + N1 * j + N1 * N2 * k + N1 * N2 * N3 * l;

     /* @@@ ijk = fscanf(Input_p,"%f",&Data[ijkl]);  */
     ijk = fscanf(Input_p,"%e",&Data[ijkl]);

     if(ijk != 1) {
       printf("$ERROR - in reading the grid data\n");
       printf("$ERROR - at ijkl: %d %d %d %d\n",i,j,k,l);
       exit(18);
     }
       }/* 5 */
      } /* 4 */
     }  /* 3 */
    }   /* 2 */
   } /* end *1* */
   else {  /* start *1* */

   Data = (float *)malloc(sizeof(float) * N1 * N2 * N3);
     if(Data == NULL) exit(19);

   for(i = 0   ; i < N1 ; i++)  { /* 2 */
    for(j = 0  ; j < N2 ; j++)  { /* 3 */
     for(k = 0 ; k < N3 ; k++)  { /* 4 */

     ijk = i + N1 * j + N1 * N2 * k;

     /* @@@ ijkl = fscanf(Input_p,"%f",&Data[ijk]); */
     ijkl = fscanf(Input_p,"%e",&Data[ijk]);
     if(ijkl != 1) {
       printf("$ERROR - in reading the grid data\n");
       printf("$ERROR - at ijk: %d %d %d\n",i,j,k);
       exit(20);
     }
     } /* 4 */
    }  /* 3 */
   }   /* 2 */
  } /* end *1* */
   fclose(Input_p);

   return(0);
}

/**************************************************************************/
int WriteInputData()
/**************************************************************************/
{
    FILE *Output_p;
    int   i,j,k,l,ijk,ijkl;
    int Items;
    float Help1;
    float Help2;
    float MinV;
	float MaxV;

    Output_p = fopen(OutputFile,"wb");
    if(Output_p == NULL) {
      printf("$ERROR - can't open output file: '%s' \n",OutputFile);
      return(1);
    }

	MinV =  1.0e+25;
	MaxV = -1.0e+25;

    i = 3;
    FWRITE(&i,sizeof(int));
    FWRITE(&TypeOfData , sizeof(int));

      FWRITE(&N3   , sizeof(int));
       FWRITE(&N2  , sizeof(int));
        FWRITE(&N1 , sizeof(int));

      Help1  = Zorig;
       Help2 = Zorig + (N3 - 1) * N3Z1;
        FWRITE(&Help1 , sizeof(float));
        FWRITE(&Help2 , sizeof(float));
      Help1  = Yorig;
       Help2 = Yorig + (N2 - 1) * N2Y1;
        FWRITE(&Help1 , sizeof(float));
        FWRITE(&Help2 , sizeof(float));
      Help1  = Xorig;
       Help2 = Xorig + (N1 - 1) * N1X1;
        FWRITE(&Help1 , sizeof(float));
        FWRITE(&Help2 , sizeof(float));

   if(MolecularOrbitals) {         

   for(k = 0   ; k < N3 ; k++)  {  /* 2 */
    for(j = 0  ; j < N2 ; j++)  {  /* 3 */
     for(i = 0 ; i < N1 ; i++)  {  /* 4 */

     ijk = i + N1 * j + N1 * N2 * k + N1 * N2 * N3 * ToSave;

     if(Data[ijk] < MinV) MinV = Data[ijk];
     if(Data[ijk] > MaxV) MaxV = Data[ijk];

     FWRITE(&Data[ijk] , sizeof(float));
     }                             /* 4 */
    }                              /* 3 */
   }                               /* 2 */
   }
   else {
   for(k = 0   ; k < N3 ; k++)  {
    for(j = 0  ; j < N2 ; j++)  { 
     for(i = 0 ; i < N1 ; i++)  {

     ijk = i + N1 * j + N1 * N2 * k;

     if(Data[ijk] < MinV) MinV = Data[ijk];
     if(Data[ijk] > MaxV) MaxV = Data[ijk];

     FWRITE(&Data[ijk] , sizeof(float));
     }
    }
   }
 }
   fclose(Output_p);

   printf("Min value: %f\n",MinV);
   printf("Max value: %f\n",MaxV);

   return(0);
}

/**************************************************************************/
int WriteCoordinateFile()
/**************************************************************************/
{
    int   i,j;
    FILE *coord_p;
    char  AtomName[3];

    coord_p = fopen(CoordinateFile,"w");
    if(coord_p == NULL) {
      printf("$ERROR - can't open output file: '%s'",CoordinateFile);
      exit(1);
    }

    fprintf(coord_p,"* ++Automatic output generated from Gaussian++\n");
    fprintf(coord_p,"* ++using the 'cube' keyword                ++\n");

    for(i = 0 ; i < TitleLines ; i++) 
        fprintf(coord_p,"* %.70s\n",TitleText[i]);

    fprintf(coord_p,"*  \n");
    fprintf(coord_p,"%5d \n",Natoms);

    for(i = 0 ; i < Natoms ; i++) {

      for(j = 0 ; j < strlen(AtomSymbols)/4 ; j++) {
         if(IA[i] == AtomSymbol_p[j]) { 
         strncpy(AtomName,&AtomSymbols[4 * j],2);
         AtomName[2] = '\0';
         break;}
      }
      fprintf(coord_p,
      "%5d%5d %-4.4s %-4.4s%10.5f%10.5f%10.5f %4.4s %-4d%10.5f \n",
      (i+1),1,"GAUS",AtomName,XC[i],YC[i],ZC[i],"GAUS",1,0.0);
      }

    fclose(coord_p);

    return(0);

}

