#! /bin/bash
#
# Au chain, 4 atoms
#
#================================================================
#
# Input flags for this script (./run.sh FLAG): 
#
MANUAL=" Usage
   run.sh [FLAG]

 where FLAG is one of the following 
 (no FLAG will print this manual page) :
 
 scf             DFT self-consistent calculation
 nscf            DFT non-self-consistent calculation
 proj            atomic projections
 dft             perform SCF, NSCF, PROJ, all together

 dos             compute DOS using atmproj wfcs
 conductor       compute COND using atmproj wfcs
 want            all WANT calculations
 all             perform all the above described steps

 check           check results with the reference outputs
 clean           delete all output files and the temporary directory
"
#
#================================================================
#

#
# source common enviroment
. ../environment.conf
#
# source low level macros for test
. ../../script/libtest.sh

#
# macros
SUFFIX="_Au4"

#
# evaluate the starting choice about what is to run 

SCF=
NSCF=
PROJ=
DOS=
CONDUCTOR=
CHECK=
CLEAN=

if [ $# = 0 ] ; then echo "$MANUAL" ; exit 0 ; fi
INPUT=`echo $1 | tr [:upper:] [:lower:]`

case $INPUT in 
   (scf)            SCF=yes ;;
   (nscf)           NSCF=yes ;;
   (proj)           PROJ=yes ;;
   (dft)            SCF=yes ; NSCF=yes ; PROJ=yes ;;
   (dos)            DOS=yes ;;
   (conductor)      CONDUCTOR=yes ;;
   (want)           DOS=yes ; CONDUCTOR=yes ;;
   (all)            SCF=yes ; NSCF=yes ; PROJ=yes ;
                    DOS=yes ; CONDUCTOR=yes ;;
   (check)          CHECK=yes ;;
   (clean)          CLEAN=yes ;;
   (*)              echo " Invalid input FLAG, type ./run.sh for help" ; exit 1 ;;
esac

#
# switches
#
if [ "$PLOT_SWITCH" = "no" ] ; then PLOT=".FALSE." ; fi


#
# initialize 
#
if [ -z "$CLEAN" ] ; then
   test_init 
fi
#


#-----------------------------------------------------------------------------

#
# running DFT SCF
#
run_dft  NAME=SCF   SUFFIX=$SUFFIX  RUN=$SCF

#
# running DFT NSCF
#
run_dft  NAME=NSCF  SUFFIX=$SUFFIX  RUN=$NSCF
   
#
# running DFT PROJ
#
if [ "$PROJ" = "yes" ] ; then
   #
   run  NAME="PROJ"  EXEC=$QE_BIN/projwfc.x  INPUT=proj$SUFFIX.in \
        OUTPUT=proj$SUFFIX.out PARALLEL=yes
fi

#
# running DFT BANDS_DFT
#
run_dft  NAME=BANDS_DFT  SUFFIX=$SUFFIX  RUN=$BANDS_DFT
   

#
# running BANDS
#
run_bands  SUFFIX=$SUFFIX  RUN=$BANDS

#
# running DOS
#
run_dos  SUFFIX=$SUFFIX  RUN=$DOS

#
# running CONDUCTOR
#
run_conductor SUFFIX=$SUFFIX  RUN=$CONDUCTOR


#
# running CHECK
#
if [ "$CHECK" = yes ] ; then
   echo "running CHECK... "
   #
   cd $TEST_HOME
   list="disentangle$SUFFIX.out wannier$SUFFIX.out"
   #
   for file in $list
   do
      ../../script/check.sh $file
   done
fi


#
# eventually clean
#
run_clean  RUN=$CLEAN


#
# exiting
exit 0


