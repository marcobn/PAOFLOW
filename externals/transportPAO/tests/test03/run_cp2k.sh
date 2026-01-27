#! /bin/bash
#
# transport through Au chains
#
#================================================================
#
# Input flags for this script (./run.sh FLAG): 
#
MANUAL=" Usage
   run.sh [FLAG]

 where FLAG is one of the following 
 (no FLAG will print this manual page) :
 

 conductor_Au01           transport through a Au chain, 1 atom per cell
 conductor_Au08_cell1     the same, using data with 8 Au atoms to 
                          describe a cell of 1 Au atom
 conductor_Au08_cell4     the same, using data with 8 Au atoms to 
                          describe a cell of 4 Au atoms
 conductor_Au16_cell1  
 conductor_Au16_cell4  
 conductor_Au16_cell8     as above 

 embed_Au08_cell1         embedding procedure, using a cell with 8 Au atoms
 embed_Au16_cell1         the same as above, 16 atoms per cell

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
SUFFIX=""

#
# evaluate the starting choice about what is to run 

#CONDUCTOR_Au01=
CONDUCTOR_Au08_cell1=
CONDUCTOR_Au08_cell4=
CONDUCTOR_Au16_cell1=
CONDUCTOR_Au16_cell4=
CONDUCTOR_Au16_cell8=
EMBED_Au08_cell1=
EMBED_Au16_cell1=
CHECK=
CLEAN=

if [ $# = 0 ] ; then echo "$MANUAL" ; exit 0 ; fi
INPUT=`echo $1 | tr [:upper:] [:lower:]`

case $INPUT in 
#   (conductor_au01)           CONDUCTOR_Au01=yes ;;
   (conductor_au08_cell1)     CONDUCTOR_Au08_cell1=yes ;;
   (conductor_au08_cell4)     CONDUCTOR_Au08_cell4=yes ;;
   (conductor_au16_cell1)     CONDUCTOR_Au16_cell1=yes ;;
   (conductor_au16_cell4)     CONDUCTOR_Au16_cell4=yes ;;
   (conductor_au16_cell8)     CONDUCTOR_Au16_cell8=yes ;;
   (embed_au08_cell1)         EMBED_Au08_cell1=yes ;;
   (embed_au16_cell1)         EMBED_Au16_cell1=yes ;;

   (want,all)       #CONDUCTOR_Au01=yes ; 
                    CONDUCTOR_Au08_cell1=yes ; CONDUCTOR_Au08_cell4=yes ;
                    CONDUCTOR_Au16_cell1=yes ; CONDUCTOR_Au16_cell4=yes ; CONDUCTOR_Au16_cell8=yes ;
                    EMBED_Au08_cell1=yes ; EMBED_Au16_cell1=yes ;;
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
# running CONDUCTOR
#
#run_conductor  NAME=CONDUCTOR_Au01        SUFFIX="_Au01"        RUN=$CONDUCTOR_Au01
run_conductor  NAME=CONDUCTOR_Au08_CELL1  SUFFIX="_Au08_cell1"  RUN=$CONDUCTOR_Au08_cell1
run_conductor  NAME=CONDUCTOR_Au08_CELL4  SUFFIX="_Au08_cell4"  RUN=$CONDUCTOR_Au08_cell4
run_conductor  NAME=CONDUCTOR_Au16_CELL1  SUFFIX="_Au16_cell1"  RUN=$CONDUCTOR_Au16_cell1
run_conductor  NAME=CONDUCTOR_Au16_CELL4  SUFFIX="_Au16_cell4"  RUN=$CONDUCTOR_Au16_cell4
run_conductor  NAME=CONDUCTOR_Au16_CELL8  SUFFIX="_Au16_cell8"  RUN=$CONDUCTOR_Au16_cell8


#
# running CHECK
#
if [ "$CHECK" = yes ] ; then
   echo "running CHECK... "
   #
   cd $TEST_HOME
   list=" "
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
rm -rf data/*.ham
#


#
# exiting
exit 0


