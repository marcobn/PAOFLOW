#! /bin/bash 
#
# Test general running utility 
#================================================================
#
# Input flags for this script (./run.sh FLAG): 
#
MANUAL=" Usage
   run.sh [-h] [-r <flag>] [<test_dirs>] 

 run the action <flag> on the <test_dirs> list 
 if <flag> is not present the action is assumed as "ALL" while
 if the <test_list> is missing ALL tests are included.
 When the command line is empty the following manual page is printed:
 
 help            print this manual    
 info            print detailed info about the implemented tests
 all             perform all the calculations
 dft             perform dft calculations only
 check           check results with the reference outputs
 update_ref      update reference results with the current output files
 clean           delete all output files and the temporary directories

 Go in the specific Test dirs in order to have a more detailed menu.
"
#
#================================================================
#

ALLOWED_ACTION="help info all dft want check update_ref clean"
ACTION=
LIST=

while getopts :hr: OPT
do
  case $OPT in
  (r) ACTION="$OPTARG" ; shift 2;;
  (h) echo "$MANUAL" ; exit 0 ;;
  (:) echo "error: $OPTARG requires an argument" ; exit 1 ;;
  (?) echo "error: unkwown option $OPTARG" ; exit 1 ;;
  esac
done
 
LIST="$*"

if [ -z "$LIST" -a -z "$ACTION" ] ; then echo "$MANUAL" ; exit 0 ; fi
if [ -z "$ACTION" ] ; then ACTION="all" ; fi
if [ -z "$LIST" ] ; then LIST=$(ls -d test*/ ) ; fi

tmp=$( echo $ACTION | tr [:upper:] [:lower:] )
ACTION=$tmp

# final call to help
if [ "$ACTION" = "help" ] ; then
   echo "$MANUAL" 
   exit 0
fi

FOUND=
for allowed in $ALLOWED_ACTION
do  
    if [ "$ACTION" = "$allowed" ] ; then FOUND="yes" ; fi
done
if [ -z "$FOUND" ] ; then
   echo "error: unkwown action = $ACTION"
   exit 2
fi


#
# input summary
#
echo "  run ACTION: " $ACTION
echo "     on LIST: " $LIST
echo 


#
# perform the required task
#
for mytest in $LIST
do
        echo "  dir $mytest does not exist "
        exit 3
    fi
    cd $mytest
    
    #
    # info
    #
    if [ "$ACTION" = "info" ] ; then
        str="$(grep @title@ README 2> /dev/null)"
        echo "${mytest%\/}     ${str#@title@ }"
        
    #
    # update_ref
    #
    elif [ "$ACTION" = "update_ref" ] ; then
        cp *.out *.dat Reference 2> /dev/null
        echo " ### $mytest : Reference updated ### " 
    #
    # other flags
    #
    else
       #
       # get different run script
       #
       SCRIPT_LIST=$( ls run*.sh  2> /dev/null )
       if [ -z "$SCRIPT_LIST" ] ; then 
           echo " ### nothing to do for $mytest ### " 
           echo
       else
           for script in $SCRIPT_LIST
           do
              echo " ### $mytest : $script  $ACTION ### " 
              ./$script $ACTION
              if [ "$ACTION" != "clean" ] ; then echo ; fi
           done
       fi
    fi
    cd ..
done
echo


exit 0





