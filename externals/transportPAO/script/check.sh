#! /bin/bash 
#
# Check to results of selected tests
#================================================================
#
MANUAL=" Usage
   check.sh [-h] [-v] [<file1>...<fileN>]

 Report a difference pattern between the selected <file>'s
 and their reference counterparts in the dir ./Reference .
 The script should be run in a specific Test dir.

"
#
#================================================================
#

VERBOSITY=low

while getopts :hv OPT
do
  case $OPT in
  (v) VERBOSITY="high" ; shift 1;;
  (h) echo "$MANUAL" ; exit 0 ;;
  (:) echo "error: $OPTARG requires an argument" ; exit 1 ;;
  (?) echo "error: unkwown option $OPTARG" ; exit 1 ;;
  esac
done

#
# account for environment variables
BIN="../../script/"
#
test -e ../environment.conf && . ../environment.conf
test -e $BIN/basedef.sh && . $BIN/basedef.sh
#
if [ "$VERBOSITY_LEVEL" = "high" ] ; then
   VERBOSITY=$VERBOSITY_LEVEL
fi
#echo "check VERBOSITY: $VERBOSITY"


#
# set file list
LIST=$*
if [ -z "$LIST" ] ; then echo "$MANUAL" ; exit 0 ; fi


#
# other checks
if [ ! -d "./Reference" ] ; then 
   echo "error: Reference directory not found"
   exit 1
fi

#-----------------------------
print_header () {
   #-----------------------------
   # 
   # used to format header printing
   #
   printf "%s" $boldon
#   printf "%s" "[1;32;40m"
   printf "\n%-15s\t%12s\t%12s\t%12s\n\n" "Variable" "This File" "Reference" "% Difference"
   printf "%s" $boldoff
}

#-----------------------------
printout () {
   #-----------------------------
   #
   # write results of the check
   # USAGE:
   #      printout $name, $val, $val_ref, $toll
   #
   if [ $# != 3 -a $# != 4 ] ; then 
      echo "ERROR: invalid syntax in printout"; exit 1
   fi
   #
   #
   echo $1 $2 $3 $4 | awk '
       { 
           #  
           # here is a simple awk script to format results  
           #  

           #
           # first few definitions about colors
           green_on="[1;32;48m"
           green_off="[0m"
           orange_on="[1;33;48m"
           orange_off="[0m"
           red_on="[1;31;48m"
           red_off="[0m"
           #
           stat_ok   = green_on"ok"green_off ;
           stat_warn = orange_on"warning"orange_off ;
           stat_fail = red_on"failed"red_off ;

           #
           # deal with data
           name=$1;
           val=$2;
           val_ref=$3;
           toll=$4;
           #
           if ( name == "iteration" ) {
              #
              diff = val - val_ref;
              rel_diff = sqrt ( ( diff / val_ref ) * ( diff / val_ref ) );
              #
              stat = stat_fail
              if ( rel_diff < toll ) {
                 stat = stat_ok
              } else if ( rel_diff < 5.0 * toll ) {
                 stat = stat_warn
              }
              #
              printf( "%-15s\t%12i\t%12i\t%12.6f\t%10s\n", name, val, val_ref, rel_diff, stat );
              #
           } else if ( name == "status" ) { 
              #
              printf( "%-15s\t%12s\t%12s\n", name, val, val_ref );
              #
           } else {
              #
              diff = val - val_ref;
              #
              if ( val_ref == 0.0 ) {
                   rel_diff = diff;
              } else {
                   rel_diff = sqrt ( ( diff / val_ref ) * ( diff / val_ref ) );
              }
              #
              stat = stat_fail
              if ( rel_diff < toll ) {
                 stat = stat_ok
              } else if ( rel_diff < 5.0 * toll ) {
                 stat = stat_warn
              }
              #
              printf( "%-15s\t%12.6f\t%12.6f\t%12.6f\t%10s\n", name, val, val_ref, rel_diff, stat );
              #
           }
        }'
}


#
# main loop over files to be checked
#
for file in $LIST
do
    
   echo 
   echo "######################### File: $file "
   #
   if [ ! -e $file ] ; then 
      #
      echo "   $file not found: skipped"  
      #
   elif [ ! -e ./Reference/$file ] ; then 
      #
      echo "   ./Reference/$file not found: skipped"  
      #
   else
      #
      print_header
      #
      OUT_FILE=$( $BIN/parse_output.awk $file )
      OUT_REF=$( $BIN/parse_output.awk ./Reference/$file )

      #
      # print each value found
      #
      for item in $OUT_FILE
      do
          name=$( echo $item | cut -f1 -d"@" )
           val=$( echo $item | cut -f2 -d"@" )
          toll=$( echo $item | cut -f3 -d"@" )

          #
          # search the reference value
          #
          for item_ref in $OUT_REF
          do 
              name_ref=$( echo $item_ref | cut -f1 -d"@" )
              if [ "$name_ref" = "$name" ] ; then 
                 val_ref=$( echo $item_ref | cut -f2 -d"@" )
              fi
          done

          #
          # print
          #
          name_tmp=$( echo $name | tr [:upper:] [:lower:] )
          name=$name_tmp
          #
          # avoid to have empty TOLL
          if [ -z "$toll" ] ; then toll=@@ ; fi
          #
          printout $name $val $val_ref $toll
          #
      done

   fi
done
echo


