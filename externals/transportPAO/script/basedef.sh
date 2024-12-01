#! /bin/sh

#
# portable definitions for echo command
# from autoconf configure
#
case `echo "testing\c"; echo 1,2,3`,`echo -n testing; echo 1,2,3` in
  *c*,-n*) ECHO_N= ECHO_C='
' ECHO_T='      ' ;;
  *c*,*  ) ECHO_N=-n ECHO_C= ECHO_T= ;;
  *)       ECHO_N= ECHO_C='\c' ECHO_T= ;;
esac

#
# check whether we need echo -e
# 
if [ "`echo -e`" = "-e" ] ; then
   ECHO_E=
else
   ECHO_E='-e'
fi

#
# definitions for bold on and bold off
boldon="[1m"
boldoff="[0m"

#
#newline
newline="
"


