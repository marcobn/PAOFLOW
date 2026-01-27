#! /bin/sh 
# generates the file include/build_date.h

# run from directory where this script is
cd `echo $0 | sed 's/\(.*\)\/.*/\1/'` # extract pathname
# come back to WanT HOME
cd ..  
#
TOPDIR=`pwd`
TARGETDIR=$TOPDIR/include
TARGETFILE=$TARGETDIR/build_date.h

if [ -e "$TARGETFILE" ] ; then rm $TARGETFILE ; fi
#
cat > $TARGETFILE << EOF

!
! Copyright (C) 2010 WanT Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file "License"
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

#define   __CONF_BUILD_DATE           "`date '+%c'`"

EOF

exit 0


