#! /bin/sh
# compute dependencies for the WanT directory tree

# run from directory where this script is
cd `echo $0 | sed 's/\(.*\)\/.*/\1/'` # extract pathname
# come back to WanT HOME
cd ..  
#
TOPDIR=`pwd`
BINDIR=$TOPDIR/config

DIR_LIST="fftqe baselib wannier transport tools embed"
SPECIAL_MODULES="etsf_io  etsf_io_tools  etsf_io_low_level \
                 mkl_dfti.f90  iotk_module iotk_base  iotk_error_interf \
                 phigemm  magma  iso_c_binding \
                 dynamicalq  environment  io_dyn_mat io_global
                 "

for DIR in $DIR_LIST
do
    # set inter-directory dependencies
    case $DIR in
        fftqe )     DEPENDS="../../include"            ;;
        baselib )   DEPENDS="../../include ../fftqe"   ;;
        wannier )   DEPENDS="../../include ../fftqe ../baselib" ;;
        transport ) DEPENDS="../../include ../fftqe ../baselib ../wannier " ;;
        tools )     DEPENDS="../../include ../fftqe ../baselib ../wannier " ;;
        embed )     DEPENDS="../../include ../fftqe ../baselib ../wannier ../transport" ;;
    esac

    # generate dependencies file
    if test -d $TOPDIR/src/$DIR
    then
        cd $TOPDIR/src/$DIR
        $BINDIR/moduledep.sh $DEPENDS > make.depend
        $BINDIR/includedep.sh $DEPENDS >> make.depend
    fi

    # handle special cases
    if [ "$DIR" = "baselib" -o "$DIR" = "fftqe" ] ; then
        mv make.depend make.depend.tmp
        sed 's/@fftw.c@/fftw.c/
             s/@fftw2.c@/fftw2.c/'  make.depend.tmp > make.depend
    fi

    # eliminate dependencies on special modules
    for module in $SPECIAL_MODULES
    do
        mv make.depend make.depend.tmp
        grep -v "@$module@" make.depend.tmp > make.depend
    done

    test -e make.depend.tmp && rm make.depend.tmp

    # check for missing dependencies
    if grep @ make.depend
    then
        notfound=1
        echo WARNING: dependencies not found in directory $DIR
    else
        echo directory $DIR : ok
    fi
    #
    # eliminate missing deps to make the script working
    mv make.depend make.depend.tmp
    awk '{
           if ( match($0, "@") ) { 
               print "#", $0 
           } else {
               print 
           }
         }' make.depend.tmp > make.depend
    #
    test -e make.depend.tmp && rm make.depend.tmp
    #
done
#
if test "$notfound" = ""
then
    echo all dependencies updated successfully
fi
