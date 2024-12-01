#! /bin/bash
#
# script to manage the lowlevel launch of codes
# needed by the test suite
#

# get script homedir
LOCAL_DIR=`echo $0 | sed 's/\(.*\)\/.*/\1/'` # extract pathname

# source environment
source $LOCAL_DIR/../environment.conf

# source base def
source $LOCAL_DIR/../../script/basedef.sh

# set redirection
case $INPUT_TYPE in
 ("from_stdin")       INPUT_REDIRECT="<" ;;
 ("from_file")        INPUT_REDIRECT="-input" ;;
 (*)                  INPUT_REDIRECT="$INPUT_TYPE" ;;
esac

# few basic definitions
TEST_HOME=$(pwd)
TEST_NAME=$(echo $TEST_HOME | awk -v FS=\/ '{print $NF}' )


#
#----------------------
test_init () {
#----------------------
#
   
   #
   # exit if TMPDIR dies not exist
   if [ ! -d $TMPDIR ] ; then 
       echo "TMPDIR = $TMPDIR   does not exist " ; exit 71 
   fi

   #
   # if the case, create local test dir
   test -d $TMPDIR/$TEST_NAME || mkdir $TMPDIR/$TEST_NAME
   #
   # create SCRATCH link
   test -e $TEST_HOME/SCRATCH && rm $TEST_HOME/SCRATCH
   cd $TEST_HOME
   ln -sf $TMPDIR/$TEST_NAME ./SCRATCH
   #
   # create HOME link
   test -e $TMPDIR/$TEST_NAME/HOME && rm $TMPDIR/$TEST_NAME/HOME
   cd $TMPDIR/$TEST_NAME
   ln -sf $TEST_HOME ./HOME
   #
   test -e $TMPDIR/$TEST_NAME/CRASH && rm $TMPDIR/$TEST_NAME/CRASH
   test -e $TEST_HOME/CRASH && rm $TEST_HOME/CRASH
   #
   cd $TEST_HOME

}

#
#----------------------
exit_if_no_etsf_support () {
#----------------------
#
   make_sys_file="$TEST_HOME/../../make.sys"
   #
   if [ ! -e "$make_sys_file" ] ; then 
      echo "ERROR: make.sys not present" ; exit 10 
   fi
   #
   str=`grep '__ETSF_IO' $make_sys_file`
   #
   if [ -z "$str" ] ; then
      echo "no ETSF-IO support... exit "
      exit 0
   fi
}


#
#----------------------
run_clean () {
#----------------------
#
   local RUN=

   for arg
   do
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
   done

   [[ "$RUN" != "yes" ]]  && return

   #
   # actual clean up
   #
   cd $TEST_HOME
      rm -rf *.out *.dat 2> /dev/null
      test -e SCRATCH && rm SCRATCH
      test -e CRASH   && rm CRASH

   cd $TMPDIR
      test -d $TEST_NAME && rm -rf $TEST_NAME
   
}


#
#----------------------
run () {
#----------------------
#
# low level tool to launch generic executables
#
   local NAME=
   local EXEC=
   local INPUT=
   local OUTPUT=
   local PARALLEL=
   local INPUT_TYPE_LOC=$INPUT_TYPE

   for arg 
   do
         [[ "$arg" == NAME=* ]]        && NAME="${arg#NAME=}"
         [[ "$arg" == EXEC=* ]]        && EXEC="${arg#EXEC=}"
         [[ "$arg" == INPUT=* ]]       && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]      && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == PARALLEL=* ]]    && PARALLEL="${arg#PARALLEL=}"
         [[ "$arg" == INPUT_TYPE=* ]]  && INPUT_TYPE_LOC="${arg#INPUT_TYPE=}"
   done

   if [ -z "$NAME" ]   ; then echo "empty NAME card"   ; exit 1 ; fi 
   if [ -z "$EXEC" ]   ; then echo "empty EXEC card"   ; exit 1 ; fi 
   if [ -z "$INPUT" ]  ; then echo "empty INPUT card"  ; exit 1 ; fi 
   if [ -z "$OUTPUT" ] ; then echo "empty OUTPUT card" ; exit 1 ; fi 
   
   if [ ! -x $EXEC ] ; then
      #
      echo "$EXEC not executable... exit "
      exit 0
      #
   fi

   if [ ! -z $NAME ] ; then
      #
      echo $ECHO_N "running $NAME calculation... $ECHO_C"
      #
   fi

   #
   if [ "$INPUT_TYPE_LOC" = "from_stdin" ] ; then
      #
      if [ "$PARALLEL" = "yes" ] ; then
         $PARA_PREFIX $EXEC $PARA_POSTFIX < $INPUT > $OUTPUT
      else
         $EXEC < $INPUT > $OUTPUT
      fi
   fi
   #
   if [ "$INPUT_TYPE_LOC" = "from_file" ] ; then
      #
      if [ "$PARALLEL" = "yes" ] ; then
         $PARA_PREFIX $EXEC $PARA_POSTFIX -input $INPUT > $OUTPUT
      else
         $EXEC -input $INPUT > $OUTPUT
      fi
   fi
   #
   if [ $? = 0 ] ; then
      if [ ! -z $NAME ] ; then echo "$ECHO_T done" ; fi
   else
      echo "$ECHO_T problems found" ; exit 1
   fi

}


#
#----------------------
run_dft () {
#----------------------
#
   local NAME=DFT
   local EXEC=$QE_BIN/pw.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done

   [[ "$RUN" != "yes" ]]  && return
   
   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/$name_tmp$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/$name_tmp$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_proj () {
#----------------------
#
   local NAME=PROJ
   local EXEC=$QE_BIN/projwfc.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done

   [[ "$RUN" != "yes" ]]  && return
   
   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/$name_tmp$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/$name_tmp$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}

#
#----------------------
run_abinit () {
#----------------------
#
   local NAME=DFT_ABINIT
   local EXEC=$ABINIT_BIN/abinip
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done

   [[ "$RUN" != "yes" ]]   &&  return
   [[ -z "$PARA_PREFIX" ]] &&  PARALELL=no
   #
   if [ "$PARALLEL" = "yes" ] ; then
      if [ -x $ABINIT_BIN/abinip ] ; then EXEC=$ABINIT_BIN/abinip ; fi
      if [ -x $ABINIT_BIN/abinit ] ; then EXEC=$ABINIT_BIN/abinit ; fi
   else
      if [ -x $ABINIT_BIN/abinis ] ; then EXEC=$ABINIT_BIN/abinis ; fi
      if [ -x $ABINIT_BIN/abinit ] ; then EXEC=$ABINIT_BIN/abinit ; fi
   fi
   
   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/$name_tmp$SUFFIX.files  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/$name_tmp$SUFFIX.log ; fi
   OUTPUT_INT=$TEST_HOME/$name_tmp$SUFFIX.out

   test -e $OUTPUT_INT && rm $OUTPUT_INT
   #
   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL \
       INPUT_TYPE="from_stdin"
}


#
#----------------------
run_export () {
#----------------------
#
 
   local NAME=EXPORT
   local EXEC=$QE_BIN/pw_export.x
   local RUN=yes
   local SUFFIX=
   local INPUT=
   local OUTPUT=
   local PARALLEL=yes
   #
   local lpara_prefix
   local lpara_postfix
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done

   [[ "$RUN" != "yes" ]]  && return

   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/pwexport$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/pwexport$SUFFIX.out ; fi

   if [ "$PARALLEL" = "yes" ] ; then
      lpara_prefix=$PARA_PREFIX
      lpara_postfix=$PARA_POSTFIX
   fi
   
   echo $ECHO_N "running $NAME calculation... $ECHO_C"
   
   if [ "$INPUT_TYPE" = "from_stdin" ] ; then
       $lpara_prefix $EXEC $lpara_postfix < $INPUT > $OUTPUT 2> /dev/null
   elif [ "$INPUT_TYPE" = "from_file" ] ; then
       $lpara_prefix $EXEC $lpara_postfix -input $INPUT > $OUTPUT 2> /dev/null
   else
       echo "$ECHO_T Invalid INPUT_TYPE = $INPUT_TYPE" ; exit 1 
   fi
   #
   if [ $? = 0 ] ; then
      echo "${ECHO_T}done"
   else
      echo "$ECHO_T problems found" ; exit 1
   fi
}


#
#----------------------
run_disentangle () {
#----------------------
#
   local NAME=DISENTANGLE
   local EXEC=$WANT_BIN/disentangle.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done

   [[ "$RUN" != "yes" ]]  && return
   
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/want$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/disentangle$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_wannier () {
#----------------------
#
   local NAME=WANNIER
   local EXEC=$WANT_BIN/wannier.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/want$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/wannier$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_bands () {
#----------------------
#
   local NAME=BANDS
   local EXEC=$WANT_BIN/bands.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/bands$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/bands$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_dos () {
#----------------------
#
   local NAME=DOS
   local EXEC=$WANT_BIN/dos.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/dos$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/dos$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_blc2wan () {
#----------------------
#
   local NAME=BLC2WAN
   local EXEC=$WANT_BIN/blc2wan.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/blc2wan$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/blc2wan$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_unfold () {
#----------------------
#
   local NAME=UNFOLD
   local EXEC=$WANT_BIN/unfold.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/unfold$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/unfold$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_plot () {
#----------------------
#
   local NAME=PLOT
   local EXEC=$WANT_BIN/plot.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/plot$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/plot$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#----------------------
run_conductor () {
#----------------------
#
   local NAME=CONDUCTOR
   local EXEC=$WANT_BIN/conductor.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/conductor$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/conductor$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}

#
#----------------------
run_embed () {
#----------------------
#
   local NAME=EMBED
   local EXEC=$WANT_BIN/embed.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local PARALLEL=yes
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
         [[ "$arg" == PARALLEL=* ]]  && PARALLEL="${arg#PARALLEL=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/embed$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/embed$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=$PARALLEL
}


#
#
#----------------------
run_current () {
#----------------------
#
   local NAME=CURRENT
   local EXEC=$WANT_BIN/current.x
   local RUN=yes
   local INPUT=
   local OUTPUT=
   local SUFFIX=
   local name_tmp
   
   for arg 
   do
         [[ "$arg" == NAME=* ]]      && NAME="${arg#NAME=}"
         [[ "$arg" == INPUT=* ]]     && INPUT="${arg#INPUT=}"
         [[ "$arg" == OUTPUT=* ]]    && OUTPUT="${arg#OUTPUT=}"
         [[ "$arg" == SUFFIX=* ]]    && SUFFIX="${arg#SUFFIX=}"
         [[ "$arg" == RUN=* ]]       && RUN="${arg#RUN=}"
   done
   
   [[ "$RUN" != "yes" ]]  && return

   name_tmp=`echo $NAME | tr [:upper:] [:lower:]`
   if [ -z "$INPUT" ]  ; then  INPUT=$TEST_HOME/current$SUFFIX.in  ; fi
   if [ -z "$OUTPUT" ] ; then OUTPUT=$TEST_HOME/current$SUFFIX.out ; fi

   run NAME=$NAME INPUT=$INPUT OUTPUT=$OUTPUT EXEC=$EXEC PARALLEL=no
}


