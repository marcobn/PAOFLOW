! 
! Copyright (C) 2011 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
!
# include "f_defs.h"
! 
!**********************************************************
   SUBROUTINE memusage( iunit )
   !**********************************************************
   !
   ! This subroutine writes a summary fo the memory 
   ! currently allocated in the data modules.
   !
   USE io_module,                ONLY : ionode 
   USE kinds,                    ONLY : dbl
   !
   USE T_kpoints_module,         ONLY : kpoints_memusage, kpoints_alloc => alloc
   USE T_smearing_module,        ONLY : smearing_memusage, smear_alloc => alloc
   USE T_hamiltonian_module,     ONLY : hamiltonian_memusage, ham_alloc => alloc
   USE T_workspace_module,       ONLY : workspace_memusage, workspace_alloc => alloc
   !
   IMPLICIT NONE
   !
   INTEGER, INTENT(IN) :: iunit
   !
   REAL(dbl) :: memsum, mtmp
   !
#ifdef HAVE_MALLINFO_FALSE
   INTEGER  :: tmem
#endif

100 FORMAT ( 4x, a20,':', f15.3, ' MB')

   memsum = 0.0_dbl
   !
   IF ( ionode ) THEN
      ! 
      WRITE( iunit, "( ' <MEMORY_USAGE>' )" ) 
      !
      IF ( smear_alloc ) THEN
          mtmp    =  smearing_memusage()
          memsum  =  memsum + mtmp
          WRITE(iunit, 100) "smearing", mtmp
      ENDIF
      IF ( kpoints_alloc ) THEN
          mtmp    =  kpoints_memusage()
          memsum  =  memsum + mtmp
          WRITE(iunit, 100) "kpoints", mtmp
      ENDIF
      IF ( ham_alloc ) THEN
          mtmp    =  hamiltonian_memusage("ham")
          memsum  =  memsum + mtmp
          WRITE(iunit, 100) "hamiltonian data", mtmp
      ENDIF
      IF ( ham_alloc ) THEN
          mtmp    =  hamiltonian_memusage("corr")
          memsum  =  memsum + mtmp
          WRITE(iunit, 100) "correlation data", mtmp
      ENDIF
      IF ( workspace_alloc ) THEN
          mtmp    =  workspace_memusage()
          memsum  =  memsum + mtmp
          WRITE(iunit, 100) "workspace", mtmp
      ENDIF
      !
      !
      WRITE( iunit, "()")
      WRITE( iunit, 100 ) "Total alloc. Memory", memsum

#ifdef HAVE_MALLINFO_FALSE
      CALL memstat( tmem )
      WRITE( iunit, 100 ) " Real alloc. Memory",  REAL( tmem )/ 1000.0_dbl
#endif

      WRITE( iunit, "( ' </MEMORY_USAGE>',/ )" ) 
      !
   ENDIF


END SUBROUTINE memusage

