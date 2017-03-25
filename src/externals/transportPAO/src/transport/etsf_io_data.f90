!
! Copyright (C) 2009 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************************
   MODULE etsf_io_data_module
   !*********************************************************
   !
   USE kinds,             ONLY : dbl
   USE mp,                ONLY : mp_bcast
   !
#ifdef __ETSF_IO
   USE etsf_io
#endif
   !
   IMPLICIT NONE
   PRIVATE
   SAVE
!
! This module contains data specific to ETSF_IO fmt
! (used eg by Abinit)
!
! The whole interface to ETSF_IO has been done with
! the contribution of Conor Hogan
!

   !
   ! general data used to readin
   !
   LOGICAL                      :: lstat
   INTEGER                      :: ncid
   !
   PUBLIC :: lstat, ncid
   !
#ifdef __ETSF_IO
   !
   TYPE(etsf_io_low_error)      :: error_data
   TYPE(etsf_groups_flags)      :: flags
   
   !
   ! dimensions, as defined by ETSF specifications
   !
   TYPE(etsf_dims)              :: dims


   !
   ! interfaces
   !
   INTERFACE etsf_io_bcast
       MODULE PROCEDURE etsf_io_bcast_dims
   END INTERFACE
   

   PUBLIC :: error_data
   PUBLIC :: flags
   PUBLIC :: dims
   !
   PUBLIC :: etsf_io_bcast

CONTAINS

!**********************************************************
   SUBROUTINE etsf_io_bcast_dims( dims_,  ionode_id )
   !**********************************************************
   IMPLICIT NONE
   !
   TYPE( etsf_dims ),    INTENT(INOUT) :: dims_
   INTEGER,              INTENT(IN)    :: ionode_id
   !
   CALL mp_bcast ( dims_%character_string_length, ionode_id ) 
   CALL mp_bcast ( dims_%max_number_of_angular_momenta, ionode_id )
   CALL mp_bcast ( dims_%max_number_of_basis_grid_points, ionode_id )
   CALL mp_bcast ( dims_%max_number_of_coefficients, ionode_id )
   CALL mp_bcast ( dims_%max_number_of_projectors, ionode_id )
   CALL mp_bcast ( dims_%max_number_of_states, ionode_id )
   CALL mp_bcast ( dims_%number_of_atoms, ionode_id )
   CALL mp_bcast ( dims_%number_of_atom_species, ionode_id )
   CALL mp_bcast ( dims_%number_of_cartesian_directions, ionode_id )
   CALL mp_bcast ( dims_%number_of_components, ionode_id )
   CALL mp_bcast ( dims_%number_of_grid_points_vector1, ionode_id )
   CALL mp_bcast ( dims_%number_of_grid_points_vector2, ionode_id )
   CALL mp_bcast ( dims_%number_of_grid_points_vector3, ionode_id )
   CALL mp_bcast ( dims_%number_of_kpoints, ionode_id )
   CALL mp_bcast ( dims_%number_of_localization_regions, ionode_id )
   CALL mp_bcast ( dims_%number_of_reduced_dimensions, ionode_id )
   CALL mp_bcast ( dims_%number_of_spinor_components, ionode_id )
   CALL mp_bcast ( dims_%number_of_spins, ionode_id )
   CALL mp_bcast ( dims_%number_of_symmetry_operations, ionode_id )
   CALL mp_bcast ( dims_%number_of_vectors, ionode_id )
   CALL mp_bcast ( dims_%real_or_complex_coefficients, ionode_id )
   CALL mp_bcast ( dims_%real_or_complex_density, ionode_id )
   CALL mp_bcast ( dims_%real_or_complex_gw_corrections, ionode_id )
   CALL mp_bcast ( dims_%real_or_complex_potential, ionode_id )
   CALL mp_bcast ( dims_%real_or_complex_wavefunctions, ionode_id )
   CALL mp_bcast ( dims_%symbol_length, ionode_id )
   !
   ! Dimensions for variables that can be splitted.
   !
   CALL mp_bcast ( dims_%my_max_number_of_coefficients, ionode_id )
   CALL mp_bcast ( dims_%my_max_number_of_states, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_components, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_grid_points_vect1, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_grid_points_vect2, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_grid_points_vect3, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_kpoints, ionode_id )
   CALL mp_bcast ( dims_%my_number_of_spins, ionode_id )
   !
END SUBROUTINE etsf_io_bcast_dims

#endif

END MODULE etsf_io_data_module
