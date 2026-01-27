!
! Copyright (C) 2006 Valerio Olevano, Matthieu Verstraete
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!---------------------------------------------------------------------------
! interface to the Nanoquanta ETSF NetCDF file format
! version 1.0
! Valerio Olevano
! Matthieu Verstraete: obtained from Valerio on 9 Jun 2006
!     last modified 1 Aug 2006
!---------------------------------------------------------------------------
! 1 Aug 2006: the read/write operations have been made into atomic routines
!  read_variable_name(ncid,dim1,dim2,&
!     variable_name)
!     or
!  read_dimension_name(ncid,dimension_name)
!     and
!  write_variable_name(ncid,dim1,dim2,&
!     variable_name)
!    which checks that the appropriate dimensions are present, or defines them
!    to get all the dimid, and calls
!  defput_variable_name(ncid,dim1id,dim2id,&
!     variable_name)
!    which defines the netcdf variable and puts the data.
!---------------------------------------------------------------------------
!  This file contains:
!
! write_wfn_etsf : write all variables and dimensions to a netcdf file
! test_wfn_etsf : get all dimensions and a few integer variables for allocation
! read_wfn_etsf : read all variables and dimensions
! write_denpot_etsf : same for densities/potentials
! test_denpot_etsf : same for densities/potentials
! read_denpot_etsf : same for densities/potentials
!
! write_variable_name* : write all individual variables for denpot and wavefunctions
! defput_variable_name* : define and put data for individual variable
!
! inqordef_dimid : with a given dimension name, inquires about its existence, and
!   defines it with the input value if it does not exist. Could also check for
!   coherence with the input value (TODO)
!
!  read_variable_name* : read in individual variables for denpot and  wavefunctions
!  read_dimension_name* : read in individual dimensions
!
!  read_dim : generic read of an individual dimension 
!------------ENDCOMMENTS-----------------ENDCOMMENTS------------------------

!===========================================================================
subroutine write_wfn_etsf(filename,fnlen,title,generating_code_and_version, &
  number_of_symmetry_operations,max_number_of_coefficients,max_number_of_states, &
  number_of_kpoints,number_of_spins,number_of_components,number_of_spinor_components, &
  number_of_atom_species,number_of_atoms,max_number_of_angular_momenta,max_number_of_projectors,&
  k_dependent_gvectors,k_dependent_number_of_states, &
  space_group,primitive_vectors, &
  atomic_numbers,valence_charges,pseudopotential_types,number_of_electrons,&
  atom_species,reduced_atom_positions, &
  reduced_symmetry_matrices,reduced_symmetry_translations, &
  reduced_coordinates_of_kpoints,kpoint_weights,monkhorst_pack_folding,kpoint_grid_vectors,kpoint_grid_shift,&
  number_of_coefficients,gvectors,gvectors_k,number_of_states, &
  kinetic_energy_cutoff,total_energy,fermi_energy,max_residual, &
  exchange_functional,correlation_functional, &
  smearing_width,smearing_scheme, &
  occupations,eigenvalues,coefficients_of_wavefunctions,&
  gw_corrections,kb_formfactor_sign,kb_formfactors,kb_formfactor_derivative)
  ! or put the arguments of variables you wish to write
  
  implicit none
  
  ! input variables
  integer,intent(in) :: fnlen
  character (len=fnlen), intent(in) :: filename ! = "file.etsf"
  character (len=*), intent(in) :: title ! = "Silicon bulk. Si 1s Corehole, etc."
  character (len=*), intent(in) :: generating_code_and_version ! = "Milan-CP 9.3.4 "

  ! parameters
  
  integer, parameter :: real_or_complex = 2
  integer, parameter :: number_of_cartesian_directions = 3
  integer, parameter :: number_of_reduced_dimensions = 3
  integer, parameter :: number_of_vectors = 3
  
  integer, intent(in) :: number_of_symmetry_operations
  integer, intent(in) :: number_of_atom_species
  integer, intent(in) :: number_of_atoms
  integer, intent(in) :: max_number_of_angular_momenta
  integer, intent(in) :: max_number_of_projectors
  integer, intent(in) :: number_of_kpoints
  integer, intent(in) :: max_number_of_states
  integer, intent(in) :: max_number_of_coefficients
  integer, intent(in) :: number_of_spinor_components
  integer, intent(in) :: number_of_spins
  integer, intent(in) :: number_of_components

  integer, parameter :: character_string_length = 80
  integer, parameter :: symbol_length = 2

  
  
  double precision, intent(in) :: primitive_vectors(number_of_cartesian_directions,number_of_vectors)
  integer, intent(in) :: space_group

  double precision, intent(in) :: atomic_numbers(number_of_atom_species)
  double precision, intent(in) :: valence_charges(number_of_atom_species)
  character (len=character_string_length) :: atom_species_names(number_of_atom_species)
  character (len=symbol_length) :: chemical_symbols(number_of_atom_species)
  character (len=character_string_length), intent(in) :: pseudopotential_types(number_of_atom_species)
  integer, intent(in) :: atom_species(number_of_atoms)
  double precision, intent(in) :: reduced_atom_positions(number_of_reduced_dimensions,number_of_atoms)
  integer, intent(in) :: number_of_electrons
  
  integer, intent(in) :: reduced_symmetry_matrices(number_of_reduced_dimensions,number_of_reduced_dimensions,&
                                                   number_of_symmetry_operations)
  double precision, intent(in) :: reduced_symmetry_translations(number_of_reduced_dimensions,&
                                                                number_of_symmetry_operations)
  
  ! Reciprocal space, G-space, k-space
  character (len=character_string_length) :: basis_set = "plane_waves"

  character (len=*), intent(in) :: k_dependent_gvectors          ! = "no"  ! or "yes"
  character (len=*), intent(in) :: k_dependent_number_of_states  ! = "no"  ! or "yes"
  double precision, intent(in) :: reduced_coordinates_of_kpoints(number_of_reduced_dimensions,number_of_kpoints)
  double precision, intent(in) :: kpoint_weights(number_of_kpoints)
  double precision, intent(in) :: kpoint_grid_shift(number_of_reduced_dimensions)
  double precision, intent(in) :: kpoint_grid_vectors(number_of_reduced_dimensions,number_of_vectors)
  integer, intent(in) :: monkhorst_pack_folding(number_of_vectors)
  integer, intent(in) :: number_of_coefficients(number_of_kpoints)
  integer, intent(in) :: gvectors(number_of_vectors,max_number_of_coefficients)
  integer, intent(in) :: gvectors_k(number_of_vectors,max_number_of_coefficients,number_of_kpoints)
  integer, intent(in) :: number_of_states(number_of_kpoints)
  
  ! Convergency data
  double precision, intent(in) :: kinetic_energy_cutoff
  double precision, intent(in) :: total_energy
  double precision, intent(in) :: fermi_energy
  double precision, intent(in) :: max_residual
  character (len=*), intent(in) :: exchange_functional ! = "LDA"
  character (len=*), intent(in) :: correlation_functional ! = "PBE"
  
  ! Electronic structure  
  double precision, intent(in) :: smearing_width
  character (len=*), intent(in) :: smearing_scheme ! = "Fermi-Dirac"
  
  double precision, intent(in) :: occupations(max_number_of_states,number_of_kpoints,number_of_spins)
  double precision, intent(in) :: eigenvalues(max_number_of_states,number_of_kpoints,number_of_spins)
  double precision, intent(in) :: coefficients_of_wavefunctions(real_or_complex,max_number_of_coefficients, &
                      number_of_spinor_components,max_number_of_states,number_of_kpoints, &
                      number_of_spins)

  double precision,intent(in) :: gw_corrections(real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins)

  integer,intent(in) :: kb_formfactor_sign(max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species)
  double precision,intent(in) :: kb_formfactors(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                     max_number_of_angular_momenta,number_of_atom_species)
  double precision,intent(in) :: kb_formfactor_derivative(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                     max_number_of_angular_momenta,number_of_atom_species)
  
  
  ! local variables
  character (len=200) :: history
  character (len=12) :: date, time, zone

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid
  ! dimensions identifiers
  integer :: ncdimid
  
  character (len=symbol_length), parameter, dimension(18) :: periodic_table = &
    (/ "H ",                              "He", &
       "Li","Be","B ","C ","N ","O ","F ","Ne", &
       "Na","Mg","Al","Si","P ","S ","Cl","Ar"  /)
  
  do s = 1, number_of_atom_species
    chemical_symbols(s) = periodic_table(int(atomic_numbers(s)+0.5))
  enddo
  atom_species_names(:) = chemical_symbols(:)
  
  ! TEST INPUT

  if (number_of_spins /= 1 .and. number_of_spins /= 2) then
    print *, 'writetsf : Error: number_of_spins must be 1 or 2 '
    print *, number_of_spins
    stop
  end if
  if (number_of_components /= 1 .and. number_of_components /= 2 &
      .and. number_of_components /= 4) then
    print *, 'writetsf : Error: number_of_components must be 1, 2, or 4 '
    print *, number_of_components
    stop
  end if

  if (number_of_spinor_components /= 1 .and. number_of_spinor_components /= 2 ) then
    print *, 'writetsf : Error: number_of_spinor_components must be 1 or 2 '
    print *, number_of_spinor_components
    stop
  end if
  
  ! OPENING
  ! we can use here NF90_64BIT_OFFSET for versions > 3.6 to allow >4GB many variables
  ! s = nf90_create(filename,nf90_64bit_offset,ncid)
  s = nf90_create(filename,nf90_clobber,ncid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

  ! GENERAL AND GLOBAL ATTRIBUTES
  s = nf90_put_att(ncid,nf90_global,"file_format","ETSF Nanoquanta")
  s = nf90_put_att(ncid,nf90_global,"file_format_version",1.0)
  s = nf90_put_att(ncid,nf90_global,"Conventions","http://www.etsf.eu/fileformats/")
  s = nf90_put_att(ncid,nf90_global,"title",title)
  call date_and_time(date,time,zone)
  history = generating_code_and_version // '   ' // &
    date(1:4) // '-' // date(5:6) // '-' // date(7:8) // '  ' // &
    time(1:2) // ':' // time(3:4) // ':' // time(5:6) // '  ' // &
    zone
  s = nf90_put_att(ncid,nf90_global,"history",history)

! this dimension is not used for the moment: define it anyway
s = nf90_def_dim(ncid,"number_of_components",number_of_components,ncdimid)

! make sure we are out of def mode before calling the write routines
s = nf90_enddef(ncid)

! call atomic routines
call write_primitive_vectors(ncid,&
           number_of_cartesian_directions,number_of_vectors,&
           primitive_vectors)
call write_space_group(ncid,space_group)
call write_atomic_numbers(ncid,&
           number_of_atom_species,&
           atomic_numbers)
call write_valence_charges(ncid,&
           number_of_atom_species,&
           valence_charges)
call write_atom_species_names(ncid,&
           character_string_length,number_of_atom_species,&
           atom_species_names)
call write_chemical_symbols(ncid,&
           symbol_length,number_of_atom_species,&
           chemical_symbols)
call write_pseudopotential_types(ncid,&
           character_string_length,number_of_atom_species,&
           pseudopotential_types)
call write_atom_species(ncid,&
           number_of_atoms,&
           atom_species)
call write_reduced_atom_positions(ncid,&
           number_of_reduced_dimensions,number_of_atoms,&
           reduced_atom_positions)
call write_number_of_electrons(ncid,number_of_electrons)
call write_reduced_symmetry_operations(ncid,&
           number_of_reduced_dimensions,number_of_symmetry_operations,&
           reduced_symmetry_matrices,reduced_symmetry_translations)
call write_reduced_coordinates_of_kpoints(ncid,&
           number_of_reduced_dimensions,number_of_kpoints,&
           reduced_coordinates_of_kpoints)
call write_kpoint_weights(ncid,&
           number_of_kpoints,&
           kpoint_weights)
call write_monkhorst_pack_folding(ncid,&
           number_of_vectors,&
           monkhorst_pack_folding)
call write_kpoint_grid_vectors(ncid,&
           number_of_reduced_dimensions,number_of_vectors,&
           kpoint_grid_vectors)
call write_kpoint_grid_shift(ncid,&
           number_of_reduced_dimensions,&
           kpoint_grid_shift)
call write_basis_set(ncid,character_string_length,basis_set)
call write_number_of_coefficients(ncid,&
           number_of_kpoints,&
           k_dependent_gvectors,number_of_coefficients)
call write_gvectors(ncid,&
           number_of_vectors,max_number_of_coefficients,number_of_kpoints,&
           k_dependent_gvectors,gvectors,gvectors_k)
call write_number_of_states(ncid,&
           number_of_kpoints,&
           k_dependent_number_of_states,number_of_states)
call write_kinetic_energy_cutoff(ncid,kinetic_energy_cutoff)
call write_total_energy(ncid,total_energy)
call write_fermi_energy(ncid,fermi_energy)
call write_max_residual(ncid,max_residual)
call write_exchange_functional(ncid,character_string_length,exchange_functional)
call write_correlation_functional(ncid,character_string_length,correlation_functional)
call write_occupations(ncid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           occupations)
call write_smearing_width(ncid,smearing_width)
call write_smearing_scheme(ncid,character_string_length,smearing_scheme)
call write_eigenvalues(ncid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           eigenvalues)
call write_gw_corrections(ncid,&
           real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins,&
           gw_corrections)
call write_kb_formfactor_sign(ncid,&
           max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_sign)
call write_kb_formfactors(ncid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactors)
call write_kb_formfactor_derivative(ncid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_derivative)
call write_coefficients_of_wavefunctions(ncid,&
           real_or_complex,max_number_of_coefficients,number_of_spinor_components,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           coefficients_of_wavefunctions)
  
  ! CLOSING FILE
    
  s = nf90_close(ncid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
end subroutine


! Valerio Olevano, April 2005 
! interface to the .etsf NetCDF file format and file specifications

!*** SHELLS ??? ***

!===========================================================================
! get dimensions from file so we can allocate the arrays
!  something other than "test" might be better for the name...
!===========================================================================
function test_wfn_etsf(filename,fnlen,number_of_symmetry_operations,&
  max_number_of_coefficients,max_number_of_states, &
  number_of_kpoints,number_of_spins,number_of_components,number_of_spinor_components, &
  number_of_atom_species,number_of_atoms,max_number_of_angular_momenta,max_number_of_projectors,&
  k_dependent_gvectors,k_dependent_number_of_states, &
  number_of_electrons)

  implicit none
  integer :: test_wfn_etsf

  ! input variables
  integer,intent(in) :: fnlen
  character (len=fnlen),intent(in) :: filename

  ! Parameters
  integer, parameter :: CSLEN = 80
  integer, parameter :: SLEN = 2
  integer, parameter :: YESLEN = 3
  double precision, parameter :: H2eV = 27.2113845d0
  
  character (len=CSLEN) :: file_format, conventions
  character (len=CSLEN) :: title
  character (len=1024) :: history
  real :: file_format_version

  ! Dimensions
  integer :: real_or_complex = 2
  integer :: number_of_cartesian_directions = 3
  integer :: number_of_reduced_dimensions = 3
  integer :: number_of_vectors = 3
  
  integer,intent(out) :: number_of_symmetry_operations
  integer,intent(out) :: number_of_atom_species
  integer,intent(out) :: number_of_atoms
  integer,intent(out) :: max_number_of_angular_momenta
  integer,intent(out) :: max_number_of_projectors
  integer,intent(out) :: number_of_kpoints
  integer,intent(out) :: max_number_of_states
  integer,intent(out) :: max_number_of_coefficients
  integer,intent(out) :: number_of_spinor_components
  integer,intent(out) :: number_of_spins
  integer,intent(out) :: number_of_components

  integer :: character_string_length = 80
  integer :: symbol_length = 2

  integer :: tl, hl, ffl, cl
  
  integer :: space_group

  integer,intent(out) :: number_of_electrons
  
  
  ! Reciprocal space, G-space, k-space
  character (len=CSLEN) :: basis_set

  character (len=YESLEN),intent(out) :: k_dependent_gvectors          ! = "no"  ! or "yes"
  character (len=YESLEN),intent(out) :: k_dependent_number_of_states  ! = "no"  ! or "yes"
  
  ! Convergency data
  double precision :: kinetic_energy_cutoff
  double precision :: total_energy
  double precision :: fermi_energy
  double precision :: max_residual
  
  ! Electronic structure  
  double precision :: smearing_width
  
  
  ! local variables
  
  integer :: tmplen

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid

  
  print *, 'checking Nanoquanta ETSF NetCDF wavefunction input file ', trim(filename)
  print *

  test_wfn_etsf = -1
  
  s = nf90_open(filename,nf90_nowrite,ncid)
  if(s /= nf90_noerr) then
    test_wfn_etsf = -1
    print *, 'unrecognized NetCDF file'
    return
  endif

  ! general
  s = nf90_inquire_attribute(ncid,nf90_global,"file_format",len=ffl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"file_format",file_format)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  if(file_format(1:15) /= "ETSF Nanoquanta") then
    test_wfn_etsf = -1
    print *, 'unrecognized .etsf NetCDF file'
    return
  endif  
  s = nf90_get_att(ncid,nf90_global,"file_format_version",file_format_version)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  write(*,'(" ",a," version ",f3.1)') file_format(1:ffl), file_format_version
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"Conventions",len=cl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"Conventions",conventions)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, conventions(1:cl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"title",len=tl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"title",title)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, title(1:tl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"history",len=hl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"history",history)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, history(1:hl)
#endif
  

  ! NetCDF dimensions declaration section

  call read_character_string_length(ncid,character_string_length)
  if(character_string_length > CSLEN) print *, 'Warning, character_string_length = ', character_string_length
  call read_symbol_length(ncid,symbol_length)
  if(symbol_length > SLEN) print *, 'Warning, symbol_length = ', symbol_length
  call read_real_or_complex(ncid,real_or_complex)
  call read_number_of_cartesian_directions(ncid,number_of_cartesian_directions)
  call read_number_of_vectors(ncid,number_of_vectors)
  call read_number_of_reduced_dimensions(ncid,number_of_reduced_dimensions)

10  format(' ',a,t35,i6)
20  format(' ',a,t38,a3)
30  format(' ',a,t34,f7.2,' [eV]')

#ifdef CHECKS
  write(*,10) 'manifold dimension ', real_or_complex
  write(*,10) 'number of cartesian directions ', number_of_cartesian_directions
  write(*,10) 'number of vectors ', number_of_vectors
  write(*,10) 'number of reduced dimensions ', number_of_reduced_dimensions
#endif

  call read_number_of_symmetry_operations(ncid,number_of_symmetry_operations)
  call read_number_of_atom_species(ncid,number_of_atom_species)
  call read_number_of_atoms(ncid,number_of_atoms)
  call read_max_number_of_angular_momenta(ncid,max_number_of_angular_momenta)
  call read_max_number_of_projectors(ncid,max_number_of_projectors)

#ifdef CHECKS
  write(*,10) 'number of symmetry operations  ', number_of_symmetry_operations
  write(*,10) 'number of atom species         ', number_of_atom_species
  write(*,10) 'number of atoms                ', number_of_atoms
  write(*,10) 'max number of angular momenta  ', max_number_of_angular_momenta
  write(*,10) 'max number of projectors       ', max_number_of_projectors
#endif

  call read_number_of_kpoints(ncid,number_of_kpoints)
  call read_max_number_of_states(ncid,max_number_of_states)
  call read_max_number_of_coefficients(ncid,max_number_of_coefficients)

#ifdef CHECKS
  write(*,10) 'number of kpoints              ', number_of_kpoints
  write(*,10) 'max number of states           ', max_number_of_states
  write(*,10) 'max number of coefficients     ', max_number_of_coefficients
#endif

  call read_number_of_spinor_components(ncid,number_of_spinor_components)
#ifdef CHECKS
  write(*,10) 'number of spinor components    ', number_of_spinor_components
#endif

  call read_number_of_spins(ncid,number_of_spins)
#ifdef CHECKS
  write(*,10) 'number of spins                ', number_of_spins
#endif

  call read_number_of_components(ncid,number_of_components)
#ifdef CHECKS
  write(*,10) 'number of components           ', number_of_components
#endif


  ! VARIABLES
  
  
  ! Unit Cell Chemical Structure

  call read_number_of_electrons(ncid,number_of_electrons)

#ifdef CHECKS
  write(*,10) 'number of electrons ', number_of_electrons
#endif
  
  ! Reciprocal G-space, plane waves  
  call read_k_dependent_gvectors(ncid,k_dependent_gvectors)

#ifdef CHECKS
  write(*,20) 'k-dependent g-vectors', k_dependent_gvectors
#endif

  ! Number of states
  call read_k_dependent_number_of_states(ncid,k_dependent_number_of_states)

#ifdef CHECKS
  write(*,20) 'k-dependent number of states', k_dependent_number_of_states
#endif
      
  
  s = nf90_close(ncid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
  test_wfn_etsf = 100
  return
  
end function test_wfn_etsf


!===========================================================================
!===========================================================================

subroutine read_wfn_etsf(filename,fnlen,&
  nsym,ng,nw,nb,nk,nas,na,mnam,mnp,&
  space_group,primitive_vectors,number_of_electrons,&
  atom_species,reduced_atom_positions, &
  reduced_symmetry_matrices,reduced_symmetry_translations, &
  reduced_coordinates_of_kpoints,kpoint_weights,gvectors,&
  fermi_energy, &
  occupations,eigenvalues,coefficients_of_wavefunctions,&
  gw_corrections,kb_formfactor_sign,kb_formfactors,kb_formfactor_derivative)

  implicit none

  ! input variables
  !   variables without intent are not accepted as input yet (hardcoded)
  integer,intent(in) :: fnlen
  character (len=fnlen),intent(in) :: filename
  ! input variables tested against values in netcdf file
  !  nsym = number_of_symmetry_operations
  !  ng is the number of g vectors in full sphere, and 
  !  nw the number read in
  !  nb=number_of_bands
  !  nk=number_of_kpoints
  !  nas = number_of_atom_species
  !  na=number_of_atoms
  !  mnam = max_number_of_angular_momenta
  !  mnp = max_number_of_projectors
  integer,intent(in) :: nsym, ng, nw, nb, nk, nas, na, mnam, mnp

  ! Parameters
  integer, parameter :: CSLEN = 80
  integer, parameter :: SLEN = 2
  integer, parameter :: YESLEN = 3
  integer, parameter :: RC = 2
  integer, parameter :: NCD = 3
  integer, parameter :: NRD = 3
  integer, parameter :: NV = 3
  double precision, parameter :: H2eV = 27.2113845d0
  integer, parameter :: NS = 1
  integer, parameter :: NSR = 1
  
  character (len=CSLEN) :: file_format, conventions
  character (len=CSLEN) :: title
  character (len=1024) :: history
  real :: file_format_version

  ! Dimensions
  integer :: real_or_complex = 2
  integer :: number_of_cartesian_directions = 3
  integer :: number_of_reduced_dimensions = 3
  integer :: number_of_vectors = 3
  
  ! these dimensions are local and checked against
  !   the input dimensions of the data variables
  integer :: number_of_symmetry_operations
  integer :: number_of_atom_species
  integer :: number_of_atoms
  integer :: max_number_of_angular_momenta
  integer :: max_number_of_projectors
  integer :: number_of_kpoints
  integer :: max_number_of_states
  integer :: max_number_of_coefficients
  integer :: number_of_spinor_components
  integer :: number_of_spins
  integer :: number_of_components

  integer :: character_string_length = 80
  integer :: symbol_length = 2

  integer :: tl, hl, ffl, cl
  
  integer,intent(out) :: space_group
  double precision,intent(out) :: primitive_vectors(NCD,NV)

  integer,intent(out) :: atom_species(na)
  double precision,intent(out) :: reduced_atom_positions(NRD,na)
  integer,intent(out) :: number_of_electrons
  
  integer,intent(out) :: reduced_symmetry_matrices(NRD,NRD,nsym)
  double precision,intent(out) :: reduced_symmetry_translations(NRD,nsym)
  
  ! Reciprocal space, G-space, k-space
  character (len=CSLEN) :: basis_set

  character (len=YESLEN) :: k_dependent_gvectors          ! = "no"  ! or "yes"
  character (len=YESLEN) :: k_dependent_number_of_states  ! = "no"  ! or "yes"
  double precision,intent(out) :: reduced_coordinates_of_kpoints(NRD,nk)
  double precision,intent(out) :: kpoint_weights(nk)

  integer,intent(out) :: gvectors(NV,ng)
  
  ! Convergency data
  double precision :: kinetic_energy_cutoff
  double precision :: total_energy
  double precision,intent(out) :: fermi_energy
  double precision :: max_residual
  
  ! Electronic structure  
  double precision :: smearing_width

  double precision,intent(out) :: occupations(nb,nk,NS)
  double precision,intent(out) :: eigenvalues(nb,nk,NS)
  double precision,intent(out) :: coefficients_of_wavefunctions(RC,nw,NSR,nb,nk,NS)

  double precision,intent(out) :: gw_corrections(RC,nb,nk,NS)

  integer,intent(out) :: kb_formfactor_sign(mnp,mnam,nas)
  double precision,intent(out) :: kb_formfactors(nw,nk,mnp,mnam,nas)
  double precision,intent(out) :: kb_formfactor_derivative(nw,nk,mnp,mnam,nas)

  
  
  ! local variables
  
  integer :: tmplen

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid
  
  print *, 'reading Nanoquanta ETSF NetCDF wavefunction input file ', trim(filename)
  print *
  
  s = nf90_open(filename,nf90_nowrite,ncid)
  if(s /= nf90_noerr) then
    print *, 'unrecognized NetCDF file'
    return
  endif

  ! general
  s = nf90_inquire_attribute(ncid,nf90_global,"file_format",len=ffl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"file_format",file_format)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  if(file_format(1:15) /= "ETSF Nanoquanta") then
    print *, 'unrecognized .etsf NetCDF file'
    return
  endif  
  s = nf90_get_att(ncid,nf90_global,"file_format_version",file_format_version)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  write(*,'(" ",a," version ",f3.1)') file_format(1:ffl), file_format_version
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"Conventions",len=cl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"Conventions",conventions)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, conventions(1:cl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"title",len=tl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"title",title)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, title(1:tl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"history",len=hl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"history",history)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, history(1:hl)
#endif
  

  ! NetCDF dimensions declaration section

  call read_character_string_length(ncid,character_string_length)
  if(character_string_length > CSLEN) print *, 'Warning, character_string_lenght = ', character_string_length
  call read_symbol_length(ncid,symbol_length)
  if(symbol_length > SLEN) print *, 'Warning, symbol_lenght = ', symbol_length

  call read_real_or_complex(ncid,real_or_complex)
  call read_number_of_cartesian_directions(ncid,number_of_cartesian_directions)
  call read_number_of_vectors(ncid,number_of_vectors)
  call read_number_of_reduced_dimensions(ncid,number_of_reduced_dimensions)
    
10  format(' ',a,t35,i6)
20  format(' ',a,t38,a3)
30  format(' ',a,t34,f7.2,' [eV]')

#ifdef CHECKS
  write(*,10) 'manifold dimension ', real_or_complex
  write(*,10) 'number of cartesian directions ', number_of_cartesian_directions
  write(*,10) 'number of vectors ', number_of_vectors
  write(*,10) 'number of reduced dimensions ', number_of_reduced_dimensions
#endif
  if(real_or_complex /= RC)  stop 'error: real or complex'
  if(number_of_cartesian_directions /= NCD) stop 'error: number of cartesian directions'
  if(number_of_vectors /= NV) stop 'error: number of vectors'
  if(number_of_reduced_dimensions /= NRD) stop 'error: number of reduced dimensions'

  call read_number_of_symmetry_operations(ncid,number_of_symmetry_operations)
  call read_number_of_atom_species(ncid,number_of_atom_species)
  call read_number_of_atoms(ncid,number_of_atoms)
  call read_max_number_of_angular_momenta(ncid,max_number_of_angular_momenta)
  call read_max_number_of_projectors(ncid,max_number_of_projectors)

#ifdef CHECKS
  write(*,10) 'number of symmetry operations  ', number_of_symmetry_operations
#endif
  if(number_of_symmetry_operations < nsym) stop 'error: not enough symmetry operations'
#ifdef CHECKS
  write(*,10) 'number of atom species         ', number_of_atom_species
#endif
  if(number_of_atom_species /= nas) stop 'error: number of atom species'
#ifdef CHECKS
  write(*,10) 'number of atoms                ', number_of_atoms
#endif
  if(number_of_atoms /= na) stop 'error: number of atoms'
#ifdef CHECKS
  write(*,10) 'max number of angular momenta  ', max_number_of_angular_momenta
#endif
  if(max_number_of_angular_momenta /= mnam) stop 'error: max number of angular momenta'
#ifdef CHECKS
  write(*,10) 'max number of projectors       ', max_number_of_projectors
#endif
  if(max_number_of_projectors /= mnp) stop 'error: max number of projectors'

  call read_number_of_kpoints(ncid,number_of_kpoints)
  call read_max_number_of_states(ncid,max_number_of_states)
  call read_max_number_of_coefficients(ncid,max_number_of_coefficients)

#ifdef CHECKS
  write(*,10) 'number of kpoints              ', number_of_kpoints
#endif
  if(number_of_kpoints /= nk) stop 'error: number of k-points'
#ifdef CHECKS
  write(*,10) 'max number of states           ', max_number_of_states
#endif
  if(max_number_of_states < nb) stop 'error: max number of states'
#ifdef CHECKS
  write(*,10) 'max number of coefficients     ', max_number_of_coefficients
#endif
  if(max_number_of_coefficients < ng) stop 'error: max number of coefficients'

  call read_number_of_spinor_components(ncid,number_of_spinor_components)
  call read_number_of_spins(ncid,number_of_spins)
  call read_number_of_components(ncid,number_of_components)

#ifdef CHECKS
  write(*,10) 'number of spinor components    ', number_of_spinor_components
  write(*,10) 'number of spins                ', number_of_spins
  write(*,10) 'number of components           ', number_of_components
#endif
  if(number_of_spinor_components /= 1) stop 'error: cannot deal with spinors'
  if(number_of_spins /= 1) stop 'error: cannot deal with spins'
  ! MJV : is this correct? What is this generic component? Do not remember
  if(number_of_components /= 1) stop 'error: cannot deal with components'

  ! VARIABLES
  
  ! Crystal Structure

call read_space_group(ncid,space_group)

#ifdef CHECKS
  write(*,10) 'space group ', space_group
#endif

call read_primitive_vectors(ncid,NCD,NV,&
     primitive_vectors)
  
#ifdef CHECKS
  write(*,'(" primitive vectors [a.u.]:",3(/3f7.3))') primitive_vectors
#endif
  
  ! Unit Cell Chemical Structure

call read_atom_species(ncid,na,&
           atom_species)

#ifdef CHECKS
  print *, 'atom species ', atom_species
#endif

call  read_reduced_atom_positions(ncid,NRD,na,&
           reduced_atom_positions)

#ifdef CHECKS
  write(*,'(" atom positions [red]:",100(/3f7.3))') reduced_atom_positions
#endif

  call read_number_of_electrons(ncid,number_of_electrons)

#ifdef CHECKS
  write(*,10) 'number of electrons ', number_of_electrons
#endif
  

  ! Symmetries
call read_reduced_symmetry_matrices(ncid,NRD,nsym,&
           reduced_symmetry_matrices)
call read_reduced_symmetry_translations(ncid,NRD,nsym,&
           reduced_symmetry_translations)

#ifdef CHECKS
  print *, 'symop ', reduced_symmetry_matrices
  print *, 'symoptr ', reduced_symmetry_translations
#endif

  ! Brillouin Zone
call read_reduced_coordinates_of_kpoints(ncid,NRD,nk,&
           reduced_coordinates_of_kpoints)

call read_kpoint_weights(ncid,nk,&
     kpoint_weights)

#ifdef CHECKS
  print *, 'kpoints ', reduced_coordinates_of_kpoints
  print *, 'kpoint_weights ', kpoint_weights
#endif
  

  ! Reciprocal G-space, plane waves  

  call read_k_dependent_gvectors(ncid,k_dependent_gvectors)
  if(k_dependent_gvectors(1:1) == 'y' .or. k_dependent_gvectors(1:1) == 'Y') then
    write(*,20) 'k-dependent g-vectors', k_dependent_gvectors
    stop 'error: cannot deal with k-dependent g-vectors'
  endif

  ! Number of states
  call read_k_dependent_number_of_states(ncid,k_dependent_number_of_states)
  if(k_dependent_number_of_states(1:1) == 'y' .or. k_dependent_number_of_states(1:1) == 'Y') then
    write(*,20) 'k-dependent number of states', k_dependent_number_of_states
    stop 'error: cannot deal with k-dependent number of states'
  endif
  
call read_reduced_coordinates_of_plane_waves(ncid,NV,ng,&
           coefficients_of_wavefunctions)
    
#ifdef CHECKS
  print *, 'g-vectors ', gvectors(:,1)
#endif


  ! Convergency data
call read_fermi_energy(ncid,fermi_energy)

#ifdef CHECKS
  write(*,30) 'Fermi energy ', fermi_energy * H2eV
#endif

  ! Electronic Structure
call read_occupations(ncid,nb,nk,NS,&
           occupations)
  
#ifdef CHECKS
  print *, 'occ ', occupations
#endif
  
call read_eigenvalues(ncid,nb,nk,NS,&
           eigenvalues)

#ifdef CHECKS
  print *, 'energies ', eigenvalues
#endif

call read_gw_corrections(ncid,RC,nb,nk,NS,&
           gw_corrections)

#ifdef CHECKS
  print *, 'gw corrections ', gw_corrections
#endif

call  read_kb_formfactor_sign(ncid,mnp,mnam,nas,&
           kb_formfactor_sign)

#ifdef CHECKS
  print *, 'KB formfactor signs ', kb_formfactor_sign
#endif

call read_kb_formfactors(ncid,nw,nk,mnp,mnam,nas,&
           kb_formfactors)

#ifdef CHECKS
  print *, 'KB formfactors ', kb_formfactors(1,1,1,1,1)
#endif

call read_kb_formfactor_derivative(ncid,nw,nk,mnp,mnam,nas,&
           kb_formfactor_derivative)

#ifdef CHECKS
  print *, 'KB formfactor derivative ', kb_formfactor_derivative(1,1,1,1,1)
#endif

call read_coefficients_of_wavefunctions(ncid,RC,nw,NSR,nb,nk,NS,&
            coefficients_of_wavefunctions)

#ifdef CHECKS
  print *, 'wavefunctions ', coefficients_of_wavefunctions(1,1,1,1,1,1)
#endif

  
  s = nf90_close(ncid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
  return

end subroutine read_wfn_etsf


!===========================================================================
!   routines for density/potential files
!===========================================================================
subroutine write_denpot_etsf(filename,fnlen,title,generating_code_and_version,&
  real_or_complex,number_of_components, &
  number_of_grid_points_vector1,number_of_grid_points_vector2,number_of_grid_points_vector3,&
  primitive_vectors,denpot,denpottype)
  
  implicit none
  
  ! input variables
  integer :: fnlen
  character (len=fnlen), intent(in) :: filename ! = "file.etsf"
  character (len=*), intent(in) :: title ! = "Silicon bulk. Si 1s Corehole, etc."
  character (len=*), intent(in) :: generating_code_and_version ! = "Milan-CP 9.3.4 "

  integer, intent(in) :: real_or_complex
  integer, intent(in) :: number_of_components
  integer, intent(in) :: number_of_grid_points_vector1
  integer, intent(in) :: number_of_grid_points_vector2
  integer, intent(in) :: number_of_grid_points_vector3
  
  ! parameters
  integer, parameter :: number_of_cartesian_directions = 3
  integer, parameter :: number_of_reduced_dimensions = 3
  integer, parameter :: number_of_vectors = 3
  integer, parameter :: character_string_length = 80

  double precision, intent(in) :: primitive_vectors(number_of_cartesian_directions,number_of_vectors)

  double precision,intent(in) :: denpot(real_or_complex,&
           number_of_grid_points_vector1,&
           number_of_grid_points_vector2,&
           number_of_grid_points_vector3,&
           number_of_components) 

  integer,intent(in) :: denpottype ! flag to signal which array is being written:
      !   1=density
      !   2=exchange
      !   3=corr
      !   4=xc
      !   etc...

  
!  character (len=*), intent(in) :: exchange_functional ! = "LDA"
!  character (len=*), intent(in) :: correlation_functional ! = "PBE"
  
  ! local variables
  character (len=200) :: history
  character (len=12) :: date, time, zone
  character (len=80) :: denpotname

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid
  
  ! TEST INPUT

  if (number_of_components /= 1 .and. number_of_components /= 2 &
      .and. number_of_components /= 4) then
    print *, 'writetsf : Error: number_of_components must be 1, 2, or 4 '
    print *, number_of_components
    stop
  end if
    
  ! OPENING
  ! we can use here NF90_64BIT_OFFSET for versions > 3.6 to allow >4GB many variables
  ! s = nf90_create(filename,nf90_64bit_offset,ncid)
  s = nf90_create(filename,nf90_clobber,ncid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)


  ! GENERAL AND GLOBAL ATTRIBUTES
  
  s = nf90_put_att(ncid,nf90_global,"file_format","ETSF Nanoquanta density/potential")
  s = nf90_put_att(ncid,nf90_global,"file_format_version",1.0)
  s = nf90_put_att(ncid,nf90_global,"Conventions","http://www.etsf.eu/fileformats/")
  s = nf90_put_att(ncid,nf90_global,"title",title)
  call date_and_time(date,time,zone)
  history = generating_code_and_version // '   ' // &
    date(1:4) // '-' // date(5:6) // '-' // date(7:8) // '  ' // &
    time(1:2) // ':' // time(3:4) // ':' // time(5:6) // '  ' // &
    zone
  s = nf90_put_att(ncid,nf90_global,"history",history)

  ! Crystal Structure
  call write_primitive_vectors(ncid,&
           number_of_cartesian_directions,number_of_vectors,&
           primitive_vectors)

  ! WARNING: for NetCDF versions prior to 3.6 the largest variable must be declared last 
  if (denpottype == 1) then
         denpotname="density"
  else if (denpottype == 2) then
         denpot="exchange_potential"
  else if (denpottype == 3) then
         denpot="correlation_potential"
  else if (denpottype == 4) then
         denpotname="exchange_correlation_potential"
  else
    stop ' error: unsupported denpottype'
  end if

  call write_denpot(ncid,&
       real_or_complex,number_of_grid_points_vector1,&
       number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components,&
       denpot,denpotname)

  ! CLOSING FILE
  s = nf90_close(ncid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
end subroutine write_denpot_etsf

!===========================================================================
! get dimensions from file so we can allocate the arrays
!===========================================================================
function test_denpot_etsf(filename,fnlen,&
  real_or_complex, number_of_components, number_of_grid_points_vector1, &
  number_of_grid_points_vector2, number_of_grid_points_vector3,denpottype)
  

  implicit none
  integer :: test_denpot_etsf

  ! input variables
  integer,intent(in) :: fnlen
  character (len=fnlen),intent(in) :: filename

  ! Parameters
  integer, parameter :: CSLEN = 80
  integer, parameter :: SLEN = 2
  integer, parameter :: YESLEN = 3
  double precision, parameter :: H2eV = 27.2113845d0
  
  character (len=CSLEN) :: file_format, conventions
  character (len=CSLEN) :: title
  character (len=1024) :: history
  real :: file_format_version

  integer :: ffl,cl,tl,hl

  ! Dimensions
  integer :: number_of_cartesian_directions = 3
  integer :: number_of_reduced_dimensions = 3
  integer :: number_of_vectors = 3
  
  integer, intent(out) :: real_or_complex
  integer,intent(out) :: number_of_components
  integer,intent(out) :: number_of_grid_points_vector1
  integer,intent(out) :: number_of_grid_points_vector2
  integer,intent(out) :: number_of_grid_points_vector3
  integer,intent(out) :: denpottype

  ! local variables
  
  integer :: tmplen

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid
  ! dimensions identifiers
  integer :: cdimid,ncddimid,nvdimid,ncdimid
  integer :: ngv1dimid,ngv2dimid,ngv3dimid
  
  ! variables identifiers
  integer :: pvid,denpotid
  
  print *, 'checking Nanoquanta ETSF NetCDF denpot input file ', trim(filename)
  print *

  test_denpot_etsf = -1
  
  s = nf90_open(filename,nf90_nowrite,ncid)
  if(s /= nf90_noerr) then
    test_denpot_etsf = -1
    print *, 'unrecognized NetCDF file'
    return
  endif
 
  !----------------------------------------------------------------
  ! general ATTRIBUTES
  !----------------------------------------------------------------
  s = nf90_inquire_attribute(ncid,nf90_global,"file_format",len=ffl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"file_format",file_format)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  if(file_format(1:15) /= "ETSF Nanoquanta") then
    test_denpot_etsf = -1
    print *, 'unrecognized .etsf NetCDF file'
    return
  endif  
  s = nf90_get_att(ncid,nf90_global,"file_format_version",file_format_version)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  write(*,'(" ",a," version ",f3.1)') file_format(1:ffl), file_format_version
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"Conventions",len=cl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"Conventions",conventions)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, conventions(1:cl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"title",len=tl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"title",title)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, title(1:tl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"history",len=hl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"history",history)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, history(1:hl)
#endif
  

  ! NetCDF dimensions declaration section

  call read_real_or_complex(ncid,real_or_complex)
  call read_number_of_cartesian_directions(ncid,number_of_cartesian_directions)
  call read_number_of_vectors(ncid,number_of_vectors)
    
10  format(' ',a,t35,i6)
20  format(' ',a,t38,a3)
30  format(' ',a,t34,f7.2,' [eV]')

#ifdef CHECKS
  write(*,10) 'manifold dimension ', real_or_complex
  write(*,10) 'number of cartesian directions ', number_of_cartesian_directions
  write(*,10) 'number of vectors ', number_of_vectors
#endif

  call read_number_of_components(ncid,number_of_components)
#ifdef CHECKS
  write(*,10) 'number of components           ', number_of_components
#endif

  call read_number_of_grid_points_vector1(ncid,number_of_grid_points_vector1)
  call read_number_of_grid_points_vector2(ncid,number_of_grid_points_vector2)
  call read_number_of_grid_points_vector3(ncid,number_of_grid_points_vector3)

  call read_denpottype(ncid,denpottype)


  s = nf90_close(ncid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
  test_denpot_etsf = 100
  return
  
end function test_denpot_etsf

!===========================================================================
!===========================================================================
subroutine read_denpot_etsf(filename,fnlen,title,&
  rc,nc,ngv1,ngv2,ngv3,dpflg,&
  real_or_complex,number_of_components, &
  number_of_grid_points_vector1,number_of_grid_points_vector2,number_of_grid_points_vector3,&
  primitive_vectors,denpot,denpottype)

  implicit none

  ! input variables
  !   variables without intent are not accepted as input yet (hardcoded)
  integer,intent(in) :: fnlen
  character (len=fnlen),intent(in) :: filename

  ! input variables tested against values in netcdf file
  !  rc= real_or_complex
  !  nc= number_of_components
  !  ngv1=number_of_grid_points_vector1
  !  ngv2=number_of_grid_points_vector2
  !  ngv3=number_of_grid_points_vector3
  !  dpflg=denpottype from test call
  integer,intent(in) :: rc,nc,ngv1,ngv2,ngv3,dpflg

  ! Parameters
  integer, parameter :: CSLEN = 80
  integer, parameter :: NCD = 3
  integer, parameter :: NRD = 3
  integer, parameter :: NV = 3
  double precision, parameter :: H2eV = 27.2113845d0
  
  character (len=CSLEN) :: file_format, conventions
  character (len=CSLEN) :: title
  character (len=1024) :: history
  real :: file_format_version

  integer :: ffl,cl,tl,hl

  ! Dimensions
  integer,intent(out) :: real_or_complex
  integer :: number_of_cartesian_directions = 3
  integer :: number_of_vectors = 3
  
  integer,intent(out) :: number_of_components
  integer,intent(out) :: number_of_grid_points_vector1
  integer,intent(out) :: number_of_grid_points_vector2
  integer,intent(out) :: number_of_grid_points_vector3

  double precision,intent(out) :: primitive_vectors(NCD,NV)

  double precision,intent(out) :: denpot(rc,ngv1,ngv2,ngv3,nc)
  integer,intent(out) :: denpottype

  
  
  ! local variables
  
  integer :: tmplen

  ! internal NetCDF variables and identifiers
  integer :: s
  integer :: ncid

  
  print *, 'reading Nanoquanta ETSF NetCDF denpot input file ', trim(filename)
  print *
  
  s = nf90_open(filename,nf90_nowrite,ncid)
  if(s /= nf90_noerr) then
    print *, 'unrecognized NetCDF file'
    return
  endif

  !----------------------------------------------------------------
  ! general ATTRIBUTES
  !----------------------------------------------------------------
  s = nf90_inquire_attribute(ncid,nf90_global,"file_format",len=ffl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"file_format",file_format)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  if(file_format(1:15) /= "ETSF Nanoquanta") then
    print *, 'unrecognized .etsf NetCDF file'
    return
  endif  
  s = nf90_get_att(ncid,nf90_global,"file_format_version",file_format_version)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  write(*,'(" ",a," version ",f3.1)') file_format(1:ffl), file_format_version
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"Conventions",len=cl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"Conventions",conventions)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, conventions(1:cl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"title",len=tl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"title",title)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, title(1:tl)
#endif

  s = nf90_inquire_attribute(ncid,nf90_global,"history",len=hl)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nf90_global,"history",history)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

#ifdef CHECKS
  print *, history(1:hl)
#endif
  

  ! NetCDF dimensions declaration section
  call read_real_or_complex(ncid,real_or_complex)
  call read_number_of_cartesian_directions(ncid,number_of_cartesian_directions)
  call read_number_of_vectors(ncid,number_of_vectors)
    
10  format(' ',a,t35,i6)
20  format(' ',a,t38,a3)
30  format(' ',a,t34,f7.2,' [eV]')

#ifdef CHECKS
  write(*,10) 'manifold dimension ', real_or_complex
  write(*,10) 'number of cartesian directions ', number_of_cartesian_directions
  write(*,10) 'number of vectors ', number_of_vectors
#endif
  if(real_or_complex /= rc)  stop 'error: real or complex'
  if(number_of_cartesian_directions /= NCD) stop 'error: number of cartesian directions'
  if(number_of_vectors /= NV) stop 'error: number of vectors'

  call read_number_of_components(ncid,number_of_components)

#ifdef CHECKS
  write(*,10) 'number of components           ', number_of_components
#endif
  if(number_of_components /= nc) stop 'error: number of components changed'

  ! VARIABLES
  
  ! Crystal Structure
  call read_primitive_vectors(ncid,&
       number_of_cartesian_directions,number_of_vectors,&
       primitive_vectors)
  
#ifdef CHECKS
#endif
  
call read_denpot(ncid,rc,ngv1,ngv2,ngv3,nc,&
     denpot)

#ifdef CHECKS
  print *, 'denpot(1,1,1,1,1) ', denpot(1,1,1,1,1)
#endif

  
  s = nf90_close(ncid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  
  return

end subroutine read_denpot_etsf


!------------------------------------------------------------------------
!     atomic routines for writing individual variables
!       dimensions are checked for and written if not present
!------------------------------------------------------------------------
subroutine write_primitive_vectors(ncid,&
           number_of_cartesian_directions,number_of_vectors,&
           primitive_vectors)
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_cartesian_directions,number_of_vectors
!data
double precision, intent(in) :: primitive_vectors(number_of_cartesian_directions,number_of_vectors)
!local
integer :: ncddimid, nvdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_cartesian_directions",number_of_cartesian_directions,ncddimid)
call inqordef_dimid(ncid,"number_of_vectors",number_of_vectors,nvdimid)

! define and put data
call defput_primitive_vectors(ncid,ncddimid, nvdimid,&
           number_of_cartesian_directions,number_of_vectors,&
           primitive_vectors)
end subroutine write_primitive_vectors

subroutine defput_primitive_vectors(ncid,ncddimid, nvdimid,&
           number_of_cartesian_directions,number_of_vectors,&
           primitive_vectors)
implicit none

!arguments
!ids
integer, intent(in) :: ncid,ncddimid, nvdimid
!dimensions
integer, intent(in) :: number_of_cartesian_directions,number_of_vectors
!data
double precision, intent(in) :: primitive_vectors(number_of_cartesian_directions,number_of_vectors)
!local
integer :: s,pvid

  ! Crystal Structure
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"primitive_vectors",nf90_double,(/ ncddimid, nvdimid /),pvid)
  s = nf90_put_att(ncid,pvid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,pvid,primitive_vectors)
end subroutine defput_primitive_vectors


subroutine write_space_group(ncid,space_group)
implicit none

!arguments
!ids
integer :: ncid
!dimensions
!data
integer, intent(in) :: space_group
!local

call defput_space_group(ncid,space_group)
end subroutine write_space_group

subroutine defput_space_group(ncid,space_group)
implicit none

!arguments
!ids
integer :: ncid
!dimensions
!data
integer, intent(in) :: space_group
!local
integer :: s,sgvid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"space_group",nf90_int,sgvid)
  s = nf90_put_att(ncid,sgvid,"valid_range",(/1,230/))
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,sgvid,space_group)
end subroutine defput_space_group


subroutine write_atomic_numbers(ncid,&
           number_of_atom_species,&
           atomic_numbers)
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_atom_species
!data
double precision, intent(in) :: atomic_numbers(number_of_atom_species)
!local
integer :: natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_atomic_numbers(ncid,natdimid,&
           number_of_atom_species,&
           atomic_numbers)
end subroutine write_atomic_numbers

subroutine defput_atomic_numbers(ncid,natdimid,&
           number_of_atom_species,&
           atomic_numbers)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,natdimid
!dimensions
integer, intent(in) :: number_of_atom_species
!data
double precision, intent(in) :: atomic_numbers(number_of_atom_species)
!local
integer :: s,anid

  ! Unit Cell Chemical Structure
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"atomic_numbers",nf90_double, (/ natdimid /), anid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,anid,atomic_numbers)
end subroutine defput_atomic_numbers


subroutine write_valence_charges(ncid,&
           number_of_atom_species,&
           valence_charges)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_atom_species
!data
double precision, intent(in) :: valence_charges(number_of_atom_species)
!local
integer :: natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_valence_charges(ncid,natdimid,&
           number_of_atom_species,&
           valence_charges)
end subroutine write_valence_charges

subroutine defput_valence_charges(ncid,natdimid,&
           number_of_atom_species,&
           valence_charges)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,natdimid
!dimensions
integer, intent(in) :: number_of_atom_species
!data
double precision, intent(in) :: valence_charges(number_of_atom_species)
!local
integer :: s,avcid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"valence_charges",nf90_double, (/ natdimid /), avcid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,avcid,valence_charges)
end subroutine defput_valence_charges


subroutine write_atom_species_names(ncid,&
           character_string_length,number_of_atom_species,&
           atom_species_names)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: character_string_length,number_of_atom_species
!data
character (len=*),intent(in) :: atom_species_names(number_of_atom_species)
!local
integer :: csldimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_atom_species_names(ncid,csldimid, natdimid,&
           number_of_atom_species,&
           atom_species_names)
end subroutine write_atom_species_names

subroutine defput_atom_species_names(ncid,csldimid, natdimid,&
           number_of_atom_species,&
           atom_species_names)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,csldimid, natdimid
!dimensions
integer,intent(in) :: number_of_atom_species
!data
character (len=*),intent(in) :: atom_species_names(number_of_atom_species)
!local
integer :: s,asnid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"atom_species_names",nf90_char, (/ csldimid, natdimid /), asnid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,asnid,atom_species_names)
end subroutine defput_atom_species_names


subroutine write_chemical_symbols(ncid,&
           symbol_length,number_of_atom_species,&
           chemical_symbols)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: symbol_length,number_of_atom_species
!data
character (len=*),intent(in) :: chemical_symbols(number_of_atom_species)
!local
integer :: sldimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"symbol_length",symbol_length,sldimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_chemical_symbols(ncid,sldimid, natdimid,&
           number_of_atom_species,&
           chemical_symbols)
end subroutine write_chemical_symbols

subroutine defput_chemical_symbols(ncid,sldimid, natdimid,&
           number_of_atom_species,&
           chemical_symbols)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,sldimid, natdimid
!dimensions
integer,intent(in) :: number_of_atom_species
!data
character (len=*),intent(in) :: chemical_symbols(number_of_atom_species)
!local
integer :: s,csid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"chemical_symbols",nf90_char, (/ sldimid, natdimid /), csid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,csid,chemical_symbols)
end subroutine defput_chemical_symbols


subroutine write_pseudopotential_types(ncid,&
           character_string_length,number_of_atom_species,&
           pseudopotential_types)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: character_string_length,number_of_atom_species
!data
character (len=*), intent(in) :: pseudopotential_types(number_of_atom_species)
!local
integer :: csldimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_pseudopotential_types(ncid,csldimid, natdimid,&
           number_of_atom_species,&
           pseudopotential_types)
end subroutine write_pseudopotential_types

subroutine defput_pseudopotential_types(ncid,csldimid, natdimid,&
           number_of_atom_species,&
           pseudopotential_types)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,csldimid, natdimid
!dimensions
integer, intent(in) :: number_of_atom_species
!data
character (len=*), intent(in) :: pseudopotential_types(number_of_atom_species)
!local
integer :: s, psptypeid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"pseudopotential_types",nf90_char, (/ csldimid, natdimid /), psptypeid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,psptypeid,pseudopotential_types)
end subroutine defput_pseudopotential_types


subroutine write_atom_species(ncid,&
           number_of_atoms,&
           atom_species)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_atoms
!data
integer, intent(in) :: atom_species(number_of_atoms)
!local
integer :: nadimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_atoms",number_of_atoms,nadimid)

! define and put data
call defput_atom_species(ncid,nadimid,&
           number_of_atoms,&
           atom_species)
end subroutine write_atom_species

subroutine defput_atom_species(ncid,nadimid,&
           number_of_atoms,&
           atom_species)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nadimid
!dimensions
integer, intent(in) :: number_of_atoms
!data
integer, intent(in) :: atom_species(number_of_atoms)
!local
integer :: s,atid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"atom_species",nf90_int, (/ nadimid /), atid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,atid,atom_species)
end subroutine defput_atom_species


subroutine write_reduced_atom_positions(ncid,&
           number_of_reduced_dimensions,number_of_atoms,&
           reduced_atom_positions)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_atoms
!data
double precision, intent(in) :: reduced_atom_positions(number_of_reduced_dimensions,number_of_atoms)
!local
integer :: nrddimid, nadimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions,nrddimid)
call inqordef_dimid(ncid,"number_of_atoms",number_of_atoms,nadimid)

! define and put data
call defput_reduced_atom_positions(ncid,nrddimid, nadimid,&
           number_of_reduced_dimensions,number_of_atoms,&
           reduced_atom_positions)
end subroutine write_reduced_atom_positions

subroutine defput_reduced_atom_positions(ncid,nrddimid, nadimid,&
           number_of_reduced_dimensions,number_of_atoms,&
           reduced_atom_positions)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nrddimid, nadimid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_atoms
!data
double precision, intent(in) :: reduced_atom_positions(number_of_reduced_dimensions,number_of_atoms)
!local
integer :: s,apid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"reduced_atom_positions",nf90_double, (/ nrddimid, nadimid /),apid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,apid,reduced_atom_positions)
end subroutine defput_reduced_atom_positions


subroutine write_number_of_electrons(ncid,number_of_electrons)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
integer, intent(in) :: number_of_electrons
!local

call defput_number_of_electrons(ncid,number_of_electrons)
end subroutine write_number_of_electrons

subroutine defput_number_of_electrons(ncid,number_of_electrons)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
integer, intent(in) :: number_of_electrons
!local
integer :: s,nelid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"number_of_electrons",nf90_int,nelid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,nelid,number_of_electrons)
end subroutine defput_number_of_electrons


subroutine write_reduced_symmetry_operations(ncid,&
           number_of_reduced_dimensions,number_of_symmetry_operations,&
           reduced_symmetry_matrices,reduced_symmetry_translations)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_symmetry_operations
!data
double precision, intent(in) :: reduced_symmetry_translations(number_of_reduced_dimensions,&
                                                              number_of_symmetry_operations)
integer, intent(in) :: reduced_symmetry_matrices(number_of_reduced_dimensions,number_of_reduced_dimensions,&
                                                 number_of_symmetry_operations)
!local
integer :: nrddimid, nsodimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions,nrddimid)
call inqordef_dimid(ncid,"number_of_symmetry_operations",number_of_symmetry_operations,nsodimid)

! define and put data
call defput_reduced_symmetry_operations(ncid,nrddimid, nsodimid,&
           number_of_reduced_dimensions,number_of_symmetry_operations,&
           reduced_symmetry_matrices,reduced_symmetry_translations)
end subroutine write_reduced_symmetry_operations

subroutine defput_reduced_symmetry_operations(ncid,nrddimid, nsodimid,&
           number_of_reduced_dimensions,number_of_symmetry_operations,&
           reduced_symmetry_matrices,reduced_symmetry_translations)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nrddimid, nsodimid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_symmetry_operations
!data
double precision, intent(in) :: reduced_symmetry_translations(number_of_reduced_dimensions,&
                                                              number_of_symmetry_operations)
integer, intent(in) :: reduced_symmetry_matrices(number_of_reduced_dimensions,number_of_reduced_dimensions,&
                                                 number_of_symmetry_operations)
!local
integer :: s,symid,tsymid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"reduced_symmetry_matrices",nf90_int,(/ nrddimid, nrddimid, nsodimid /), symid)
  s = nf90_def_var(ncid,"reduced_symmetry_translations",nf90_double,(/ nrddimid, nsodimid /), tsymid)
  if (all(abs(reduced_symmetry_translations) < 1e-9)) then
! WARNING only the translations have the symmorphic attribute in this writing
    s= nf90_put_att(ncid,symid,"symmorphic","yes")
    s= nf90_put_att(ncid,tsymid,"symmorphic","yes")
  else
    s= nf90_put_att(ncid,symid,"symmorphic","no")
    s= nf90_put_att(ncid,tsymid,"symmorphic","no")
  endif
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,tsymid,reduced_symmetry_translations)
  s = nf90_put_var(ncid,symid,reduced_symmetry_matrices)
end subroutine defput_reduced_symmetry_operations


subroutine write_reduced_coordinates_of_kpoints(ncid,&
           number_of_reduced_dimensions,number_of_kpoints,&
           reduced_coordinates_of_kpoints)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_kpoints
!data
double precision, intent(in) :: reduced_coordinates_of_kpoints(number_of_reduced_dimensions,number_of_kpoints)
!local
integer :: nrddimid, nkdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions,nrddimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)

! define and put data
call defput_reduced_coordinates_of_kpoints(ncid,nrddimid, nkdimid,&
           number_of_reduced_dimensions,number_of_kpoints,&
           reduced_coordinates_of_kpoints)
end subroutine write_reduced_coordinates_of_kpoints

subroutine defput_reduced_coordinates_of_kpoints(ncid,nrddimid, nkdimid,&
           number_of_reduced_dimensions,number_of_kpoints,&
           reduced_coordinates_of_kpoints)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nrddimid, nkdimid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_kpoints
!data
double precision, intent(in) :: reduced_coordinates_of_kpoints(number_of_reduced_dimensions,number_of_kpoints)
!local
integer :: s,kvid

  ! Brillouin Zone
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"reduced_coordinates_of_kpoints",nf90_double, (/ nrddimid, nkdimid /), kvid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kvid,reduced_coordinates_of_kpoints)
end subroutine defput_reduced_coordinates_of_kpoints


subroutine write_kpoint_weights(ncid,&
           number_of_kpoints,&
           kpoint_weights)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
double precision, intent(in) :: kpoint_weights(number_of_kpoints)
!local
integer :: nkdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)

! define and put data
call defput_kpoint_weights(ncid,nkdimid,&
           number_of_kpoints,&
           kpoint_weights)
end subroutine write_kpoint_weights

subroutine defput_kpoint_weights(ncid,nkdimid,&
           number_of_kpoints,&
           kpoint_weights)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nkdimid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
double precision, intent(in) :: kpoint_weights(number_of_kpoints)
!local
integer :: s, kwid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kpoint_weights",nf90_double, (/ nkdimid /), kwid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kwid,kpoint_weights)
end subroutine defput_kpoint_weights


subroutine write_monkhorst_pack_folding(ncid,&
           number_of_vectors,&
           monkhorst_pack_folding)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_vectors
!data
integer, intent(in) :: monkhorst_pack_folding(number_of_vectors)
!local
integer :: nvdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_vectors",number_of_vectors,nvdimid)

! define and put data
call defput_monkhorst_pack_folding(ncid,nvdimid,&
           number_of_vectors,&
           monkhorst_pack_folding)
end subroutine write_monkhorst_pack_folding

subroutine defput_monkhorst_pack_folding(ncid,nvdimid,&
           number_of_vectors,&
           monkhorst_pack_folding)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nvdimid
!dimensions
integer, intent(in) :: number_of_vectors
!data
integer, intent(in) :: monkhorst_pack_folding(number_of_vectors)
!local
integer :: s, mpfid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"monkhorst_pack_folding",nf90_int, (/ nvdimid /), mpfid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,mpfid,monkhorst_pack_folding)
end subroutine defput_monkhorst_pack_folding


subroutine write_kpoint_grid_vectors(ncid,&
           number_of_reduced_dimensions,number_of_vectors,&
           kpoint_grid_vectors)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_vectors
!data
double precision, intent(in) :: kpoint_grid_vectors(number_of_reduced_dimensions,number_of_vectors)
!local
integer :: nrddimid, nvdimid 

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions,nrddimid)
call inqordef_dimid(ncid,"number_of_vectors",number_of_vectors,nvdimid)

! define and put data
call defput_kpoint_grid_vectors(ncid,nrddimid, nvdimid,&
           number_of_reduced_dimensions,number_of_vectors,&
           kpoint_grid_vectors)
end subroutine write_kpoint_grid_vectors

subroutine defput_kpoint_grid_vectors(ncid,nrddimid, nvdimid,&
           number_of_reduced_dimensions,number_of_vectors,&
           kpoint_grid_vectors)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nrddimid, nvdimid 
!dimensions
integer, intent(in) :: number_of_reduced_dimensions,number_of_vectors
!data
double precision, intent(in) :: kpoint_grid_vectors(number_of_reduced_dimensions,number_of_vectors)
!local
integer :: s, kgvid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kpoint_grid_vectors",nf90_double, (/ nrddimid, nvdimid /), kgvid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kgvid,kpoint_grid_vectors)
end subroutine defput_kpoint_grid_vectors


subroutine write_kpoint_grid_shift(ncid,&
           number_of_reduced_dimensions,&
           kpoint_grid_shift)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: number_of_reduced_dimensions
!data
double precision, intent(in) :: kpoint_grid_shift(number_of_reduced_dimensions)
!local
integer :: nrddimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions,nrddimid)

! define and put data
call defput_kpoint_grid_shift(ncid,nrddimid,&
           number_of_reduced_dimensions,&
           kpoint_grid_shift)
end subroutine write_kpoint_grid_shift

subroutine defput_kpoint_grid_shift(ncid,nrddimid,&
           number_of_reduced_dimensions,&
           kpoint_grid_shift)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,nrddimid
!dimensions
integer,intent(in) :: number_of_reduced_dimensions
!data
double precision, intent(in) :: kpoint_grid_shift(number_of_reduced_dimensions)
!local
integer :: s,kgsid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kpoint_grid_shift",nf90_double, (/ nrddimid /), kgsid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kgsid,kpoint_grid_shift)
end subroutine defput_kpoint_grid_shift


subroutine write_basis_set(ncid,character_string_length,basis_set)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: character_string_length
!data
character (len=*), intent(in) :: basis_set
!local
integer :: csldimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)

! define and put data
call defput_basis_set(ncid,csldimid,basis_set)
end subroutine write_basis_set

subroutine defput_basis_set(ncid,csldimid,basis_set)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,csldimid
!dimensions
!data
character (len=*), intent(in) :: basis_set
!local
integer :: s,bsid

  ! Reciprocal G-space, plane waves  
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"basis_set",nf90_char,(/ csldimid /),bsid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,bsid,basis_set)
end subroutine defput_basis_set


subroutine write_number_of_coefficients(ncid,&
           number_of_kpoints,&
           k_dependent_gvectors,number_of_coefficients)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_gvectors          ! = "no"  ! or "yes"
integer, intent(in) :: number_of_coefficients(number_of_kpoints)
!local
integer :: nkdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)

! define and put data
call defput_number_of_coefficients(ncid,nkdimid,&
           number_of_kpoints,&
           k_dependent_gvectors,number_of_coefficients)
end subroutine write_number_of_coefficients

subroutine defput_number_of_coefficients(ncid,nkdimid,&
           number_of_kpoints,&
           k_dependent_gvectors,number_of_coefficients)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nkdimid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_gvectors          ! = "no"  ! or "yes"
integer, intent(in) :: number_of_coefficients(number_of_kpoints)
!local
integer :: s,ngkid

  s = nf90_redef(ncid)
  if(k_dependent_gvectors(1:1) == 'y') then
    s = nf90_def_var(ncid,"number_of_coefficients",nf90_int,(/ nkdimid /),ngkid)
    s = nf90_put_att(ncid,ngkid,"k_dependent","yes")
  else
    s = nf90_def_var(ncid,"number_of_coefficients",nf90_int,ngkid)
    s = nf90_put_att(ncid,ngkid,"k_dependent","no")
  endif
  s = nf90_enddef(ncid)
  if(k_dependent_gvectors(1:1) == 'y') then
    s = nf90_put_var(ncid,ngkid,number_of_coefficients)
  endif
end subroutine defput_number_of_coefficients


subroutine write_gvectors(ncid,&
           number_of_vectors,max_number_of_coefficients,number_of_kpoints,&
           k_dependent_gvectors,gvectors,gvectors_k)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_vectors,max_number_of_coefficients,number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_gvectors          ! = "no"  ! or "yes"
integer, intent(in) :: gvectors(number_of_vectors,max_number_of_coefficients)
integer, intent(in) :: gvectors_k(number_of_vectors,max_number_of_coefficients,number_of_kpoints)
!local
integer :: nvdimid, mngdimid, nkdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_vectors",number_of_vectors,nvdimid)
call inqordef_dimid(ncid,"max_number_of_coefficients",max_number_of_coefficients,mngdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)

! define and put data
call defput_gvectors(ncid,nvdimid, mngdimid, nkdimid,&
           number_of_vectors,max_number_of_coefficients,number_of_kpoints,&
           k_dependent_gvectors,gvectors,gvectors_k)
end subroutine write_gvectors

subroutine defput_gvectors(ncid,nvdimid, mngdimid, nkdimid,&
           number_of_vectors,max_number_of_coefficients,number_of_kpoints,&
           k_dependent_gvectors,gvectors,gvectors_k)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nvdimid, mngdimid, nkdimid
!dimensions
integer, intent(in) :: number_of_vectors,max_number_of_coefficients,number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_gvectors          ! = "no"  ! or "yes"
integer, intent(in) :: gvectors(number_of_vectors,max_number_of_coefficients)
integer, intent(in) :: gvectors_k(number_of_vectors,max_number_of_coefficients,number_of_kpoints)
!local
integer :: s,gvid

  s = nf90_redef(ncid)
  if(k_dependent_gvectors(1:1) == 'y') then
    s = nf90_def_var(ncid,"reduced_coordinates_of_plane_waves",nf90_int,(/ nvdimid, mngdimid, nkdimid /),gvid)
    s = nf90_put_att(ncid,gvid,"k_dependent","yes")
  else
    s = nf90_def_var(ncid,"reduced_coordinates_of_plane_waves",nf90_int,(/ nvdimid, mngdimid /),gvid)
    s = nf90_put_att(ncid,gvid,"k_dependent","no")
  endif
  s = nf90_enddef(ncid)
  if(k_dependent_gvectors(1:1) == 'y') then
    s = nf90_put_var(ncid,gvid,gvectors_k)
  else
    s = nf90_put_var(ncid,gvid,gvectors)
  endif
end subroutine defput_gvectors


subroutine write_number_of_states(ncid,&
           number_of_kpoints,&
           k_dependent_number_of_states,number_of_states)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_number_of_states  ! = "no"  ! or "yes"
integer, intent(in) :: number_of_states(number_of_kpoints)
!local
integer :: nkdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)

! define and put data
call defput_number_of_states(ncid,nkdimid,&
           number_of_kpoints,&
           k_dependent_number_of_states,number_of_states)
end subroutine write_number_of_states

subroutine defput_number_of_states(ncid,nkdimid,&
           number_of_kpoints,&
           k_dependent_number_of_states,number_of_states)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,nkdimid
!dimensions
integer, intent(in) :: number_of_kpoints
!data
character (len=*), intent(in) :: k_dependent_number_of_states  ! = "no"  ! or "yes"
integer, intent(in) :: number_of_states(number_of_kpoints)
!local
integer :: s,nsid

  ! Number of states
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"number_of_states",nf90_int,(/ nkdimid /),nsid)
  if(k_dependent_number_of_states(1:1) == 'y') then
    s = nf90_put_att(ncid,nsid,"k_dependent","yes")
  else
    s = nf90_put_att(ncid,nsid,"k_dependent","no")
  endif
  s = nf90_enddef(ncid)
  if(k_dependent_number_of_states(1:1) == 'y') then
    s = nf90_put_var(ncid,nsid,number_of_states)
  endif
end subroutine defput_number_of_states
  

subroutine write_kinetic_energy_cutoff(ncid,kinetic_energy_cutoff)
!use netcdf
implicit none

!arguments
!ids
integer :: ncid
!dimensions
!data
double precision, intent(in) :: kinetic_energy_cutoff
!local

call defput_kinetic_energy_cutoff(ncid,kinetic_energy_cutoff)
end subroutine write_kinetic_energy_cutoff
  
subroutine defput_kinetic_energy_cutoff(ncid,kinetic_energy_cutoff)
!use netcdf
implicit none

!arguments
!ids
integer :: ncid
!dimensions
!data
double precision, intent(in) :: kinetic_energy_cutoff
!local
integer :: s,kecid

  ! Convergency data
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kinetic_energy_cutoff",nf90_double,kecid)
  ! s = nf90_put_att(ncid,kecid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kecid,kinetic_energy_cutoff)
end subroutine defput_kinetic_energy_cutoff


subroutine write_total_energy(ncid,total_energy)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: total_energy
!local

call defput_total_energy(ncid,total_energy)
end subroutine write_total_energy

subroutine defput_total_energy(ncid,total_energy)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: total_energy
!local
integer :: s,totalenergyid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"total_energy",nf90_double,totalenergyid)
  ! s = nf90_put_att(ncid,totalenergyid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,totalenergyid,total_energy)
end subroutine defput_total_energy


subroutine write_fermi_energy(ncid,fermi_energy)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: fermi_energy
!local

call defput_fermi_energy(ncid,fermi_energy)
end subroutine write_fermi_energy

subroutine defput_fermi_energy(ncid,fermi_energy)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: fermi_energy
!local
integer :: s,fermienergyid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"fermi_energy",nf90_double,fermienergyid)
  ! s = nf90_put_att(ncid,fermienergyid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,fermienergyid,fermi_energy)
end subroutine defput_fermi_energy


subroutine write_max_residual(ncid,max_residual)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: max_residual
!local

call defput_max_residual(ncid,max_residual)
end subroutine write_max_residual

subroutine defput_max_residual(ncid,max_residual)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: max_residual
!local
integer :: s,maxresidualid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"max_residual",nf90_double,maxresidualid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,maxresidualid,max_residual)
end subroutine defput_max_residual


subroutine write_exchange_functional(ncid,character_string_length,exchange_functional)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: character_string_length
!data
character (len=*), intent(in) :: exchange_functional ! = "PBE"
!local
integer :: s,csldimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)

! define and put data
call defput_exchange_functional(ncid,csldimid,exchange_functional)
end subroutine write_exchange_functional

subroutine defput_exchange_functional(ncid,csldimid,exchange_functional)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,csldimid
!dimensions
!data
character (len=*), intent(in) :: exchange_functional ! = "PBE"
!local
integer :: s,xfid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"exchange_functional",nf90_char,(/ csldimid /),xfid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,xfid,exchange_functional)
end subroutine defput_exchange_functional


subroutine write_correlation_functional(ncid,character_string_length,correlation_functional)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: character_string_length
!data
character (len=*), intent(in) :: correlation_functional ! = "PBE"
!local
integer :: s,csldimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)

! define and put data
call defput_correlation_functional(ncid,csldimid,correlation_functional)
end subroutine write_correlation_functional

subroutine defput_correlation_functional(ncid,csldimid,correlation_functional)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,csldimid
!dimensions
!data
character (len=*), intent(in) :: correlation_functional ! = "PBE"
!local
integer :: s,cfid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"correlation_functional",nf90_char,(/ csldimid /),cfid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,cfid,correlation_functional)
end subroutine defput_correlation_functional
  

subroutine write_occupations(ncid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           occupations)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: occupations(max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s, mnsdimid, nkdimid, nscdimid


!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"max_number_of_states",max_number_of_states,mnsdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"number_of_spins",number_of_spins,nscdimid)

! define and put data
call defput_occupations(ncid,mnsdimid, nkdimid, nscdimid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           occupations)
end subroutine write_occupations
  
subroutine defput_occupations(ncid,mnsdimid, nkdimid, nscdimid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           occupations)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,mnsdimid, nkdimid, nscdimid
!dimensions
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: occupations(max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s,occupationid

  ! Electronic Structure
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"occupations",nf90_double,(/ mnsdimid, nkdimid, nscdimid/),occupationid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,occupationid,occupations)
end subroutine defput_occupations


subroutine write_smearing_width(ncid,smearing_width)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: smearing_width
!local
integer :: s

call defput_smearing_width(ncid,smearing_width)
end subroutine write_smearing_width

subroutine defput_smearing_width(ncid,smearing_width)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
!data
double precision, intent(in) :: smearing_width
!local
integer :: s,swid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"smearing_width",nf90_double,swid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,swid,smearing_width)
end subroutine defput_smearing_width


subroutine write_smearing_scheme(ncid,character_string_length,smearing_scheme)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: character_string_length
!data
character (len=*), intent(in) :: smearing_scheme ! = "Fermi-Dirac"
!local
integer :: s,csldimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"character_string_length",character_string_length,csldimid)

! define and put data
call defput_smearing_scheme(ncid,csldimid,smearing_scheme)
end subroutine write_smearing_scheme

subroutine defput_smearing_scheme(ncid,csldimid,smearing_scheme)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,csldimid
!dimensions
!data
character (len=*), intent(in) :: smearing_scheme ! = "Fermi-Dirac"
!local
integer :: s,smsid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"smearing_scheme",nf90_char,(/ csldimid /),smsid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,smsid,smearing_scheme)
end subroutine defput_smearing_scheme


subroutine write_eigenvalues(ncid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           eigenvalues)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: eigenvalues(max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s,mnsdimid, nkdimid, nscdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"max_number_of_states",max_number_of_states,mnsdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"number_of_spins",number_of_spins,nscdimid)

! define and put data
call defput_eigenvalues(ncid,mnsdimid, nkdimid, nscdimid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           eigenvalues)
end subroutine write_eigenvalues

subroutine defput_eigenvalues(ncid,mnsdimid, nkdimid, nscdimid,&
           max_number_of_states,number_of_kpoints,number_of_spins,&
           eigenvalues)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,mnsdimid, nkdimid, nscdimid
!dimensions
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: eigenvalues(max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s,energyid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"eigenvalues",nf90_double,(/ mnsdimid, nkdimid, nscdimid/), energyid)
  ! s = nf90_put_att(ncid,energyid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,energyid,eigenvalues)
end subroutine defput_eigenvalues


subroutine write_gw_corrections(ncid,&
           real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins,&
           gw_corrections)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision,intent(in) :: gw_corrections(real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s,cdimid, mnsdimid, nkdimid, nscdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"real_or_complex",real_or_complex,cdimid)
call inqordef_dimid(ncid,"max_number_of_states",max_number_of_states,mnsdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"number_of_spins",number_of_spins,nscdimid)

! define and put data
call defput_gw_corrections(ncid,cdimid, mnsdimid, nkdimid, nscdimid,&
           real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins,&
           gw_corrections)
end subroutine write_gw_corrections

subroutine defput_gw_corrections(ncid,cdimid, mnsdimid, nkdimid, nscdimid,&
           real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins,&
           gw_corrections)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,cdimid, mnsdimid, nkdimid, nscdimid
!dimensions
integer,intent(in) :: real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision,intent(in) :: gw_corrections(real_or_complex,max_number_of_states,number_of_kpoints,number_of_spins)
!local
integer :: s,gwcorrid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"gw_corrections",nf90_double,(/ cdimid, mnsdimid, nkdimid, nscdimid/), gwcorrid)
  ! s = nf90_put_att(ncid,gwcorrid,"units","atomic units")
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,gwcorrid,gw_corrections)
end subroutine defput_gw_corrections
 

subroutine write_kb_formfactor_sign(ncid,&
           max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_sign)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species
!data
integer,intent(in) :: kb_formfactor_sign(max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species)

!local
integer :: s,mnpdimid, mnamdimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"max_number_of_projectors",max_number_of_projectors,mnpdimid)
call inqordef_dimid(ncid,"max_number_of_angular_momenta",max_number_of_angular_momenta,mnamdimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_kb_formfactor_sign(ncid,mnpdimid, mnamdimid, natdimid,&
           max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_sign)
end subroutine write_kb_formfactor_sign
 
subroutine defput_kb_formfactor_sign(ncid,mnpdimid, mnamdimid, natdimid,&
           max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_sign)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,mnpdimid, mnamdimid, natdimid
!dimensions
integer,intent(in) :: max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species
!data
integer,intent(in) :: kb_formfactor_sign(max_number_of_projectors,max_number_of_angular_momenta,number_of_atom_species)

!local
integer :: s,kbffsid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kb_formfactor_sign",nf90_int, &
                  (/ mnpdimid, mnamdimid, natdimid/),kbffsid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kbffsid,kb_formfactor_sign)
end subroutine defput_kb_formfactor_sign


subroutine write_kb_formfactors(ncid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactors)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: max_number_of_coefficients,number_of_kpoints,max_number_of_projectors
integer,intent(in) :: max_number_of_angular_momenta,number_of_atom_species
!data
double precision,intent(in) :: kb_formfactors(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                   max_number_of_angular_momenta,number_of_atom_species)
!local
integer :: s,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"max_number_of_coefficients",max_number_of_coefficients,mngdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"max_number_of_projectors",max_number_of_projectors,mnpdimid)
call inqordef_dimid(ncid,"max_number_of_angular_momenta",max_number_of_angular_momenta,mnamdimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_kb_formfactors(ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactors)
end subroutine write_kb_formfactors

subroutine defput_kb_formfactors(ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactors)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid
!dimensions
integer,intent(in) :: max_number_of_coefficients,number_of_kpoints,max_number_of_projectors
integer,intent(in) :: max_number_of_angular_momenta,number_of_atom_species
!data
double precision,intent(in) :: kb_formfactors(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                   max_number_of_angular_momenta,number_of_atom_species)
!local
integer :: s,kbffid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kb_formfactors",nf90_double, &
                  (/ mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid/),kbffid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kbffid,kb_formfactors)
end subroutine defput_kb_formfactors


subroutine write_kb_formfactor_derivative(ncid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_derivative)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid
!dimensions
integer,intent(in) :: max_number_of_coefficients,number_of_kpoints,max_number_of_projectors
integer,intent(in) :: max_number_of_angular_momenta,number_of_atom_species
!data
double precision,intent(in) :: kb_formfactor_derivative(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                     max_number_of_angular_momenta,number_of_atom_species)

!local
integer :: s,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"max_number_of_coefficients",max_number_of_coefficients,mngdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"max_number_of_projectors",max_number_of_projectors,mnpdimid)
call inqordef_dimid(ncid,"max_number_of_angular_momenta",max_number_of_angular_momenta,mnamdimid)
call inqordef_dimid(ncid,"number_of_atom_species",number_of_atom_species,natdimid)

! define and put data
call defput_kb_formfactor_derivative(ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_derivative)

end subroutine write_kb_formfactor_derivative

subroutine defput_kb_formfactor_derivative(ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid,&
           max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
           max_number_of_angular_momenta,number_of_atom_species,&
           kb_formfactor_derivative)
!use netcdf
implicit none

!arguments
!ids
integer,intent(in) :: ncid,mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid
!dimensions
integer,intent(in) :: max_number_of_coefficients,number_of_kpoints,max_number_of_projectors
integer,intent(in) :: max_number_of_angular_momenta,number_of_atom_species
!data
double precision,intent(in) :: kb_formfactor_derivative(max_number_of_coefficients,number_of_kpoints,max_number_of_projectors,&
                                     max_number_of_angular_momenta,number_of_atom_species)

!local
integer :: s,kbffdid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"kb_formfactor_derivative",nf90_double, &
                  (/ mngdimid, nkdimid, mnpdimid, mnamdimid, natdimid/),kbffdid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,kbffdid,kb_formfactor_derivative)

end subroutine defput_kb_formfactor_derivative


subroutine write_coefficients_of_wavefunctions(ncid,&
                  real_or_complex,max_number_of_coefficients,number_of_spinor_components,&
                  max_number_of_states,number_of_kpoints,number_of_spins,&
                  coefficients_of_wavefunctions)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: real_or_complex,max_number_of_coefficients,number_of_spinor_components
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: coefficients_of_wavefunctions(real_or_complex,max_number_of_coefficients, &
                    number_of_spinor_components,max_number_of_states,number_of_kpoints, &
                    number_of_spins)
!local
integer :: cdimid, mngdimid, nsrcdimid, mnsdimid, nkdimid, nscdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"real_or_complex",real_or_complex,cdimid)
call inqordef_dimid(ncid,"max_number_of_coefficients",max_number_of_coefficients,mngdimid)
call inqordef_dimid(ncid,"number_of_spinor_components",number_of_spinor_components,nsrcdimid)
call inqordef_dimid(ncid,"max_number_of_states",max_number_of_states,mnsdimid)
call inqordef_dimid(ncid,"number_of_kpoints",number_of_kpoints,nkdimid)
call inqordef_dimid(ncid,"number_of_spins",number_of_spins,nscdimid)

! define and put data
call defput_coefficients_of_wavefunctions(ncid,cdimid, mngdimid, nsrcdimid, mnsdimid, nkdimid, nscdimid,&
                  real_or_complex,max_number_of_coefficients,number_of_spinor_components,&
                  max_number_of_states,number_of_kpoints,number_of_spins,&
                  coefficients_of_wavefunctions)

end subroutine write_coefficients_of_wavefunctions

subroutine defput_coefficients_of_wavefunctions(ncid,cdimid, mngdimid, nsrcdimid, mnsdimid, nkdimid, nscdimid,&
                  real_or_complex,max_number_of_coefficients,number_of_spinor_components,&
                  max_number_of_states,number_of_kpoints,number_of_spins,&
                  coefficients_of_wavefunctions)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,cdimid, mngdimid, nsrcdimid, mnsdimid, nkdimid, nscdimid
!dimensions
integer, intent(in) :: real_or_complex,max_number_of_coefficients,number_of_spinor_components
integer, intent(in) :: max_number_of_states,number_of_kpoints,number_of_spins
!data
double precision, intent(in) :: coefficients_of_wavefunctions(real_or_complex,max_number_of_coefficients, &
                    number_of_spinor_components,max_number_of_states,number_of_kpoints, &
                    number_of_spins)
!local
integer :: s,wavefunctionid

  ! WARNING: for NetCDF versions prior to 3.6 the largest variable must be declared last 
  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,"coefficients_of_wavefunctions",nf90_double, &
                  (/ cdimid, mngdimid, nsrcdimid, mnsdimid, nkdimid, nscdimid /),wavefunctionid)
  s = nf90_enddef(ncid)
  s = nf90_put_var(ncid,wavefunctionid,coefficients_of_wavefunctions)

end subroutine defput_coefficients_of_wavefunctions


subroutine write_denpot(ncid,&
         real_or_complex,number_of_grid_points_vector1,&
         number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components,&
         denpot,denpotname)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid
!dimensions
integer, intent(in) :: real_or_complex,number_of_grid_points_vector1
integer, intent(in) :: number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components
!data
double precision, intent(in) :: denpot(real_or_complex,number_of_grid_points_vector1,&
         number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components)
character(len=*), intent(in) :: denpotname
!local
integer :: cdimid, ng1dimid, ng2dimid, ng3dimid, ncdimid

!check for dimensions and put them if they are not there yet
call inqordef_dimid(ncid,"real_or_complex",real_or_complex,cdimid)
call inqordef_dimid(ncid,"number_of_grid_points_vector1",number_of_grid_points_vector1,ng1dimid)
call inqordef_dimid(ncid,"number_of_grid_points_vector2",number_of_grid_points_vector2,ng2dimid)
call inqordef_dimid(ncid,"number_of_grid_points_vector3",number_of_grid_points_vector3,ng3dimid)
call inqordef_dimid(ncid,"number_of_components",number_of_components,ncdimid)

! define and put data
call defput_denpot(ncid,cdimid, ng1dimid, ng2dimid, ng3dimid, ncdimid,&
     real_or_complex,number_of_grid_points_vector1,&
     number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components,&
     denpot,denpotname)
end subroutine write_denpot

subroutine defput_denpot(ncid,cdimid, ng1dimid, ng2dimid, ng3dimid, ncdimid,&
         real_or_complex,number_of_grid_points_vector1,&
         number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components,&
         denpot,denpotname)
!use netcdf
implicit none

!arguments
!ids
integer, intent(in) :: ncid,cdimid, ng1dimid, ng2dimid, ng3dimid, ncdimid
!dimensions
integer, intent(in) :: real_or_complex,number_of_grid_points_vector1
integer, intent(in) :: number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components
!data
double precision, intent(in) :: denpot(real_or_complex,number_of_grid_points_vector1,&
         number_of_grid_points_vector2,number_of_grid_points_vector3,number_of_components)
character(len=*), intent(in) :: denpotname
!local
integer :: s,denpotid

  s = nf90_redef(ncid)
  s = nf90_def_var(ncid,denpotname,nf90_double, &
                  (/ cdimid, ng1dimid, ng2dimid, ng3dimid, ncdimid /),denpotid)
  ! unit attribute is atomic units always
  !   for density this is electrons/bohr^3
  !   for potentials this is Ha/bohr^3
  s = nf90_put_att(ncid,denpotid,"units","atomic units")
  ! denpot data
  s = nf90_enddef(ncid)

  s = nf90_put_var(ncid,denpotid,denpot)
end subroutine defput_denpot


!------------------------------------------------------------------------
!   generic routine to search for a dimid and create it if
!     it does not exist yet
!------------------------------------------------------------------------
subroutine inqordef_dimid(ncid,dimname,dimval,dimid)

!use netcdf
implicit none

integer, intent(in) :: ncid,dimval
character(len=*), intent(in) :: dimname
integer, intent(out) :: dimid

!local
integer :: s

s = nf90_inq_dimid(ncid,dimname,dimid)
if(s /= nf90_noerr) then
  s = nf90_redef(ncid)
  s = nf90_def_dim(ncid,dimname,dimval,dimid)
  s = nf90_enddef(ncid)
end if

end subroutine inqordef_dimid

!------------------------------------------------------------------------
!     atomic routines for reading individual variables
!       dimensions are checked for in each case
!------------------------------------------------------------------------
subroutine read_primitive_vectors(ncid,NCD,NV,&
     primitive_vectors)

!use netcdf
implicit none

integer, intent(in) :: ncid,NCD,NV
double precision, intent(out) :: primitive_vectors(NCD,NV)

!local
integer :: s,pvid

s = nf90_inq_varid(ncid,"primitive_vectors",pvid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
s = nf90_get_var(ncid,pvid,primitive_vectors)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_primitive_vectors

subroutine read_number_of_grid_points_vector1(ncid,number_of_grid_points_vector1)

 !use netcdf
 implicit none

 integer, intent(in) :: ncid
 integer, intent(out) :: number_of_grid_points_vector1

 !local
 integer :: s,ngv1dimid
 
 s = nf90_inq_dimid(ncid,"number_of_grid_points_vector1",ngv1dimid)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
 s = nf90_inquire_dimension(ncid,ngv1dimid,len=number_of_grid_points_vector1)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_number_of_grid_points_vector1

subroutine read_number_of_grid_points_vector2(ncid,number_of_grid_points_vector2)

 !!use netcdf
 implicit none

 integer, intent(in) :: ncid
 integer, intent(out) :: number_of_grid_points_vector2

 !local
 integer :: s,ngv2dimid
 
 s = nf90_inq_dimid(ncid,"number_of_grid_points_vector2",ngv2dimid)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
 s = nf90_inquire_dimension(ncid,ngv2dimid,len=number_of_grid_points_vector2)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_number_of_grid_points_vector2

subroutine read_number_of_grid_points_vector3(ncid,number_of_grid_points_vector3)

 !use netcdf
 implicit none

 integer, intent(in) :: ncid
 integer, intent(out) :: number_of_grid_points_vector3

 !local
 integer :: s,ngv3dimid
 
 s = nf90_inq_dimid(ncid,"number_of_grid_points_vector3",ngv3dimid)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
 s = nf90_inquire_dimension(ncid,ngv3dimid,len=number_of_grid_points_vector3)
   if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_number_of_grid_points_vector3

subroutine read_denpottype(ncid,denpottype)

!use netcdf
implicit none

integer, intent(in) :: ncid
integer, intent(out) :: denpottype

!local
integer :: s,denpotid

  denpottype=-1
  s = nf90_inq_varid(ncid,"density",denpotid)
  if (s == nf90_noerr) then
    denpottype=1
  end if
  s = nf90_inq_varid(ncid,"exchange_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=2
  end if
  s = nf90_inq_varid(ncid,"correlation_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=3
  end if
  s = nf90_inq_varid(ncid,"exchange_correlation_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=4
  end if
  if (denpottype==-1) then
    write(*,*) ' error: none of the known data variables have been found amongst:'
    write(*,*) ' (1) density'
    write(*,*) ' (2) exchange_potential'
    write(*,*) ' (3) correlation_potential'
    write(*,*) ' (4) exchange_correlation_potential'
    stop 
  end if
end subroutine read_denpottype

subroutine read_denpot(ncid,rc,ngv1,ngv2,ngv3,nc,&
           denpot)

!use netcdf
implicit none

integer, intent(in) :: ncid,rc,ngv1,ngv2,ngv3,nc)
double precision, intent(out) :: denpot(rc,ngv1,ngv2,ngv3,nc)

!local
integer :: s,denpotid,denpottype

  denpottype=-1
  s = nf90_inq_varid(ncid,"density",denpotid)
  if (s == nf90_noerr) then
    denpottype=1
#ifdef CHECKS
    write(*,*) 'Found a density file'
#endif
  end if
  s = nf90_inq_varid(ncid,"exchange_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=2
#ifdef CHECKS
    write(*,*) 'Found an exchange_potential file'
#endif
  end if
  s = nf90_inq_varid(ncid,"correlation_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=3
#ifdef CHECKS
    write(*,*) 'Found a correlation_potential file'
#endif
  end if
  s = nf90_inq_varid(ncid,"exchange_correlation_potential",denpotid)
  if (s == nf90_noerr) then
    denpottype=4
#ifdef CHECKS
    write(*,*) 'Found an exchange_correlation_potential file'
#endif
  end if
  if (denpottype==-1) then
    write(*,*) ' error: none of the known data variables have been found amongst:'
    write(*,*) ' (1) density'
    write(*,*) ' (2) exchange_potential'
    write(*,*) ' (3) correlation_potential'
    write(*,*) ' (4) exchange_correlation_potential'
    stop 
  end if
  s = nf90_get_var(ncid,denpotid,denpot,count=(/rc,ngv1,ngv2,ngv3,nc/))
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_denpot

subroutine read_space_group(ncid,space_group)

!use netcdf
implicit none

integer, intent(in) :: ncid
integer, intent(out) :: space_group

integer :: s,sgvid

  s = nf90_inq_varid(ncid,"space_group",sgvid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,sgvid,space_group)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_space_group

subroutine read_atom_species(ncid,na,&
           atom_species)

!use netcdf
implicit none

integer, intent(in) :: ncid,na
integer, intent(out) :: atom_species(na) 

integer :: s,atid

  s = nf90_inq_varid(ncid,"atom_species",atid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,atid,atom_species)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_atom_species

subroutine read_reduced_atom_positions(ncid,NRD,na,&
           reduced_atom_positions)

!use netcdf
implicit none

integer, intent(in) :: ncid,NRD,na
double precision, intent(out) :: reduced_atom_positions(NRD,na)

integer :: s,apid

  s = nf90_inq_varid(ncid,"reduced_atom_positions",apid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,apid,reduced_atom_positions)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_reduced_atom_positions

subroutine read_reduced_symmetry_matrices(ncid,NRD,nsym,&
           reduced_symmetry_matrices)

!use netcdf
implicit none

integer, intent(in) :: ncid,NRD,nsym
integer, intent(out) :: reduced_symmetry_matrices(NRD,NRD,nsym)

integer :: s,symid

  s = nf90_inq_varid(ncid,"reduced_symmetry_matrices",symid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,symid,reduced_symmetry_matrices)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_reduced_symmetry_matrices

subroutine read_reduced_symmetry_translations(ncid,NRD,nsym,&
           reduced_symmetry_translations)

!use netcdf
implicit none

integer, intent(in) :: ncid,NRD,nsym
double precision, intent(out) :: reduced_symmetry_translations(NRD,nsym)

integer :: s,tsymid

  s = nf90_inq_varid(ncid,"reduced_symmetry_translations",tsymid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,tsymid,reduced_symmetry_translations)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_reduced_symmetry_translations

subroutine read_reduced_coordinates_of_kpoints(ncid,NRD,nk,&
           reduced_coordinates_of_kpoints)

!use netcdf
implicit none

integer, intent(in) :: ncid,NRD,nk
double precision, intent(out) :: reduced_coordinates_of_kpoints(NRD,nk)

integer :: s,kvid

  s = nf90_inq_varid(ncid,"reduced_coordinates_of_kpoints",kvid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,kvid,reduced_coordinates_of_kpoints)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_reduced_coordinates_of_kpoints

subroutine read_kpoint_weights(ncid,nk,&
           kpoint_weights)

!use netcdf
implicit none

integer, intent(in) :: ncid,nk
double precision, intent(out) :: kpoint_weights(nk)

integer :: s,kwid

  s = nf90_inq_varid(ncid,"kpoint_weights",kwid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,kwid,kpoint_weights)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_kpoint_weights

subroutine read_k_dependent_gvectors(ncid,k_dependent_gvectors)

  !use netcdf
  implicit none

  integer, intent(in) :: ncid
  integer, parameter :: YESLEN = 3
  character (len=YESLEN),intent(out) :: k_dependent_gvectors          ! = "no"  ! or "yes"

  ! local vars
  integer :: s,ngkid,tmplen

  s = nf90_inq_varid(ncid,"number_of_coefficients",ngkid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_inquire_attribute(ncid,ngkid,"k_dependent",len=tmplen)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,ngkid,"k_dependent",k_dependent_gvectors)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_k_dependent_gvectors

subroutine read_k_dependent_number_of_states(ncid,k_dependent_number_of_states)

  !use netcdf
  implicit none

  integer, intent(in) :: ncid
  integer, parameter :: YESLEN = 3
  character (len=YESLEN),intent(out) :: k_dependent_number_of_states          ! = "no"  ! or "yes"

  ! local vars
  integer :: s,nsid,tmplen

  s = nf90_inq_varid(ncid,"number_of_states",nsid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_inquire_attribute(ncid,nsid,"k_dependent",len=tmplen)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_att(ncid,nsid,"k_dependent",k_dependent_number_of_states)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_k_dependent_number_of_states

subroutine read_reduced_coordinates_of_plane_waves(ncid,NV,ng,&
           reduced_coordinates_of_plane_waves)

!use netcdf
implicit none

integer, intent(in) :: ncid,NV,ng
integer, intent(out) :: reduced_coordinates_of_plane_waves(NV,ng)

integer :: s,gvid

  s = nf90_inq_varid(ncid,"reduced_coordinates_of_plane_waves",gvid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,gvid,reduced_coordinates_of_plane_waves,count=(/NV,ng/))
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_reduced_coordinates_of_plane_waves

subroutine read_fermi_energy(ncid,fermi_energy)

!use netcdf
implicit none

integer, intent(in) :: ncid
double precision, intent(out) :: fermi_energy

integer :: s,fermienergyid

  s = nf90_inq_varid(ncid,"fermi_energy",fermienergyid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,fermienergyid,fermi_energy)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_fermi_energy

subroutine read_occupations(ncid,nb,nk,NS,&
           occupations)

!use netcdf
implicit none

integer, intent(in) :: ncid,nb,nk,NS
double precision, intent(out) :: occupations(nb,nk,NS)

integer :: s,occupationid

  s = nf90_inq_varid(ncid,"occupations",occupationid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,occupationid,occupations,count=(/nb,nk,NS/))
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_occupations

subroutine read_eigenvalues(ncid,nb,nk,NS,&
           eigenvalues)

!use netcdf
implicit none

integer, intent(in) :: ncid,nb,nk,NS
double precision, intent(out) :: eigenvalues(nb,nk,NS)

integer :: s,energyid

  s = nf90_inq_varid(ncid,"eigenvalues",energyid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,energyid,eigenvalues,count=(/nb,nk,NS/))
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_eigenvalues

subroutine read_gw_corrections(ncid,RC,nb,nk,NS,&
           gw_corrections)

!use netcdf
implicit none

integer, intent(in) :: ncid,RC,nb,nk,NS
double precision, intent(out) :: gw_corrections(RC,nb,nk,NS)

integer :: s,gwcorrid

  s = nf90_inq_varid(ncid,"gw_corrections",gwcorrid)
  if(s /= nf90_noerr) then
    gw_corrections = 0
  else  
    s = nf90_get_var(ncid,gwcorrid,gw_corrections,count=(/RC,nb,nk,NS/))
      if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  endif
end subroutine read_gw_corrections

subroutine read_kb_formfactor_sign(ncid,mnp,mnam,nas,&
           kb_formfactor_sign)

!use netcdf
implicit none

integer, intent(in) :: ncid,mnp,mnam,nas
integer, intent(out) :: kb_formfactor_sign(mnp,mnam,nas)

integer :: s,kbffsid

  s = nf90_inq_varid(ncid,"kb_formfactor_sign",kbffsid)
  if(s /= nf90_noerr) then
    kb_formfactor_sign = 0
  else  
    s = nf90_get_var(ncid,kbffsid,kb_formfactor_sign,count=(/mnp,mnam,nas/))
      if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  endif
end subroutine read_kb_formfactor_sign

subroutine read_kb_formfactors(ncid,nw,nk,mnp,mnam,nas,&
           kb_formfactors)

!use netcdf
implicit none

integer, intent(in) :: ncid,nw,nk,mnp,mnam,nas
double precision, intent(out) :: kb_formfactors(nw,nk,mnp,mnam,nas)

integer :: s,kbffid

  s = nf90_inq_varid(ncid,"kb_formfactors",kbffid)
  if(s /= nf90_noerr) then
    kb_formfactors = 0
  else  
    s = nf90_get_var(ncid,kbffid,kb_formfactors,count=(/nw,nk,mnp,mnam,nas/))
      if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  endif
end subroutine read_kb_formfactors

subroutine read_kb_formfactor_derivative(ncid,nw,nk,mnp,mnam,nas,&
           kb_formfactor_derivative)

!use netcdf
implicit none

integer, intent(in) :: ncid,nw,nk,mnp,mnam,nas
double precision, intent(out) :: kb_formfactor_derivative(nw,nk,mnp,mnam,nas)

integer :: s,kbffdid

  s = nf90_inq_varid(ncid,"kb_formfactor_derivative",kbffdid)
  if(s /= nf90_noerr) then
    kb_formfactor_derivative = 0
  else  
    s = nf90_get_var(ncid,kbffdid,kb_formfactor_derivative,count=(/nw,nk,mnp,mnam,nas/))
      if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  endif
end subroutine read_kb_formfactor_derivative

subroutine read_coefficients_of_wavefunctions(ncid,RC,nw,NSR,nb,nk,NS,&
           coefficients_of_wavefunctions)

!use netcdf
implicit none

integer, intent(in) :: ncid,RC,nw,NSR,nb,nk,NS
double precision, intent(out) :: coefficients_of_wavefunctions(RC,nw,NSR,nb,nk,NS)

integer :: s,wavefunctionid

  s = nf90_inq_varid(ncid,"coefficients_of_wavefunctions",wavefunctionid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_get_var(ncid,wavefunctionid,coefficients_of_wavefunctions,count=(/RC,nw,NSR,nb,nk,NS/))
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
end subroutine read_coefficients_of_wavefunctions

!------------------------------------------------------------------------
!     atomic routines for reading integer variables
!------------------------------------------------------------------------
subroutine read_number_of_electrons(ncid,number_of_electrons)

 !use netcdf
 implicit none
 
 integer, intent(in) :: ncid
 integer, intent(out) :: number_of_electrons
 
 integer :: s,nelid
 
 s = nf90_inq_varid(ncid,"number_of_electrons",nelid)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
 s = nf90_get_var(ncid,nelid,number_of_electrons)
  if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

end subroutine read_number_of_electrons

!-------------------------------------------------------------
!   routines to read in dimensions
!-------------------------------------------------------------
subroutine read_character_string_length(ncid,character_string_length)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: character_string_length
  call read_dim(ncid,"character_string_length",character_string_length)
end subroutine read_character_string_length

subroutine read_symbol_length(ncid,symbol_length)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: symbol_length
  call read_dim(ncid,"symbol_length",symbol_length)
end subroutine read_symbol_length

subroutine read_real_or_complex(ncid,real_or_complex)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: real_or_complex
  call read_dim(ncid,"real_or_complex",real_or_complex)
end subroutine read_real_or_complex

subroutine read_number_of_cartesian_directions(ncid,number_of_cartesian_directions)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_cartesian_directions
  call read_dim(ncid,"number_of_cartesian_directions",number_of_cartesian_directions)
end subroutine read_number_of_cartesian_directions

subroutine read_number_of_vectors(ncid,number_of_vectors)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_vectors
  call read_dim(ncid,"number_of_vectors",number_of_vectors)
end subroutine read_number_of_vectors

subroutine read_number_of_reduced_dimensions(ncid,number_of_reduced_dimensions)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_reduced_dimensions
  call read_dim(ncid,"number_of_reduced_dimensions",number_of_reduced_dimensions)
end subroutine read_number_of_reduced_dimensions

subroutine read_number_of_symmetry_operations(ncid,number_of_symmetry_operations)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_symmetry_operations
  call read_dim(ncid,"number_of_symmetry_operations",number_of_symmetry_operations)
end subroutine read_number_of_symmetry_operations

subroutine read_number_of_atom_species(ncid,number_of_atom_species)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_atom_species
  call read_dim(ncid,"number_of_atom_species",number_of_atom_species)
end subroutine read_number_of_atom_species

subroutine read_number_of_atoms(ncid,number_of_atoms)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_atoms
  call read_dim(ncid,"number_of_atoms",number_of_atoms)
end subroutine read_number_of_atoms

subroutine read_max_number_of_angular_momenta(ncid,max_number_of_angular_momenta)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: max_number_of_angular_momenta
  call read_dim(ncid,"max_number_of_angular_momenta",max_number_of_angular_momenta)
end subroutine read_max_number_of_angular_momenta

subroutine read_max_number_of_projectors(ncid,max_number_of_projectors)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: max_number_of_projectors
  call read_dim(ncid,"max_number_of_projectors",max_number_of_projectors)
end subroutine read_max_number_of_projectors

subroutine read_number_of_kpoints(ncid,number_of_kpoints)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_kpoints
  call read_dim(ncid,"number_of_kpoints",number_of_kpoints)
end subroutine read_number_of_kpoints

subroutine read_max_number_of_states(ncid,max_number_of_states)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: max_number_of_states
  call read_dim(ncid,"max_number_of_states",max_number_of_states)
end subroutine read_max_number_of_states

subroutine read_max_number_of_coefficients(ncid,max_number_of_coefficients)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: max_number_of_coefficients
  call read_dim(ncid,"max_number_of_coefficients",max_number_of_coefficients)
end subroutine read_max_number_of_coefficients

subroutine read_number_of_spinor_components(ncid,number_of_spinor_components)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_spinor_components
  call read_dim(ncid,"number_of_spinor_components",number_of_spinor_components)
end subroutine read_number_of_spinor_components

subroutine read_number_of_spins(ncid,number_of_spins)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_spins
  call read_dim(ncid,"number_of_spins",number_of_spins)
end subroutine read_number_of_spins

subroutine read_number_of_components(ncid,number_of_components)
  implicit none
  integer, intent(in) :: ncid
  integer, intent(out) :: number_of_components
  call read_dim(ncid,"number_of_components",number_of_components)
end subroutine read_number_of_components


!
!   generic routine to search for a dimid and read its value
!
subroutine read_dim(ncid,dimname,dimval)

!use netcdf
implicit none

integer, intent(in) :: ncid
character(len=*), intent(in) :: dimname
integer, intent(out) :: dimval

!local
integer :: s,dimid

  s = nf90_inq_dimid(ncid,dimname,dimid)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)
  s = nf90_inquire_dimension(ncid,dimid,len=dimval)
    if(s /= nf90_noerr) call handle_etsf_netcdf_err(s)

end subroutine read_dim

!===========================================================================
!   error handling routine
!===========================================================================
subroutine handle_etsf_netcdf_err(status)
  !use netcdf
  implicit none
  integer, intent(in) :: status
  if(status /= nf90_noerr) then
    print *, trim(nf90_strerror(status))
    stop 'error netcdf'
  endif
end subroutine
