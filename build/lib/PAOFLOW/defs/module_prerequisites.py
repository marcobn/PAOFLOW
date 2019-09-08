

report_pre_reqs = 'SUGGESTION: %s must be called before %s'

module_pre_reqs = { 'add_external_fields'      : ['projectability'],\
                    'pao_hamiltonian'          : ['projectability'],\
                    'z2_pack'                  : ['pao_hamiltonian'],\
                    'doubling_Hamiltonian'     : ['pao_hamiltonian'],\
                    'cutting_Hamiltonian'      : ['pao_hamiltonian'],\
                    'wave_function_projection' : ['pao_hamiltonian'],\
                    'bands'                    : ['pao_hamiltonian'],\
                    'topology'                 : ['bands', 'spin_operator'],\
                    'interpolated_hamiltonian' : ['pao_hamiltonian'],\
                    'pao_eigh'                 : ['pao_hamiltonian'],\
                    'dos'                      : ['pao_eigh'],\
                    'fermi_surface'            : ['pao_eigh'],\
                    'spin_texture'             : ['pao_eigh'],
                    'gradient_and_momenta'     : ['pao_eigh'],\
                    'adaptive_smearing'        : ['gradient_and_momenta'],\
                    'anomalous_Hall'           : ['gradient_and_momenta'],\
                    'spin_Hall'                : ['gradient_and_momenta', 'spin_operator'],\
                    'transport'                : ['gradient_and_momenta'],\
                    'dielectric_tensor'        : ['gradient_and_momenta'] }


key_error_strings = { 'HRs'  : 'HRs must first be calculated with \'build_pao_hamiltonian\'',\
                      'Hks'  : 'Hks must first be calculated with \'build_pao_hamiltonian\'',\
                      '1' : '2' }
