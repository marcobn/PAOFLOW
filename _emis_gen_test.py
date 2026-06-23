from src.PAOFLOW.gen import paoflow_driver as d

cfg = dict(
    properties=['dos', 'emissivity'],
    savedir='si.save',
    upfs=['Si.upf'],
    basisdir='BASIS',
    outputdir='output',
    npool=1,
    smearing='gauss',
    spin_orbit=False,
    std_basis='standard',
    ibrav=2,
    nk=30,
    emin=-6.0,
    emax=6.0,
    ne=1000,
    do_pdos=True,
    interpolate=False,
    nfft=0,
    use_intersite_v=False,
)

script = d.build_run_script(cfg)
if isinstance(script, (list, tuple)):
    script = '\n'.join(script)
compile(script, 'gen.py', 'exec')
print('OK: generated script compiles')
for ln in script.splitlines():
    if 'dielectric_tensor' in ln or 'emis' in ln.lower() or 'run_optical' in ln:
        print(ln)
