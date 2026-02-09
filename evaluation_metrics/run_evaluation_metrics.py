from subprocess import Popen
import os
variables={
    2:['tos','psl','pr','tas'],
    3:['pr','psl','tas','monmaxtasmax','tos','mrso','siconc'] 
}
# Modify the list 
method_folders=[
    '../DMDc',
    '../DMDc-opt',
#    '../DMDc-p',
#    '../LR-Forcing',
]
method_names=[
    'DMDc',
    'DMDc-opt',    
#    'DMDc-p',
#    'LR-Forcing'
]

os.makedirs(f'logs',exist_ok=True)
str_method_folders=','.join(method_folders)
str_method_names=','.join(method_names)
for tier in [2,3]:
    for variable in variables[tier]:
        name=variable[:3]+str(tier)+'ev'
        command=f'srun -n 1 -J {name} python3 evaluation_metrics.py {str_method_folders} {str_method_names} {variable} {tier}'
        print(command)
        Popen(command+f' 1> logs/log-{name}.txt 2> logs/err-{name}.txt &',shell=True).communicate()


