from subprocess import Popen
#import os

variables={
    2:['tas','pr','psl','tos'],
    3:['pr','psl','tas','monmaxtasmax','tos','mrso','siconc']
}

cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxtasmax": "Aday",
    "mrso": "Lmon"
}

for tier in [2,3]:
    for variable in variables[tier]:
        input_path='/data/databases/ForceSMIP'
        output_path = f'Tier{tier}'
        ##output_path = os.path.join('./ForceSMIP',f'Evaluation-Tier{tier}',cmipTable[variable],variable)

        # SLURM run command
        command=f'srun -n 1 python3 dmdc_p.py {input_path} {output_path} {variable} {tier}'
        print(command)
        
        # Logging
        name=variable[:3]+str(tier)
        Popen(f'mkdir -p logsF',shell=True).communicate()
        Popen(command+f' 1> logsF/log-{name}.txt 2> logsF/err-{name}.txt &',shell=True).communicate()