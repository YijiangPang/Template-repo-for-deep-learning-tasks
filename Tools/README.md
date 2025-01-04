# convenient-slurm-commands
Check Tools.hpcc.sbatch. Before submit the task, estimate the two important parameters of the task “--mem=1000M” and “--time=00:05:00" in advance, which denote required memory and the total run time.
```
sbatch *.sbatch     #submit jobs
squeue -u <username>    #List all current jobs for a user
scancel <jobid>     #cancel one job
```


# Useful Ubuntu commands
```
df -h  #check disk space
du -cBG --max-depth=1 2> >(grep -v 'Permission denied') | sort -n  #check current folder space
```

# VScode update bug
```
Terminate vscode process -  “ps uxa | grep .vscode-server | awk '{print $2}' | xargs kill -9”  or  “pkill -9 vscode-server” 
Delete the vscode folder ~/?vscode?
```


# Useful notes

Using wget to download file from google drive
```
pip install gdown
gdown --id ID -O PATH
```
